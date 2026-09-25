"""Stage 3: pair features, two-stage LightGBM matcher, threshold tuning, submission writing.

Stage 1 (per pair): blocking scores/ranks, rapidfuzz similarities on name / core
name / address, digit-token agreement, missingness, how common the name/address
is, and candidate-list context. Two stage-1 models are trained on disjoint S1
halves (A, B) so every training pair gets an out-of-fold probability p1.

Stage 2 (per pair, with context): stage-1 features + p1 + how p1 compares with the
S1's other candidates and with the best competing S1 for the same target.

Decision: keep pairs with p2 >= threshold, then each target goes only to its
highest-probability S1 (labels show every target belongs to at most one S1).
The threshold maximises exact macro F0.5 on the 'tune' half of the held-out S1s (with
competing non-hold S1s taking part in target ownership); the score is reported on the other
'final' half. Hold S1s are never used for fitting or for the lexicon.
"""
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .integrity import code_digest, digest, environment, files_digest, seal, verify, verify_stage
from .diagnostics import error_rows, summarize_errors
from .validation import load_truth, validate_pairs, validate_records

DIGITS = re.compile(r"\d+")

# Frequency of exact core name / address among targets and references (same country).
REC_SQL = """
CREATE TABLE rec AS
WITH r AS (SELECT * FROM read_parquet({glob}))
SELECT r.*,
  regexp_extract_all(r.addr, '[0-9]+') AS nums,
  sum((src > 1)::INT) OVER (PARTITION BY country, core) AS fc_t,
  sum((src = 1)::INT) OVER (PARTITION BY country, core) AS fc_s,
  CASE WHEN addr = '' THEN -1 ELSE sum((src > 1)::INT) OVER (PARTITION BY country, addr) END AS fa_t
FROM r
"""

GROUP_SQL = """
CREATE OR REPLACE TABLE c AS
SELECT *,
  count(*) OVER (PARTITION BY sid) AS n_c,
  s / max(s) OVER (PARTITION BY sid) AS s_rel,
  max(s) OVER (PARTITION BY sid) - s AS s_gap,
  count(*) OVER (PARTITION BY tid) AS t_nc,
  rank() OVER (PARTITION BY tid ORDER BY s DESC) AS t_rank,
  s / max(s) OVER (PARTITION BY tid) AS t_rel
FROM read_parquet({cands})
"""

BASE = ["s", "sn", "sa", "nk", "r_s", "r_n", "r_a", "n_c", "s_rel", "s_gap", "t_nc", "t_rank", "t_rel",
        "tsrc", "s_fct", "s_fcs", "s_fat", "t_fct", "t_fcs", "t_fat"]
FEATS = BASE + ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "c_ratio", "c_tset", "c_nospace",
                "c_exact", "a_ratio", "a_tsort", "a_tset", "a_partial", "t_addr_empty", "len_n_s", "len_n_t",
                "len_a_s", "len_a_t", "num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt",
                "t_num_cnt", "hn_ratio", "hn_rel", "raw_ratio", "raw_partial", "sup_core", "sup_addr", "skel_ratio",
                "hn_logdiff", "hn_sub1"]
CTX = ["p1", "p1_rank", "p1_max", "p1_sum", "p1_n05", "p1_rel", "p1_trank", "p1_tother", "p1_tdiff", "p1_tn05",
       "p1_srank", "p1_sn05", "p1_smax", "anc_name", "anc_core", "anc_addr", "anc_n"]
FEATS2 = FEATS + CTX
ANCHOR_P, ANCHOR_K, ANCHOR_MIN = 0.3, 4, 0.02
# Early rejection reduces stage-2 work, not the reported candidate set. Organizer clarification
# rows 133/152 requires the input to the FIRST scoring model, including subsequently rejected pairs.
CAND_P = 0.01

CTX_SQL = """
SELECT sid, tid, p1,
  rank() OVER (PARTITION BY sid ORDER BY p1 DESC) AS p1_rank,
  max(p1) OVER (PARTITION BY sid) AS p1_max,
  sum(p1) OVER (PARTITION BY sid) AS p1_sum,
  sum((p1 > 0.5)::INT) OVER (PARTITION BY sid) AS p1_n05,
  rank() OVER (PARTITION BY tid ORDER BY p1 DESC) AS p1_trank,
  max(p1) OVER (PARTITION BY tid) AS tmax,
  nth_value(p1, 2) OVER (PARTITION BY tid ORDER BY p1 DESC, sid
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS t2,
  sum((p1 > 0.5)::INT) OVER (PARTITION BY tid) AS p1_tn05,
  rank() OVER (PARTITION BY sid, r.src ORDER BY p1 DESC) AS p1_srank,
  sum((p1 > 0.5)::INT) OVER (PARTITION BY sid, r.src) AS p1_sn05,
  max(p1) OVER (PARTITION BY sid, r.src) AS p1_smax
FROM p1tab JOIN rec r ON r.rid = p1tab.tid
"""


def _num_feats(sa, ta):
    """Digit-token agreement; hn_* compare the first number (house/plot number): typo vs neighbour.
    hn_logdiff = log1p |a - b|, hn_sub1 = same length and exactly one digit differs (typo-like, while
    decoys shift the number by a small amount: 353 vs 355)."""
    out = np.zeros((len(sa), 10), np.float32)
    for i, (x, y) in enumerate(zip(sa, ta)):
        a, b = DIGITS.findall(x), DIGITS.findall(y)
        sa_, sb_ = set(a), set(b)
        inter = len(sa_ & sb_)
        hn = (fuzz.ratio(a[0], b[0]) / 100, abs(int(a[0][:9]) - int(b[0][:9])) / max(int(a[0][:9]), int(b[0][:9]), 1))             if a and b else (-1, -1)
        out[i] = (inter / len(sa_ | sb_) if sa_ and sb_ else -1, inter,
                  float(bool(sa_) and bool(sb_) and not inter),
                  float(bool(a) and bool(b) and a[0] == b[0]), len(a), len(b), *hn,
                  *((np.log1p(abs(int(a[0][:9]) - int(b[0][:9]))),
                     float(len(a[0]) == len(b[0]) and sum(p != q for p, q in zip(a[0], b[0])) == 1))
                    if a and b else (-1, -1)))
    return out


def features(df, threads=1):
    """Stage-1 feature matrix (float32, column order FEATS) for joined candidate rows."""
    sn, tn = df["sname"].tolist(), df["tname"].tolist()
    sc, tc = df["scor"].tolist(), df["tcore"].tolist()
    sa, ta = df["saddr"].tolist(), df["taddr"].tolist()
    d = lambda a, b, f: cpdist(a, b, scorer=f, workers=threads or 1, dtype=np.float32)  # noqa: E731
    cols = {c: df[c].to_numpy(np.float32) for c in BASE}
    cols.update(n_ratio=d(sn, tn, fuzz.ratio), n_tsort=d(sn, tn, fuzz.token_sort_ratio),
                n_tset=d(sn, tn, fuzz.token_set_ratio), n_partial=d(sn, tn, fuzz.partial_ratio),
                n_jw=d(sn, tn, JaroWinkler.normalized_similarity), c_ratio=d(sc, tc, fuzz.ratio),
                c_tset=d(sc, tc, fuzz.token_set_ratio),
                c_nospace=d(df["scor_ns"].tolist(), df["tcor_ns"].tolist(), fuzz.ratio),
                a_ratio=d(sa, ta, fuzz.ratio), a_tsort=d(sa, ta, fuzz.token_sort_ratio),
                a_tset=d(sa, ta, fuzz.token_set_ratio), a_partial=d(sa, ta, fuzz.partial_ratio))
    rs, rt = df["srawn"].tolist(), df["trawn"].tolist()
    cols.update(raw_ratio=d(rs, rt, fuzz.ratio), raw_partial=d(rs, rt, fuzz.partial_ratio))
    ks, kt = df["sskel"].tolist(), df["tskel"].tolist()
    skel = d(ks, kt, fuzz.ratio)
    skel[(df["sskel"].to_numpy() == "") | (df["tskel"].to_numpy() == "")] = -1
    cols["skel_ratio"] = skel
    # Computed in SQL (_pairs): exact core, empty target address, lengths, cluster support
    # (true matches come in groups of near-identical variants; decoys stand alone), digit-token counts.
    for k in ("c_exact", "t_addr_empty", "len_n_s", "len_n_t", "len_a_s", "len_a_t", "sup_core", "sup_addr",
              "num_common", "s_num_cnt", "t_num_cnt"):
        cols[k] = df[k].to_numpy(np.float32)
    # Digit tokens: set agreement; hn_* compare the first number (house/plot number): typo vs neighbour.
    sd, td, inter = df["s_nd"].to_numpy(np.float64), df["t_nd"].to_numpy(np.float64), cols["num_common"]
    both = (sd > 0) & (td > 0)
    cols["num_jac"] = np.where(both, inter / np.maximum(sd + td - inter, 1), -1).astype(np.float32)
    cols["num_conflict"] = (both & (inter == 0)).astype(np.float32)
    shn, thn = df["s_hn"].to_numpy(object), df["t_hn"].to_numpy(object)
    has = (shn != "") & (thn != "")
    cols["num_first_eq"] = (has & (shn == thn)).astype(np.float32)
    cols["hn_ratio"] = np.where(has, d(df["s_hn"].tolist(), df["t_hn"].tolist(), fuzz.ratio) / 100, -1).astype(np.float32)
    sv = pd.to_numeric(df["s_hv"]).to_numpy(np.float64, na_value=0)
    tv = pd.to_numeric(df["t_hv"]).to_numpy(np.float64, na_value=0)
    diff = np.abs(sv - tv)
    cols["hn_rel"] = np.where(has, diff / np.maximum(np.maximum(sv, tv), 1), -1).astype(np.float32)
    cols["hn_logdiff"] = np.where(has, np.log1p(diff), -1).astype(np.float32)
    cols["hn_sub1"] = np.where(has, df["hn_one"].to_numpy(bool), -1).astype(np.float32)
    return np.column_stack([cols[c] for c in FEATS])


def _pairs(con, where):
    """Candidate rows (with group features) joined to both records' normalized text."""
    return con.execute(f"""
        SELECT c.*, a.name AS sname, a.core AS scor, a.addr AS saddr, a.rawn AS srawn, a.skel AS sskel,
               b.name AS tname, b.core AS tcore, b.addr AS taddr, b.src AS tsrc, b.rawn AS trawn, b.skel AS tskel,
               a.fc_t AS s_fct, a.fc_s AS s_fcs, a.fa_t AS s_fat,
               b.fc_t AS t_fct, b.fc_s AS t_fcs, b.fa_t AS t_fat,
               (a.core = b.core)::FLOAT AS c_exact, (b.addr = '')::FLOAT AS t_addr_empty,
               length(a.name)::FLOAT AS len_n_s, length(b.name)::FLOAT AS len_n_t,
               length(a.addr)::FLOAT AS len_a_s, length(b.addr)::FLOAT AS len_a_t,
               -- cluster support within this reference's candidate list (all its candidates are in one slice)
               (count(*) OVER (PARTITION BY c.sid, b.core))::FLOAT AS sup_core,
               (CASE WHEN b.addr = '' THEN -1 ELSE count(*) OVER (PARTITION BY c.sid, b.addr) END)::FLOAT AS sup_addr,
               replace(a.core, ' ', '') AS scor_ns, replace(b.core, ' ', '') AS tcor_ns,
               len(list_distinct(a.nums)) AS s_nd, len(list_distinct(b.nums)) AS t_nd,
               len(list_intersect(list_distinct(a.nums), list_distinct(b.nums)))::FLOAT AS num_common,
               len(a.nums)::FLOAT AS s_num_cnt, len(b.nums)::FLOAT AS t_num_cnt,
               coalesce(a.nums[1], '') AS s_hn, coalesce(b.nums[1], '') AS t_hn,
               TRY_CAST(left(a.nums[1], 9) AS DOUBLE) AS s_hv, TRY_CAST(left(b.nums[1], 9) AS DOUBLE) AS t_hv,
               coalesce(length(a.nums[1]) = length(b.nums[1]) AND levenshtein(a.nums[1], b.nums[1]) = 1, false) AS hn_one
        FROM c JOIN rec a ON a.rid = c.sid JOIN rec b ON b.rid = c.tid WHERE {where}
        ORDER BY c.sid, c.tid""").df()


def lit(value):
    """SQL string literal with quotes escaped (paths may contain apostrophes)."""
    return "'" + str(value).replace("'", "''") + "'"


def normalized_sql(directory):
    """Explicit three-file parquet expression; never ingest an unmanifested wildcard match."""
    return "[" + ",".join(lit(Path(directory) / f"norm_s{i}.parquet") for i in (1,2,3)) + "]"


def configure(con, mem, threads, temp_dir):
    """Validated DuckDB resource settings."""
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", str(mem)):
        raise ValueError(f"memory limit must look like '8GB', got {mem!r}")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError(f"threads must be a positive int, got {threads!r}")
    con.execute(f"SET memory_limit={lit(mem)}; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={lit(temp_dir)}")
    if threads:
        con.execute(f"SET threads={threads}")


def scratch(work):
    """Directory for spill/temporary files: $FASTER_TMP if set (e.g. /tmp on Kaggle), else the work dir."""
    return os.environ.get("FASTER_TMP") or str(work)


def _connect(work, split, mem, threads):
    """Load a verified stage; reject incompatible schemas and candidate endpoints."""
    verify_stage(work, split)
    con = duckdb.connect()
    configure(con, mem, threads, Path(scratch(work)) / "duck_tmp")
    w = Path(work) / split
    con.execute(REC_SQL.format(glob=normalized_sql(w)))
    validate_records(con)
    columns = {r[0] for r in con.execute("DESCRIBE rec").fetchall()}
    if not {"rawn", "skel"}.issubset(columns):
        raise ValueError("Prepared data lacks raw-name/skeleton features; rebuild")
    if con.execute("SELECT count(*) FROM rec WHERE skel IS NULL").fetchone()[0]:
        raise ValueError("skeleton missing for some records")
    con.execute(GROUP_SQL.format(cands=lit(w / "cands.parquet")))
    validate_pairs(con, "c")
    return con


def _context(con, sid, tid, p1, anchor_where="TRUE", chunks=8, threads=1):
    """Stage-2 context columns (order CTX) aligned to (sid, tid) via a DuckDB window pass."""
    check_scores(p1)
    con.register("p1df", pd.DataFrame({"sid": sid, "tid": tid, "p1": p1}))
    con.execute("CREATE OR REPLACE TABLE p1tab AS SELECT * FROM p1df")
    con.unregister("p1df")
    con.execute(f"CREATE OR REPLACE TABLE ctx AS {CTX_SQL}")
    _anchors(con, anchor_where, chunks, threads)
    return con


def _anchors(con, where, chunks, threads=1):
    """Cluster consistency: a reference's true matches are variants of each other, while a decoy
    (another business at the same address) is not. For each plausible pair (p1 >= ANCHOR_MIN), compare
    the target with the S1's other strongest candidates (top ANCHOR_K with p1 >= ANCHOR_P) and keep the
    best name / core-name / address similarity and the number of such anchors. Table `anc`."""
    con.execute(f"""CREATE OR REPLACE TABLE anchors AS SELECT sid, tid AS aid FROM (
        SELECT sid, tid, p1, row_number() OVER (PARTITION BY sid ORDER BY p1 DESC, tid) AS r FROM p1tab)
        WHERE r <= {ANCHOR_K} AND p1 >= {ANCHOR_P}""")
    parts = []
    for k in range(chunks):
        df = con.execute(f"""
            SELECT q.sid, q.tid, t.name AS tn, t.core AS tc, t.addr AS ta, x.name AS an, x.core AS ac, x.addr AS aa
            FROM p1tab q JOIN anchors a ON a.sid = q.sid AND a.aid <> q.tid
            JOIN rec t ON t.rid = q.tid JOIN rec x ON x.rid = a.aid
            WHERE q.p1 >= {ANCHOR_MIN} AND q.sid % {chunks} = {k} AND ({where})""").df()
        if not len(df):
            continue
        d = lambda a, b, f: cpdist(a, b, scorer=f, workers=threads or 1, dtype=np.float32)  # noqa: E731
        addr = d(df.ta.tolist(), df.aa.tolist(), fuzz.ratio)
        addr[(df.ta.to_numpy() == "") | (df.aa.to_numpy() == "")] = -1
        g = pd.DataFrame({"sid": df.sid, "tid": df.tid, "n": d(df.tn.tolist(), df.an.tolist(), fuzz.token_set_ratio),
                          "c": d(df.tc.tolist(), df.ac.tolist(), fuzz.ratio), "a": addr})
        parts.append(g.groupby(["sid", "tid"], sort=False).agg(anc_name=("n", "max"), anc_core=("c", "max"),
                                                              anc_addr=("a", "max"), anc_n=("n", "size")).reset_index())
    anc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"sid": [], "tid": [], "anc_name": [], "anc_core": [], "anc_addr": [], "anc_n": []})
    con.register("ancdf", anc)
    con.execute("CREATE OR REPLACE TABLE anc AS SELECT * FROM ancdf")
    con.unregister("ancdf")


def _ctx_matrix(con, keys):
    """Fetch context rows for keys (DataFrame sid, tid) in keys' order."""
    con.register("k", keys[["sid", "tid"]].reset_index(drop=True).assign(ix=np.arange(len(keys))))
    m = con.execute("""
        SELECT x.p1, x.p1_rank, x.p1_max, x.p1_sum, x.p1_n05, coalesce(x.p1 / nullif(x.p1_max, 0), 0) AS p1_rel, x.p1_trank,
               CASE WHEN x.p1_trank = 1 THEN coalesce(x.t2, 0) ELSE x.tmax END AS p1_tother,
               x.p1 - CASE WHEN x.p1_trank = 1 THEN coalesce(x.t2, 0) ELSE x.tmax END AS p1_tdiff, x.p1_tn05,
               x.p1_srank, x.p1_sn05, x.p1_smax,
               coalesce(n.anc_name, -1) AS anc_name, coalesce(n.anc_core, -1) AS anc_core,
               coalesce(n.anc_addr, -1) AS anc_addr, coalesce(n.anc_n, 0) AS anc_n
        FROM k JOIN ctx x USING (sid, tid) LEFT JOIN anc n USING (sid, tid) ORDER BY k.ix""").fetchnumpy()
    con.unregister("k")
    result = np.column_stack([m[c].astype(np.float32) for c in CTX])
    if len(result) != len(keys) or not np.isfinite(result).all():
        raise ValueError("Context keys missing/duplicated or non-finite features")
    return result


def check_scores(p, thr=None, name="scores"):
    """Probabilities must be finite and in [0,1]; NaN would otherwise pass a SQL `p >= thr` filter."""
    p = np.asarray(p, dtype=np.float64)
    if p.size and not (np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0):
        raise ValueError(f"{name}: non-finite or out-of-range probabilities")
    if thr is not None and not (np.isfinite(thr) and 0.0 <= thr <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {thr!r}")


def decide(sid, tid, p, thr):
    """Threshold, then give each target only to its best S1 (ties: lowest sid, same rule as the SQL
    writer). Returns {sid: [tid,...]}."""
    check_scores(p, thr)
    df = pd.DataFrame({"sid": sid, "tid": tid, "p": p})
    df = df[df.p >= thr].sort_values(["p", "sid"], ascending=[False, True], kind="mergesort").drop_duplicates("tid")
    return df.groupby("sid")["tid"].apply(list).to_dict()


def macro_f05(truth, pred, all_sids):
    """Exact per-reference macro F0.5, retaining empty and unretrieved references."""
    if not len(all_sids) or len(set(all_sids)) != len(all_sids):
        raise ValueError("macro_f05 needs at least one reference")
    tot = 0.0
    for s in all_sids:
        t, q = truth.get(s, set()), set(pred.get(s, ()))
        if not t:
            tot += not q
        elif q:
            tp = len(t & q)
            tot += 5 * tp / (5 * tp + 4 * (len(q) - tp) + (len(t) - tp))
    return tot / len(all_sids)


def hold_rule(col, hold_mod):
    """SQL predicate selecting holdout S1s (~1/hold_mod of references)."""
    if type(hold_mod) is not int or hold_mod < 2:
        raise ValueError(f"hold_mod must be an int >= 2, got {hold_mod!r}")
    return f"(md5_number_lower({col} || 'hold26') % {hold_mod} = 0)"


def _params(threads, seed, leaves=127, lr=0.05):
    """Return fixed-round CPU settings; reject invalid resources before fitting."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(leaves) is not int or leaves < 2 or not np.isfinite(lr) or not 0 < lr <= 1:
        raise ValueError("leaves must be >= 2 and learning rate in (0,1]")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("threads must be a positive integer")
    return dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=threads or 1, verbose=-1, seed=seed, deterministic=True, force_col_wise=True)




_GPU = None

def gpu_available():
    """True if XGBoost can train on a CUDA GPU on this machine (checked once with a tiny fit)."""
    global _GPU
    if _GPU is None:
        try:
            import xgboost as xgb
            d = xgb.DMatrix(np.random.default_rng(26).random((256, 4)).astype(np.float32), label=np.arange(256) % 2)
            probe = xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0, "nthread": 1}, d, 2)
            device = json.loads(probe.save_config())["learner"]["generic_param"]["device"]
            _GPU = device.startswith("cuda")
        except Exception:
            _GPU = False
    return _GPU


def resolve_backend(backend):
    """'auto' -> 'xgb' when a CUDA GPU is usable (Kaggle T4, local RTX), else 'lgb' on CPU."""
    if backend == "auto":
        return "xgb" if gpu_available() else "lgb"
    if backend not in ("xgb", "lgb"):
        raise ValueError(f"backend must be auto|xgb|lgb, got {backend!r}")
    return backend


def _xgb_params(P, device):
    """XGBoost settings equivalent to the LightGBM parameters in P."""
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": device,
            "learning_rate": P["learning_rate"], "max_leaves": P["num_leaves"], "grow_policy": "lossguide",
            "max_depth": 0, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "max_bin": 256, "seed": P["seed"], "nthread": P["num_threads"] or 1, "verbosity": 1}


def _cupy():
    """CuPy module when installed and a CUDA GPU is usable, else None (then XGBoost copies host data)."""
    if not gpu_available():
        return None
    try:
        import cupy
        return cupy
    except ImportError:
        return None


def _xgb_train(P, make, n, rounds):
    """Train on the GPU when usable; if the GPU runs out of memory, rebuild the matrix on the host and
    train on CPU. `make(gpu)` builds the quantized matrix (built directly in GPU memory
    from CuPy blocks when gpu is True). No evaluation set (it only cost GPU memory)."""
    import xgboost as xgb
    device = "cuda" if gpu_available() else "cpu"
    try:
        d = make(device == "cuda")
        cp = _cupy()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()  # staging blocks must not starve XGBoost
        booster = xgb.train(_xgb_params(P, device), d, rounds)
        booster.set_attr(training_device=model_device(booster), cpu_retry="false")
        return booster
    except Exception as e:  # XGBoostError or CuPy OutOfMemoryError
        if device != "cuda" or not is_memory_failure(e):
            raise
        print(f"  [xgboost] GPU out of memory ({n:,} rows); retrying this booster on CPU", flush=True)
        d = None
        cp = _cupy()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        booster = xgb.train(_xgb_params(P, "cpu"), make(False), rounds)
        booster.set_attr(training_device=model_device(booster), cpu_retry="true")
        return booster


def _fit(backend, P, X, y, w, names, rounds, valid=None):
    """Train one booster: XGBoost (GPU if available, CPU otherwise) or LightGBM (CPU)."""
    if backend == "lgb":
        vs = [lgb.Dataset(valid[0], valid[1], feature_name=list(names))] if valid is not None else []
        return lgb.train(P, lgb.Dataset(X, y, feature_name=list(names), weight=w), rounds, valid_sets=vs,
                         callbacks=[lgb.log_evaluation(100)] if vs else [])
    import xgboost as xgb
    return _xgb_train(P, lambda gpu: xgb.QuantileDMatrix(X, y, weight=w, feature_names=list(names),
                      nthread=P["num_threads"] or 1), len(y), rounds)


def _fit_parts(backend, P, loaders, y, w, names, rounds):
    """_fit for training rows kept on disk: `loaders` return consecutive row blocks matching y / w.
    QuantileDMatrix still holds the combined quantized data in memory: this is not external-memory
    training or a guarantee that full data fits. LightGBM concatenates the float matrices."""
    if backend == "lgb":
        return _fit(backend, P, np.vstack([f() for f in loaders]), y, w, names, rounds)
    import xgboost as xgb

    class Blocks(xgb.DataIter):
        """Repeatable feature batches aligned with labels and optional reference weights."""

        def __init__(self, cp):
            """Initialize one repeatable batch iterator with an optional GPU array module."""
            self.i = self.off = 0
            self.cp = cp
            super().__init__()

        def next(self, input_data):
            """Supply one aligned feature/label/weight block to XGBoost."""
            if self.i == len(loaders):
                return False
            X = loaders[self.i]()
            n = len(X)
            if X.ndim != 2 or X.shape[1] != len(names) or self.off + n > len(y):
                raise ValueError("Streamed feature shape does not match labels/schema")
            lab = y[self.off:self.off + n].astype(np.float32)
            wt = None if w is None else w[self.off:self.off + n]
            if self.cp is not None:  # build the quantized matrix directly in GPU memory
                X, lab = self.cp.asarray(X), self.cp.asarray(lab)
                wt = None if wt is None else self.cp.asarray(wt)
            input_data(data=X, label=lab, feature_names=list(names), weight=wt)
            self.i, self.off = self.i + 1, self.off + n
            return True

        def reset(self):
            """Rewind and verify that a completed pass consumed every label."""
            if self.i == len(loaders) and self.off != len(y):
                raise ValueError("Streamed rows do not match the labels")
            self.i = self.off = 0

    return _xgb_train(P, lambda gpu: xgb.QuantileDMatrix(Blocks(_cupy() if gpu else None), max_bin=256,
                      nthread=P["num_threads"] or 1), len(y), rounds)


def model_device(booster):
    """Read the actual booster device, never infer GPU execution from hardware availability."""
    if isinstance(booster, lgb.Booster):
        return "cpu"
    return json.loads(booster.save_config())["learner"]["generic_param"]["device"]


def is_memory_failure(error):
    """Recognize allocation failures without masking unrelated model or schema errors."""
    import xgboost as xgb
    eligible = isinstance(error, (MemoryError, xgb.core.XGBoostError)) or type(error).__module__.startswith("cupy")
    message = str(error).lower()
    return eligible and (isinstance(error, MemoryError) or any(t in message for t in
        ("out of memory", "outofmemory", "bad allocation", "memoryallocation", "memory allocation", "alloc_failed")))


def _pred(m, X, threads=None):
    """Predict with bounded GPU transfers; preserve CPU fallback for the remaining batches."""
    if isinstance(m, lgb.Booster):
        return m.predict(X, num_threads=threads or 1)
    cp = _cupy() if model_device(m).startswith("cuda") else None
    if cp is not None and len(X):
        out = np.empty(len(X), np.float64)
        for start in range(0, len(X), 65536):
            block = None
            stop = min(start+65536, len(X))
            try:
                if cp is None:
                    out[start:stop] = m.inplace_predict(X[start:stop])
                else:
                    block = cp.asarray(X[start:stop])
                    out[start:stop] = cp.asnumpy(m.inplace_predict(block))
            except Exception as error:
                if cp is None or not is_memory_failure(error):
                    raise
                block = None
                cp.get_default_memory_pool().free_all_blocks()
                m.set_param({"device":"cpu"})
                cp = None
                out[start:stop] = m.inplace_predict(X[start:stop])
            finally:
                block = None
                if cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
        return out
    return np.asarray(m.inplace_predict(X), dtype=np.float64)


def _names(m):
    """Ordered feature names a saved booster was trained with."""
    return m.feature_name() if isinstance(m, lgb.Booster) else list(m.feature_names or [])


def _save(m, base):
    """Save a booster as BASE.txt (LightGBM) or BASE.json (XGBoost)."""
    m.save_model(str(base) + (".txt" if isinstance(m, lgb.Booster) else ".json"))


def _load(base):
    """Load BASE.json (XGBoost, placed on the GPU when available) or BASE.txt (LightGBM)."""
    js, txt = Path(str(base) + ".json"), Path(str(base) + ".txt")
    if js.exists():
        import xgboost as xgb
        b = xgb.Booster(model_file=str(js))
        trained = b.attr("training_device") or "cpu"
        b.set_param({"device": "cuda" if trained.startswith("cuda") and gpu_available() else "cpu"})
        return b
    return lgb.Booster(model_file=str(txt))


def _importance(m, names):
    """Gain importance per feature for either booster type."""
    if isinstance(m, lgb.Booster):
        return dict(zip(names, m.feature_importance("gain").round().tolist()))
    g = m.get_score(importance_type="gain")
    return {n: round(float(g.get(n, 0.0)), 1) for n in names}


def pair_weights(sid, mode):
    """Per-pair training weights. 'none': every pair 1. 'ref': each reference's pairs sum to a
    constant (the metric averages per reference); 'sqrt': halfway (1/sqrt(n_candidates)). Mean 1."""
    if mode not in ("none", "ref", "sqrt"):
        raise ValueError("weight must be none|ref|sqrt")
    if mode == "none":
        return None
    if not len(sid):
        raise ValueError("Cannot weight an empty reference list")
    n = pd.Series(sid).map(pd.Series(sid).value_counts()).to_numpy(np.float64)
    w = 1.0 / n if mode == "ref" else 1.0 / np.sqrt(n)
    return (w / w.mean()).astype(np.float32)


def _predict_stage1(con, models, where, threads, chunks, tmp=None, tag=""):
    """Featurize candidate rows matching `where` chunk by chunk; return keys + averaged p1.
    If tmp is given, stage-1 matrices are cached there as .npy for stage 2."""
    keys, p1 = [], []
    for k in range(chunks):
        df = _pairs(con, f"({where}) AND c.sid % {chunks} = {k}")
        if not len(df):
            continue
        X = features(df, threads)
        p1.append(np.mean([_pred(m, X, threads) for m in models], axis=0))
        keys.append(df[["sid", "tid"]])
        if tmp is not None:
            np.save(Path(tmp) / f"{tag}{k}.npy", X)
        print(f"  [stage1 {tag}] chunk {k + 1}/{chunks}: {len(df):,} pairs", flush=True)
    if not keys:
        return pd.DataFrame({"sid": [], "tid": []}, dtype="int32"), np.empty(0)
    probabilities = np.concatenate(p1)
    check_scores(probabilities, name="stage1")
    return pd.concat(keys, ignore_index=True), probabilities


def _check_args(n_train, hold_mod, chunks, rounds):
    """Reject nonsensical split sizes and work settings before reading data."""
    for name, v, lo in (("n_train", n_train, 2), ("hold_mod", hold_mod, 2), ("chunks", chunks, 1), ("rounds", rounds, 1)):
        if type(v) is not int or v < lo:
            raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")


def _load_truth(con, gt_path):
    """Compatibility entry point for the shared label-integrity gate."""
    return load_truth(con, gt_path)


def _score_all(con, models, m2, threads, chunks, tmp_parent):
    """Return every first-model input; early-rejected pairs have a cascade decision score of zero."""
    Path(tmp_parent).mkdir(parents=True, exist_ok=True)
    own = Path(tempfile.mkdtemp(prefix="fe_", dir=tmp_parent))
    try:
        keys, p1 = _predict_stage1(con, models, "TRUE", threads, chunks, tmp=own, tag="x")
        keep = p1 >= CAND_P  # learned candidate filter: only these pairs reach the matcher
        kept = keys[keep].reset_index(drop=True)
        _context(con, kept.sid.values, kept.tid.values, p1[keep], chunks=chunks, threads=threads)
        p2, off = np.zeros(len(keys), dtype=np.float64), 0
        for k in range(chunks):
            path = own / f"x{k}.npy"
            if not path.exists():
                continue
            X = np.load(path, allow_pickle=False)
            s = keep[off:off + len(X)]
            kk = keys.iloc[off:off + len(X)][s]
            if s.any():
                p2[off:off + len(X)][s] = _pred(m2, np.hstack([X[s], _ctx_matrix(con, kk)]), threads)
            off += len(X)
        if off != len(keys):
            raise ValueError("Stage-2 candidate coverage mismatch")
        check_scores(p2)
        return keys, p1, p2
    finally:
        shutil.rmtree(own)


def evaluate_threshold(sid, tid, p, truth, tune_sids, final_sids, grid=None):
    """Choose with tune labels, then evaluate final labels exactly once."""
    if not tune_sids or not final_sids or set(tune_sids) & set(final_sids):
        raise ValueError("Tune and final sets must be nonempty and disjoint")
    refine = grid is None  # an explicit grid is used exactly as given
    grid = grid if grid is not None else [round(float(t), 3) for t in np.arange(.2, .96, .025)]
    scores = {t: macro_f05(truth, decide(sid, tid, p, t), tune_sids) for t in grid}
    best = max(scores, key=lambda t: (scores[t], t))
    for t in (np.arange(max(0.0, best - 0.02), min(1.0, best + 0.02) + 1e-9, 0.0025) if refine else []):
        t = round(float(t), 4)  # refinement around the best coarse value, tune labels only
        if t not in scores:
            scores[t] = macro_f05(truth, decide(sid, tid, p, t), tune_sids)
    threshold = max(scores, key=lambda t: (scores[t], t))
    final = macro_f05(truth, decide(sid, tid, p, threshold), final_sids)
    return {"threshold": threshold, "tune_macro_f05": scores[threshold],
            "final_macro_f05": final, "grid_tune": scores}


def model_path(work, name):
    """Resolve one model/ensemble subdirectory without allowing traversal or absolute paths."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Model name must be a single safe directory name")
    result = Path(work) / name
    if result.is_symlink():
        raise ValueError("Linked model directories are not supported")
    return result


def train(work, gt_path, n_train=10_000, hold_mod=22, mem="4GB", threads=1, seed=26, rounds=600,
          chunks=8, leaves=127, lr=0.05, weight="none", backend="lgb", model_dir="model", stage2_all=False):
    """Fit classifiers and stacker, then evaluate disjoint tune/final reference groups.

    Stacker inputs use classifier-level cross-fitting. The learned lexicon uses
    non-hold references, so this is not full-pipeline cross-fitting. Every training
    candidate is featurized once; the stage-1 matrices are cached in a run-owned scratch
    directory (removed afterwards), streamed into the boosters and reused for the production-scoring
    evaluation pass. Stage 2 uses the same bounded reference sample unless stage2_all is explicit.
    Complete-graph context/evaluation still scans all candidates and needs measured resources.
    """
    Path(scratch(work)).mkdir(parents=True, exist_ok=True)
    own = Path(tempfile.mkdtemp(prefix="fe_train_", dir=scratch(work)))
    try:
        return _train(work, gt_path, n_train, hold_mod, mem, threads, seed, rounds, chunks, leaves, lr,
                      weight, backend, model_dir, own, stage2_all)
    finally:
        shutil.rmtree(own, ignore_errors=True)


def _train(work, gt_path, n_train, hold_mod, mem, threads, seed, rounds, chunks, leaves, lr, weight, backend,
           model_dir, own, stage2_all=False):
    """Body of train(); `own` is the scratch directory for cached stage-1 matrices.

    Every training candidate (all references: fit groups A/B, holdout, others) is featurized exactly once,
    one sid-slice at a time; matrices live on disk and boosters read them block by block. n_train sets the
    stage-1 references (A + B; above the non-holdout count means all). Stage 2 uses A/B by default;
    stage2_all explicitly expands fitting to all non-holdout references. Early rejection skips stage 2."""
    _check_args(n_train, hold_mod, chunks, rounds)
    P = _params(threads, seed, leaves, lr)
    if weight not in ("none", "ref", "sqrt"):
        raise ValueError("weight must be none|ref|sqrt")
    backend = resolve_backend(backend)
    out = model_path(work, model_dir)
    if out.exists():
        raise FileExistsError("Model destination exists; use a new versioned WORK directory")
    stage = verify_stage(work, "train")
    if stage["inputs"]["settings"]["hold_mod"] != hold_mod:
        raise ValueError("Training/lexicon holdout split mismatch")
    if stage["inputs"]["files"].get(Path(gt_path).name) != files_digest([gt_path])[Path(gt_path).name]:
        raise ValueError("Training labels differ from labels used by the lexicon")
    t0 = time.time()
    lg = lambda msg: print(f"[train {time.time() - t0:.0f}s] {msg}", flush=True)  # noqa: E731
    con = _connect(work, "train", mem, threads)
    label_stats = _load_truth(con, gt_path)
    # Disjoint S1 groups by a seeded hash of the entity id (never used as a feature).
    # Holdout = fixed hash rule shared with the lexicon step (which excludes these S1s).
    ab_salt = "ab" if seed == 26 else f"ab{seed}"
    n_ref = con.execute(f"SELECT count(*) FROM rec WHERE src = 1 AND NOT {hold_rule('entity_id', hold_mod)}").fetchone()[0]
    n_fit = min(n_train, n_ref)
    half = n_fit // 2
    con.execute(f"""CREATE TABLE split AS SELECT rid AS sid,
          CASE WHEN ho THEN 'hold' WHEN h < {half} THEN 'A' WHEN h < {n_fit} THEN 'B' END AS g
        FROM (SELECT rid, ho, row_number() OVER (PARTITION BY ho ORDER BY md5(entity_id || {lit(ab_salt)}), entity_id) - 1 AS h
              FROM (SELECT rid, entity_id, {hold_rule('entity_id', hold_mod)} AS ho FROM rec WHERE src = 1))""")
    counts = dict(con.execute("SELECT g, count(*) FROM split GROUP BY g").fetchall())
    if any(not counts.get(g) for g in ("A", "B", "hold")):
        raise ValueError("Insufficient references for train/hold groups")
    code = con.execute("SELECT sid, CASE g WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'hold' THEN 2 ELSE 3 END AS c FROM split").df()
    group_of = pd.Series(code.c.to_numpy(np.int8), index=code.sid.to_numpy())
    del code
    # One featurization pass over every candidate (group 0 = A, 1 = B, 2 = holdout, 3 = other references).
    sids, tids, grps, ys, files = [], [], [], [], []
    for k in range(chunks):  # one sid-slice of text rows in memory at a time
        df = _pairs(con, f"c.sid % {chunks} = {k}")
        if not len(df):
            continue
        path = own / f"x{k}.npy"
        np.save(path, features(df, threads))
        kk = df[["sid", "tid"]].reset_index(drop=True)
        del df
        lab = con.execute(f"SELECT sid, tid, 1::TINYINT AS y FROM gt WHERE sid % {chunks} = {k}").df()
        sids.append(kk.sid.to_numpy(np.int32)); tids.append(kk.tid.to_numpy(np.int32))
        grps.append(kk.sid.map(group_of).to_numpy(np.int8))
        ys.append(kk.merge(lab, on=["sid", "tid"], how="left").y.fillna(0).to_numpy(np.int8))
        files.append(path)
        lg(f"features {k + 1}/{chunks}: {len(kk):,} pairs")
    sid, tid, grp, y = (np.concatenate(v) for v in (sids, tids, grps, ys))
    del sids, tids, grps, ys
    bounds = np.cumsum([0] + [np.load(f, mmap_mode="r").shape[0] for f in files])
    if bounds[-1] != len(sid) or len(sid) != con.execute("SELECT count(*) FROM c").fetchone()[0]:
        raise ValueError("Featurized rows must cover every candidate exactly once")
    for i, name in enumerate(("A", "B", "hold")):
        sel = grp == i
        if not sel.any() or (i < 2 and len(np.unique(y[sel])) != 2):
            raise ValueError(f"{name} needs naturally retrieved positives and negatives")
        lg(f"{name}: {int(sel.sum()):,} pairs, pos rate {y[sel].mean():.3f}")

    def loaders(sel, ctx=None):
        """Row-block readers for the selected rows, in array order (optionally with cached context)."""
        out = []
        for i, f in enumerate(files):
            s = sel[bounds[i]:bounds[i + 1]]
            if s.any():
                out.append((lambda f=f, s=s: np.load(f)[s]) if ctx is None else
                           (lambda f=f, s=s, c=ctx[i]: np.hstack([np.load(f)[s], np.load(c)])))
        return out

    m = {}
    for i, name in enumerate(("A", "B")):
        sel = grp == i
        m[name] = _fit_parts(backend, P, loaders(sel), y[sel], pair_weights(sid[sel], weight), FEATS, rounds)
    lg("stage-1 models fitted")
    # p1: out-of-fold for A/B (training context); production p1 = mean of both models for every row.
    p1 = np.empty(len(sid)); p1p = np.empty(len(sid))
    for i, f in enumerate(files):
        a, b = bounds[i], bounds[i + 1]
        X = np.load(f)
        pa, pb = _pred(m["A"], X, threads), _pred(m["B"], X, threads)
        p1p[a:b] = np.mean([pa, pb], axis=0)
        p1[a:b] = np.where(grp[a:b] == 0, pb, np.where(grp[a:b] == 1, pa, p1p[a:b]))
    del X, pa, pb
    np.save(own / "p1p.npy", p1p)
    del p1p
    # Candidate filter, then stage-2 context over the kept candidates only (training rows: out-of-fold p1).
    # A/B use out-of-fold p1; explicit stage2_all also includes other non-hold references with the production mean.
    check_scores(p1, name="training stage1")
    kt = p1 >= CAND_P
    _context(con, sid[kt], tid[kt], p1[kt], chunks=chunks, threads=threads)
    fit2 = ((grp != 2) if stage2_all else (grp < 2)) & kt
    w2 = pair_weights(sid[fit2], weight)
    lg(f"stage 2 rows: {int(fit2.sum()):,} of {int((grp != 2).sum()):,} non-holdout blocked pairs (p1 >= {CAND_P})")
    del kt, p1
    ctx = {}
    for i in range(len(files)):
        s = fit2[bounds[i]:bounds[i + 1]]
        if s.any():
            ctx[i] = own / f"ctx{i}.npy"
            np.save(ctx[i], _ctx_matrix(con, pd.DataFrame({"sid": sid[bounds[i]:bounds[i + 1]][s],
                                                            "tid": tid[bounds[i]:bounds[i + 1]][s]})))
    con.execute("DROP TABLE IF EXISTS anc; DROP TABLE IF EXISTS anchors; DROP TABLE IF EXISTS ctx; DROP TABLE IF EXISTS p1tab")
    lg("stage-2 context built")
    # Hold labels never reach the learner, early stopping or a training log.
    m2 = _fit_parts(backend, P, loaders(fit2, ctx), y[fit2], w2, FEATS2, rounds)
    for c in ctx.values():
        c.unlink()
    lg("stage-2 model fitted")
    # One complete production pass: stage-2 scores compete only with stage-2 scores, on filtered candidates.
    p1 = np.load(own / "p1p.npy")
    kp = p1 >= CAND_P
    _context(con, sid[kp], tid[kp], p1[kp], chunks=chunks, threads=threads)
    p2 = np.zeros(len(sid), dtype=np.float64)
    for i, f in enumerate(files):
        a, b = bounds[i], bounds[i + 1]
        s = kp[a:b]
        if s.any():
            keys = pd.DataFrame({"sid": sid[a:b][s], "tid": tid[a:b][s]})
            p2[a:b][s] = _pred(m2, np.hstack([np.load(f)[s], _ctx_matrix(con, keys)]), threads)
    hold_sids = con.execute("SELECT sid FROM split WHERE g = 'hold'").df().sid.tolist()
    lab_hold = con.execute("SELECT sid, tid FROM gt WHERE sid IN (SELECT sid FROM split WHERE g = 'hold')").df()
    hb = grp == 2
    cand = {"p1_min": CAND_P, "export_scope": "input_to_first_scoring_model",
            "hold_candidates_per_ref": int(hb.sum()) / len(hold_sids),
            "hold_stage2_pairs_per_ref": int((hb & kp).sum()) / len(hold_sids),
            "blocked_link_recall": int(y[hb].sum()) / max(len(lab_hold), 1),
            "stage2_link_recall": int(y[hb & kp].sum()) / max(len(lab_hold), 1)}
    lg(f"candidate filter: {cand}")
    check_scores(p1)
    check_scores(p2)
    tune_set = set(con.execute(f"""SELECT r.rid FROM rec r JOIN split s ON s.sid = r.rid
        WHERE s.g='hold' AND md5_number_lower(r.entity_id || 'tune26') % 2 = 0""").df().rid)
    tune_sids = [s for s in hold_sids if s in tune_set]
    final_sids = [s for s in hold_sids if s not in tune_set]
    truth = lab_hold.groupby("sid")["tid"].apply(set).to_dict()
    keep = p2 >= 0.1  # searched thresholds are all >= 0.18 and decide() drops lower scores first: exact
    result = evaluate_threshold(sid[keep], tid[keep], p2[keep], truth, tune_sids, final_sids)
    res = {"stage2": result}
    hold = grp == 2
    hk = pd.DataFrame({"sid": sid[hold], "tid": tid[hold], "p_stage1": p1[hold], "p_stage2": p2[hold]})
    # Preserve the actual GLOBAL decision. Saved hold-only probabilities cannot reconstruct
    # losses to competitors outside hold, which made historical postmortems ambiguous.
    hold_set = set(hold_sids)
    selected = {s: targets for s,targets in decide(sid[keep], tid[keep], p2[keep], result['threshold']).items()
                if s in hold_set}
    complete_truth = {s: truth.get(s, set()) for s in hold_sids}
    hits = dict.fromkeys(hold_sids, 0)
    hits.update({int(s): int(n) for s,n in hk.loc[y[hold].astype(bool),'sid'].value_counts().items()})
    countries = dict(con.execute("SELECT r.rid,r.country FROM rec r JOIN split s ON s.sid=r.rid WHERE s.g='hold'").fetchall())
    errors = error_rows(complete_truth, selected, hold_sids, hits, countries)
    diagnostics = {half: summarize_errors([r for r in errors if (r['sid'] in tune_set) == is_tune])
                   for half,is_tune in (('tune',True),('final',False))}
    for half in ('tune','final'):
        if abs(diagnostics[half]['all']['macro_f05']-result[half+'_macro_f05']) > 1e-10:
            raise RuntimeError('Diagnostic and production decision metrics disagree')
    lg(f"HOLDOUT stage2: threshold={result['threshold']} final macro F0.5={result['final_macro_f05']:.5f}")
    recall_ceiling = cand["blocked_link_recall"]
    devices = {name: "cpu" if isinstance(b, lgb.Booster) else b.attr("training_device") or model_device(b)
               for name,b in (("A",m["A"]),("B",m["B"]),("stage2",m2))}
    metrics = {"holdout_macro_f05": res["stage2"]["final_macro_f05"], "threshold": res["stage2"]["threshold"],
               "params": {"n_train": n_train, "rounds": rounds, "leaves": leaves, "lr": lr, "hold_mod": hold_mod,
                          "chunks": chunks, "seed": seed, "weight": weight, "backend": backend,
                          "gpu": any(d.startswith("cuda") for d in devices.values()),
                          "devices": devices, "stage2_all": stage2_all},
               "references_fit": {"A": int(counts["A"]), "B": int(counts["B"]),
                                  "stage2": int(np.unique(sid[fit2]).size)},
               "candidate_filter": cand,
               "error_diagnostics": diagnostics,
               "labels": label_stats, "stage2": res["stage2"],
               "evaluation": "One final evaluation; production scores/global ownership; competitor pool includes fitted references.",
               "cross_fit_scope": "classifier only; lexicon fitted on all non-hold references",
               "holdout_s1": len(hold_sids), "tune_s1": len(tune_sids), "final_s1": len(final_sids),
               "candidate_link_recall": recall_ceiling, "feature_names_stage1": FEATS, "feature_names_stage2": FEATS2,
               "importance2": _importance(m2, FEATS2),
               "seconds": time.time() - t0}
    out = model_path(work, model_dir)
    out.mkdir(parents=True, exist_ok=False)
    _save(m["A"], out / "model_A"); _save(m["B"], out / "model_B"); _save(m2, out / "model_2")
    extension = ".txt" if backend == "lgb" else ".json"
    hk.assign(y=y[hold].astype(bool), p=hk["p_stage2"]).to_parquet(out / "hold_preds.parquet", index=False)
    pd.DataFrame({"sid": hold_sids}).to_parquet(out / "hold_sids.parquet", index=False)
    pd.DataFrame(errors).assign(half=['tune' if r['sid'] in tune_set else 'final' for r in errors]).to_parquet(
        out / 'hold_error_rows.parquet', index=False)
    pd.DataFrame({'sid': hold_sids,
                  'matched_tids': [sorted(selected.get(s, ())) for s in hold_sids]}).to_parquet(
        out / 'hold_decisions.parquet', index=False)
    # Persist split membership by entity id (DuckDB's hash() may change across versions).
    ids = con.execute("SELECT r.entity_id, r.rid, s.g FROM split s JOIN rec r ON r.rid = s.sid WHERE s.g IS NOT NULL").df()
    ids["half"] = np.where(ids.rid.isin(tune_set), "tune", np.where(ids.g == "hold", "final", ""))
    ids[["entity_id", "g", "half"]].to_parquet(out / "split_ids.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    seal(out, "model_manifest.json", {"code": code_digest(), "environment": environment(),
        "train_stage": stage, "labels": files_digest([gt_path]), "params": metrics["params"],
        "lexicon": digest(Path(work) / "lexicon.json")},
        [n + extension for n in ("model_A", "model_B", "model_2")] +
        ["metrics.json", "hold_preds.parquet", "hold_sids.parquet", "split_ids.parquet",
         "hold_error_rows.parquet", "hold_decisions.parquet"])
    con.close()
    return metrics


def load_models(model_dir):
    """Load the three boosters and refuse any whose feature list differs from this code's."""
    md = Path(model_dir)
    meta = verify(md, "model_manifest.json")
    if meta["inputs"]["code"] != code_digest() or meta["inputs"]["environment"] != environment():
        raise ValueError("Model code/environment fingerprint mismatch; use saved source or retrain")
    for base in ("model_A", "model_B", "model_2"):
        if sum((md / (base + ext)).exists() for ext in (".txt", ".json")) != 1:
            raise ValueError("Missing or ambiguous model format: " + base)
    m1 = [_load(md / f"model_{g}") for g in "AB"]
    m2 = _load(md / "model_2")
    for m, want, name in ((m1[0], FEATS, "lgb_A"), (m1[1], FEATS, "lgb_B"), (m2, FEATS2, "lgb2")):
        if _names(m) != list(want):
            raise ValueError(f"{name}.txt was trained with {len(_names(m))} features that differ from this code's "
                             f"{len(want)}; use the git tag the model was built with, or retrain")
    thr = json.loads((md / "metrics.json").read_text())["threshold"]
    check_scores([], thr)
    return m1, m2, thr


def predict(work, split, out_dir, mem="20GB", threads=None, chunks=8, tmp=None, model_dir="model"):
    """Verify model/data provenance, score every candidate and seal new output files."""
    if type(chunks) is not int or chunks < 1:
        raise ValueError("chunks must be a positive integer")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError("Prediction destination exists; choose a new versioned path")
    m1, m2, thr = load_models(model_path(work, model_dir))
    model_meta = verify(model_path(work, model_dir), "model_manifest.json")
    if model_meta["inputs"]["lexicon"] != digest(Path(work) / "lexicon.json"):
        raise ValueError("Model/test lexicon mismatch")
    con = _connect(work, split, mem, threads)
    try:
        keys, p1, p2 = _score_all(con, m1, m2, threads, chunks, tmp or scratch(work))
        blocked = con.execute("SELECT count(*) FROM c").fetchone()[0]
        # Never trim c after first-stage scoring: it is the officially required candidate set.
        out.mkdir(parents=True, exist_ok=False)
        scores = out / "scores.parquet"
        pd.DataFrame({"sid": keys.sid.values, "tid": keys.tid.values, "p1": p1, "p": p2,
                      "stage2_scored": p1 >= CAND_P}).to_parquet(scores, index=False)
        n_match, n_s1 = write_outputs(con, scores, thr, out)
        seal(out, "prediction_manifest.json", {"model": digest(model_path(work, model_dir) / "model_manifest.json"),
             "stage": verify_stage(work, split), "code": code_digest(), "threshold": thr,
             "pairs": len(keys), "blocked_pairs": blocked, "early_reject_p1": CAND_P,
             "candidate_scope": "input_to_first_scoring_model",
             "references": n_s1, "matches": n_match},
             ["scores.parquet", "matching_results.tsv", "candidate_pairs.tsv"])
        print(f"[predict {split}] wrote {n_s1:,} rows and {n_match:,} links", flush=True)
    finally:
        con.close()


def write_outputs(con, scores_path, thr, out, chunks=64):
    """Decision in DuckDB (p >= thr, then each target to its best S1) and stream both TSVs.
    Every S1 in `rec` gets exactly one row; lists may be empty. Aggregation runs in S1 chunks
    so memory stays bounded. Returns (links, S1 rows)."""
    out = Path(out)
    if type(chunks) is not int or chunks < 1:
        raise ValueError("chunks must be a positive integer")
    validate_records(con)
    if any((out / name).exists() for name in ("matching_results.tsv", "candidate_pairs.tsv")):
        raise FileExistsError("Refusing to overwrite existing TSV outputs")
    con.execute(f"CREATE OR REPLACE TABLE sc AS SELECT sid, tid, p FROM read_parquet({lit(scores_path)})")
    bad = con.execute("SELECT count(*) FROM sc WHERE p IS NULL OR isnan(p) OR p < 0 OR p > 1 OR sid IS NULL OR tid IS NULL").fetchone()[0]
    dup = con.execute("SELECT count(*) - count(DISTINCT (sid, tid)) FROM sc").fetchone()[0]
    if bad or dup:
        raise ValueError(f"scores table invalid: {bad} bad rows, {dup} duplicate pairs")
    check_scores([], thr)
    validate_pairs(con, "sc")
    if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='c'").fetchone()[0]:
        diff = con.execute("""SELECT count(*) FROM (
            (SELECT sid,tid FROM c EXCEPT SELECT sid,tid FROM sc)
            UNION ALL (SELECT sid,tid FROM sc EXCEPT SELECT sid,tid FROM c))""").fetchone()[0]
        if diff:
            raise ValueError("Scores must cover exactly the final candidate set")
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"""CREATE OR REPLACE TABLE pr AS SELECT sid, tid FROM (
          SELECT sid, tid, row_number() OVER (PARTITION BY tid ORDER BY p DESC, sid) AS r
          FROM sc WHERE p >= {thr}) WHERE r = 1""")
    con.execute("CREATE OR REPLACE TABLE ids AS SELECT rid, entity_id, src FROM rec")
    for fname, col, table in (("matching_results.tsv", "matched_entity_ids", "pr"),
                              ("candidate_pairs.tsv", "candidate_entity_ids", "sc")):
        with open(out / fname, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for k in range(chunks):
                cur = con.execute(f"""
                    SELECT s.entity_id, coalesce(string_agg(t.entity_id, ',' ORDER BY t.rid), '')
                    FROM ids s LEFT JOIN (SELECT * FROM {table} WHERE sid % {chunks} = {k}) x ON x.sid = s.rid
                    LEFT JOIN ids t ON t.rid = x.tid
                    WHERE s.src = 1 AND s.rid % {chunks} = {k} GROUP BY s.rid, s.entity_id ORDER BY s.rid""")
                while rows := cur.fetchmany(200_000):
                    f.writelines(f"{a}\t{b}\n" for a, b in rows)
    return (con.execute("SELECT count(*) FROM pr").fetchone()[0],
            con.execute("SELECT count(*) FROM rec WHERE src = 1").fetchone()[0])


def rewrite(work, split, scores_path, out, thr=None, mem="8GB", threads=None):
    """Refuse unverified legacy score rewrites; verified predictions use immutable destinations."""
    raise ValueError("Threshold rewrites are disabled for verified runs; use predict with the selected "
                     "model threshold. Historical scores require their matching saved code.")
