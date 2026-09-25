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

DIGITS = re.compile(r"\d+")

# Frequency of exact core name / address among targets and references (same country).
REC_SQL = """
CREATE TABLE rec AS
WITH r AS (SELECT * FROM read_parquet({glob}))
SELECT r.*,
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
  row_number() OVER (PARTITION BY tid ORDER BY s DESC, sid) AS t_rank,
  s / max(s) OVER (PARTITION BY tid) AS t_rel
FROM read_parquet({cands})
"""

BASE = ["s", "sn", "sa", "nk", "r_s", "r_n", "r_a", "n_c", "s_rel", "s_gap", "t_nc", "t_rank", "t_rel",
        "tsrc", "s_fct", "s_fcs", "s_fat", "t_fct", "t_fcs", "t_fat"]
FEATS = BASE + ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "c_ratio", "c_tset", "c_nospace",
                "c_exact", "a_ratio", "a_tsort", "a_tset", "a_partial", "t_addr_empty", "len_n_s", "len_n_t",
                "len_a_s", "len_a_t", "num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt",
                "t_num_cnt", "hn_ratio", "hn_rel", "raw_ratio", "raw_partial", "sup_core", "sup_addr"]
CTX = ["p1", "p1_rank", "p1_max", "p1_sum", "p1_n05", "p1_rel", "p1_trank", "p1_tother", "p1_tdiff", "p1_tn05",
       "p1_srank", "p1_sn05", "p1_smax", "anc_name", "anc_core", "anc_addr", "anc_n"]
FEATS2 = FEATS + CTX
ANCHOR_P, ANCHOR_K, ANCHOR_MIN = 0.3, 4, 0.02  # anchors: top-4 candidates with p1 >= 0.3; scored pairs: p1 >= 0.02

CTX_SQL = """
SELECT sid, tid, p1,
  row_number() OVER (PARTITION BY sid ORDER BY p1 DESC, tid) AS p1_rank,
  max(p1) OVER (PARTITION BY sid) AS p1_max,
  sum(p1) OVER (PARTITION BY sid) AS p1_sum,
  sum((p1 > 0.5)::INT) OVER (PARTITION BY sid) AS p1_n05,
  row_number() OVER (PARTITION BY tid ORDER BY p1 DESC, sid) AS p1_trank,
  max(p1) OVER (PARTITION BY tid) AS tmax,
  nth_value(p1, 2) OVER (PARTITION BY tid ORDER BY p1 DESC, sid
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS t2,
  sum((p1 > 0.5)::INT) OVER (PARTITION BY tid) AS p1_tn05,
  row_number() OVER (PARTITION BY sid, r.src ORDER BY p1 DESC, tid) AS p1_srank,
  sum((p1 > 0.5)::INT) OVER (PARTITION BY sid, r.src) AS p1_sn05,
  max(p1) OVER (PARTITION BY sid, r.src) AS p1_smax
FROM p1tab JOIN rec r ON r.rid = p1tab.tid
"""


def _num_feats(sa, ta):
    """Digit-token agreement; hn_* compare the first number (house/plot number): typo vs neighbour."""
    out = np.zeros((len(sa), 8), np.float32)
    for i, (x, y) in enumerate(zip(sa, ta)):
        a, b = DIGITS.findall(x), DIGITS.findall(y)
        sa_, sb_ = set(a), set(b)
        inter = len(sa_ & sb_)
        hn = (fuzz.ratio(a[0], b[0]) / 100, abs(int(a[0][:9]) - int(b[0][:9])) / max(int(a[0][:9]), int(b[0][:9]), 1))             if a and b else (-1, -1)
        out[i] = (inter / len(sa_ | sb_) if sa_ and sb_ else -1, inter,
                  float(bool(sa_) and bool(sb_) and not inter),
                  float(bool(a) and bool(b) and a[0] == b[0]), len(a), len(b), *hn)
    return out


def features(df):
    """Stage-1 feature matrix (float32, column order FEATS) for joined candidate rows."""
    sn, tn = df["sname"].tolist(), df["tname"].tolist()
    sc, tc = df["scor"].tolist(), df["tcore"].tolist()
    sa, ta = df["saddr"].tolist(), df["taddr"].tolist()
    d = lambda a, b, f: cpdist(a, b, scorer=f, workers=-1, dtype=np.float32)  # noqa: E731
    cols = {c: df[c].to_numpy(np.float32) for c in BASE}
    cols.update(n_ratio=d(sn, tn, fuzz.ratio), n_tsort=d(sn, tn, fuzz.token_sort_ratio),
                n_tset=d(sn, tn, fuzz.token_set_ratio), n_partial=d(sn, tn, fuzz.partial_ratio),
                n_jw=d(sn, tn, JaroWinkler.normalized_similarity), c_ratio=d(sc, tc, fuzz.ratio),
                c_tset=d(sc, tc, fuzz.token_set_ratio),
                c_nospace=d([x.replace(" ", "") for x in sc], [x.replace(" ", "") for x in tc], fuzz.ratio),
                a_ratio=d(sa, ta, fuzz.ratio), a_tsort=d(sa, ta, fuzz.token_sort_ratio),
                a_tset=d(sa, ta, fuzz.token_set_ratio), a_partial=d(sa, ta, fuzz.partial_ratio))
    cols["c_exact"] = np.fromiter((x == y for x, y in zip(sc, tc)), np.float32, len(sc))
    rs, rt = df["srawn"].tolist(), df["trawn"].tolist()
    cols.update(raw_ratio=d(rs, rt, fuzz.ratio), raw_partial=d(rs, rt, fuzz.partial_ratio))
    # Cluster support: true matches come in groups of near-identical variants; decoys stand alone.
    cols["sup_core"] = df.groupby(["sid", "tcore"]).sid.transform("size").to_numpy(np.float32)
    sup_a = df.groupby(["sid", "taddr"]).sid.transform("size").to_numpy(np.float32)
    cols["sup_addr"] = np.where(df["taddr"].to_numpy() == "", -1, sup_a).astype(np.float32)
    cols["t_addr_empty"] = np.fromiter((not x for x in ta), np.float32, len(ta))
    for k, v in (("len_n_s", sn), ("len_n_t", tn), ("len_a_s", sa), ("len_a_t", ta)):
        cols[k] = np.fromiter(map(len, v), np.float32, len(v))
    nf = _num_feats(sa, ta)
    for j, k in enumerate(["num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt", "t_num_cnt",
                           "hn_ratio", "hn_rel"]):
        cols[k] = nf[:, j]
    return np.column_stack([cols[c] for c in FEATS])


def _pairs(con, where):
    """Candidate rows (with group features) joined to both records' normalized text."""
    return con.execute(f"""
        SELECT c.*, a.name AS sname, a.core AS scor, a.addr AS saddr, a.rawn AS srawn,
               b.name AS tname, b.core AS tcore, b.addr AS taddr, b.src AS tsrc, b.rawn AS trawn,
               a.fc_t AS s_fct, a.fc_s AS s_fcs, a.fa_t AS s_fat,
               b.fc_t AS t_fct, b.fc_s AS t_fcs, b.fa_t AS t_fat
        FROM c JOIN rec a ON a.rid = c.sid JOIN rec b ON b.rid = c.tid WHERE {where}
        ORDER BY c.sid, c.tid""").df()


def lit(value):
    """SQL string literal with quotes escaped (paths may contain apostrophes)."""
    return "'" + str(value).replace("'", "''") + "'"


def configure(con, mem, threads, temp_dir):
    """Validated DuckDB resource settings."""
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", str(mem)):
        raise ValueError(f"memory limit must look like '8GB', got {mem!r}")
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError(f"threads must be a positive int, got {threads!r}")
    con.execute(f"SET memory_limit={lit(mem)}; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={lit(temp_dir)}")
    if threads:
        con.execute(f"SET threads={threads}")


def scratch(work):
    """Directory for spill/temporary files: $FASTER_TMP if set (e.g. /tmp on Kaggle), else the work dir."""
    return os.environ.get("FASTER_TMP") or str(work)


def _connect(work, split, mem, threads):
    """Open DuckDB with resource limits and load `rec` (records + name/address frequencies) and
    `c` (candidates + candidate-list context) for one split."""
    con = duckdb.connect()
    configure(con, mem, threads, Path(scratch(work)) / "duck_tmp")
    w = Path(work) / split
    con.execute(REC_SQL.format(glob=lit(w / "norm_s*.parquet")))
    if "rawn" not in [r[0] for r in con.execute("DESCRIBE rec").fetchall()]:  # runs prepared before rawn existed
        con.execute("ALTER TABLE rec ADD COLUMN rawn VARCHAR; UPDATE rec SET rawn = replace(name, ' ', '')")
    con.execute(GROUP_SQL.format(cands=lit(w / "cands.parquet")))
    return con


def _context(con, sid, tid, p1, anchor_where="TRUE", chunks=8):
    """Stage-2 context columns (order CTX) aligned to (sid, tid) via a DuckDB window pass,
    plus cluster-consistency (anchor) features for pairs whose S1 matches `anchor_where`."""
    con.register("p1df", pd.DataFrame({"sid": sid, "tid": tid, "p1": p1}))
    con.execute("CREATE OR REPLACE TABLE p1tab AS SELECT * FROM p1df")
    con.unregister("p1df")
    con.execute(f"CREATE OR REPLACE TABLE ctx AS {CTX_SQL}")
    _anchors(con, anchor_where, chunks)
    return con


def _anchors(con, where, chunks):
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
        d = lambda a, b, f: cpdist(a, b, scorer=f, workers=-1, dtype=np.float32)  # noqa: E731
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
        SELECT x.p1, x.p1_rank, x.p1_max, x.p1_sum, x.p1_n05, x.p1 / x.p1_max AS p1_rel, x.p1_trank,
               CASE WHEN x.p1_trank = 1 THEN coalesce(x.t2, 0) ELSE x.tmax END AS p1_tother,
               x.p1 - CASE WHEN x.p1_trank = 1 THEN coalesce(x.t2, 0) ELSE x.tmax END AS p1_tdiff, x.p1_tn05,
               x.p1_srank, x.p1_sn05, x.p1_smax,
               coalesce(n.anc_name, -1) AS anc_name, coalesce(n.anc_core, -1) AS anc_core,
               coalesce(n.anc_addr, -1) AS anc_addr, coalesce(n.anc_n, 0) AS anc_n
        FROM k JOIN ctx x USING (sid, tid) LEFT JOIN anc n USING (sid, tid) ORDER BY k.ix""").fetchnumpy()
    con.unregister("k")
    return np.column_stack([m[c].astype(np.float32) for c in CTX])


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
    """Official metric: F0.5 per S1, averaged over `all_sids` (singletons included: an empty
    prediction for an S1 with no true matches scores 1, any prediction scores 0)."""
    if not len(all_sids):
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
    if not isinstance(hold_mod, int) or hold_mod < 2:
        raise ValueError(f"hold_mod must be an int >= 2, got {hold_mod!r}")
    return f"(hash({col} || 'hold26') % {hold_mod} = 0)"


def _params(threads, seed, leaves=127, lr=0.05):
    """LightGBM hyperparameters shared by all three boosters."""
    return dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=threads or 0, verbose=-1, seed=seed)


_GPU = None


def gpu_available():
    """True if XGBoost can train on a CUDA GPU on this machine (checked once with a tiny fit)."""
    global _GPU
    if _GPU is None:
        try:
            import xgboost as xgb
            d = xgb.DMatrix(np.random.rand(256, 4).astype(np.float32), label=np.arange(256) % 2)
            xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0}, d, 2)
            _GPU = True
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


def _fit(backend, P, X, y, w, names, rounds, valid=None):
    """Train one booster: XGBoost (GPU if available, CPU otherwise) or LightGBM (CPU)."""
    if backend == "lgb":
        vs = [lgb.Dataset(valid[0], valid[1], feature_name=list(names))] if valid is not None else []
        return lgb.train(P, lgb.Dataset(X, y, feature_name=list(names), weight=w), rounds, valid_sets=vs,
                         callbacks=[lgb.log_evaluation(100)] if vs else [])
    import xgboost as xgb
    xp = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
          "device": "cuda" if gpu_available() else "cpu", "learning_rate": P["learning_rate"],
          "max_leaves": P["num_leaves"], "grow_policy": "lossguide", "max_depth": 0, "min_child_weight": 5,
          "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "max_bin": 256, "seed": P["seed"],
          "nthread": P["num_threads"] or 0, "verbosity": 1}
    d = xgb.QuantileDMatrix(X, y, weight=w, feature_names=list(names))  # quantized: ~1 byte per value
    ev = []
    if valid is not None:
        ev = [(xgb.QuantileDMatrix(valid[0], valid[1], ref=d, feature_names=list(names)), "hold")]
    return xgb.train(xp, d, rounds, evals=ev, verbose_eval=100 if ev else False)


def _pred(m, X, threads=None):
    """Match probability for the rows of X, from either booster type."""
    if isinstance(m, lgb.Booster):
        return m.predict(X, num_threads=threads or 0)
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
        b.set_param({"device": "cuda" if gpu_available() else "cpu"})
        return b
    return lgb.Booster(model_file=str(txt))


def _importance(m, names):
    """Gain importance per feature for either booster type."""
    if isinstance(m, lgb.Booster):
        return dict(zip(names, m.feature_importance("gain").round().tolist()))
    g = m.get_score(importance_type="gain")
    return {n: round(float(g.get(n, 0.0)), 1) for n in names}


def _predict_stage1(con, models, where, threads, chunks, tmp=None, tag=""):
    """Featurize candidate rows matching `where` chunk by chunk; return keys + averaged p1.
    If tmp is given, stage-1 matrices are cached there as .npy for stage 2."""
    keys, p1 = [], []
    for k in range(chunks):
        df = _pairs(con, f"({where}) AND c.sid % {chunks} = {k}")
        if not len(df):
            continue
        X = features(df)
        p1.append(np.mean([_pred(m, X, threads) for m in models], axis=0))
        keys.append(df[["sid", "tid"]])
        if tmp is not None:
            np.save(Path(tmp) / f"{tag}{k}.npy", X)
        print(f"  [stage1 {tag}] chunk {k + 1}/{chunks}: {len(df):,} pairs", flush=True)
    if not keys:
        return pd.DataFrame({"sid": [], "tid": []}, dtype="int32"), np.empty(0)
    return pd.concat(keys, ignore_index=True), np.concatenate(p1)


def _check_args(n_train, hold_mod, chunks, rounds):
    """Fail early on invalid training settings instead of producing empty groups later."""
    for name, v, lo in (("n_train", n_train, 2), ("hold_mod", hold_mod, 2), ("chunks", chunks, 1), ("rounds", rounds, 1)):
        if not isinstance(v, int) or v < lo:
            raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")


def _load_truth(con, gt_path):
    """Labels -> (sid, tid) rids. Refuses any label that does not resolve to a known record
    (an unknown target must never silently turn a reference into a fake singleton)."""
    con.execute(f"""CREATE TABLE gt_raw AS SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
          FROM read_parquet({lit(gt_path)}) WHERE coalesce(matched_entity_ids, '') <> ''""")
    con.execute("""CREATE TABLE gt AS SELECT a.rid AS sid, b.rid AS tid FROM gt_raw g
        JOIN rec a ON a.entity_id = g.s AND a.src = 1 JOIN rec b ON b.entity_id = g.t AND b.src > 1""")
    n_raw, n_ok = (con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("gt_raw", "gt"))
    n_lab = con.execute(f"SELECT count(*) FROM read_parquet({lit(gt_path)})").fetchone()[0]
    n_s1 = con.execute("SELECT count(*) FROM rec WHERE src = 1").fetchone()[0]
    dup_t = con.execute("SELECT count(*) - count(DISTINCT tid) FROM gt").fetchone()[0]
    dup_p = con.execute("SELECT count(*) - count(DISTINCT (sid, tid)) FROM gt").fetchone()[0]
    if n_raw != n_ok or n_lab != n_s1 or dup_p:
        raise ValueError(f"label integrity: {n_raw} raw links, {n_ok} resolved; {n_lab} label rows vs "
                         f"{n_s1} S1 records; {dup_p} duplicate pairs")
    return {"links": n_ok, "label_rows": n_lab, "targets_with_2plus_owners": dup_t}


def pair_weights(sid, mode):
    """Per-pair training weights. 'none': every pair 1. 'ref': each reference's pairs sum to a
    constant (the metric averages per reference); 'sqrt': halfway (1/sqrt(n_candidates)). Mean 1."""
    if mode == "none":
        return None
    n = pd.Series(sid).map(pd.Series(sid).value_counts()).to_numpy(np.float64)
    w = 1.0 / n if mode == "ref" else 1.0 / np.sqrt(n)
    return (w / w.mean()).astype(np.float32)


def train(work, gt_path, n_train=300_000, hold_mod=22, mem="20GB", threads=None, seed=26, rounds=600,
          chunks=8, leaves=127, lr=0.05, weight="none", backend="auto"):
    """Train the two-stage matcher on the train split and write WORK/model/.

    1. Validate labels; split S1s into A, B (matcher training) and hold (hash rule shared with the
       lexicon), hold further into tune/final halves.
    2. Stage 1: LightGBM on A and on B; out-of-fold p1 for A/B, averaged p1 for hold and for the
       non-sampled S1s that compete for the same targets.
    3. Stage 2: LightGBM on A+B with stage-1 features plus competition context built from p1.
    4. Threshold tuned on the tune half (global target ownership), macro F0.5 reported on the
       final half. Saves boosters, metrics.json, hold predictions and split ids.
    """
    _check_args(n_train, hold_mod, chunks, rounds)
    t0 = time.time()
    lg = lambda msg: print(f"[train {time.time() - t0:.0f}s] {msg}", flush=True)  # noqa: E731
    con = _connect(work, "train", mem, threads)
    label_stats = _load_truth(con, gt_path)
    # Disjoint S1 groups by a seeded hash of the entity id (never used as a feature).
    # Holdout = fixed hash rule shared with the lexicon step (which excludes these S1s).
    half = n_train // 2
    con.execute(f"""CREATE TABLE split AS SELECT rid AS sid,
          CASE WHEN ho THEN 'hold' WHEN h < {half} THEN 'A' WHEN h < {n_train} THEN 'B' END AS g
        FROM (SELECT rid, ho, row_number() OVER (PARTITION BY ho ORDER BY hash(entity_id || 'ab')) - 1 AS h
              FROM (SELECT rid, entity_id, {hold_rule('entity_id', hold_mod)} AS ho FROM rec WHERE src = 1))""")
    data = {}
    for g in ("A", "B", "hold"):
        lab = con.execute(f"SELECT sid, tid FROM gt WHERE sid IN (SELECT sid FROM split WHERE g = '{g}')").df()
        key = set(zip(lab.sid, lab.tid))
        keys, Xs = [], []
        for k in range(chunks):  # chunked so only one slice of text rows is in memory at a time
            df = _pairs(con, f"c.sid IN (SELECT sid FROM split WHERE g = '{g}') AND c.sid % {chunks} = {k}")
            keys.append(df[["sid", "tid"]].copy())
            Xs.append(features(df))
            del df
        kk = pd.concat(keys, ignore_index=True)
        y = np.fromiter(((s, t) in key for s, t in zip(kk.sid, kk.tid)), np.int8, len(kk))
        data[g] = {"keys": kk, "X": np.vstack(Xs), "y": y, "lab": lab}
        del Xs
        lg(f"{g}: {len(kk):,} pairs, pos rate {y.mean():.3f}")
    P = _params(threads, seed, leaves, lr)
    if weight not in ("none", "ref", "sqrt"):
        raise ValueError(f"weight must be none|ref|sqrt, got {weight!r}")
    backend = resolve_backend(backend)
    lg(f"booster backend: {backend} ({'GPU' if backend == 'xgb' and gpu_available() else 'CPU'})")
    m = {g: _fit(backend, P, data[g]["X"], data[g]["y"], pair_weights(data[g]["keys"].sid.values, weight), FEATS,
                 rounds) for g in ("A", "B")}
    lg("stage-1 models fitted")
    # p1: out-of-fold for A/B, averaged for everything else (holdout + all other train S1s).
    data["A"]["p1"] = _pred(m["B"], data["A"]["X"], threads)
    data["B"]["p1"] = _pred(m["A"], data["B"]["X"], threads)
    data["hold"]["p1"] = np.mean([_pred(m[g], data["hold"]["X"], threads) for g in "AB"], axis=0)
    # Competing S1s outside the sample only matter through shared targets.
    con.execute("""CREATE TABLE tsample AS SELECT DISTINCT tid FROM c
                   WHERE sid IN (SELECT sid FROM split WHERE g IS NOT NULL)""")
    ok, op1 = _predict_stage1(con, [m["A"], m["B"]], "c.sid IN (SELECT sid FROM split WHERE g IS NULL) "
                              "AND c.tid IN (SELECT tid FROM tsample)", threads, chunks, tag="others")
    allk = pd.concat([data[g]["keys"] for g in ("A", "B", "hold")] + [ok], ignore_index=True)
    allp = np.concatenate([data[g]["p1"] for g in ("A", "B", "hold")] + [op1])
    _context(con, allk.sid.values, allk.tid.values, allp,
             anchor_where="q.sid IN (SELECT sid FROM split WHERE g IS NOT NULL)", chunks=chunks)
    for g in ("A", "B", "hold"):
        data[g]["X2"] = np.hstack([data[g]["X"], _ctx_matrix(con, data[g]["keys"])])
        del data[g]["X"]  # stage-1 matrix no longer needed; keeps peak memory down
    lg("stage-2 context built")
    X2 = np.vstack([data["A"].pop("X2"), data["B"].pop("X2")])
    y2 = np.concatenate([data["A"]["y"], data["B"]["y"]])
    hd = data["hold"]
    w2 = pair_weights(np.concatenate([data["A"]["keys"].sid.values, data["B"]["keys"].sid.values]), weight)
    m2 = _fit(backend, P, X2, y2, w2, FEATS2, rounds, valid=(hd["X2"], hd["y"]))
    del X2
    hold_sids = con.execute("SELECT sid FROM split WHERE g = 'hold'").df().sid.tolist()
    # Hold is split in two: thresholds are tuned on "tune" and the score is reported on "final".
    tune_set = set(con.execute(f"""SELECT r.rid FROM rec r JOIN split s ON s.sid = r.rid
                                  WHERE s.g = 'hold' AND hash(r.entity_id || 'tune{seed}') % 2 = 0""").df().rid)
    tune_sids = [s for s in hold_sids if s in tune_set]
    final_sids = [s for s in hold_sids if s not in tune_set]
    truth = hd["lab"].groupby("sid")["tid"].apply(set).to_dict()
    hk = hd["keys"]
    # Competitors outside hold (out-of-fold / averaged stage-1 scores) take part in target
    # ownership, as they do for the real test decision.
    comp = pd.concat([data["A"]["keys"].assign(p=data["A"]["p1"]), data["B"]["keys"].assign(p=data["B"]["p1"]),
                      ok.assign(p=op1)], ignore_index=True)
    comp = comp[comp.tid.isin(set(hk.tid))]
    res = {}
    for name, p in (("stage1", hd["p1"]), ("stage2", _pred(m2, hd["X2"], threads))):
        check_scores(p, name=f"hold {name}")
        allsid = np.concatenate([hk.sid.values, comp.sid.values])
        alltid = np.concatenate([hk.tid.values, comp.tid.values])
        allp = np.concatenate([p, comp.p.values])
        grid_t, grid_f, grid_local = {}, {}, {}
        for t in np.arange(0.2, 0.96, 0.025):
            t = round(float(t), 3)
            dec = decide(allsid, alltid, allp, t)  # global ownership, metric over hold only
            grid_t[t] = macro_f05(truth, dec, tune_sids)
            grid_f[t] = macro_f05(truth, dec, final_sids)
            grid_local[t] = macro_f05(truth, decide(hk.sid.values, hk.tid.values, p, t), hold_sids)
        thr = max(grid_t, key=grid_t.get)
        res[name] = {"threshold": thr, "tune_macro_f05": grid_t[thr], "final_macro_f05": grid_f[thr],
                     "local_ownership_hold_f05": grid_local[thr], "grid_tune": grid_t, "grid_final": grid_f}
        lg(f"HOLDOUT {name}: thr={thr} (tuned on {len(tune_sids):,}) -> final-half macro F0.5 = {grid_f[thr]:.5f} "
           f"| tune-half {grid_t[thr]:.5f} | hold-only ownership {grid_local[thr]:.5f}")
        hk = hk.assign(**{f"p_{name}": p})
    recall_ceiling = len(set(zip(hk.sid, hk.tid)) & set(zip(hd["lab"].sid, hd["lab"].tid))) / max(len(hd["lab"]), 1)
    metrics = {"holdout_macro_f05": res["stage2"]["final_macro_f05"], "threshold": res["stage2"]["threshold"],
               "params": {"n_train": n_train, "rounds": rounds, "leaves": leaves, "lr": lr, "hold_mod": hold_mod,
                          "chunks": chunks, "seed": seed, "weight": weight, "backend": backend,
                          "gpu": bool(backend == "xgb" and gpu_available())},
               "labels": label_stats, "stage1": res["stage1"], "stage2": res["stage2"],
               "holdout_s1": len(hold_sids), "tune_s1": len(tune_sids), "final_s1": len(final_sids),
               "candidate_link_recall": recall_ceiling, "feature_names_stage1": FEATS, "feature_names_stage2": FEATS2,
               "importance2": _importance(m2, FEATS2),
               "seconds": time.time() - t0}
    out = Path(work) / "model"
    out.mkdir(parents=True, exist_ok=True)
    _save(m["A"], out / "model_A"); _save(m["B"], out / "model_B"); _save(m2, out / "model_2")
    hk.assign(y=hd["y"], p=hk["p_stage2"]).to_parquet(out / "hold_preds.parquet", index=False)
    pd.DataFrame({"sid": hold_sids}).to_parquet(out / "hold_sids.parquet", index=False)
    # Persist split membership by entity id (DuckDB's hash() may change across versions).
    ids = con.execute("SELECT r.entity_id, r.rid, s.g FROM split s JOIN rec r ON r.rid = s.sid WHERE s.g IS NOT NULL").df()
    ids["half"] = np.where(ids.rid.isin(tune_set), "tune", np.where(ids.g == "hold", "final", ""))
    ids[["entity_id", "g", "half"]].to_parquet(out / "split_ids.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def load_models(model_dir):
    """Load the three boosters and refuse any whose feature list differs from this code's."""
    md = Path(model_dir)
    legacy = not any((md / f"model_2{e}").exists() for e in (".txt", ".json"))  # older runs: lgb_*.txt
    bases = ["lgb_A", "lgb_B", "lgb2"] if legacy else ["model_A", "model_B", "model_2"]
    m1 = [_load(md / b) for b in bases[:2]]
    m2 = _load(md / bases[2])
    for m, want, name in ((m1[0], FEATS, bases[0]), (m1[1], FEATS, bases[1]), (m2, FEATS2, bases[2])):
        if _names(m) != list(want):
            raise ValueError(f"{name} was trained with {len(_names(m))} features that differ from this code's "
                             f"{len(want)}; use the git tag the model was built with, or retrain")
    thr = json.loads((md / "metrics.json").read_text())["threshold"]
    check_scores([], thr)
    return m1, m2, thr


def predict(work, split, out_dir, mem="20GB", threads=None, chunks=8, tmp=None):
    """Score every candidate pair of `split` with the saved boosters and write the two TSVs.

    Stage-1 features are cached per chunk in a run-owned temp dir, stage-2 context is built from
    all stage-1 scores (so competing S1s are seen), scores are validated and saved to
    scores.parquet, then write_outputs applies threshold + one-owner-per-target.
    """
    if not isinstance(chunks, int) or chunks < 1:
        raise ValueError(f"chunks must be a positive int, got {chunks!r}")
    t0 = time.time()
    m1, m2, thr = load_models(Path(work) / "model")  # fail before any expensive work
    con = _connect(work, split, mem, threads)
    # Run-owned temp dir: cleanup can only ever delete what this call created.
    Path(tmp or scratch(work)).mkdir(parents=True, exist_ok=True)
    own = Path(tempfile.mkdtemp(prefix="fe_", dir=tmp or scratch(work)))
    try:
        keys, p1 = _predict_stage1(con, m1, "TRUE", threads, chunks, tmp=own, tag="x")
        check_scores(p1, name="stage-1 test scores")
        _context(con, keys.sid.values, keys.tid.values, p1, chunks=chunks)
        p2, off = [], 0
        for k in range(chunks):
            f = own / f"x{k}.npy"
            if not f.exists():  # empty chunk: no candidates in this bucket
                continue
            X = np.load(f)
            kk = keys.iloc[off:off + len(X)]
            off += len(X)
            p2.append(_pred(m2, np.hstack([X, _ctx_matrix(con, kk)]), threads))
            print(f"  [stage2 {split}] chunk {k + 1}/{chunks} {time.time() - t0:.0f}s", flush=True)
        if off != len(keys):
            raise RuntimeError(f"stage-2 consumed {off} rows but stage 1 produced {len(keys)}")
    finally:
        shutil.rmtree(own, ignore_errors=True)
    p2 = np.concatenate(p2) if p2 else np.empty(0)
    check_scores(p2, name="stage-2 test scores")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sid": keys.sid.values, "tid": keys.tid.values, "p1": p1, "p": p2}).to_parquet(
        out / "scores.parquet", index=False)
    del keys, p1, p2
    n_match, n_s1 = write_outputs(con, out / "scores.parquet", thr, out)
    print(f"[predict {split}] wrote {n_s1:,} rows, {n_match:,} links ({n_match / n_s1:.2f}/S1) "
          f"{time.time() - t0:.0f}s", flush=True)


def write_outputs(con, scores_path, thr, out, chunks=64):
    """Decision in DuckDB (p >= thr, then each target to its best S1) and stream both TSVs.
    Every S1 in `rec` gets exactly one row; lists may be empty. Aggregation runs in S1 chunks
    so memory stays bounded. Returns (links, S1 rows)."""
    out = Path(out)
    con.execute(f"CREATE OR REPLACE TABLE sc AS SELECT sid, tid, p FROM read_parquet({lit(scores_path)})")
    bad = con.execute("SELECT count(*) FROM sc WHERE p IS NULL OR isnan(p) OR p < 0 OR p > 1 OR sid IS NULL OR tid IS NULL").fetchone()[0]
    dup = con.execute("SELECT count(*) - count(DISTINCT (sid, tid)) FROM sc").fetchone()[0]
    if bad or dup:
        raise ValueError(f"scores table invalid: {bad} bad rows, {dup} duplicate pairs")
    check_scores([], thr)
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
    """Re-emit both TSVs from a saved scores.parquet (e.g. after a crash or to try another threshold)."""
    thr = thr if thr is not None else json.loads((Path(work) / "model" / "metrics.json").read_text())["threshold"]
    con = duckdb.connect()
    configure(con, mem, threads, Path(scratch(work)) / "duck_tmp")
    con.execute(f"CREATE TABLE rec AS SELECT rid, entity_id, src FROM read_parquet({lit(Path(work) / split / 'norm_s*.parquet')})")
    n_match, n_s1 = write_outputs(con, scores_path, thr, out)
    print(f"[rewrite {split}] thr={thr} {n_s1:,} rows, {n_match:,} links ({n_match / n_s1:.2f}/S1)", flush=True)
