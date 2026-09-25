"""Stage 3: pair features, two-stage LightGBM matcher, threshold tuning, submission writing.

Stage 1 (per pair): blocking scores/ranks, rapidfuzz similarities on name / core
name / address, digit-token agreement, missingness, how common the name/address
is, and candidate-list context. Two stage-1 models are trained on disjoint S1
halves (A, B) so every training pair gets an out-of-fold probability p1.

Stage 2 (per pair, with context): stage-1 features + p1 + how p1 compares with the
S1's other candidates and with the best competing S1 for the same target.

Decision: keep pairs with p2 >= threshold, then each target goes only to its
highest-probability S1 (labels show every target belongs to at most one S1).
The threshold maximises exact macro F0.5 on held-out S1s (never used in fitting).
"""
import json
import os
import re
import shutil
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
WITH r AS (SELECT * FROM read_parquet('{w}/norm_s*.parquet'))
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
FROM read_parquet('{cands}')
"""

BASE = ["s", "sn", "sa", "nk", "r_s", "r_n", "r_a", "n_c", "s_rel", "s_gap", "t_nc", "t_rank", "t_rel",
        "tsrc", "s_fct", "s_fcs", "s_fat", "t_fct", "t_fcs", "t_fat"]
FEATS = BASE + ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "c_ratio", "c_tset", "c_nospace",
                "c_exact", "a_ratio", "a_tsort", "a_tset", "a_partial", "t_addr_empty", "len_n_s", "len_n_t",
                "len_a_s", "len_a_t", "num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt",
                "t_num_cnt", "hn_ratio", "hn_rel", "raw_ratio", "raw_partial", "sup_core", "sup_addr"]
CTX = ["p1", "p1_rank", "p1_max", "p1_sum", "p1_n05", "p1_rel", "p1_trank", "p1_tother", "p1_tdiff", "p1_tn05"]
FEATS2 = FEATS + CTX

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
  sum((p1 > 0.5)::INT) OVER (PARTITION BY tid) AS p1_tn05
FROM p1tab
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


def scratch(work):
    """Directory for spill/temporary files: $FASTER_TMP if set (e.g. /tmp on Kaggle), else the work dir."""
    return os.environ.get("FASTER_TMP") or str(work)


def _connect(work, split, mem, threads):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{Path(scratch(work)) / 'duck_tmp'}'")
    if threads:
        con.execute(f"SET threads={threads}")
    w = Path(work) / split
    con.execute(REC_SQL.format(w=w))
    if "rawn" not in [r[0] for r in con.execute("DESCRIBE rec").fetchall()]:  # runs prepared before rawn existed
        con.execute("ALTER TABLE rec ADD COLUMN rawn VARCHAR; UPDATE rec SET rawn = replace(name, ' ', '')")
    con.execute(GROUP_SQL.format(cands=w / "cands.parquet"))
    return con


def _context(con, sid, tid, p1):
    """Stage-2 context columns (order CTX) aligned to (sid, tid) via a DuckDB window pass."""
    con.register("p1df", pd.DataFrame({"sid": sid, "tid": tid, "p1": p1}))
    con.execute("CREATE OR REPLACE TABLE p1tab AS SELECT * FROM p1df")
    con.unregister("p1df")
    con.execute(f"CREATE OR REPLACE TABLE ctx AS {CTX_SQL}")
    return con


def _ctx_matrix(con, keys):
    """Fetch context rows for keys (DataFrame sid, tid) in keys' order."""
    con.register("k", keys[["sid", "tid"]].reset_index(drop=True).assign(ix=np.arange(len(keys))))
    m = con.execute("""
        SELECT x.p1, x.p1_rank, x.p1_max, x.p1_sum, x.p1_n05, x.p1 / x.p1_max AS p1_rel, x.p1_trank,
               CASE WHEN x.p1_trank = 1 THEN coalesce(x.t2, 0) ELSE x.tmax END AS p1_tother,
               x.p1 - CASE WHEN x.p1_trank = 1 THEN coalesce(x.t2, 0) ELSE x.tmax END AS p1_tdiff, x.p1_tn05
        FROM k JOIN ctx x USING (sid, tid) ORDER BY k.ix""").fetchnumpy()
    con.unregister("k")
    return np.column_stack([m[c].astype(np.float32) for c in CTX])


def decide(sid, tid, p, thr):
    """Threshold, then give each target only to its best S1. Returns {sid: [tid,...]}."""
    df = pd.DataFrame({"sid": sid, "tid": tid, "p": p})
    df = df[df.p >= thr].sort_values("p", ascending=False).drop_duplicates("tid")
    return df.groupby("sid")["tid"].apply(list).to_dict()


def macro_f05(truth, pred, all_sids):
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
    return f"(hash({col} || 'hold26') % {hold_mod} = 0)"


def _params(threads, seed, leaves=127, lr=0.05):
    return dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=threads or 0, verbose=-1, seed=seed)


def _predict_stage1(con, models, where, threads, chunks, tmp=None, tag=""):
    """Featurize candidate rows matching `where` chunk by chunk; return keys + averaged p1.
    If tmp is given, stage-1 matrices are cached there as .npy for stage 2."""
    keys, p1 = [], []
    for k in range(chunks):
        df = _pairs(con, f"({where}) AND c.sid % {chunks} = {k}")
        if not len(df):
            continue
        X = features(df)
        p1.append(np.mean([m.predict(X, num_threads=threads or 0) for m in models], axis=0))
        keys.append(df[["sid", "tid"]])
        if tmp is not None:
            np.save(Path(tmp) / f"{tag}{k}.npy", X)
        print(f"  [stage1 {tag}] chunk {k + 1}/{chunks}: {len(df):,} pairs", flush=True)
    if not keys:
        return pd.DataFrame({"sid": [], "tid": []}, dtype="int32"), np.empty(0)
    return pd.concat(keys, ignore_index=True), np.concatenate(p1)


def train(work, gt_path, n_train=300_000, hold_mod=22, mem="20GB", threads=None, seed=26, rounds=600,
          chunks=8, leaves=127, lr=0.05):
    t0 = time.time()
    lg = lambda msg: print(f"[train {time.time() - t0:.0f}s] {msg}", flush=True)  # noqa: E731
    con = _connect(work, "train", mem, threads)
    con.execute(f"""CREATE TABLE gt AS SELECT a.rid AS sid, b.rid AS tid FROM (
          SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
          FROM '{gt_path}' WHERE coalesce(matched_entity_ids, '') <> '') g
        JOIN rec a ON a.entity_id = g.s JOIN rec b ON b.entity_id = g.t""")
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
    m = {g: lgb.train(P, lgb.Dataset(data[g]["X"], data[g]["y"], feature_name=FEATS), rounds) for g in ("A", "B")}
    lg("stage-1 models fitted")
    # p1: out-of-fold for A/B, averaged for everything else (holdout + all other train S1s).
    data["A"]["p1"] = m["B"].predict(data["A"]["X"], num_threads=threads or 0)
    data["B"]["p1"] = m["A"].predict(data["B"]["X"], num_threads=threads or 0)
    data["hold"]["p1"] = np.mean([m[g].predict(data["hold"]["X"], num_threads=threads or 0) for g in "AB"], axis=0)
    # Competing S1s outside the sample only matter through shared targets.
    con.execute("""CREATE TABLE tsample AS SELECT DISTINCT tid FROM c
                   WHERE sid IN (SELECT sid FROM split WHERE g IS NOT NULL)""")
    ok, op1 = _predict_stage1(con, [m["A"], m["B"]], "c.sid IN (SELECT sid FROM split WHERE g IS NULL) "
                              "AND c.tid IN (SELECT tid FROM tsample)", threads, chunks, tag="others")
    allk = pd.concat([data[g]["keys"] for g in ("A", "B", "hold")] + [ok], ignore_index=True)
    allp = np.concatenate([data[g]["p1"] for g in ("A", "B", "hold")] + [op1])
    _context(con, allk.sid.values, allk.tid.values, allp)
    for g in ("A", "B", "hold"):
        data[g]["X2"] = np.hstack([data[g]["X"], _ctx_matrix(con, data[g]["keys"])])
        del data[g]["X"]  # stage-1 matrix no longer needed; keeps peak memory down
    lg("stage-2 context built")
    X2 = np.vstack([data["A"].pop("X2"), data["B"].pop("X2")])
    y2 = np.concatenate([data["A"]["y"], data["B"]["y"]])
    hd = data["hold"]
    m2 = lgb.train(P, lgb.Dataset(X2, y2, feature_name=FEATS2), rounds,
                   valid_sets=[lgb.Dataset(hd["X2"], hd["y"], feature_name=FEATS2)],
                   callbacks=[lgb.log_evaluation(100)])
    hold_sids = con.execute("SELECT sid FROM split WHERE g = 'hold'").df().sid.tolist()
    truth = hd["lab"].groupby("sid")["tid"].apply(set).to_dict()
    hk = hd["keys"]
    res = {}
    for name, p in (("stage1", hd["p1"]), ("stage2", m2.predict(hd["X2"], num_threads=threads or 0))):
        grid = {round(t, 3): macro_f05(truth, decide(hk.sid.values, hk.tid.values, p, t), hold_sids)
                for t in np.arange(0.2, 0.96, 0.025)}
        thr = max(grid, key=grid.get)
        res[name] = {"macro_f05": grid[thr], "threshold": thr, "grid": grid}
        lg(f"HOLDOUT {name}: macro F0.5 = {grid[thr]:.5f} at thr={thr}")
        hk = hk.assign(**{f"p_{name}": p})
    recall_ceiling = len(set(zip(hk.sid, hk.tid)) & set(zip(hd["lab"].sid, hd["lab"].tid))) / max(len(hd["lab"]), 1)
    metrics = {"holdout_macro_f05": res["stage2"]["macro_f05"], "threshold": res["stage2"]["threshold"],
               "params": {"n_train": n_train, "rounds": rounds, "leaves": leaves, "lr": lr, "hold_mod": hold_mod},
               "stage1": res["stage1"], "stage2": res["stage2"], "holdout_s1": len(hold_sids),
               "candidate_link_recall": recall_ceiling,
               "importance2": dict(zip(FEATS2, m2.feature_importance("gain").round().tolist())),
               "seconds": time.time() - t0}
    out = Path(work) / "model"
    out.mkdir(parents=True, exist_ok=True)
    m["A"].save_model(str(out / "lgb_A.txt")); m["B"].save_model(str(out / "lgb_B.txt"))
    m2.save_model(str(out / "lgb2.txt"))
    hk.assign(y=hd["y"], p=hk["p_stage2"]).to_parquet(out / "hold_preds.parquet", index=False)
    pd.DataFrame({"sid": hold_sids}).to_parquet(out / "hold_sids.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def predict(work, split, out_dir, mem="20GB", threads=None, chunks=8, tmp=None):
    t0 = time.time()
    md = Path(work) / "model"
    m1 = [lgb.Booster(model_file=str(md / f"lgb_{g}.txt")) for g in "AB"]
    m2 = lgb.Booster(model_file=str(md / "lgb2.txt"))
    thr = json.loads((md / "metrics.json").read_text())["threshold"]
    con = _connect(work, split, mem, threads)
    tmp = Path(tmp or Path(scratch(work)) / "tmp_fe")
    tmp.mkdir(parents=True, exist_ok=True)
    keys, p1 = _predict_stage1(con, m1, "TRUE", threads, chunks, tmp=tmp, tag="x")
    _context(con, keys.sid.values, keys.tid.values, p1)
    p2, off = [], 0
    for k in range(chunks):
        X = np.load(tmp / f"x{k}.npy")
        kk = keys.iloc[off:off + len(X)]
        off += len(X)
        p2.append(m2.predict(np.hstack([X, _ctx_matrix(con, kk)]), num_threads=threads or 0))
        print(f"  [stage2 {split}] chunk {k + 1}/{chunks} {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sid": keys.sid.values, "tid": keys.tid.values, "p1": p1, "p": np.concatenate(p2)}).to_parquet(
        out / "scores.parquet", index=False)
    del keys, p1, p2
    n_match, n_s1 = write_outputs(con, out / "scores.parquet", thr, out)
    print(f"[predict {split}] wrote {n_s1:,} rows, {n_match:,} links ({n_match / n_s1:.2f}/S1) "
          f"{time.time() - t0:.0f}s", flush=True)


def write_outputs(con, scores_path, thr, out, chunks=16):
    """Decision in DuckDB (p >= thr, then each target to its best S1) and stream both TSVs.
    Every S1 in `rec` gets exactly one row; lists may be empty. Aggregation runs in S1 chunks
    so memory stays bounded. Returns (links, S1 rows)."""
    out = Path(out)
    con.execute(f"CREATE OR REPLACE TABLE sc AS SELECT sid, tid, p FROM read_parquet('{scores_path}')")
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
    con.execute(f"SET memory_limit='{mem}'; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{Path(scratch(work)) / 'duck_tmp'}'")
    if threads:
        con.execute(f"SET threads={threads}")
    con.execute(f"CREATE TABLE rec AS SELECT rid, entity_id, src FROM read_parquet('{Path(work) / split}/norm_s*.parquet')")
    n_match, n_s1 = write_outputs(con, scores_path, thr, out)
    print(f"[rewrite {split}] thr={thr} {n_s1:,} rows, {n_match:,} links ({n_match / n_s1:.2f}/S1)", flush=True)
