"""Stage 3: pair features, LightGBM matcher, threshold tuning, submission writing.

Features per (S1, target) candidate: blocking scores/ranks, rapidfuzz string
similarities on name / core name / address, digit-token agreement, missingness,
and competition context (how many S1s claim the same target, this pair's rank
on both sides). Decision: keep pairs with p >= threshold, then each target goes
only to its highest-probability S1 (training labels show every target belongs
to at most one S1). The threshold maximises exact macro F0.5 on held-out S1s.
"""
import json
import re
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

FEATS = ["s", "sn", "sa", "nk", "r_s", "r_n", "r_a", "n_c", "s_rel", "s_gap", "t_nc", "t_rank", "t_rel",
         "tsrc", "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "c_ratio", "c_tset", "c_nospace",
         "a_ratio", "a_tsort", "a_tset", "a_partial", "t_addr_empty", "len_n_s", "len_n_t", "len_a_s",
         "len_a_t", "num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt", "t_num_cnt"]


def _num_feats(sa, ta):
    out = np.zeros((len(sa), 6), np.float32)
    for i, (x, y) in enumerate(zip(sa, ta)):
        a, b = DIGITS.findall(x), DIGITS.findall(y)
        sa_, sb_ = set(a), set(b)
        inter = len(sa_ & sb_)
        out[i] = (inter / len(sa_ | sb_) if sa_ and sb_ else -1, inter,
                  float(bool(sa_) and bool(sb_) and not inter),
                  float(bool(a) and bool(b) and a[0] == b[0]), len(a), len(b))
    return out


def features(df):
    """df has blocking/group columns + sname,score,saddr,tname,tcore,taddr,tsrc. Returns float32 matrix."""
    sn, tn = df["sname"].tolist(), df["tname"].tolist()
    sc, tc = df["score"].tolist(), df["tcore"].tolist()
    sa, ta = df["saddr"].tolist(), df["taddr"].tolist()
    d = lambda a, b, f: cpdist(a, b, scorer=f, workers=-1, dtype=np.float32)  # noqa: E731
    cols = {c: df[c].to_numpy(np.float32) for c in FEATS[:14]}
    cols.update(n_ratio=d(sn, tn, fuzz.ratio), n_tsort=d(sn, tn, fuzz.token_sort_ratio),
                n_tset=d(sn, tn, fuzz.token_set_ratio), n_partial=d(sn, tn, fuzz.partial_ratio),
                n_jw=d(sn, tn, JaroWinkler.normalized_similarity), c_ratio=d(sc, tc, fuzz.ratio),
                c_tset=d(sc, tc, fuzz.token_set_ratio),
                c_nospace=d([x.replace(" ", "") for x in sc], [x.replace(" ", "") for x in tc], fuzz.ratio),
                a_ratio=d(sa, ta, fuzz.ratio), a_tsort=d(sa, ta, fuzz.token_sort_ratio),
                a_tset=d(sa, ta, fuzz.token_set_ratio), a_partial=d(sa, ta, fuzz.partial_ratio))
    cols["t_addr_empty"] = np.array([not x for x in ta], np.float32)
    for k, v in (("len_n_s", sn), ("len_n_t", tn), ("len_a_s", sa), ("len_a_t", ta)):
        cols[k] = np.fromiter(map(len, v), np.float32, len(v))
    nf = _num_feats(sa, ta)
    for j, k in enumerate(["num_jac", "num_common", "num_conflict", "num_first_eq", "s_num_cnt", "t_num_cnt"]):
        cols[k] = nf[:, j]
    return np.column_stack([cols[c] for c in FEATS])


def _pairs(con, where):
    """Candidate rows (with group features) joined to both records' normalized text."""
    return con.execute(f"""
        SELECT c.*, a.name AS sname, a.core AS score, a.addr AS saddr,
               b.name AS tname, b.core AS tcore, b.addr AS taddr, b.src AS tsrc
        FROM c JOIN rec a ON a.rid = c.sid JOIN rec b ON b.rid = c.tid WHERE {where}""").df()


def _connect(work, split, mem, threads):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET preserve_insertion_order=false")
    if threads:
        con.execute(f"SET threads={threads}")
    w = Path(work) / split
    con.execute(f"CREATE TABLE rec AS SELECT * FROM read_parquet('{w}/norm_s*.parquet')")
    con.execute(GROUP_SQL.format(cands=w / "cands.parquet"))
    return con


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


def train(work, gt_path, n_train=300_000, n_hold=100_000, mem="20GB", threads=None, seed=26, rounds=600):
    t0 = time.time()
    con = _connect(work, "train", mem, threads)
    con.execute(f"""CREATE TABLE gt AS SELECT a.rid AS sid, b.rid AS tid FROM (
          SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
          FROM '{gt_path}' WHERE coalesce(matched_entity_ids, '') <> '') g
        JOIN rec a ON a.entity_id = g.s JOIN rec b ON b.entity_id = g.t""")
    # Disjoint S1 groups by a seeded hash of the entity id (never the numeric value as a feature).
    con.execute(f"""CREATE TABLE split AS SELECT rid AS sid,
          CASE WHEN h < {n_train} THEN 'train' WHEN h < {n_train + n_hold} THEN 'hold' END AS g
        FROM (SELECT rid, row_number() OVER (ORDER BY hash(entity_id || '{seed}')) - 1 AS h FROM rec WHERE src = 1)""")
    data = {}
    for g in ("train", "hold"):
        df = _pairs(con, f"c.sid IN (SELECT sid FROM split WHERE g = '{g}')")
        lab = con.execute(f"SELECT sid, tid FROM gt WHERE sid IN (SELECT sid FROM split WHERE g = '{g}')").df()
        key = set(zip(lab.sid, lab.tid))
        y = np.fromiter(((s, t) in key for s, t in zip(df.sid, df.tid)), np.int8, len(df))
        data[g] = (df[["sid", "tid"]].copy(), features(df), y, lab)
        print(f"[train] {g}: {len(df):,} pairs, pos={y.mean():.3f}, {time.time() - t0:.0f}s", flush=True)
    (_, X, y, _), (hk, Xh, yh, hlab) = data["train"], data["hold"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=threads or 0, verbose=-1, seed=seed)
    model = lgb.train(params, lgb.Dataset(X, y, feature_name=FEATS), rounds,
                      valid_sets=[lgb.Dataset(Xh, yh, feature_name=FEATS)],
                      callbacks=[lgb.log_evaluation(100)])
    ph = model.predict(Xh, num_threads=threads or 0)
    hold_sids = con.execute("SELECT sid FROM split WHERE g = 'hold'").df().sid.tolist()
    truth = hlab.groupby("sid")["tid"].apply(set).to_dict()
    grid = {round(t, 3): macro_f05(truth, decide(hk.sid.values, hk.tid.values, ph, t), hold_sids)
            for t in np.arange(0.2, 0.96, 0.025)}
    thr = max(grid, key=grid.get)
    recall_ceiling = len(set(zip(hk.sid, hk.tid)) & set(zip(hlab.sid, hlab.tid))) / max(len(hlab), 1)
    metrics = {"holdout_macro_f05": grid[thr], "threshold": thr, "grid": grid,
               "holdout_s1": len(hold_sids), "candidate_link_recall": recall_ceiling,
               "importance": dict(zip(FEATS, model.feature_importance("gain").round().tolist())),
               "seconds": time.time() - t0}
    out = Path(work) / "model"
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "lgb.txt"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[train] HOLDOUT macro F0.5 = {grid[thr]:.5f} at thr={thr}  ({time.time() - t0:.0f}s)", flush=True)
    return metrics


def predict(work, split, out_dir, mem="20GB", threads=None, chunks=8):
    t0 = time.time()
    model_dir = Path(work) / "model"
    model = lgb.Booster(model_file=str(model_dir / "lgb.txt"))
    thr = json.loads((model_dir / "metrics.json").read_text())["threshold"]
    con = _connect(work, split, mem, threads)
    sids, tids, ps = [], [], []
    for k in range(chunks):
        df = _pairs(con, f"c.sid % {chunks} = {k}")
        ps.append(model.predict(features(df), num_threads=threads or 0))
        sids.append(df.sid.values); tids.append(df.tid.values)
        print(f"[predict {split}] chunk {k + 1}/{chunks} {len(df):,} pairs {time.time() - t0:.0f}s", flush=True)
    sid, tid, p = np.concatenate(sids), np.concatenate(tids), np.concatenate(ps)
    pred = decide(sid, tid, p, thr)
    ids = con.execute("SELECT rid, entity_id, src FROM rec").df()
    name = dict(zip(ids.rid, ids.entity_id))
    s1 = ids[ids.src == 1].rid.tolist()
    cand = pd.DataFrame({"sid": sid, "tid": tid}).groupby("sid")["tid"].apply(list).to_dict()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, col, mapping in (("matching_results.tsv", "matched_entity_ids", pred),
                                ("candidate_pairs.tsv", "candidate_entity_ids", cand)):
        with open(out / fname, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for s in s1:
                f.write(name[s] + "\t" + ",".join(name[t] for t in mapping.get(s, ())) + "\n")
    pd.DataFrame({"sid": sid, "tid": tid, "p": p}).to_parquet(out / "scores.parquet", index=False)
    n_match = sum(map(len, pred.values()))
    print(f"[predict {split}] wrote {len(s1):,} rows, {n_match:,} links ({n_match / len(s1):.2f}/S1) {time.time() - t0:.0f}s", flush=True)
