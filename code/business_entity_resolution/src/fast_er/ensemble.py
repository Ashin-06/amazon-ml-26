"""Average several models of the same code version (different seeds / A-B samples) and write outputs.

All members must share the stage-1 outputs (same candidates) and the hold / tune / final halves (the
tune split is fixed; only the A/B training sample varies with the seed).

Evaluate:  python ensemble.py eval  WORK GT_PARQUET model model_s27 model_s28
Apply:     python ensemble.py apply WORK SPLIT OUT_DIR THRESHOLD out/scores.parquet out_s27/scores.parquet ...
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.model import check_scores, configure, decide, lit, macro_f05, write_outputs  # noqa: E402


def evaluate(work, gt_path, model_dirs):
    """Mean of the members' hold probabilities; threshold tuned on the tune half, F0.5 on the final half
    (hold-only target ownership, applied identically to single members for a fair comparison)."""
    work = Path(work)
    preds = [pd.read_parquet(work / d / "hold_preds.parquet", columns=["sid", "tid", "p"]) for d in model_dirs]
    base = preds[0][["sid", "tid"]]
    for d, p in zip(model_dirs, preds):
        if not p[["sid", "tid"]].equals(base):
            raise ValueError(f"{d}: hold pairs differ from {model_dirs[0]} (members must share candidates and hold)")
    con = duckdb.connect()
    ids = con.execute(f"""SELECT r.rid, s.half FROM read_parquet({lit(work / 'train' / 'norm_s1.parquet')}) r
                          JOIN read_parquet({lit(work / model_dirs[0] / 'split_ids.parquet')}) s USING (entity_id)
                          WHERE s.g = 'hold'""").df()
    tune, final = ids.rid[ids.half == "tune"].tolist(), ids.rid[ids.half == "final"].tolist()
    t2r = con.execute(f"SELECT entity_id, rid FROM read_parquet({lit(work / 'train' / 'norm_s*.parquet')})").df()
    t2r = dict(zip(t2r.entity_id, t2r.rid))
    g = con.execute(f"SELECT * FROM read_parquet({lit(gt_path)})").df()
    hold = set(tune) | set(final)
    truth = {t2r[s]: {t2r[x] for x in m.split(",")} for s, m in zip(g.source1_entity_id, g.matched_entity_ids)
             if m and t2r.get(s) in hold}

    def best(p):
        """(threshold, tune F0.5, final F0.5) with coarse + refined search on the tune half."""
        res = {}
        def at(t):
            t = round(float(t), 4)
            if t not in res:
                d = decide(base.sid.values, base.tid.values, p, t)
                res[t] = (macro_f05(truth, d, tune), macro_f05(truth, d, final))
        for t in np.arange(0.3, 0.96, 0.025):
            at(t)
        b = max(res, key=lambda k: res[k][0])
        for t in np.arange(max(0.0, b - 0.02), min(1.0, b + 0.02) + 1e-9, 0.0025):
            at(t)
        b = max(res, key=lambda k: res[k][0])
        return b, res[b][0], res[b][1]

    out = {}
    for d, p in zip(model_dirs, preds):
        out[d] = best(p.p.values)
        print(f"{d:14s} thr {out[d][0]:.4f} | tune {out[d][1]:.5f} | FINAL {out[d][2]:.5f}", flush=True)
    mean = np.mean([p.p.values for p in preds], axis=0)
    check_scores(mean, name="ensemble hold")
    out["ensemble"] = best(mean)
    print(f"{'ENSEMBLE':14s} thr {out['ensemble'][0]:.4f} | tune {out['ensemble'][1]:.5f} | FINAL {out['ensemble'][2]:.5f}")
    (work / "ensemble_metrics.json").write_text(json.dumps({"members": model_dirs, "results": out}, indent=1))
    return out


def apply(work, split, out_dir, thr, score_paths, mem="8GB", threads=None):
    """Average the members' test probabilities pair by pair and write both TSVs with `thr`."""
    con = duckdb.connect()
    configure(con, mem, threads, Path(work) / "duck_tmp")
    parts = [f"SELECT sid, tid, p FROM read_parquet({lit(p)})" for p in score_paths]
    n = [con.execute(f"SELECT count(*) FROM read_parquet({lit(p)})").fetchone()[0] for p in score_paths]
    if len(set(n)) != 1:
        raise ValueError(f"members scored different numbers of pairs: {n}")
    con.execute(f"CREATE TABLE ens AS SELECT sid, tid, avg(p) AS p, count(*) AS k FROM ({' UNION ALL '.join(parts)}) "
                "GROUP BY sid, tid")
    if con.execute(f"SELECT count(*) FROM ens WHERE k <> {len(score_paths)}").fetchone()[0]:
        raise ValueError("members scored different pair sets")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT sid, tid, p FROM ens) TO {lit(out / 'scores.parquet')} (FORMAT parquet)")
    con.execute(f"CREATE TABLE rec AS SELECT rid, entity_id, src FROM read_parquet({lit(Path(work) / split / 'norm_s*.parquet')})")
    links, rows = write_outputs(con, out / "scores.parquet", thr, out)
    print(f"[ensemble {split}] {len(score_paths)} members, thr {thr}: {rows:,} rows, {links:,} links")


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "apply":
        apply(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6:])
    else:
        sys.exit(__doc__)
