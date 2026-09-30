# Does a cross-encoder blend improve the complete-reference macro F0.5? (go/no-go, holdout only)
# Blend = logistic regression on [logit(p), xe, logit(p)*xe], fitted on TUNE-half band pairs only; pairs outside the
# band keep p. Cutoff chosen on the tune half (same grid for baseline and blend); final half evaluated once, all
# final-half references in the denominator (incl. no-candidate references). Ownership: one S1 per target among
# holdout pairs (hold-only diagnostic unless the run saved global decisions). Saved tune/final split (split_ids).
# Usage: python work/blend_eval.py HOLD_PREDS SPLIT_IDS NORM_DIR XE_SCORES [country breakdown]
import sys, duckdb, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package src/ (fast_er)
from fast_er.model import decide, macro_f05
hp_path, split_path, norm, xe_path = sys.argv[1:5]
con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='1500MB'")
hp = pd.read_parquet(hp_path, columns=["sid", "tid", "p", "y"])
xe = pd.read_parquet(xe_path)
sp = pd.read_parquet(split_path)
con.register("sp", sp[sp.g == "hold"])
ref = con.execute(f"""SELECT r.rid, sp.half, r.country FROM read_parquet('{norm}/norm_s1.parquet') r
                      JOIN sp ON sp.entity_id = r.entity_id""").df()
half = dict(zip(ref.rid, ref.half)); country = dict(zip(ref.rid, ref.country))
tune = sorted(ref.rid[ref.half == "tune"]); final = sorted(ref.rid[ref.half == "final"])
con.register("ref", ref[["rid"]])
lk = con.execute(f"""
  WITH h AS (SELECT r.rid, r.entity_id FROM read_parquet('{norm}/norm_s1.parquet') r JOIN ref USING (rid)),
       g AS (SELECT h.rid AS sid, unnest(string_split(g.matched_entity_ids, ',')) AS t
             FROM 'work/pq/train_ground_truth.parquet' g JOIN h ON h.entity_id = g.source1_entity_id
             WHERE coalesce(g.matched_entity_ids, '') <> '')
  SELECT g.sid, t.rid AS tid FROM g JOIN read_parquet(['{norm}/norm_s2.parquet', '{norm}/norm_s3.parquet']) t ON t.entity_id = g.t""").df()
truth = lk.groupby("sid").tid.apply(set).to_dict()
d = hp.merge(xe, on=["sid", "tid"], how="left")
band = d.xe.notna().values
assert band.sum() == len(xe), "xe pairs must align exactly with (sid, tid) of the holdout band"
lp = np.log(np.clip(d.p, 1e-6, 1 - 1e-6) / (1 - np.clip(d.p, 1e-6, 1 - 1e-6))).values
X = np.c_[lp, d.xe.fillna(0), lp * d.xe.fillna(0)]
is_tune = d.sid.map(half).eq("tune").values
fit = band & is_tune
lr = LogisticRegression(max_iter=2000).fit(X[fit], d.y.values[fit])
pb = d.p.values.copy()
pb[band] = lr.predict_proba(X[band])[:, 1]
print(f"blend fitted on {int(fit.sum()):,} tune-half band pairs; coefficients {np.round(lr.coef_[0], 3)}, intercept {lr.intercept_[0]:.3f}")


def pick(p):
    grid = np.round(np.arange(0.40, 0.991, 0.01), 3)
    best = max(grid, key=lambda t: macro_f05(truth, decide(d.sid.values, d.tid.values, p, float(t)), tune))
    dec = decide(d.sid.values, d.tid.values, p, float(best))
    return best, macro_f05(truth, dec, tune), macro_f05(truth, dec, final), dec


for name, p in (("baseline (model score)", d.p.values), ("blend (model + cross-encoder)", pb)):
    t, ft, ff, dec = pick(p)
    by = {c: macro_f05(truth, dec, [s for s in final if country[s] == c]) for c in sorted(set(country.values()))}
    print(f"{name:32s} cutoff {t:.2f} | tune {ft:.5f} | FINAL {ff:.5f} | " + " ".join(f"{c} {v:.5f}" for c, v in by.items()))
