# Apply the cross-encoder blend to a run's TEST scores. The blend (logistic regression on [logit(p), xe, logit(p)*xe])
# is fitted exactly as in work/blend_eval.py: on the TUNE half of the holdout's uncertain band (0.02 < p < 0.995).
# Test pairs in the same band get the blended probability; every other pair keeps p. Aligned by (sid, tid); fails if a
# band pair has no cross-encoder score. Output: parquet (sid, tid, p) usable by work/rethreshold.py, + coefficients JSON.
# Usage: python work/apply_blend.py HOLD_PREDS SPLIT_IDS NORM_TRAIN XE_HOLD TEST_SCORES XE_TEST OUT_SCORES OUT_JSON [LO HI]
# (band LO < p < HI, default 0.02 0.995; must match the band exported for cross-encoder scoring)
import json, sys, duckdb, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
hp_path, split_path, norm, xe_hold, test_scores, xe_test, out, out_json = sys.argv[1:9]
lo, hi = (float(sys.argv[9]), float(sys.argv[10])) if len(sys.argv) > 10 else (0.02, 0.995)
con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='2GB'; SET temp_directory='work/duck_tmp_rt'")
sp = pd.read_parquet(split_path)
con.register("sp", sp[sp.half == "tune"])
tune = set(con.execute(f"SELECT r.rid FROM read_parquet('{norm}/norm_s1.parquet') r JOIN sp ON sp.entity_id = r.entity_id").df().rid)
d = pd.read_parquet(hp_path, columns=["sid", "tid", "p", "y"]).merge(pd.read_parquet(xe_hold), on=["sid", "tid"], how="inner")
d = d[d.sid.isin(tune)]
lp = np.log(np.clip(d.p, 1e-6, 1 - 1e-6) / (1 - np.clip(d.p, 1e-6, 1 - 1e-6))).values
lr = LogisticRegression(max_iter=2000).fit(np.c_[lp, d.xe, lp * d.xe], d.y.values)
b0, (b1, b2, b3) = float(lr.intercept_[0]), map(float, lr.coef_[0])
n_band = con.execute(f"SELECT count(*) FROM read_parquet('{test_scores}') WHERE p > {lo} AND p < {hi}").fetchone()[0]
n_xe = con.execute(f"""SELECT count(*) FROM read_parquet('{test_scores}') s JOIN read_parquet('{xe_test}') x USING (sid, tid)
                       WHERE s.p > {lo} AND s.p < {hi}""").fetchone()[0]
if n_band != n_xe or n_xe != con.execute(f"SELECT count(*) FROM read_parquet('{xe_test}')").fetchone()[0]:
    raise SystemExit(f"cross-encoder scores do not cover the test band exactly: band {n_band}, matched {n_xe}")
lpx = "ln(least(greatest(s.p, 1e-6), 1 - 1e-6) / (1 - least(greatest(s.p, 1e-6), 1 - 1e-6)))"
con.execute(f"""COPY (SELECT s.sid, s.tid,
      CASE WHEN x.xe IS NULL THEN s.p ELSE 1 / (1 + exp(-({b0} + {b1} * {lpx} + {b2} * x.xe + {b3} * {lpx} * x.xe))) END AS p
    FROM read_parquet('{test_scores}') s LEFT JOIN read_parquet('{xe_test}') x USING (sid, tid)) TO '{out}' (FORMAT parquet)""")
n_out, bad = con.execute(f"SELECT count(*), count(*) FILTER (WHERE p IS NULL OR isnan(p) OR p < 0 OR p > 1) FROM '{out}'").fetchone()
n_in = con.execute(f"SELECT count(*) FROM read_parquet('{test_scores}')").fetchone()[0]
if n_out != n_in or bad:
    raise SystemExit(f"output mismatch: {n_out} vs {n_in} rows, {bad} invalid probabilities")
info = {"intercept": b0, "coef_logit_p": b1, "coef_xe": b2, "coef_interaction": b3, "fit_pairs": int(len(d)),
        "band": [lo, hi], "test_band_pairs": int(n_band), "rows": int(n_out)}
json.dump(info, open(out_json, "w"), indent=1)
print(info)
