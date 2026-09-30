"""Apply saved blend coefficients (models/blend/blend.json) to test scores: pairs in the band LO < p < HI get
sigmoid(intercept + a*logit(p) + b*xe + c*logit(p)*xe); every other pair keeps p. Same formula as apply_blend.py,
which fits the coefficients on the tune half of the holdout (full retraining).
usage: python apply_saved_blend.py BLEND.json TEST_SCORES.parquet TEST_XE.parquet OUT.parquet"""
import json
import sys

import duckdb

blend, scores, xe, out = sys.argv[1:5]
b = json.load(open(blend))
lo, hi = b["band"]
q = lambda p: "'" + p.replace("\\", "/") + "'"
lp = "ln(greatest(least(s.p, 1 - 1e-6), 1e-6) / (1 - greatest(least(s.p, 1 - 1e-6), 1e-6)))"
z = f"({b['intercept']} + {b['coef_logit_p']} * {lp} + {b['coef_xe']} * x.xe + {b['coef_interaction']} * {lp} * x.xe)"
con = duckdb.connect()
con.execute("SET threads=4; SET memory_limit='6GB'")
missing = con.execute(f"SELECT count(*) FROM read_parquet({q(scores)}) s ANTI JOIN read_parquet({q(xe)}) x USING(sid, tid) "
                      f"WHERE s.p > {lo} AND s.p < {hi}").fetchone()[0]
if missing:
    raise SystemExit(f"{missing} band pairs have no cross-encoder score")
con.execute(f"""COPY (SELECT s.sid, s.tid, CASE WHEN s.p > {lo} AND s.p < {hi} THEN 1 / (1 + exp(-{z})) ELSE s.p END AS p
    FROM read_parquet({q(scores)}) s LEFT JOIN read_parquet({q(xe)}) x USING(sid, tid) ORDER BY s.sid, s.tid) TO {q(out)} (FORMAT parquet)""")
print("wrote", out)
