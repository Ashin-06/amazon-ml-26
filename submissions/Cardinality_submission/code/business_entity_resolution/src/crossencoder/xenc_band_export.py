# Export the uncertain band (0.02 < p < 0.995) of a run's holdout pairs, with raw "name | address" text, for
# cross-encoder scoring. Usage: python work/xenc_band_export.py SCORES NORM_DIR PQ_DIR OUT.parquet [train|test] [LO HI]
# (LO < p < HI; default 0.02 0.995; HI > 1 includes p = 1)
import sys, duckdb, pandas as pd
hp_path, norm, pq, out = sys.argv[1:5]
split = sys.argv[5] if len(sys.argv) > 5 else "train"
lo, hi = (float(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 7 else (0.02, 0.995)
con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='1500MB'")
# band filtered in DuckDB before pandas (test scores have ~165M rows; review 2026-09-26 P1)
b = con.execute(f"SELECT sid, tid, p{', y' if split == 'train' else ''} FROM read_parquet('{hp_path}') "
                f"WHERE p > {lo} AND p < {hi} ORDER BY sid, tid").df()
con.register("b", b)
txt = f"""(SELECT n.rid, coalesce(r.business_name, '') || ' | ' || coalesce(r.business_address, '') AS txt
          FROM read_parquet(['{norm}/norm_s1.parquet','{norm}/norm_s2.parquet','{norm}/norm_s3.parquet']) n
          JOIN read_parquet(['{pq}/{split}_source1.parquet','{pq}/{split}_source2.parquet','{pq}/{split}_source3.parquet']) r
          ON r.entity_id = n.entity_id WHERE n.rid IN (SELECT sid FROM b UNION SELECT tid FROM b))"""
d = con.execute(f"SELECT b.*, x.txt AS a, z.txt AS b FROM b JOIN {txt} x ON x.rid = b.sid JOIN {txt} z ON z.rid = b.tid").df()
assert len(d) == len(b), (len(d), len(b))
d.to_parquet(out, index=False)
print(out, len(d), "pairs", ("positives " + str(int(d.y.sum()))) if "y" in d else "")
