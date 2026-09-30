"""True (reference, target) pairs of the held-out references, in the pipeline's record ids, for choosing the
rescue thresholds on the tune half. usage: python src/rescue/make_hold_truth.py DATASET_DIR NORM_TRAIN_DIR HOLD_PREDS OUT"""
import sys

import duckdb

data, norm, hold_preds, out = sys.argv[1:5]
con = duckdb.connect()
con.execute("SET threads=4; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE ids AS SELECT rid, entity_id FROM read_parquet(['{norm}/norm_s1.parquet','{norm}/norm_s2.parquet','{norm}/norm_s3.parquet'])")
con.execute(f"CREATE TABLE hs AS SELECT DISTINCT sid FROM read_parquet('{hold_preds}')")
con.execute(f"""COPY (SELECT a.rid sid, b.rid tid FROM (SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t
    FROM read_csv('{data}/train/train_ground_truth.tsv', delim='\t', header=true, quote='', all_varchar=true)
    WHERE coalesce(matched_entity_ids,'')<>'') g JOIN ids a ON a.entity_id=g.s JOIN ids b ON b.entity_id=g.t
    WHERE a.rid IN (SELECT sid FROM hs)) TO '{out}' (FORMAT parquet)""")
print(out, con.execute(f"SELECT count(*), count(DISTINCT sid) FROM '{out}'").fetchone())
