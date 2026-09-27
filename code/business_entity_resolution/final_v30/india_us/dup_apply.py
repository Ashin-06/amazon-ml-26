"""Test: add (R, B) when B is unassigned, an exact name+address duplicate of a target A assigned to R, B has a unique
such R, and (R, B) is already a v24 candidate pair. usage: python dup_apply.py BASE.tsv OUT.tsv"""
import sys, duckdb
from pathlib import Path
base, out = sys.argv[1:3]; T = 'work/v21L/run/test'
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE r AS SELECT rid, entity_id, src, country, name, addr FROM read_parquet(['{T}/norm_s1.parquet','{T}/norm_s2.parquet','{T}/norm_s3.parquet'])")
con.execute(f"CREATE TABLE sub AS SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t FROM read_csv('{base}', delim='\t', header=true, quote='', all_varchar=true) WHERE coalesce(matched_entity_ids,'')<>''")
con.execute("CREATE TABLE acc AS SELECT a.rid sid, t.rid tid FROM sub JOIN r a ON a.entity_id=sub.s JOIN r t ON t.entity_id=sub.t")
con.execute("""CREATE TABLE add AS SELECT DISTINCT a.sid, b.rid tid FROM acc a JOIN r ta ON ta.rid=a.tid JOIN r b ON b.src>1 AND b.rid<>ta.rid AND b.country=ta.country AND b.name=ta.name AND b.addr=ta.addr AND ta.addr<>''
   WHERE b.rid NOT IN (SELECT tid FROM acc)""")
con.execute("CREATE TABLE add1 AS SELECT * FROM add QUALIFY count(*) OVER (PARTITION BY tid)=1")
con.execute("CREATE TABLE addc AS SELECT a.* FROM add1 a SEMI JOIN read_parquet('work/v24/kaggle/out/scores.parquet') c ON c.sid=a.sid AND c.tid=a.tid")
print('duplicate additions', con.execute("SELECT count(*) FROM add1").fetchone(), 'in candidate set', con.execute("SELECT count(*) FROM addc").fetchone(),
      con.execute("SELECT r.country, count(*) FROM addc JOIN r ON r.rid=addc.sid GROUP BY 1").fetchall())
con.execute(f"COPY (SELECT s.entity_id source1_entity_id, string_agg(t.entity_id, ',' ORDER BY t.entity_id) matched_entity_ids FROM addc a JOIN r s ON s.rid=a.sid JOIN r t ON t.rid=a.tid GROUP BY 1) TO '{out}.extra.tsv' (DELIMITER '\t', HEADER, QUOTE '')")
