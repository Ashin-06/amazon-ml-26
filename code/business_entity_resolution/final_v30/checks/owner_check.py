"""Ownership mistakes: target assigned to reference A although another reference B has exactly the target's
normalized name AND address (and A does not). Counts per country for a matching file."""
import sys, duckdb
F = sys.argv[1]; T = 'work/v21L/run/test'
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE r AS SELECT entity_id, src, country, name, addr FROM read_parquet(['{T}/norm_s1.parquet','{T}/norm_s2.parquet','{T}/norm_s3.parquet'])")
con.execute(f"CREATE TABLE sub AS SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t FROM read_csv('{F}', delim='\t', header=true, quote='', all_varchar=true) WHERE coalesce(matched_entity_ids,'')<>''")
con.execute("""CREATE TABLE x AS SELECT sub.s, sub.t, a.country, (a.name=t.name AND a.addr=t.addr) a_exact,
   (SELECT any_value(b.entity_id) FROM r b WHERE b.src=1 AND b.country=t.country AND b.name=t.name AND b.addr=t.addr AND t.addr<>'' AND b.entity_id<>sub.s) better
   FROM sub JOIN r a ON a.entity_id=sub.s JOIN r t ON t.entity_id=sub.t""")
print(con.execute("SELECT country, count(*) links, count(*) FILTER (WHERE better IS NOT NULL AND NOT a_exact) wrong_owner FROM x GROUP BY 1 ORDER BY 1").fetchall())
con.execute("COPY (SELECT s, t, better FROM x WHERE better IS NOT NULL AND NOT a_exact) TO 'work/comp/v25/owner_fix.parquet'")
