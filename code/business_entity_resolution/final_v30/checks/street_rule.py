"""Pattern: identical normalized name, identical address once digits are removed (same street/city), different
house number. Holdout: precision among all such pairs, among blocked-but-rejected ones, and never-retrieved ones.
Test: how many French pairs of this pattern are not accepted."""
import duckdb
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'; T='work/v21L/run/test'
con=duckdb.connect(); con.execute("SET threads=6; SET memory_limit='7GB'")
strip="trim(regexp_replace(regexp_replace(addr,'[0-9]+[a-z]*',' ','g'),' +',' ','g'))"
def pairs(P, sids_sql):
    con.execute(f"CREATE OR REPLACE TABLE r AS SELECT rid, entity_id, src, country, name, addr, {strip} street, regexp_extract(addr,'[0-9]+') num FROM read_parquet(['{P}/norm_s1.parquet','{P}/norm_s2.parquet','{P}/norm_s3.parquet'])")
    con.execute(f"""CREATE OR REPLACE TABLE sp AS SELECT s.rid sid, t.rid tid, s.country FROM r s JOIN r t ON t.src>1 AND t.country=s.country AND t.name=s.name AND t.street=s.street
       WHERE s.src=1 AND s.rid IN ({sids_sql}) AND length(s.street)>=8 AND s.num<>'' AND t.num<>'' AND s.num<>t.num""")
pairs(N, f"SELECT DISTINCT sid FROM read_parquet('{M}/hold_preds.parquet')")
con.execute("CREATE TABLE gt AS SELECT * FROM 'work/comp/hold_truth.parquet'")
con.execute(f"CREATE TABLE lk AS SELECT sid, tid, lgb_small_tree q FROM read_parquet('{S}/final_v1/locked_predictions.parquet')")
con.execute(f"CREATE TABLE hp AS SELECT sid, tid, p FROM read_parquet('{M}/hold_preds.parquet')")
print('HOLDOUT (India/US):', con.execute("""SELECT sp.country, CASE WHEN hp.sid IS NULL THEN 'not_retrieved' WHEN coalesce(lk.q,hp.p)>=0.77 THEN 'accepted' ELSE 'rejected' END st,
   count(*) n, round(avg((gt.sid IS NOT NULL)::int),3) prec FROM sp LEFT JOIN gt USING(sid,tid) LEFT JOIN hp USING(sid,tid) LEFT JOIN lk USING(sid,tid) GROUP BY ALL ORDER BY ALL""").fetchall())
pairs(T, f"SELECT rid FROM read_parquet('{T}/norm_s1.parquet') WHERE country='france'")
con.execute("CREATE TABLE sub AS SELECT DISTINCT unnest(string_split(matched_entity_ids, ',')) t, source1_entity_id s FROM read_csv('submissions/v27_matching_results.tsv', delim='\t', header=true, quote='', all_varchar=true) WHERE coalesce(matched_entity_ids,'')<>''")
con.execute("CREATE TABLE b AS SELECT sid, tid, p FROM read_parquet('work/v24x4f/scores_blend.parquet') SEMI JOIN sp USING(sid,tid)")
print('FRANCE test:', con.execute("""SELECT CASE WHEN a.s IS NOT NULL THEN 'accepted' WHEN o.t IS NOT NULL THEN 'target_owned_by_other' WHEN b.sid IS NULL THEN 'not_retrieved' ELSE 'rejected' END st, count(*)
   FROM sp JOIN r s ON s.rid=sp.sid JOIN r t ON t.rid=sp.tid LEFT JOIN sub a ON a.s=s.entity_id AND a.t=t.entity_id LEFT JOIN (SELECT DISTINCT t FROM sub) o ON o.t=t.entity_id
   LEFT JOIN b USING(sid,tid) GROUP BY 1 ORDER BY 1""").fetchall())
con.execute("COPY (SELECT sp.sid, sp.tid, b.p FROM sp LEFT JOIN b USING(sid,tid)) TO 'work/comp/v25/fr_street_pairs.parquet'")
