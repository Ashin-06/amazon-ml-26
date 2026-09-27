"""Train labels: for name-only targets whose normalized name equals exactly 2 references' normalized names,
how well do candidate tie-break signals pick the true owner? (accuracy only where the signal discriminates)"""
import duckdb, pandas as pd
pd.set_option('display.width', 250)
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='7GB'")
N = 'work/v21L/run/train'; D = '6ab10eb3b23ba_student_resource/student_resource/dataset/train'
rd = lambda f: f"read_csv('{D}/{f}', delim='\t', header=true, quote='', all_varchar=true)"
con.execute(f"CREATE TABLE r AS SELECT rid, entity_id, src, country, name, addr FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
con.execute(f"CREATE TABLE raw AS SELECT entity_id, business_name bn FROM {rd('train_source1.tsv')} UNION ALL SELECT entity_id, business_name FROM {rd('train_source2.tsv')} UNION ALL SELECT entity_id, business_name FROM {rd('train_source3.tsv')}")
con.execute(f"CREATE TABLE gl AS SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t FROM {rd('train_ground_truth.tsv')} WHERE coalesce(matched_entity_ids,'')<>''")
con.execute("CREATE TABLE nn AS SELECT country, name, list(entity_id) ids FROM r WHERE src=1 GROUP BY ALL HAVING count(*)=2")
con.execute("""CREATE TABLE x AS SELECT t.entity_id te, t.src tsrc, gl.s own, unnest(nn.ids) cand FROM r t JOIN nn USING(country, name) JOIN gl ON gl.t=t.entity_id
   WHERE t.src>1 AND t.addr='' AND list_contains(nn.ids, gl.s)""")
# per-candidate signals
con.execute("CREATE TABLE lk AS SELECT gl.s, gl.t, r.src, r.addr='' ae FROM gl JOIN r ON r.entity_id=gl.t")
con.execute("""CREATE TABLE f AS SELECT x.te, x.cand, x.cand=x.own is_own, x.tsrc,
   (rt.bn = rc.bn) raw_eq, (lower(rt.bn)=lower(rc.bn)) rawci_eq,
   (rt.bn = upper(rt.bn)) = (rc.bn = upper(rc.bn)) case_same,
   (SELECT count(*) FROM lk WHERE lk.s=x.cand AND lk.t<>x.te) n_other,
   (SELECT count(*) FROM lk WHERE lk.s=x.cand AND lk.t<>x.te AND lk.src=x.tsrc) n_same_src,
   (SELECT count(*) FROM lk WHERE lk.s=x.cand AND lk.t<>x.te AND lk.ae) n_other_nameonly
   FROM x JOIN raw rt ON rt.entity_id=x.te JOIN raw rc ON rc.entity_id=x.cand""")
print('targets', con.execute("SELECT count(DISTINCT te) FROM f").fetchone())
for sig in ['case_same', 'n_other', 'n_same_src', 'n_other_nameonly']:
    r = con.execute(f"""WITH g AS (SELECT te, max({sig}::double) mx, min({sig}::double) mn, max(CASE WHEN is_own THEN {sig}::double END) ov FROM f GROUP BY te)
       SELECT count(*) FILTER (WHERE mx<>mn) discr, avg(CASE WHEN ov=mx THEN 1.0 ELSE 0.0 END) FILTER (WHERE mx<>mn) own_is_max FROM g""").fetchone()
    print(f'{sig:18s} discriminates {r[0]:6d} targets; owner has the higher value {r[1]}')
