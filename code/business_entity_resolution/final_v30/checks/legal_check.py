import duckdb
con=duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
LF="['inc','llc','corp','ltd','co','pvt','lp','llp','pc','pllc','sarl','sas','sa','eurl','sasu','sci','ei','snc','private','limited']"
SP='C:/Users/ashin/AppData/Local/Temp/claude/D--Amazon-ML-26/a6f59d5d-823b-4509-a3c7-7eb2c7f6f181/scratchpad'
def kind(P, pairs, KEY='rid'):
    con.execute(f"CREATE OR REPLACE TABLE r AS SELECT rid, entity_id, country, name, addr FROM read_parquet(['{P}/norm_s1.parquet','{P}/norm_s2.parquet','{P}/norm_s3.parquet'])")
    con.execute(f"""CREATE OR REPLACE TABLE k AS SELECT pp.*, s.country, s.addr<>t.addr AND t.addr<>'' diffaddr,
      list_sort(list_filter(string_split(s.name,' '), w-> list_contains({LF},w))) sl, list_sort(list_filter(string_split(t.name,' '), w-> list_contains({LF},w))) tl,
      list_sort(list_filter(string_split(s.name,' '), w-> NOT list_contains({LF},w))) sb, list_sort(list_filter(string_split(t.name,' '), w-> NOT list_contains({LF},w))) tb
      FROM ({pairs}) pp JOIN r s ON s.{KEY}=pp.sid JOIN r t ON t.{KEY}=pp.tid""")
    return "sb=tb AND len(sl)>0 AND len(tl)>0 AND sl<>tl AND diffaddr"
cond=kind('work/v21L/run/train', f"SELECT h.sid, h.tid, h.y::int y, coalesce(l.lgb_small_tree,h.p) q FROM read_parquet('work/v24/kaggle/work/model/hold_preds.parquet') h LEFT JOIN read_parquet('artifacts/push995-20260927-0915/final_v1/locked_predictions.parquet') l USING(sid,tid) WHERE h.p>0.001")
print('HOLDOUT legal-swap @ different address:', con.execute(f"SELECT country, count(*) FILTER (WHERE q>=0.77) accepted, round(avg(y) FILTER (WHERE q>=0.77),3) accepted_precision FROM k WHERE {cond} GROUP BY 1").fetchall())
cond=kind('work/v21L/run/test', "SELECT source1_entity_id sid, unnest(string_split(matched_entity_ids, ',')) tid FROM read_csv('submissions/v30_frxe2_matching_results.tsv', delim='\t', header=true, quote='', all_varchar=true) WHERE coalesce(matched_entity_ids,'')<>''", 'entity_id')
print('v30 accepted legal-swap @ different address:', con.execute(f"SELECT country, count(*), round(count(*)*1.0/any_value(n),4) per_link FROM k JOIN (SELECT country c, count(*) n FROM k GROUP BY 1) z ON z.c=k.country WHERE {cond} GROUP BY 1").fetchall())
