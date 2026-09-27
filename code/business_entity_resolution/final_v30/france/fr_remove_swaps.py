"""Remove accepted French links where the target sits at the reference's exact address but swaps a real
reference word for another real French word ("faune club sarl" -> "faune ecole sarl"). France has 6x the US
rate of such same-address swaps, i.e. mostly sibling businesses, not corruptions (US analog: 90% true at the
base rate, so at 6x the excess is decoys). usage: python fr_remove_swaps.py BASE.tsv TEST_NORM_DIR OUT.tsv
"""
import sys, csv, duckdb
base, norm, out = sys.argv[1:4]
q = lambda p: "'" + p.replace(chr(92), '/') + "'"
LEG = "['inc','llc','corp','ltd','co','pvt','lp','llp','pc','pllc','sarl','sas','sa','eurl','sasu','sci','ei','snc','the','of','&','and','a','c','l','s','p','d','de','du','des','la','le','et','st','ste','ets']"
KEEP = "['fils','groupe','associes','developpement','frs','svc','ctr','france','partners','assoc']"
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE r AS SELECT entity_id, src, name, addr FROM read_parquet([{q(norm+'/norm_s1.parquet')},{q(norm+'/norm_s2.parquet')},{q(norm+'/norm_s3.parquet')}]) WHERE country='france'")
con.execute("CREATE TABLE voc AS SELECT w FROM (SELECT unnest(string_split(name,' ')) w FROM r WHERE src=1) GROUP BY w HAVING count(*)>=20 AND length(w)>=4")
con.execute(f"CREATE TABLE sub AS SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t FROM read_csv({q(base)}, delim='\t', header=true, quote='', all_varchar=true) WHERE coalesce(matched_entity_ids,'')<>''")
con.execute(f"""CREATE TABLE p AS SELECT sub.s, sub.t,
  list_filter(string_split(a.name,' '), w-> NOT list_contains({LEG},w) AND NOT list_contains({KEEP},w) AND NOT list_contains(string_split(b.name,' '),w)) dropped,
  list_filter(string_split(b.name,' '), w-> NOT list_contains({LEG},w) AND NOT list_contains({KEEP},w) AND NOT list_contains(string_split(a.name,' '),w)) added
  FROM sub JOIN r a ON a.entity_id=sub.s AND a.src=1 JOIN r b ON b.entity_id=sub.t WHERE a.addr=b.addr AND a.addr<>''""")
con.execute("""CREATE TABLE rm AS SELECT s, t FROM p, (SELECT list(w) vl FROM voc) v WHERE len(dropped)>=1 AND len(added)>=1
  AND list_bool_and(list_transform(added, w-> list_contains(vl, w)))
  AND list_bool_and(list_transform(added, a-> list_bool_and(list_transform(dropped, d-> length(d)>=4 AND levenshtein(a,d)>2 AND NOT starts_with(a,d) AND NOT starts_with(d,a)))))""")
rm = {}
for s, t in con.execute("SELECT s, t FROM rm").fetchall():
    rm.setdefault(s, set()).add(t)
print('links removed', sum(map(len, rm.values())), 'references', len(rm))
with open(base, encoding='utf-8', newline='') as fi, open(out, 'w', encoding='utf-8', newline='') as fo:
    rd = csv.reader(fi, delimiter='\t', quoting=csv.QUOTE_NONE); wr = csv.writer(fo, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
    wr.writerow(next(rd))
    for row in rd:
        if row[0] in rm:
            row = [row[0], ','.join(x for x in row[1].split(',') if x and x not in rm[row[0]])]
        wr.writerow(row)
