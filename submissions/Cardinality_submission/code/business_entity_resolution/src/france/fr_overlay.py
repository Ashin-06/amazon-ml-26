"""France-only recall overlay on an existing matching file (no model change).

Adds French candidate pairs the models reject but whose name/address pattern is a true-match pattern at
>=94% precision in the India/US labels (French words the models never saw):
  F1 same exact address, target = subset of reference content words + a French generic suffix
     (fils/groupe/associes/developpement/frs/svc; "france" too) with no decoy suffix
     (India/US analog: Services/Center/Partners suffix at same address, 99.9% true on holdout).
  F2 reference alone at that address, target is one token absent from every French reference name
     (made-up rebrand / glued name / acronym), candidate score >= F2_MIN (India/US analog: 87-95% true).
  F4 target has no address and its core name belongs to exactly one French reference, score >= 0.3
     (India/US analog: 96.8% true on holdout; France assigns 90% of these vs 96-97% India/US).
  F3 (optional) different address, same content words (any order), legal form equal or added/dropped but
     never swapped (US: swapped legal form at another address is 31% true), best-scoring reference for that
     target, score >= 0.6 (India/US analog 87.6-99.3% true).
  F5 (optional) like F3 but the target may drop reference words / add a generic suffix, score >= 0.5
     (India/US analog: different-address subset 91.9% overall, 71% in the 0.5-0.77 band).
Priority per target F1 > F4 > F2 > F3 > F5.
Only pairs in the exported candidate set, only targets nobody owns yet, one reference per target (highest score).
usage: python work/fr_overlay.py BASE.tsv SCORES.parquet TEST_NORM_DIR OUT.tsv [F2_MIN] [RULES]
"""
import sys, duckdb, csv
base, scores, norm, out = sys.argv[1:5]
f2min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
rules = sys.argv[6].split(',') if len(sys.argv) > 6 else ['F1', 'F2', 'F4']
q = lambda p: "'" + p.replace(chr(92), '/') + "'"
LEG = "['inc','llc','corp','ltd','co','pvt','lp','llp','pc','pllc','sarl','sas','sa','eurl','sasu','sci','ei','snc','the','of','&','and','a','c','l','s','p','d','de','du','des','la','le','et','st','ste','ets']"
TRUE = "['fils','groupe','associes','developpement','frs','svc','france']"
LF = "['inc','llc','corp','ltd','co','pvt','lp','llp','pc','pllc','sarl','sas','sa','eurl','sasu','sci','ei','snc','private','limited']"
DECOY = "['holding','participations','distribution','intl','groupement','ent','ind','sol','bros','group','india']"
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE r AS SELECT rid, entity_id, src, name, core, addr FROM read_parquet([{q(norm+'/norm_s1.parquet')},{q(norm+'/norm_s2.parquet')},{q(norm+'/norm_s3.parquet')}]) WHERE country='france'")
con.execute("CREATE TABLE vocab AS SELECT DISTINCT unnest(string_split(name,' ')) w FROM r WHERE src=1")
con.execute("CREATE TABLE lone AS SELECT addr FROM r WHERE src=1 AND addr<>'' GROUP BY 1 HAVING count(*)=1")
con.execute(f"CREATE TABLE sub AS SELECT source1_entity_id s, unnest(string_split(matched_entity_ids, ',')) t FROM read_csv({q(base)}, delim='\t', header=true, quote='', all_varchar=true) WHERE coalesce(matched_entity_ids,'')<>''")
con.execute("CREATE TABLE uc AS SELECT core FROM r WHERE src=1 GROUP BY 1 HAVING count(*)=1")
con.execute(f"""CREATE TABLE c AS SELECT b.sid, b.tid, b.p, s.entity_id se, t.entity_id te, s.name sn, t.name tn, s.addr sa, t.addr ta, s.core scr, t.core tcr, max(b.p) OVER (PARTITION BY b.tid) pmax
  FROM read_parquet({q(scores)}) b JOIN r s ON s.rid=b.sid AND s.src=1 JOIN r t ON t.rid=b.tid AND t.src>1
  WHERE t.entity_id NOT IN (SELECT t FROM sub)""")
con.execute(f"""CREATE TABLE f AS SELECT *,
  list_filter(string_split(sn,' '), w-> NOT list_contains({LEG},w) AND NOT list_contains({TRUE},w)) sc,
  list_filter(string_split(tn,' '), w-> NOT list_contains({LEG},w) AND NOT list_contains({TRUE},w) AND NOT list_contains({DECOY},w)) tc,
  len(list_filter(string_split(tn,' '), w-> list_contains({TRUE},w) AND NOT list_contains(string_split(sn,' '),w))) ntrue,
  len(list_filter(string_split(tn,' '), w-> list_contains({DECOY},w) AND NOT list_contains(string_split(sn,' '),w))) ndecoy
  FROM c WHERE (sa=ta AND ta<>'') OR (ta='' AND tcr=scr) OR (sa<>ta AND ta<>'' AND sa<>'' AND p>=0.5)""")
con.execute(f"""CREATE TABLE adds AS SELECT 'F1' rl, se, te, p FROM f
  WHERE ntrue>=1 AND ndecoy=0 AND len(list_intersect(sc,tc))>=1 AND len(list_filter(tc, w-> NOT list_contains(sc,w)))=0
  AND sa=ta
  UNION ALL SELECT 'F4', se, te, p FROM f WHERE ta='' AND tcr=scr AND scr IN (SELECT core FROM uc) AND p>=0.3
  UNION ALL SELECT 'F3', se, te, p FROM f WHERE sa<>ta AND ta<>'' AND p>=0.6 AND ndecoy=0 AND p=pmax
    AND list_sort(list_filter(string_split(sn,' '), w-> NOT list_contains({LF},w))) = list_sort(list_filter(string_split(tn,' '), w-> NOT list_contains({LF},w)))
    AND NOT (len(list_filter(string_split(sn,' '), w-> list_contains({LF},w)))>0 AND len(list_filter(string_split(tn,' '), w-> list_contains({LF},w)))>0
             AND list_sort(list_filter(string_split(sn,' '), w-> list_contains({LF},w))) <> list_sort(list_filter(string_split(tn,' '), w-> list_contains({LF},w))))
  UNION ALL SELECT 'F5', se, te, p FROM f WHERE sa<>ta AND ta<>'' AND p>=0.5 AND ndecoy=0 AND p=pmax
    AND len(list_intersect(sc,tc))>=1 AND len(list_filter(tc, w-> NOT list_contains(sc,w)))=0
    AND NOT (len(list_filter(string_split(sn,' '), w-> list_contains({LF},w)))>0 AND len(list_filter(string_split(tn,' '), w-> list_contains({LF},w)))>0
             AND list_sort(list_filter(string_split(sn,' '), w-> list_contains({LF},w))) <> list_sort(list_filter(string_split(tn,' '), w-> list_contains({LF},w))))
  UNION ALL SELECT 'F2', se, te, p FROM f
  WHERE sa=ta AND sa IN (SELECT addr FROM lone) AND ntrue=0 AND ndecoy=0 AND len(tc)=1 AND len(list_intersect(sc,tc))=0
    AND tc[1] NOT IN (SELECT w FROM vocab) AND length(tc[1])>=2 AND p>={f2min}""")
PRIO = "CASE rl WHEN 'F1' THEN 4 WHEN 'F4' THEN 3 WHEN 'F2' THEN 2 WHEN 'F3' THEN 1 ELSE 0 END"
con.execute(f"""CREATE TABLE pick AS SELECT te, se, rl FROM (SELECT *, row_number() OVER (PARTITION BY te ORDER BY {PRIO} DESC, p DESC, se) k
  FROM adds WHERE rl IN ({','.join(repr(x) for x in rules)})) WHERE k=1""")  # deterministic: rule priority, score, then reference ID
print('added by rule', con.execute("SELECT rl, count(*), count(DISTINCT se) FROM pick GROUP BY 1").fetchall(), flush=True)
con.execute(f"COPY (SELECT a.rl, s.name sn, t.name tn, s.addr sa, a.p FROM pick k JOIN adds a USING(te,se) JOIN r s ON s.entity_id=a.se JOIN r t ON t.entity_id=a.te) TO {q(out+'.adds.parquet')}")
extra = {}
for se, te in con.execute("SELECT se, te FROM pick ORDER BY se, te").fetchall():
    extra.setdefault(se, []).append(te)
n = 0
with open(base, encoding='utf-8', newline='') as fi, open(out, 'w', encoding='utf-8', newline='') as fo:
    rd = csv.reader(fi, delimiter='\t', quoting=csv.QUOTE_NONE); wr = csv.writer(fo, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
    wr.writerow(next(rd))
    for row in rd:
        if row[0] in extra:
            ids = [x for x in row[1].split(',') if x] + extra[row[0]]; row = [row[0], ','.join(ids)]; n += 1
        wr.writerow(row)
print('references changed', n, 'links added', sum(map(len, extra.values())))
