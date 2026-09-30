"""Matching file: India/US rows from INUS.tsv; France rows = v2 French scores >= 0.925 with one owner per target
(highest score, ties by reference). usage: python fr_v2_base.py INUS.tsv FR_SCORES.parquet OUT.tsv"""
import sys, csv, duckdb
inus, scores, out = sys.argv[1:4]; N = 'work/v21L/run/test'
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='5GB'")
con.execute(f"CREATE TABLE ids AS SELECT rid, entity_id, country FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
con.execute(f"""CREATE TABLE w AS SELECT a.entity_id s, b.entity_id t FROM (SELECT * FROM read_parquet('{scores}') WHERE p>=0.925
   QUALIFY row_number() OVER (PARTITION BY tid ORDER BY p DESC, sid)=1) x JOIN ids a ON a.rid=x.sid AND a.country='france' JOIN ids b ON b.rid=x.tid""")
fr = {}
for s, t in con.execute("SELECT s, t FROM w ORDER BY s, t").fetchall(): fr.setdefault(s, []).append(t)
french = set(r[0] for r in con.execute("SELECT entity_id FROM ids WHERE country='france' AND rid IN (SELECT rid FROM read_parquet('" + N + "/norm_s1.parquet'))").fetchall())
n = 0
with open(inus, encoding='utf-8', newline='') as fi, open(out, 'w', encoding='utf-8', newline='') as fo:
    rd = csv.reader(fi, delimiter='\t', quoting=csv.QUOTE_NONE); wr = csv.writer(fo, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
    wr.writerow(next(rd))
    for row in rd:
        if row[0] in french: row = [row[0], ','.join(fr.get(row[0], []))]; n += 1
        wr.writerow(row)
print('French rows rewritten', n, 'French links', sum(map(len, fr.values())))
