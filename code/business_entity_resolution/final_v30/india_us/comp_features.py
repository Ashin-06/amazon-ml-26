"""Target-side name competition features for the v24 correction stacker.

For each band pair (s, t): among ALL candidate references of target t (full candidate set, not only the
held-out ones), the best competing reference's core token-set ratio and name Jaro-Winkler, the gap to this
pair, the number of near-identical competitors (core token-set >= 95) and the target's candidate count.
IDs only join records. usage: python comp_features.py BAND_PAIRS.parquet CANDS_GLOB NORM_DIR OUT.parquet
"""
import sys, duckdb, numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
band, cands, norm, out = sys.argv[1:5]
q = lambda p: "'" + p.replace(chr(92), '/') + "'"
con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='5GB'; SET preserve_insertion_order=true")
con.execute(f"CREATE TABLE b AS SELECT row_number() OVER () - 1 ix, sid, tid FROM read_parquet({q(band)})")
con.execute("CREATE TABLE bt AS SELECT DISTINCT tid FROM b")
con.execute(f"CREATE TABLE c AS SELECT DISTINCT sid, tid FROM read_parquet({cands}) WHERE tid IN (SELECT tid FROM bt) UNION SELECT sid, tid FROM b")
con.execute(f"CREATE TABLE r AS SELECT rid, name, core FROM read_parquet([{q(norm+'/norm_s1.parquet')},{q(norm+'/norm_s2.parquet')},{q(norm+'/norm_s3.parquet')}])")
n = con.execute("SELECT count(*) FROM c").fetchone()[0]; print('competition pairs', n, flush=True)
parts = []
for k in range(8):
    d = con.execute(f"SELECT c.sid, c.tid, s.name sn, s.core sc, t.name tn, t.core tc FROM c JOIN r s ON s.rid=c.sid JOIN r t ON t.rid=c.tid WHERE hash(c.tid)%8={k}").df()
    d['ct'] = cpdist(d.sc.tolist(), d.tc.tolist(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    d['jw'] = cpdist(d.sn.tolist(), d.tn.tolist(), scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    parts.append(d[['sid', 'tid', 'ct', 'jw']]); print('chunk', k, len(d), flush=True)
s = pd.concat(parts, ignore_index=True); con.register('s', s)
# best / second best per target so "best other" excludes the pair itself
con.execute("""CREATE TABLE agg AS SELECT tid, count(*) ncand,
  max(ct) ct1, list_sort(list(ct), 'DESC')[2] ct2, max(jw) jw1, list_sort(list(jw), 'DESC')[2] jw2, count(*) FILTER (WHERE ct>=95) n95
  FROM s GROUP BY tid""")
f = con.execute("""SELECT b.ix, s.ct, s.jw, a.ncand, a.n95 - (s.ct>=95)::int n95_other,
   CASE WHEN s.ct=a.ct1 THEN coalesce(a.ct2,0) ELSE a.ct1 END ct_other,
   CASE WHEN s.jw=a.jw1 THEN coalesce(a.jw2,0) ELSE a.jw1 END jw_other
   FROM b JOIN s USING(sid,tid) JOIN agg a USING(tid) ORDER BY b.ix""").df()
if len(f) != con.execute("SELECT count(*) FROM b").fetchone()[0] or not np.array_equal(f.ix.to_numpy(), np.arange(len(f))):
    raise ValueError('alignment')
out_df = pd.DataFrame({'comp_ncand': np.log1p(f.ncand), 'comp_n95_other': np.log1p(f.n95_other.clip(lower=0)),
    'comp_ct_other': f.ct_other, 'comp_jw_other': f.jw_other, 'comp_ct_gap': f.ct - f.ct_other, 'comp_jw_gap': f.jw - f.jw_other}).astype(np.float32)
out_df.to_parquet(out, index=False); print('wrote', out, out_df.shape, out_df.describe().T[['mean', 'min', 'max']])
