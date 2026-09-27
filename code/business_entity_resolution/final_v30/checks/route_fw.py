"""Holdout: size/recall of a rescue route (same first core word + same leading house number, not yet a candidate)."""
import duckdb
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'
SP='C:/Users/ashin/AppData/Local/Temp/claude/D--Amazon-ML-26/a6f59d5d-823b-4509-a3c7-7eb2c7f6f181/scratchpad'
con=duckdb.connect(); con.execute("SET threads=6; SET memory_limit='7GB'")
con.execute(f"""CREATE TABLE r AS SELECT rid, src, country, string_split(core,' ')[1] fw, regexp_extract(addr,'[0-9]+') num FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])""")
con.execute(f"CREATE TABLE hs AS SELECT DISTINCT sid FROM read_parquet('{M}/hold_preds.parquet')")
con.execute("CREATE TABLE s AS SELECT r.* FROM r JOIN hs ON hs.sid=r.rid WHERE num<>'' AND fw<>''")
con.execute("CREATE TABLE p AS SELECT s.rid sid, t.rid tid FROM s JOIN r t ON t.src>1 AND t.country=s.country AND t.fw=s.fw AND t.num=s.num")
con.execute(f"CREATE TABLE truth AS SELECT sid, tid FROM read_parquet('{SP}/lt.parquet')")
con.execute(f"CREATE TABLE fr AS SELECT sid FROM read_parquet('{S}/final_v1/error_by_reference.parquet')")
con.execute(f"CREATE TABLE cand AS SELECT sid, tid FROM read_parquet('{M}/hold_preds.parquet')")
con.execute("CREATE TABLE x AS SELECT p.*, (t.sid IS NOT NULL) y FROM p SEMI JOIN fr USING(sid) LEFT JOIN truth t USING(sid,tid) ANTI JOIN cand USING(sid,tid)")
print('new final-half pairs', con.execute("SELECT count(*), sum(y::int), count(DISTINCT sid) FROM x").fetchall())
print('final-half retrieval misses total', con.execute(f"SELECT count(*) FROM read_parquet('{SP}/lt.parquet') WHERE k='miss'").fetchall())
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
con.execute(f"CREATE TABLE nm AS SELECT rid, name, addr FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
d=con.execute("SELECT x.y, a.name sn, b.name tn, a.addr sa, b.addr ta FROM x JOIN nm a ON a.rid=x.sid JOIN nm b ON b.rid=x.tid").df()
d['nj']=cpdist(d.sn.tolist(),d.tn.tolist(),scorer=JaroWinkler.normalized_similarity,workers=-1)
d['nr']=cpdist(d.sn.tolist(),d.tn.tolist(),scorer=fuzz.ratio,workers=-1)
d['at']=cpdist(d.sa.tolist(),d.ta.tolist(),scorer=fuzz.token_set_ratio,workers=-1)
d['score']=d.nr/100+d['at']/100
d=d.sort_values('score',ascending=False).reset_index(drop=True); cy=d.y.cumsum()
for k in (200,500,1000,2000,5000,20000):
    print(f'top {k}: true {int(cy.iloc[k-1])} precision {cy.iloc[k-1]/k:.3f} recall {cy.iloc[k-1]/d.y.sum():.3f}')
