"""Holdout: rescue route B = same country + identical core name (core shared by <= CAP targets), target has an
address, pair not produced by blocking and not by route A. Score = name ratio + address token-set ratio."""
import duckdb, numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'
con=duckdb.connect(); con.execute("SET threads=6; SET memory_limit='7GB'")
con.execute(f"CREATE TABLE r AS SELECT rid, src, country, name, core, addr, string_split(core,' ')[1] fw, regexp_extract(addr,'[0-9]+') num FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
con.execute("CREATE TABLE tc AS SELECT country, core, count(*) nt FROM r WHERE src>1 AND addr<>'' GROUP BY ALL")
con.execute(f"CREATE TABLE hs AS SELECT DISTINCT sid FROM read_parquet('{M}/hold_preds.parquet')")
con.execute("CREATE TABLE s AS SELECT r.* FROM r JOIN hs ON hs.sid=r.rid")
con.execute("""CREATE TABLE p AS SELECT s.rid sid, t.rid tid, s.name sn, t.name tn, s.addr sa, t.addr ta, tc.nt FROM s JOIN r t ON t.src>1 AND t.country=s.country AND t.core=s.core AND t.addr<>''
   JOIN tc ON tc.country=s.country AND tc.core=s.core WHERE tc.nt<=300 AND NOT (t.fw=s.fw AND t.num=s.num AND s.num<>'')""")
con.execute(f"CREATE TABLE p2 AS SELECT p.* FROM p ANTI JOIN read_parquet('{M}/hold_preds.parquet') c ON c.sid=p.sid AND c.tid=p.tid")
d=con.execute("SELECT * FROM p2").df()
gt=pd.read_parquet('work/comp/hold_truth.parquet'); d=d.merge(gt.assign(y=1),on=['sid','tid'],how='left').fillna({'y':0})
d['score']=(cpdist(d.sn.tolist(),d.tn.tolist(),scorer=fuzz.ratio,workers=-1)+cpdist(d.sa.tolist(),d.ta.tolist(),scorer=fuzz.token_set_ratio,workers=-1))/100
print('route B new pairs', len(d), 'true', int(d.y.sum()))
d=d.sort_values('score',ascending=False).reset_index(drop=True); cy=d.y.cumsum()
for k in (100,200,400,800,1600):
    if k<=len(d): print(f'top {k}: true {int(cy.iloc[k-1])} precision {cy.iloc[k-1]/k:.3f} score>={d.score.iloc[k-1]:.3f}')
d[['sid','tid','score','y']].to_parquet('work/comp/routeB_hold.parquet',index=False)
