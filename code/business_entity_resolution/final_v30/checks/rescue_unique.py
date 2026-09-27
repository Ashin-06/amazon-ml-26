"""Holdout test: add (reference, name-only target) pairs never retrieved as candidates when the target's core
name equals exactly one reference's core in the whole pool (and nobody owns the target). Exact macro F0.5."""
import numpy as np, pandas as pd, duckdb
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'
con=duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE r AS SELECT rid, src, country, name, core, addr FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
con.execute("CREATE TABLE uc AS SELECT country, core, any_value(rid) sid FROM r WHERE src=1 GROUP BY ALL HAVING count(*)=1")
con.execute("CREATE TABLE pairs AS SELECT uc.sid, t.rid tid FROM r t JOIN uc ON uc.country=t.country AND uc.core=t.core WHERE t.src>1 AND t.addr=''")
con.execute(f"CREATE TABLE fr AS SELECT sid, tp, fp, fn FROM read_parquet('{S}/final_v1/error_by_reference.parquet')")
con.execute("CREATE TABLE lt AS SELECT sid, tid FROM read_parquet('C:/Users/ashin/AppData/Local/Temp/claude/D--Amazon-ML-26/a6f59d5d-823b-4509-a3c7-7eb2c7f6f181/scratchpad/lt.parquet')")
con.execute(f"CREATE TABLE cand AS SELECT sid, tid FROM read_parquet('{M}/hold_preds.parquet')")
con.execute(f"CREATE TABLE owned AS SELECT DISTINCT tid FROM read_parquet('{S}/final_v1/locked_predictions.parquet') WHERE lgb_small_tree>=0.77")
con.execute("""CREATE TABLE x AS SELECT p.*, (lt.sid IS NOT NULL) y, (c.sid IS NOT NULL) incand, p.tid IN (SELECT tid FROM owned) own FROM pairs p
  JOIN fr USING(sid) LEFT JOIN lt USING(sid,tid) LEFT JOIN cand c USING(sid,tid)""")
print(con.execute("SELECT incand, own, count(*), avg(y::int) FROM x GROUP BY ALL ORDER BY ALL").fetchall())
add=con.execute("SELECT sid, y FROM x WHERE NOT incand AND NOT own").df()
er=con.execute("SELECT * FROM fr").df().set_index('sid')
g=add.groupby('sid').y.agg(['sum','count']); tp,fp,fn=er.tp.copy(),er.fp.copy(),er.fn.copy()
tp.loc[g.index]+=g['sum']; fp.loc[g.index]+=g['count']-g['sum']; fn.loc[g.index]-=g['sum']
def f05(tp,fp,fn):
    P=np.where(tp+fp>0,tp/np.maximum(tp+fp,1),1.0); R=np.where(tp+fn>0,tp/np.maximum(tp+fn,1),1.0)
    f=np.where(tp>0,1.25*P*R/np.maximum(0.25*P+R,1e-12),0.0); return np.where((tp+fp==0)&(tp+fn==0),1.0,f).mean()
print('added', len(add), 'precision', round(add.y.mean(),4) if len(add) else None, 'final F0.5', round(f05(er.tp.values,er.fp.values,er.fn.values),6), '->', round(f05(tp.values,fp.values,fn.values),6))
