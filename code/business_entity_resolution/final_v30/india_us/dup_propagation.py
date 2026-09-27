"""Holdout: target B unassigned, target A assigned to reference R, A and B have identical normalized name AND
identical non-empty address -> add (R, B). Exact macro F0.5 on the final half (rule has no tuned parameter)."""
import duckdb, numpy as np, pandas as pd
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'
con=duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
con.execute(f"CREATE TABLE r AS SELECT rid, src, country, name, addr FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
for half, refs_f, pred_f in (('tune', f'{S}/stack_v4/tune_references.parquet', f'{S}/stack_v4/oof_predictions.parquet'), ('final', f'{S}/final_v1/error_by_reference.parquet', f'{S}/final_v1/locked_predictions.parquet')):
    con.execute(f"CREATE OR REPLACE TABLE refs AS SELECT * FROM read_parquet('{refs_f}')")
    con.execute(f"""CREATE OR REPLACE TABLE acc AS SELECT sid, tid FROM (SELECT sid, tid, lgb_small_tree q FROM read_parquet('{pred_f}') WHERE sid IN (SELECT sid FROM refs))
       WHERE q>=0.77 QUALIFY row_number() OVER (PARTITION BY tid ORDER BY q DESC, sid)=1""")
    con.execute("""CREATE OR REPLACE TABLE add AS SELECT DISTINCT a.sid, b.rid tid FROM acc a JOIN r ta ON ta.rid=a.tid JOIN r b ON b.src>1 AND b.rid<>ta.rid AND b.country=ta.country AND b.name=ta.name AND b.addr=ta.addr AND ta.addr<>''
       WHERE b.rid NOT IN (SELECT tid FROM acc)""")
    con.execute("CREATE OR REPLACE TABLE add1 AS SELECT * FROM add QUALIFY count(*) OVER (PARTITION BY tid)=1")
    ad=con.execute("SELECT a.sid, (g.sid IS NOT NULL)::int y FROM add1 a LEFT JOIN 'work/comp/hold_truth.parquet' g USING(sid,tid)").df()
    er=con.execute("SELECT sid, tp, fp, fn FROM refs").df().set_index('sid')
    if half=='tune':  # tune_references counts are for the v24 base; approximate
        pass
    g=ad.groupby('sid').y.agg(['sum','count']); tp,fp,fn=er.tp.copy(),er.fp.copy(),er.fn.copy()
    tp.loc[g.index]+=g['sum']; fp.loc[g.index]+=g['count']-g['sum']; fn.loc[g.index]-=g['sum']
    def f05(tp,fp,fn):
        P=np.where(tp+fp>0,tp/np.maximum(tp+fp,1),1.0); R=np.where(tp+fn>0,tp/np.maximum(tp+fn,1),1.0)
        f=np.where(tp>0,1.25*P*R/np.maximum(0.25*P+R,1e-12),0.0); return np.where((tp+fp==0)&(tp+fn==0),1.0,f).mean()
    print(half, 'added', len(ad), 'precision', round(ad.y.mean(),3) if len(ad) else None, 'delta', round(f05(tp.values,fp.values,fn.values)-f05(er.tp.values,er.fp.values,er.fn.values),6))
