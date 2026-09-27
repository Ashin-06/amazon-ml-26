"""Holdout tests of an external proposal: (1) sibling name propagation for name-only targets,
(2) rarity-gated thresholds. Exact macro F0.5 with held-pool ownership; tune = OOF, final = locked model."""
import importlib.util, warnings, numpy as np, pandas as pd, duckdb
warnings.filterwarnings('ignore')
spec=importlib.util.spec_from_file_location('metric','artifacts/leaderboard-push-20260926-1205/tune_blend_audit.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'
con=duckdb.connect(); con.execute("SET threads=6")
con.execute(f"CREATE TABLE r AS SELECT rid, src, country, name, core, addr FROM read_parquet(['{N}/norm_s1.parquet','{N}/norm_s2.parquet','{N}/norm_s3.parquet'])")
con.execute("CREATE TABLE nc AS SELECT country, core, count(*) ns FROM r WHERE src=1 GROUP BY ALL")
def prep(f, refs):
    con.register('refs', refs[['sid']])
    return con.execute(f"""SELECT o.sid, o.tid, o.lgb_small_tree q, h.y::int y, t.addr='' te, t.name tname, coalesce(nc.ns,0) ns
      FROM read_parquet('{f}') o JOIN read_parquet('{M}/hold_preds.parquet') h USING(sid,tid) JOIN r t ON t.rid=o.tid
      LEFT JOIN nc ON nc.country=t.country AND nc.core=t.core WHERE o.sid IN (SELECT sid FROM refs)""").df()
tref=pd.read_parquet(f'{S}/stack_v4/tune_references.parquet'); fref=pd.read_parquet(f'{S}/final_v1/error_by_reference.parquet')
sets={'tune':(prep(f'{S}/stack_v4/oof_predictions.parquet',tref),tref),'final':(prep(f'{S}/final_v1/locked_predictions.parquet',fref),fref)}
def F(d,refs,q): return float(np.mean(m.scores_for_threshold(d.sid.to_numpy(),d.tid.to_numpy(),q,d.y.to_numpy(),refs.sid.to_numpy(),refs.true_links.to_numpy(),0.77)))
for half,(d,refs) in sets.items():
    base=F(d,refs,d.q.to_numpy())
    # Pillar 1: name-only target t; exactly one candidate ref already has an accepted sibling target with the identical name
    acc=d[d.q>=0.77][['sid','tname']].drop_duplicates(); acc['sib']=True
    x=d.merge(acc,on=['sid','tname'],how='left'); x['sib']=x.sib.fillna(False).astype(bool)
    x['nsib']=x.groupby('tid').sib.transform('sum')
    pick=(x.te & x.sib & (x.nsib==1) & (x.q<0.77)).to_numpy()
    q=x.q.to_numpy().copy(); q[pick]=0.771
    print(half,'P1 sibling propagation: pairs',int(pick.sum()),'precision',round(float(x.y[pick].mean()),3) if pick.any() else None,'F',round(base,6),'->',round(F(x,refs,q),6),flush=True)
    # Pillar 2: rarity-gated thresholds (target core frequency among references)
    res=[]
    for t_rare in (0.55,0.65,0.77):
        for t_common in (0.77,0.85,0.95):
            q=d.q.to_numpy().copy(); rare=(d.ns<=1).to_numpy(); common=(d.ns>=5).to_numpy()
            q[rare]+=0.77-t_rare; q[common]+=0.77-t_common
            res.append((F(d,refs,q),t_rare,t_common))
    best=max(res); print(half,'P2 rarity thresholds best',[round(best[0],6),best[1],best[2]],'base',round(base,6),flush=True)
