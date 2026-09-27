"""Holdout: accept rejected band pairs when the target's name is near-identical to this reference (name JW >= a)
and clearly less similar to every other candidate reference of that target (best other JW <= b).
Grid on tune (OOF correction scores), best cell scored once on final (locked scores)."""
import importlib.util, warnings, numpy as np, pandas as pd, duckdb
warnings.filterwarnings('ignore')
spec=importlib.util.spec_from_file_location('metric','artifacts/leaderboard-push-20260926-1205/tune_blend_audit.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; N='work/v21L/run/train'
d=pd.read_parquet(f'{S}/stack_v4/pairs_labels_masked.parquet',columns=['sid','tid','xe'])
band=d[d.xe.notna()][['sid','tid']].reset_index(drop=True)
comp=pd.read_parquet('work/comp/hold_comp.parquet'); band['jw']=comp.comp_jw_gap+comp.comp_jw_other; band['jwo']=comp.comp_jw_other
con=duckdb.connect()
te=con.execute(f"SELECT rid tid, addr='' te FROM read_parquet(['{N}/norm_s2.parquet','{N}/norm_s3.parquet'])").df()
band=band.merge(te,on='tid',how='left')
def load(f,refs):
    x=pd.read_parquet(f,columns=['sid','tid','lgb_small_tree']).rename(columns={'lgb_small_tree':'q'})
    x=x.merge(pd.read_parquet(f'{M}/hold_preds.parquet',columns=['sid','tid','y']),on=['sid','tid'])
    x=x[x.sid.isin(refs.sid)].merge(band,on=['sid','tid'],how='left').reset_index(drop=True); x['y']=x.y.astype(int); return x
tref=pd.read_parquet(f'{S}/stack_v4/tune_references.parquet'); fref=pd.read_parquet(f'{S}/final_v1/error_by_reference.parquet')
sets={'tune':(load(f'{S}/stack_v4/oof_predictions.parquet',tref),tref),'final':(load(f'{S}/final_v1/locked_predictions.parquet',fref),fref)}
F=lambda x,refs,q: float(np.mean(m.scores_for_threshold(x.sid.to_numpy(),x.tid.to_numpy(),q,x.y.to_numpy(),refs.sid.to_numpy(),refs.true_links.to_numpy(),0.77)))
def apply(x,a,b,only_te):
    k=(x.jw>=a)&(x.jwo<=b)&(x.q<0.77)&x.jw.notna()
    if only_te: k&=x.te.fillna(False)
    q=x.q.to_numpy().copy(); q[k.to_numpy()]=0.771; return q,int(k.sum()),float(x.y[k].mean()) if k.any() else None
x,refs=sets['tune']; base=F(x,refs,x.q.to_numpy()); res=[]
for a in (0.9,0.93,0.95,0.97):
    for b in (0.7,0.8,0.85,0.9):
        for o in (True,False):
            q,n,pr=apply(x,a,b,o); res.append((F(x,refs,q)-base,a,b,o,n,pr))
res.sort(reverse=True); print('tune base',round(base,6),'top',res[:4])
_,a,b,o,_,_=res[0]; x,refs=sets['final']; q,n,pr=apply(x,a,b,o)
print('FINAL at tune-best',a,b,o,'pairs',n,'precision',pr,'delta',F(x,refs,q)-F(x,refs,x.q.to_numpy()))
