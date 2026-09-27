"""Holdout test of a count-aware rule: references with at most K accepted links get their best unowned candidate
if its correction score >= a (one extra link per reference). Grid on tune (OOF), best cell once on final."""
import importlib.util, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
spec=importlib.util.spec_from_file_location('metric','artifacts/leaderboard-push-20260926-1205/tune_blend_audit.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
S='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'
h=pd.read_parquet(f'{M}/hold_preds.parquet',columns=['sid','tid','y'])
def load(f,refs):
    x=pd.read_parquet(f,columns=['sid','tid','lgb_small_tree']).rename(columns={'lgb_small_tree':'q'}).merge(h,on=['sid','tid'])
    x=x[x.sid.isin(refs.sid)].reset_index(drop=True); x['y']=x.y.astype(int); return x
tref=pd.read_parquet(f'{S}/stack_v4/tune_references.parquet'); fref=pd.read_parquet(f'{S}/final_v1/error_by_reference.parquet')
sets={'tune':(load(f'{S}/stack_v4/oof_predictions.parquet',tref),tref),'final':(load(f'{S}/final_v1/locked_predictions.parquet',fref),fref)}
F=lambda x,refs,q: float(np.mean(m.scores_for_threshold(x.sid.to_numpy(),x.tid.to_numpy(),q,x.y.to_numpy(),refs.sid.to_numpy(),refs.true_links.to_numpy(),0.77)))
def rule(x,K,a):
    w=x.sort_values(['tid','q','sid'],ascending=[True,False,True]).drop_duplicates('tid')      # current owner per target
    acc=w[w.q>=0.77]; n=acc.groupby('sid').size(); owned=set(acc.tid)
    c=x[(x.q<0.77)&(x.q>=a)&~x.tid.isin(owned)]
    c=c[c.sid.map(n).fillna(0)<=K].sort_values(['sid','q'],ascending=[True,False]).drop_duplicates('sid')
    c=c.sort_values(['tid','q'],ascending=[True,False]).drop_duplicates('tid')
    q=x.q.to_numpy().copy(); idx=x.reset_index().merge(c[['sid','tid']],on=['sid','tid'])['index'].to_numpy(); q[idx]=0.771
    return q,len(idx),float(x.y.iloc[idx].mean()) if len(idx) else None
x,refs=sets['tune']; base=F(x,refs,x.q.to_numpy()); res=[]
for K in (0,1,2,3):
    for a in (0.3,0.4,0.5,0.6,0.7):
        q,n,pr=rule(x,K,a); res.append((F(x,refs,q)-base,K,a,n,pr))
res.sort(reverse=True); print('tune top', [tuple(round(v,6) if isinstance(v,float) else v for v in r) for r in res[:5]])
_,K,a,_,_=res[0]; x,refs=sets['final']; q,n,pr=rule(x,K,a)
print('FINAL at tune-best K',K,'a',a,'pairs',n,'precision',pr,'delta',round(F(x,refs,q)-F(x,refs,x.q.to_numpy()),6))
