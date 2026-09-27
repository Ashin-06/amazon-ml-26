"""Count-aware rule re-measured with the v25 correction (LightGBM 600, cutoff 0.74): references with <= K accepted
links get their best unowned candidate if a <= score < 0.74. Tune = grouped OOF, final = model fitted on all tune."""
import hashlib, importlib.util, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings('ignore')
P='artifacts/push995-20260927-0915'; M='work/v24/kaggle/work/model'; CUT=0.74
spec=importlib.util.spec_from_file_location('metric','artifacts/leaderboard-push-20260926-1205/tune_blend_audit.py'); mt=importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
d=pd.read_parquet(f'{P}/stack_v4/pairs_labels_masked.parquet'); X=pd.read_parquet(f'{P}/stack_v4/band_features.parquet').to_numpy()
tref=pd.read_parquet(f'{P}/stack_v4/tune_references.parquet'); fref=pd.read_parquet(f'{P}/final_v1/error_by_reference.parquet')
ix=np.flatnonzero(d.xe.notna().to_numpy()); y=d.y.to_numpy()
folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in tref.sid}
rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8)
CFG=dict(n_estimators=600,num_leaves=15,max_depth=5,min_child_samples=100,learning_rate=.04,reg_lambda=15,colsample_bytree=.9)
mk=lambda: lgb.LGBMClassifier(**CFG,n_jobs=6,verbosity=-1,random_state=26,deterministic=True,force_col_wise=True)
oof=d.p.to_numpy().copy(); bp=np.zeros(len(ix))
for fold in range(3):
    fit=(rf>=0)&(rf!=fold); chk=rf==fold; bp[chk]=mk().fit(X[fit],y[ix[fit]]).predict_proba(X[chk])[:,1]
oof[ix]=bp
tune=y[ix]>=0; fin=d.p.to_numpy().copy(); fin[ix]=mk().fit(X[tune],y[ix[tune]]).predict_proba(X)[:,1]
h=pd.read_parquet(f'{M}/hold_preds.parquet',columns=['sid','tid','y'])
base=d[['sid','tid']].merge(h,on=['sid','tid'],how='left'); base['y']=base.y.astype(int)
def F(x,refs,q): return float(np.mean(mt.scores_for_threshold(x.sid.to_numpy(),x.tid.to_numpy(),q,x.y.to_numpy(),refs.sid.to_numpy(),refs.true_links.to_numpy(),CUT)))
def rule(x,K,a):
    w=x.sort_values(['tid','q','sid'],ascending=[True,False,True]).drop_duplicates('tid'); acc=w[w.q>=CUT]
    n=acc.groupby('sid').size(); owned=set(acc.tid)
    c=x[(x.q<CUT)&(x.q>=a)&~x.tid.isin(owned)]; c=c[c.sid.map(n).fillna(0)<=K]
    c=c.sort_values(['sid','q'],ascending=[True,False]).drop_duplicates('sid').sort_values(['tid','q'],ascending=[True,False]).drop_duplicates('tid')
    q=x.q.to_numpy().copy(); idx=x.reset_index().merge(c[['sid','tid']],on=['sid','tid'])['index'].to_numpy(); q[idx]=CUT+0.001
    return q,len(idx),float(x.y.iloc[idx].mean()) if len(idx) else None
res=[]
xt=base.assign(q=oof); xt=xt[xt.sid.isin(tref.sid)].reset_index(drop=True); b0=F(xt,tref,xt.q.to_numpy())
for K in (1,2,3):
    for a in (0.5,0.55,0.6,0.65,0.7):
        q,n,pr=rule(xt,K,a); res.append((F(xt,tref,q)-b0,K,a,n,pr))
res.sort(reverse=True); print('tune base',round(b0,6),'top',[tuple(round(v,6) if isinstance(v,float) else v for v in r) for r in res[:4]])
_,K,a,_,_=res[0]; xf=base.assign(q=fin); xf=xf[xf.sid.isin(fref.sid)].reset_index(drop=True); b1=F(xf,fref,xf.q.to_numpy()); q,n,pr=rule(xf,K,a)
print('FINAL v25 base',round(b1,6),'| rule K',K,'a',a,'pairs',n,'precision',pr,'delta',round(F(xf,fref,q)-b1,6))
