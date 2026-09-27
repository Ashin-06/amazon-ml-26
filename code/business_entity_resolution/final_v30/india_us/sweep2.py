"""Wider tune-only sweep of the correction LightGBM (+ 5-seed averaging); winner scored once on final."""
import hashlib, importlib.util, itertools, time, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings('ignore')
P='artifacts/push995-20260927-0915'
spec=importlib.util.spec_from_file_location('st',f'{P}/stacker.py'); st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
spec2=importlib.util.spec_from_file_location('metric','artifacts/leaderboard-push-20260926-1205/tune_blend_audit.py'); mt=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mt)
d=pd.read_parquet(f'{P}/stack_v4/pairs_labels_masked.parquet'); refs=pd.read_parquet(f'{P}/stack_v4/tune_references.parquet')
X=pd.read_parquet(f'{P}/stack_v4/band_features.parquet').to_numpy()
ix=np.flatnonzero(d.xe.notna().to_numpy())
folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8); y=d.y.to_numpy()
def mk(c,seed): return lgb.LGBMClassifier(**c,n_jobs=6,verbosity=-1,random_state=seed,deterministic=True,force_col_wise=True)
def oof(c,seeds=(26,)):
    prob=d.p.to_numpy().copy(); bp=np.zeros(len(ix))
    for fold in range(3):
        fit=(rf>=0)&(rf!=fold); chk=rf==fold
        bp[chk]=np.mean([mk(c,s).fit(X[fit],y[ix[fit]]).predict_proba(X[chk])[:,1] for s in seeds],axis=0)
    prob[ix]=bp; t,v,_=st.choose_cutoff(d,prob,refs); return float(v.mean()),t
res=[]
for n,lr,mc,nl in itertools.product((450,600,900),(0.03,0.04),(50,100,200),(15,23)):
    c=dict(n_estimators=n,num_leaves=nl,max_depth=5 if nl==15 else 6,min_child_samples=mc,learning_rate=lr,reg_lambda=15,colsample_bytree=.9)
    s,t=oof(c); res.append((s,t,c)); print(round(s,6),t,n,lr,mc,nl,flush=True)
res.sort(key=lambda r:-r[0]); top=res[0]; print('BEST SINGLE',top,flush=True)
sb,tb=oof(top[2],seeds=(26,27,28,29,30)); print('BEST x5 seeds',round(sb,6),tb,flush=True)
c=top[2]; seeds=(26,27,28,29,30) if sb>top[0] else (26,); cut=tb if sb>top[0] else top[1]
tune=y[ix]>=0; p=d.p.to_numpy().copy()
p[ix]=np.mean([mk(c,s).fit(X[tune],y[ix[tune]]).predict_proba(X)[:,1] for s in seeds],axis=0)
M='work/v24/kaggle/work/model'; fref=pd.read_parquet(f'{P}/final_v1/error_by_reference.parquet')
dd=d[['sid','tid']].merge(pd.read_parquet(f'{M}/hold_preds.parquet',columns=['sid','tid','y']),on=['sid','tid'],how='left'); fin=dd.sid.isin(fref.sid).to_numpy()
sc=lambda q,cut: float(np.mean(mt.scores_for_threshold(dd.sid.to_numpy()[fin],dd.tid.to_numpy()[fin],q[fin],dd.y.to_numpy()[fin].astype(int),fref.sid.to_numpy(),fref.true_links.to_numpy(),cut)))
print('FINAL winner',c,'seeds',len(seeds),'cut',cut,round(sc(p,cut),6),'| v25 was 0.988918, v24stack 0.988792')
