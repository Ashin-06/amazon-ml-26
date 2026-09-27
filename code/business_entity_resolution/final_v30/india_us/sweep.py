"""Fast sweep of stronger correction models: grouped 3-fold OOF on tune (exact macro F0.5, held-pool
ownership); the tune winner is then fitted on all tune and scored once on the untouched final half."""
import hashlib, importlib.util, sys, time, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings('ignore')
P='artifacts/push995-20260927-0915'
spec=importlib.util.spec_from_file_location('st',f'{P}/stacker.py'); st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
spec2=importlib.util.spec_from_file_location('metric','artifacts/leaderboard-push-20260926-1205/tune_blend_audit.py'); mt=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mt)
d=pd.read_parquet(f'{P}/stack_v4/pairs_labels_masked.parquet'); refs=pd.read_parquet(f'{P}/stack_v4/tune_references.parquet')
f=pd.read_parquet(f'{P}/stack_v4/band_features.parquet'); comp=pd.read_parquet('work/comp/hold_comp.parquet')
ix=np.flatnonzero(d.xe.notna().to_numpy())
folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8); y=d.y.to_numpy()
CFG={'base320':dict(n_estimators=320,num_leaves=15,max_depth=5,min_child_samples=100,learning_rate=.04,reg_lambda=15,colsample_bytree=.9),
     'lgb800':dict(n_estimators=800,num_leaves=31,max_depth=7,min_child_samples=60,learning_rate=.03,reg_lambda=10,colsample_bytree=.8,subsample=.8,subsample_freq=1),
     'lgb1500':dict(n_estimators=1500,num_leaves=63,max_depth=-1,min_child_samples=40,learning_rate=.015,reg_lambda=5,colsample_bytree=.7,subsample=.8,subsample_freq=1),
     'lgb600d':dict(n_estimators=600,num_leaves=15,max_depth=5,min_child_samples=100,learning_rate=.04,reg_lambda=15,colsample_bytree=.9)}
def model(c): return lgb.LGBMClassifier(**c,n_jobs=6,verbosity=-1,random_state=26,deterministic=True,force_col_wise=True)
res=[]
for fs,X in (('base',f.to_numpy()),('comp',np.hstack([f.to_numpy(),comp.to_numpy()]))):
    for name,c in CFG.items():
        t0=time.time(); prob=d.p.to_numpy().copy(); bp=np.zeros(len(ix))
        for fold in range(3):
            fit=(rf>=0)&(rf!=fold); chk=rf==fold
            bp[chk]=model(c).fit(X[fit],y[ix[fit]]).predict_proba(X[chk])[:,1]
        prob[ix]=bp; t,v,_=st.choose_cutoff(d,prob,refs)
        res.append((float(v.mean()),fs,name,t)); print(fs,name,'cutoff',t,'tune OOF',round(float(v.mean()),6),f'{time.time()-t0:.0f}s',flush=True)
best=max(res); print('TUNE WINNER',best,flush=True)
# final: fit winner on all tune band pairs, score untouched final half at the tune cutoff
_,fs,name,t=best; X=f.to_numpy() if fs=='base' else np.hstack([f.to_numpy(),comp.to_numpy()])
tune=y[ix]>=0; mdl=model(CFG[name]).fit(X[tune],y[ix[tune]])
M='work/v24/kaggle/work/model'; fref=pd.read_parquet(f'{P}/final_v1/error_by_reference.parquet')
h=pd.read_parquet(f'{M}/hold_preds.parquet',columns=['sid','tid','y']); dd=d[['sid','tid','p']].merge(h,on=['sid','tid'],how='left')
p=d.p.to_numpy().copy(); p[ix]=mdl.predict_proba(X)[:,1]
fin=dd.sid.isin(fref.sid).to_numpy()
sc=lambda q,cut: float(np.mean(mt.scores_for_threshold(dd.sid.to_numpy()[fin],dd.tid.to_numpy()[fin],q[fin],dd.y.to_numpy()[fin].astype(int),fref.sid.to_numpy(),fref.true_links.to_numpy(),cut)))
lock=pd.read_parquet(f'{P}/final_v1/locked_predictions.parquet'); lq=d[['sid','tid']].merge(lock,on=['sid','tid'],how='left').lgb_small_tree.to_numpy()
print('FINAL current v24stack', round(sc(lq,0.77),6), '| new', fs, name, 'cut', t, round(sc(p,t),6))
