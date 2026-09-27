"""OOF tune: correction model with all features vs without the e5-base cross-encoder score (value of xe5)."""
import os; os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib, importlib.util, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
P='artifacts/push995-20260927-0915'
spec=importlib.util.spec_from_file_location('st',f'{P}/stacker.py'); st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
d=pd.read_parquet(f'{P}/stack_v4/pairs_labels_masked.parquet'); refs=pd.read_parquet(f'{P}/stack_v4/tune_references.parquet')
f=pd.read_parquet(f'{P}/stack_v4/band_features.parquet')
ix=np.flatnonzero(d.xe.notna().to_numpy())
folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8); y=d.y.to_numpy()
for name,cols in (('all',list(f.columns)),('no_xe5',[c for c in f.columns if c not in ('xe5','xe5_present')])):
    X=f[cols].to_numpy(); prob=d.p.to_numpy().copy(); bp=np.zeros(len(ix))
    for fold in range(3):
        fit=(rf>=0)&(rf!=fold); chk=rf==fold
        bp[chk]=st.train_one('lgb_small_tree',X[fit],y[ix[fit]]).predict_proba(X[chk])[:,1]
    prob[ix]=bp; t,v,curve=st.choose_cutoff(d,prob,refs)
    print(name,'cutoff',t,'tune OOF',round(float(v.mean()),6),flush=True)
