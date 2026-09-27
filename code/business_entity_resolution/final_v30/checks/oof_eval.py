"""OOF tune comparison: v24stack correction features with vs without target-side competition features."""
import os; os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib, importlib.util, sys, numpy as np, pandas as pd
P='artifacts/push995-20260927-0915'
spec=importlib.util.spec_from_file_location('st',f'{P}/stacker.py'); st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
d=pd.read_parquet(f'{P}/stack_v4/pairs_labels_masked.parquet'); refs=pd.read_parquet(f'{P}/stack_v4/tune_references.parquet')
f=pd.read_parquet(f'{P}/stack_v4/band_features.parquet'); comp=pd.read_parquet(sys.argv[1] if len(sys.argv)>1 else 'work/comp/hold_comp.parquet')
ix=np.flatnonzero(d.xe.notna().to_numpy())
folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8); y=d.y.to_numpy()
for name,X in (('base',f.to_numpy()),('comp',np.hstack([f.to_numpy(),comp.to_numpy()]))):
    prob=d.p.to_numpy().copy(); bp=np.zeros(len(ix))
    for fold in range(3):
        fit=(rf>=0)&(rf!=fold); chk=rf==fold
        mdl=st.train_one('lgb_small_tree',X[fit],y[ix[fit]]); bp[chk]=mdl.predict_proba(X[chk])[:,1]
    prob[ix]=bp
    t,v,curve=st.choose_cutoff(d,prob,refs)
    print(name,'cutoff',t,'tune OOF',float(v.mean()),'at .77',curve.get('0.77'),flush=True)
