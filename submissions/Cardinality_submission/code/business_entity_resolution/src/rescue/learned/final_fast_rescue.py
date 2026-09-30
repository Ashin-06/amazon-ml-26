"""Freeze the tune-selected rescue model, then evaluate the complete final pipeline once."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
HERE=Path(__file__).parent;PKG=Path(__file__).resolve().parents[3];ROOT=Path.cwd()  # PKG: code/models; ROOT: data working directory
P=ROOT/'artifacts/after-0982910-20260927'/'fast_rescue_v1';OUT=ROOT/'artifacts/after-0982910-20260927'/'fast_rescue_final_v1'
OUT.mkdir(exist_ok=False)
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
r=module('rescue',HERE/'fast_rescue.py');st=module('stack',PKG/'src/correction/stacker.py');r.st=st
report=json.loads((P/'audit.json').read_text());arm=report['selected']
if not arm:raise ValueError('No tune-selected model')
if hashlib.sha256((HERE/'fast_rescue.py').read_bytes()).hexdigest()!=report['source_sha256']:raise ValueError('Feature source drift')
x=pd.read_parquet(P/'pairs_masked.parquet');f=pd.read_parquet(P/'features.parquet')
old=ROOT/'artifacts/push995-20260927-0915';d=pd.read_parquet(old/'stack_v4/pairs_labels_masked.parquet')
oof=pd.read_parquet(ROOT/'artifacts/after-0982910-20260927'/'xenc7_pilot_v1/oof_predictions.parquet');cut=report['results'][arm]['cutoff']
if not d[['sid','tid']].equals(oof[['sid','tid']]):raise ValueError('Base alignment')
owned=set(d.loc[oof.v25_control.ge(.74),'tid'])
fit=x.y.ge(0).to_numpy()&~x.tid.isin(owned).to_numpy()
with threadpool_limits(limits=1):model=r.fit(arm,f.to_numpy()[fit],x.y.to_numpy()[fit]);p=model.predict_proba(f.to_numpy())[:,1]
if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Predictions')
joblib.dump(model,OUT/'rescue_model.joblib')
x[['sid','tid']].assign(p=p).to_parquet(OUT/'locked_predictions.parquet',index=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest={'arm':arm,'cutoff':cut,'feature_names':list(f.columns),'fit_pairs':int(fit.sum()),
 'feature_source_sha256':sha(HERE/'fast_rescue.py'),'evaluation_source_sha256':sha(__file__),
 'model_sha256':sha(OUT/'rescue_model.joblib'),'prediction_sha256':sha(OUT/'locked_predictions.parquet'),
 'pilot_sha256':sha(P/'audit.json'),'feature_sha256':sha(P/'features.parquet'),'masked_pairs_sha256':sha(P/'pairs_masked.parquet'),
 'scope':'India/US rescue only; natural existing route; frozen before one reused-final check; base full model uses held-pool ownership.'}
(OUT/'locked_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
# Everything above is fixed before opening the final labels.
refs=pd.read_parquet(st.MD/'hold_error_rows.parquet');refs=refs[refs.half.eq('final')].sort_values('sid').reset_index(drop=True)
if set(refs.sid)&set(x.loc[x.y.ge(0),'sid']):raise ValueError('Tune/final overlap')
truth=pd.read_parquet(ROOT/'work/comp/hold_truth.parquet',columns=['sid','tid'])
x=x.drop(columns='y').merge(truth.assign(y=1),on=['sid','tid'],how='left',validate='one_to_one').fillna({'y':0})
f0=pd.read_parquet(old/'stack_v4/band_features.parquet');ix=np.flatnonzero(d.xe.notna());base_model=joblib.load(ROOT/'work/comp/v25/v25_lgb600.joblib')
bp=d.p.to_numpy().copy()
with threadpool_limits(limits=1):bp[ix]=base_model.predict_proba(f0.to_numpy())[:,1]
order=np.lexsort((d.sid.to_numpy(),-bp,d.tid.to_numpy()));w=order[np.r_[True,d.tid.to_numpy()[order][1:]!=d.tid.to_numpy()[order][:-1]]]
base=d.iloc[w[bp[w]>=.74]][['sid','tid']].merge(truth.assign(y=1),on=['sid','tid'],how='left',validate='one_to_one').fillna({'y':0})
owned=set(base.tid);fixed=r.choose(x,x.score.to_numpy(),owned,1.88);learned=r.choose(x,p,owned,cut)
b=r.values(base,fixed,refs);v=r.values(base,learned,refs);result=r.compare(v,b)
result.update(baseline=float(b.mean()),by_country={co:r.compare(v[refs.country.eq(co)],b[refs.country.eq(co)]) for co in refs.country.unique()},
 fixed_pairs=int(fixed.sid.isin(refs.sid).sum()),learned_pairs=int(learned.sid.isin(refs.sid).sum()),
 learned_true=int(learned.loc[learned.sid.isin(refs.sid),'y'].sum()))
result['eligible_for_export']=result['delta']>0 and result['paired_95_interval'][0]>0
(OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
