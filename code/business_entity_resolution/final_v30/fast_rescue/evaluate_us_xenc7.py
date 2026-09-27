"""Exploratory US-only xenc7 comparison selected on tune, final checked once."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import importlib.util,json
import numpy as np,pandas as pd,joblib,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
HERE=Path(__file__).parent;ROOT=HERE.parent.parent;OUT=HERE/'xenc7_us_v1';OUT.mkdir(exist_ok=False)
sp=importlib.util.spec_from_file_location('evaluate',HERE/'evaluate_xenc7.py');ev=importlib.util.module_from_spec(sp);sp.loader.exec_module(ev)
m=ev.base_module();P=ev.PILOT
report=json.loads((HERE/'xenc7_pilot_v1/audit.json').read_text())
if ev.sha(HERE/'evaluate_xenc7.py')!=report['source_sha256']:raise ValueError('Pilot source drift')
for p,h in report['input_sha256'].items():
 if ev.sha(p)!=h:raise ValueError('Input drift')
d=pd.read_parquet(P/'pairs_labels_masked.parquet');f=pd.read_parquet(HERE/'xenc7_pilot_v1/band_features.parquet')
refs=pd.read_parquet(P/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
ev.fold_assignments(d,refs);oof=pd.read_parquet(HERE/'xenc7_pilot_v1/oof_predictions.parquet')
if not d[['sid','tid']].equals(oof[['sid','tid']]):raise ValueError('OOF alignment')
# Select the US slice only; labels outside tune remain masked.
us=refs.country.eq('us').to_numpy();v=m.winner_values(d,oof.neural_tree.to_numpy(),refs,.74);b=m.winner_values(d,oof.v25_control.to_numpy(),refs,.74)
tune=ev.comparison(v[us],b[us],refs[us].reset_index(drop=True))
passed=tune['delta']>=.0001 and tune['paired_95_interval'][0]>0
print('US tune',json.dumps(tune),'gate',passed,flush=True)
ev.write_json(OUT/'tune_gate.json',{'result':tune,'passed':passed,'source_sha256':ev.sha(__file__)})
if not passed:raise SystemExit(0)
ix=np.flatnonzero(d.xe.notna());fit=d.y.to_numpy()[ix]>=0
with threadpool_limits(limits=1):
 model=ev.train(m,'neural_tree',f.to_numpy()[fit],d.y.to_numpy()[ix[fit]])
 new=d.p.to_numpy().copy();new[ix]=model.predict_proba(f.to_numpy())[:,1]
 current_model=joblib.load(ROOT/'work/comp/v25/v25_lgb600.joblib')
 old=d.p.to_numpy().copy();old[ix]=current_model.predict_proba(f.iloc[:,:63].to_numpy())[:,1]
# Country is used only to select a fixed model, never as an identity feature.
countries=pd.read_parquet(m.NORM/'norm_s1.parquet',columns=['rid','country']).set_index('rid').country
take=d.sid.map(countries).eq('us').to_numpy();p=np.where(take,new,old)
joblib.dump(model,OUT/'us_model.joblib')
locked=d[['sid','tid']].assign(selected=p,v25=old)
locked.to_parquet(OUT/'locked_predictions.parquet',index=False)
manifest={'scope':'US only; v25 India unchanged','cutoff':.74,'arm':'neural_tree','feature_names':list(f.columns),
 'model_sha256':ev.sha(OUT/'us_model.joblib'),'prediction_sha256':ev.sha(OUT/'locked_predictions.parquet'),
 'source_sha256':ev.sha(__file__),'evaluation_source_sha256':ev.sha(HERE/'evaluate_xenc7.py'),
 'pilot_sha256':ev.sha(HERE/'xenc7_pilot_v1/audit.json'),'input_sha256':report['input_sha256'],
 'rescue_cutoff':1.88,'protocol':'Exploratory country selection on tune, frozen before one reused-final comparison. Held-pool ownership; French rows must remain byte identical to selected French variant.'}
ev.write_json(OUT/'locked_manifest.json',manifest)
# Open final truth only after all weights and decisions are frozen.
allrefs=pd.read_parquet(m.MD/'hold_error_rows.parquet').sort_values('sid').reset_index(drop=True)
truth=pd.read_parquet(ROOT/'work/comp/hold_truth.parquet',columns=['sid','tid'])
route=pd.read_parquet(HERE/'v26_combined_audit_v1/route_pairs.parquet')
final_refs=allrefs[allrefs.half.eq('final')].reset_index(drop=True)
if set(final_refs.sid)&set(refs.sid):raise ValueError('Identity overlap')
def evaluate(prob):
 order=np.lexsort((d.sid.to_numpy(),-prob,d.tid.to_numpy()));w=order[np.r_[True,d.tid.to_numpy()[order][1:]!=d.tid.to_numpy()[order][:-1]]]
 w=w[prob[w]>=.74];base=d.iloc[w][['sid','tid']].copy()
 rescue=route[(route.score>=1.88)&~route.tid.isin(base.tid)].sort_values(['tid','score','sid'],ascending=[True,False,True]).drop_duplicates('tid')
 links=pd.concat([base,rescue[['sid','tid']]],ignore_index=True)
 if links.tid.duplicated().any():raise ValueError('Owner conflict')
 links=links.merge(truth.assign(y=1),on=['sid','tid'],how='left',validate='one_to_one').fillna({'y':0})
 return m.metric.evaluate_winners(links.sid.to_numpy(),np.ones(len(links)),links.y.to_numpy(),final_refs.sid.to_numpy(),final_refs.true_links.to_numpy(),.5)
base=evaluate(old);selected=evaluate(p)
result=ev.comparison(selected,base,final_refs)
us=final_refs.country.eq('us').to_numpy();us_result=ev.comparison(selected[us],base[us],final_refs[us].reset_index(drop=True))
passed=us_result['delta']>0 and us_result['paired_95_interval'][0]>0 and result['delta']>0
result.update({'us':us_result,'eligible_for_export':passed,'baseline':float(base.mean()),'manifest':manifest})
ev.write_json(OUT/'audit.json',result)
print(json.dumps(result,indent=2),flush=True)
