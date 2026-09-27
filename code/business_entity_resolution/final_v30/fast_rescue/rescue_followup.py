"""Bounded tuning-only attempts to improve the frozen learned rescue matcher."""
from pathlib import Path
import importlib.util,json,time,hashlib
import numpy as np,pandas as pd,joblib,lightgbm as lgb
from threadpoolctl import threadpool_limits
H=Path(__file__).parent;ROOT=H.parent.parent;OUT=H/'rescue_followup_v1'
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def interaction_features(f):
 f=f.copy()
 for name in ['name_ratio','name_tsort','core_ratio','raw_ratio']:
  n=f[name]/100.;a=f.addr_ratio/100.
  f[name+'_addr_product']=n*a
  f[name+'_addr_min']=np.minimum(n,a)
  f[name+'_number_conflict']=n*((f.numbers_drop+f.numbers_add)>0)
  f[name+'_rare']=n/(1+f.log_ns)
 f['addr_without_extra_digits']=f.addr_ratio*((f.numbers_drop+f.numbers_add)==0)
 f['addr_exact_core_overlap']=f.addr_exact*f.core_tset/100.
 f['core_exact_addr_overlap']=f.core_exact*f.addr_ratio/100.
 return f.astype(np.float32)
def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic()
 r=module('rescue',H/'fast_rescue.py');st=module('stack',ROOT/'artifacts/push995-20260927-0915/stacker.py');r.st=st
 x=pd.read_parquet(H/'fast_rescue_v1/pairs_masked.parquet');f=pd.read_parquet(H/'fast_rescue_v1/features.parquet')
 old=pd.read_parquet(H/'fast_rescue_v1/oof_predictions.parquet');assert x[['sid','tid']].equals(old[['sid','tid']])
 pdir=ROOT/'artifacts/push995-20260927-0915/stack_v4'
 refs=pd.read_parquet(pdir/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
 d=pd.read_parquet(pdir/'pairs_labels_masked.parquet');oof=pd.read_parquet(H/'xenc7_pilot_v1/oof_predictions.parquet');assert d[['sid','tid']].equals(oof[['sid','tid']])
 bp=oof.v25_control.to_numpy();order=np.lexsort((d.sid.to_numpy(),-bp,d.tid.to_numpy()));w=order[np.r_[True,d.tid.to_numpy()[order][1:]!=d.tid.to_numpy()[order][:-1]]]
 base=d.iloc[w[bp[w]>=.74]][['sid','tid','y']];owned=set(base.tid)
 baseline=r.values(base,r.choose(x,old.linear.to_numpy(),owned,.6900000000000002),refs)
 folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
 rf=x.sid.map(folds).fillna(-1).to_numpy();eligible=~x.tid.isin(owned).to_numpy();y=x.y.to_numpy()
 assert np.all(y[rf<0]==-1)
 allp={'linear_tree_blend':(old.linear.to_numpy()+old.tree.to_numpy())/2}
 with threadpool_limits(limits=1):
  for arm in ['interactions','linear_wider_tree_blend']:
   X=interaction_features(f).to_numpy() if arm=='interactions' else f.to_numpy();p=np.zeros(len(x))
   for fold in range(3):
    tr=(rf>=0)&(rf!=fold)&eligible;val=rf==fold;other=rf<0
    if arm=='interactions':m=r.fit('linear',X[tr],y[tr])
    else:
     m=lgb.LGBMClassifier(n_estimators=200,num_leaves=15,max_depth=4,min_child_samples=75,learning_rate=.04,reg_lambda=30,n_jobs=1,verbosity=-1,random_state=27,deterministic=True,force_col_wise=True);m.fit(X[tr],y[tr])
    p[val]=m.predict_proba(X[val])[:,1];p[other]+=m.predict_proba(X[other])[:,1]/3
    joblib.dump(m,OUT/f'{arm}_fold{fold}.joblib')
   if arm!='interactions':p=(p+old.linear.to_numpy())/2
   allp[arm]=p
 results={};saved=x[['sid','tid']].copy()
 for arm,p in allp.items():
  curves=[]
  for cut in np.r_[np.arange(.5,.991,.01),.995]:
   v=r.values(base,r.choose(x,p,owned,float(cut)),refs);curves.append((float(v.mean()),float(cut),v))
  _,cut,v=max(curves,key=lambda a:(a[0],a[1]));z=r.compare(v,baseline);z['cutoff']=cut;z['by_country']={co:r.compare(v[refs.country.eq(co)],baseline[refs.country.eq(co)]) for co in refs.country.unique()}
  z['passes_gate']=z['delta']>=.00005 and z['paired_95_interval'][0]>0 and all(t['delta']>=0 for t in z['by_country'].values())
  results[arm]=z;saved[arm]=p;print(arm,json.dumps(z),flush=True)
 good=[k for k,v in results.items() if v['passes_gate']];selected=max(good,key=lambda a:results[a]['score']) if good else None
 saved.to_parquet(OUT/'oof_predictions.parquet',index=False)
 audit={'selected':selected,'baseline_v28_tune':float(baseline.mean()),'results':results,'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'final_labels_opened':False}
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('Selected',selected,'seconds',audit['seconds'],flush=True)
if __name__=='__main__':main()
