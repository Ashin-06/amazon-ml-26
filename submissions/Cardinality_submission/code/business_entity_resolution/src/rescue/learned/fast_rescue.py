"""Fast grouped rescue-matcher pilot on existing natural candidates only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import hashlib,importlib.util,json,time
import duckdb,joblib,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from threadpoolctl import threadpool_limits
HERE=Path(__file__).parent;PKG=Path(__file__).resolve().parents[3];ROOT=Path.cwd()  # PKG: code/models; ROOT: data working directory
OUT=ROOT/'artifacts/after-0982910-20260927'/'fast_rescue_v1'

def features(x,out):
 c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='650MB';SET max_temp_directory_size='3GB';SET preserve_insertion_order=false")
 c.execute('SET temp_directory='+st.q(out/'sql_tmp'))
 paths='['+','.join(st.q(st.NORM/f'norm_s{i}.parquet') for i in (1,2,3))+']'
 c.execute(f'CREATE VIEW r AS SELECT rid,src,country,name,core,addr,rawn FROM read_parquet({paths})')
 c.register('keys',x[['sid','tid']].assign(ix=np.arange(len(x))))
 t=c.execute('''SELECT k.ix,s.name sn,t.name tn,s.core sc,t.core tc,s.addr sa,t.addr ta,s.rawn sr,t.rawn tr,s.country co,t.src tsrc
 FROM keys k JOIN r s ON s.rid=k.sid JOIN r t ON t.rid=k.tid ORDER BY k.ix''').df()
 if len(t)!=len(x) or not np.array_equal(t.ix,np.arange(len(x))):raise ValueError('Feature join')
 c.register('names',t[['co','sc']].rename(columns={'co':'country','sc':'core'}).drop_duplicates())
 freq=c.execute('SELECT r.country,r.core,count(*) FILTER(WHERE src=1) ns,count(*) FILTER(WHERE src>1) nt FROM r SEMI JOIN names USING(country,core) GROUP BY 1,2').df().rename(columns={'country':'co','core':'sc'})
 t=t.merge(freq,on=['co','sc'],how='left',validate='many_to_one').sort_values('ix').reset_index(drop=True);c.close()
 f=pd.DataFrame({'score':x.score.to_numpy(),'log_ns':np.log1p(t.ns),'log_nt':np.log1p(t.nt),'source3':t.tsrc.eq(3).astype(float)})
 for aa,bb,prefix in [('sn','tn','name'),('sc','tc','core'),('sa','ta','addr'),('sr','tr','raw')]:
  a=t[aa].tolist();b=t[bb].tolist()
  for name,fun in [('ratio',fuzz.ratio),('tset',fuzz.token_set_ratio),('tsort',fuzz.token_sort_ratio),('jw',JaroWinkler.normalized_similarity)]:
   f[prefix+'_'+name]=cpdist(a,b,scorer=fun,workers=1,dtype=np.float32)
  f[prefix+'_exact']=(t[aa].eq(t[bb])&t[aa].ne('')).astype(float)
  f[prefix+'_len_s']=t[aa].str.len();f[prefix+'_len_t']=t[bb].str.len()
  sa=t[aa].str.split().map(set);sb=t[bb].str.split().map(set)
  f[prefix+'_drop']=[len(a-b) for a,b in zip(sa,sb)];f[prefix+'_add']=[len(b-a) for a,b in zip(sa,sb)]
 a=t.sa.str.findall(r'\d+').map(set);b=t.ta.str.findall(r'\d+').map(set)
 f['numbers_drop']=[len(aa-bb) for aa,bb in zip(a,b)];f['numbers_add']=[len(bb-aa) for aa,bb in zip(a,b)]
 if not np.isfinite(f.to_numpy()).all():raise ValueError('Invalid features')
 return f.astype(np.float32)

def fit(arm,X,y):
 if arm=='linear':m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=27))
 else:m=lgb.LGBMClassifier(n_estimators=200,num_leaves=7,max_depth=3,min_child_samples=50,learning_rate=.04,reg_lambda=20,n_jobs=1,verbosity=-1,random_state=27,deterministic=True,force_col_wise=True)
 m.fit(X,y);return m

def choose(x,p,owned,cut):
 use=(p>=cut)&~x.tid.isin(owned).to_numpy();a=x.loc[use,['sid','tid','y']].assign(p=p[use])
 return a.sort_values(['tid','p','sid'],ascending=[True,False,True]).drop_duplicates('tid')

def values(base,added,refs):
 b=pd.concat([base[['sid','tid','y']],added[['sid','tid','y']]],ignore_index=True)
 if b.tid.duplicated().any():raise ValueError('Ownership')
 return st.metric.evaluate_winners(b.sid.to_numpy(),np.ones(len(b)),b.y.to_numpy(),refs.sid.to_numpy(),refs.true_links.to_numpy(),.5)

def compare(v,b):
 d=v-b;se=float(d.std(ddof=1)/np.sqrt(len(d)))
 return {'score':float(v.mean()),'delta':float(d.mean()),'paired_95_interval':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)]}

def main():
 global st
 OUT.mkdir(exist_ok=False);start=time.monotonic()
 spec=importlib.util.spec_from_file_location('stack',PKG/'src/correction/stacker.py');st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
 pilot=ROOT/'artifacts/push995-20260927-0915/stack_v4'
 refs=pd.read_parquet(pilot/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
 x=pd.read_parquet(HERE/'v26_combined_audit_v1/route_pairs.parquet')
 c=duckdb.connect();c.register('pairs',x);c.register('tune',refs[['sid']])
 # Final labels are masked inside the query; no final truth reaches pandas.
 x=c.execute(f'''SELECT p.sid,p.tid,p.score,CASE WHEN r.sid IS NULL THEN -1 ELSE CAST(g.sid IS NOT NULL AS INTEGER) END y
 FROM pairs p LEFT JOIN tune r ON p.sid=r.sid LEFT JOIN read_parquet({st.q(ROOT/'work/comp/hold_truth.parquet')}) g ON g.sid=p.sid AND g.tid=p.tid''').df();c.close()
 if x.duplicated(['sid','tid']).any():raise ValueError('Duplicate route')
 f=features(x,OUT);f.to_parquet(OUT/'features.parquet',index=False);x.to_parquet(OUT/'pairs_masked.parquet',index=False)
 d=pd.read_parquet(pilot/'pairs_labels_masked.parquet');oof=pd.read_parquet(HERE/'xenc7_pilot_v1/oof_predictions.parquet')
 if not d[['sid','tid']].equals(oof[['sid','tid']]):raise ValueError('Base OOF alignment')
 p=oof.v25_control.to_numpy();order=np.lexsort((d.sid.to_numpy(),-p,d.tid.to_numpy()));w=order[np.r_[True,d.tid.to_numpy()[order][1:]!=d.tid.to_numpy()[order][:-1]]]
 base=d.iloc[w[p[w]>=.74]][['sid','tid','y']];owned=set(base.tid)
 fixed=choose(x,x.score.to_numpy(),owned,1.88);baseline=values(base,fixed,refs)
 eligible=~x.tid.isin(owned).to_numpy();folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
 rf=x.sid.map(folds).fillna(-1).to_numpy();X=f.to_numpy();y=x.y.to_numpy()
 print('Natural route pairs',len(x),'eligible tune',int(((rf>=0)&eligible).sum()),'positive',int(y[(rf>=0)&eligible].sum()),flush=True)
 results={};saved=x[['sid','tid']].copy()
 with threadpool_limits(limits=1):
  for arm in ('linear','tree'):
   p=np.zeros(len(x))
   for fold in range(3):
    train=(rf>=0)&(rf!=fold)&eligible;check=rf==fold;other=rf<0
    model=fit(arm,X[train],y[train]);p[check]=model.predict_proba(X[check])[:,1];p[other]+=model.predict_proba(X[other])[:,1]/3
    joblib.dump(model,OUT/f'{arm}_fold{fold}.joblib')
   curve=[]
   for cut in np.r_[np.arange(.5,.991,.01),.995]:
    v=values(base,choose(x,p,owned,float(cut)),refs);curve.append((float(v.mean()),float(cut),v))
   score,cut,v=max(curve,key=lambda a:(a[0],a[1]));r=compare(v,baseline);r['cutoff']=cut;r['by_country']={co:compare(v[refs.country.eq(co)],baseline[refs.country.eq(co)]) for co in refs.country.unique()}
   results[arm]=r;saved[arm]=p;print(arm,r,flush=True)
 good=[a for a,r in results.items() if r['delta']>=.0001 and r['paired_95_interval'][0]>0]
 selected=max(good,key=lambda a:results[a]['score']) if good else None
 report={'selected':selected,'results':results,'baseline_tune':float(baseline.mean()),'feature_names':list(f.columns),'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'protocol':'Three S1 tuning folds; deployment-cohort fit; final labels masked. Exact complete-reference F0.5; base OOF ownership. No final or test inference yet.'}
 saved.to_parquet(OUT/'oof_predictions.parquet',index=False);(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('Selected',selected,'seconds',report['seconds'],flush=True)
if __name__=='__main__':main()
