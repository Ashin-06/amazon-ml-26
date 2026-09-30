"""Export the locked fast rescue matcher, preserving v27 France and candidate scope."""
from pathlib import Path
import csv,hashlib,importlib.util,itertools,json,time
import duckdb,joblib,numpy as np,pandas as pd,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import os
HERE=Path(__file__).parent;PKG=Path(__file__).resolve().parents[3];ROOT=Path.cwd()  # PKG: code/models; ROOT: data working directory

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def write_rows(base_rows,fr_rows,out,fr_ids,extra):
 """Preserve French rows verbatim in value; replace only India/US rescue decisions."""
 w=csv.writer(out,delimiter='\t',quoting=csv.QUOTE_NONE,lineterminator='\n')
 w.writerow(['source1_entity_id','matched_entity_ids']);n=links=0
 for a,b in itertools.zip_longest(base_rows,fr_rows):
  if a is None or b is None or len(a)!=2 or len(b)!=2 or a[0]!=b[0]:raise ValueError('Reference coverage/order mismatch')
  if a[0] in fr_ids:row=b
  else:
   old=list(filter(None,a[1].split(',')));added=extra.get(a[0],[])
   if set(old)&set(added) or len(set(added))!=len(added):raise ValueError('Duplicate new link')
   row=[a[0],','.join(old+added)]
  w.writerow(row);n+=1;links+=len(list(filter(None,row[1].split(','))))
 return n,links

def main():
 out=Path(os.environ.get('FAST_RESCUE_OUT',ROOT/'artifacts/after-0982910-20260927'/'fast_rescue_test_v1'));out.mkdir(exist_ok=False);start=time.monotonic()
 final=PKG/'models/rescue_learned';a=json.loads((final/'audit.json').read_text());manifest=json.loads((final/'locked_manifest.json').read_text())
 if not a['eligible_for_export']:raise ValueError('Final gate failed')
 if sha(final/'rescue_model.joblib')!=manifest['model_sha256'] or sha(HERE/'frozen_fast_rescue.py')!=manifest['feature_source_sha256']:raise ValueError('Model or feature source mismatch')
 sp=importlib.util.spec_from_file_location('rescue',HERE/'frozen_fast_rescue.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
 sp=importlib.util.spec_from_file_location('stack',PKG/'src/correction/stacker.py');st=importlib.util.module_from_spec(sp);sp.loader.exec_module(st)
 st.NORM=ROOT/'work/v21L/run/test';r.st=st
 base=ROOT/'submissions/v25stack_t074_fr0925_matching_results.tsv';fr=ROOT/'submissions/v27_matching_results.tsv';route=ROOT/'work/comp/v25/rescue_test_pairs.parquet'
 candidates=ROOT/'work/comp/v25/candidate_pairs_v27.tsv'
 inputs=[base,fr,route,candidates,*[st.NORM/f'norm_s{i}.parquet' for i in (1,2,3)]]
 stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
 x=pd.read_parquet(route)
 if x.duplicated(['sid','tid']).any() or not x.score.ge(1.5).all():raise ValueError('Route scope')
 model=joblib.load(final/'rescue_model.joblib');prob=[]
 for k in range(0,len(x),80000):
  batch=x.iloc[k:k+80000].reset_index(drop=True);tmp=out/f'features_{k:07d}';tmp.mkdir()
  f=r.features(batch,tmp)
  if list(f.columns)!=manifest['feature_names']:raise ValueError('Feature order')
  with threadpool_limits(limits=1):p=model.predict_proba(f.to_numpy())[:,1]
  if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Nonfinite/out-of-range scores')
  prob.append(p);print('Scored',k+len(batch),'of',len(x),flush=True)
 p=np.concatenate(prob);x[['sid','tid']].assign(p=p).to_parquet(out/'rescue_predictions.parquet',index=False)
 c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='600MB';SET max_temp_directory_size='3GB'");c.execute('SET temp_directory='+st.q(out/'ownership_tmp'))
 c.register('route',x[['tid','te']].drop_duplicates())
 rd=f"read_csv({st.q(base)},delim='\t',header=true,all_varchar=true,quote='',escape='')"
 owned=c.execute(f"SELECT DISTINCT r.tid FROM route r SEMI JOIN (SELECT unnest(string_split(matched_entity_ids,',')) te FROM {rd} WHERE coalesce(matched_entity_ids,'')<>'') b USING(te)").df()
 mask=x.tid.isin(owned.tid)
 if not np.array_equal(mask.to_numpy(),x.owned.to_numpy()):raise ValueError('Original rescue baseline ownership changed')
 fr_ids=set(c.execute(f"SELECT entity_id FROM read_parquet({st.q(st.NORM/'norm_s1.parquet')}) WHERE country='france'").df().entity_id);c.close()
 picked=r.choose(x.assign(y=-1),p,set(owned.tid),manifest['cutoff']).merge(x[['sid','tid','se','te']],on=['sid','tid'],validate='one_to_one')
 extra=picked.groupby('se').te.apply(lambda z:sorted(z)).to_dict();picked.to_parquet(out/'selected_rescue_pairs.parquet',index=False)
 dest=out/'matching_results.tsv'
 with base.open(encoding='utf-8',newline='') as f,fr.open(encoding='utf-8',newline='') as g,dest.open('x',encoding='utf-8',newline='') as z:
  aa=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE);bb=csv.reader(g,delimiter='\t',quoting=csv.QUOTE_NONE)
  expected=['source1_entity_id','matched_entity_ids']
  if next(aa)!=expected or next(bb)!=expected:raise ValueError('Header')
  n,links=write_rows(aa,bb,z,fr_ids,extra)
 if n!=1732544:raise ValueError('Reference count')
 if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=stamp for p,stamp in stamps.items()):raise ValueError('Input changed')
 result={'complete':True,'matching':str(dest),'matching_sha256':sha(dest),'rows':n,'links':links,'learned_rescue_links':len(picked),'cutoff':manifest['cutoff'],
 'candidate':str(candidates),'candidate_sha256':sha(candidates),'inputs':{str(p):sha(p) for p in inputs},
 'model_manifest_sha256':sha(final/'locked_manifest.json'),'source_sha256':sha(__file__),'seconds':time.monotonic()-start,
 'scope':'India/US learned rescue replaces fixed route; France copied from v27; existing v27 candidates unchanged; not yet independently validated or published.'}
 (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
