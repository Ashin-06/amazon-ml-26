"""Bounded tuning-only first-word corruption retrieval pilot."""
from pathlib import Path
import importlib.util,json,time,hashlib
import duckdb,joblib,numpy as np,pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from threadpoolctl import threadpool_limits
H=Path(__file__).parent;ROOT=H.parent.parent;OUT=H/'second_word_route_v1'
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();r=module('rescue',H/'fast_rescue.py');st=module('stack',ROOT/'artifacts/push995-20260927-0915/stacker.py');r.st=st
 old=ROOT/'artifacts/push995-20260927-0915/stack_v4';refs=pd.read_parquet(old/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
 c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='700MB';SET max_temp_directory_size='5GB';SET preserve_insertion_order=false");c.execute('SET temp_directory='+st.q(OUT/'sql_tmp'));c.register('refs',refs[['sid']])
 paths='['+','.join(st.q(st.NORM/f'norm_s{i}.parquet') for i in (1,2,3))+']'
 c.execute(f"CREATE TABLE r AS SELECT rid,src,country,name,addr,string_split(core,' ')[1] fw,string_split(core,' ')[2] sw,regexp_extract(addr,'[0-9]+') num FROM read_parquet({paths})")
 c.execute("CREATE TABLE s AS SELECT r.* FROM r JOIN refs ON refs.sid=r.rid WHERE src=1 AND num<>'' AND sw<>''")
 join='FROM s JOIN r t ON t.src>1 AND s.country=t.country AND s.num=t.num AND s.sw=t.sw AND s.fw<>t.fw'
 count=c.execute('SELECT count(*) '+join).fetchone()[0];print('Raw join',count,flush=True)
 if count>12000000:
  (OUT/'audit.json').write_text(json.dumps({'raw_pairs':count,'stopped':'12-million-pair budget exceeded','eligible':False}),encoding='utf-8');return
 c.execute('CREATE VIEW oldcand AS SELECT sid,tid FROM read_parquet('+st.q(st.MD/'hold_preds.parquet')+')')
 parts=[]
 for k in range(16):
  x=c.execute(f'SELECT s.rid sid,t.rid tid,s.name sn,t.name tn,s.addr sa,t.addr ta {join} WHERE hash(s.rid)%16={k} AND NOT EXISTS(SELECT 1 FROM oldcand o WHERE o.sid=s.rid AND o.tid=t.rid)').df()
  if len(x):
   score=(cpdist(x.sn.tolist(),x.tn.tolist(),scorer=fuzz.ratio,workers=1,dtype=np.float32)+cpdist(x.sa.tolist(),x.ta.tolist(),scorer=fuzz.token_set_ratio,workers=1,dtype=np.float32))/100
   parts.append(x[['sid','tid']].assign(score=score).loc[score>=1.5])
  print('Chunk',k,flush=True)
 if not parts:
  (OUT/'audit.json').write_text(json.dumps({'raw_pairs':count,'new_pairs':0,'eligible':False}),encoding='utf-8');return
 x=pd.concat(parts,ignore_index=True);assert not x.duplicated(['sid','tid']).any();x.to_parquet(OUT/'unlabelled_pairs.parquet',index=False)
 f=r.features(x,OUT);prob=np.zeros(len(x));folds=x.sid.map(lambda s:int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3).to_numpy()
 with threadpool_limits(limits=1):
  for k in range(3):
   m=joblib.load(H/f'fast_rescue_v1/linear_fold{k}.joblib');use=folds==k
   if use.any():prob[use]=m.predict_proba(f.to_numpy()[use])[:,1]
 x[['sid','tid']].assign(p=prob).to_parquet(OUT/'locked_predictions.parquet',index=False)
 # Only tuning labels enter pandas after route and all model decisions are frozen.
 c.register('pairs',x)
 x=c.execute(f'SELECT p.*,CAST(g.sid IS NOT NULL AS INTEGER) y FROM pairs p JOIN refs r ON p.sid=r.sid LEFT JOIN read_parquet({st.q(ROOT/"work/comp/hold_truth.parquet")}) g ON p.sid=g.sid AND p.tid=g.tid').df();c.close()
 # Preserve prediction alignment explicitly after SQL join.
 pred=pd.read_parquet(OUT/'locked_predictions.parquet');x=pred.merge(x,on=['sid','tid'],validate='one_to_one');assert len(x)==len(pred)
 d=pd.read_parquet(old/'pairs_labels_masked.parquet');bp=pd.read_parquet(H/'xenc7_pilot_v1/oof_predictions.parquet');assert d[['sid','tid']].equals(bp[['sid','tid']])
 p=bp.v25_control.to_numpy();o=np.lexsort((d.sid.to_numpy(),-p,d.tid.to_numpy()));w=o[np.r_[True,d.tid.to_numpy()[o][1:]!=d.tid.to_numpy()[o][:-1]]];base=d.iloc[w[p[w]>=.74]][['sid','tid','y']]
 rx=pd.read_parquet(H/'fast_rescue_v1/pairs_masked.parquet');rp=pd.read_parquet(H/'fast_rescue_v1/oof_predictions.parquet');assert rx[['sid','tid']].equals(rp[['sid','tid']])
 rescue=r.choose(rx,rp.linear.to_numpy(),set(base.tid),.6900000000000002);combined=pd.concat([base,rescue[['sid','tid','y']]],ignore_index=True)
 add=r.choose(x,x.p.to_numpy(),set(combined.tid),.6900000000000002)
 b=r.values(combined,add.iloc[:0],refs);v=r.values(combined,add,refs);a=r.compare(v,b);a.update(raw_pairs=count,new_candidates=len(x),new_true_candidates=int(x.y.sum()),accepted=len(add),accepted_true=int(add.y.sum()),by_country={co:r.compare(v[refs.country.eq(co)],b[refs.country.eq(co)]) for co in refs.country.unique()},seconds=time.monotonic()-start)
 a['eligible']=a['delta']>=.0001 and a['paired_95_interval'][0]>0 and all(z['delta']>=0 for z in a['by_country'].values());a['final_labels_opened']=False
 (OUT/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf-8');print(json.dumps(a,indent=2))
if __name__=='__main__':main()
