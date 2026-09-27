from pathlib import Path
import importlib.util,json,hashlib,time
import duckdb,numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).parent;ROOT=H.parent.parent;OUT=H/'second_word_adaptation_v1'
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();r=mod('rescue',H/'fast_rescue.py');st=mod('stack',ROOT/'artifacts/push995-20260927-0915/stacker.py');r.st=st
 old=ROOT/'artifacts/push995-20260927-0915/stack_v4';refs=pd.read_parquet(old/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
 x=pd.read_parquet(H/'second_word_route_v1/unlabelled_pairs.parquet');f=r.features(x,OUT).assign(new_route=1.)
 c=duckdb.connect();c.register('pairs',x);c.register('refs',refs[['sid']]);labels=c.execute(f'SELECT p.sid,p.tid,CAST(g.sid IS NOT NULL AS INTEGER) y FROM pairs p JOIN refs r ON r.sid=p.sid LEFT JOIN read_parquet({st.q(ROOT/"work/comp/hold_truth.parquet")}) g ON g.sid=p.sid AND g.tid=p.tid').df();c.close()
 x=x.merge(labels,on=['sid','tid'],validate='one_to_one');assert x[['sid','tid']].equals(pd.read_parquet(H/'second_word_route_v1/unlabelled_pairs.parquet')[['sid','tid']])
 ox=pd.read_parquet(H/'fast_rescue_v1/pairs_masked.parquet');of=pd.read_parquet(H/'fast_rescue_v1/features.parquet').assign(new_route=0.)
 op=pd.read_parquet(H/'fast_rescue_v1/oof_predictions.parquet');assert ox[['sid','tid']].equals(op[['sid','tid']])
 d=pd.read_parquet(old/'pairs_labels_masked.parquet');bp=pd.read_parquet(H/'xenc7_pilot_v1/oof_predictions.parquet');assert d[['sid','tid']].equals(bp[['sid','tid']])
 p=bp.v25_control.to_numpy();o=np.lexsort((d.sid.to_numpy(),-p,d.tid.to_numpy()));w=o[np.r_[True,d.tid.to_numpy()[o][1:]!=d.tid.to_numpy()[o][:-1]]];base=d.iloc[w[p[w]>=.74]][['sid','tid','y']]
 rescue=r.choose(ox,op.linear.to_numpy(),set(base.tid),.6900000000000002);combined=pd.concat([base,rescue[['sid','tid','y']]],ignore_index=True)
 # Old route training excludes base ownership; new route excludes existing v28 ownership.
 eligible_old=ox.y.ge(0)&~ox.tid.isin(base.tid);eligible_new=~x.tid.isin(combined.tid)
 xx=pd.concat([ox.loc[eligible_old],x.loc[eligible_new]],ignore_index=True);ff=pd.concat([of.loc[eligible_old],f.loc[eligible_new]],ignore_index=True)
 folds=lambda ids:ids.map(lambda s:int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3).to_numpy()
 trainfold=folds(xx.sid);valfold=folds(x.sid);prob=np.zeros(len(x))
 with threadpool_limits(limits=1):
  for k in range(3):
   tr=trainfold!=k;va=valfold==k;m=r.fit('linear',ff.to_numpy()[tr],xx.y.to_numpy()[tr]);prob[va]=m.predict_proba(f.to_numpy()[va])[:,1];joblib.dump(m,OUT/f'fold{k}.joblib')
 baseline=r.values(combined,combined.iloc[:0],refs);curves=[]
 for cut in np.r_[np.arange(.5,.991,.01),.995]:
  added=r.choose(x,prob,set(combined.tid),float(cut));v=r.values(combined,added,refs);curves.append((float(v.mean()),float(cut),v,added))
 _,cut,v,added=max(curves,key=lambda z:(z[0],z[1]));a=r.compare(v,baseline);a.update(cutoff=cut,added=len(added),added_true=int(added.y.sum()),by_country={co:r.compare(v[refs.country.eq(co)],baseline[refs.country.eq(co)]) for co in refs.country.unique()},seconds=time.monotonic()-start,final_labels_opened=False)
 a['eligible']=a['delta']>=.00005 and a['paired_95_interval'][0]>0 and all(z['delta']>=0 for z in a['by_country'].values())
 x[['sid','tid']].assign(p=prob).to_parquet(OUT/'oof_predictions.parquet',index=False);(OUT/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf-8');print(json.dumps(a,indent=2))
if __name__=='__main__':main()
