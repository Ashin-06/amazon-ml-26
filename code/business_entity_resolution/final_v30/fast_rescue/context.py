"""Grouped experiment using stronger, label-free sibling records for v24."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import duckdb,joblib,numpy as np,pandas as pd,psutil,pyarrow as pa
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];PREV=ROOT/'artifacts/push995-20260927-0915'
spec=importlib.util.spec_from_file_location('base',PREV/'stacker.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"

def context_features(d,band,norm,out):
    """Use other high-confidence targets; IDs only join, labels never consumed."""
    c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='400MB';SET max_temp_directory_size='2GB';SET preserve_insertion_order=false")
    c.execute('SET temp_directory='+q(out/'context_tmp'))
    scores=d[['sid','tid','p','xe']];keys=d.loc[band,['sid','tid']].reset_index(drop=True).assign(ix=np.arange(int(band.sum()),dtype=np.int32))
    c.register('scores',scores);c.register('keys',keys)
    paths='['+','.join(q(norm/f'norm_s{i}.parquet') for i in (2,3))+']'
    c.execute(f'CREATE VIEW targets AS SELECT rid,src,name,core,addr,rawn FROM read_parquet({paths})')
    feature={};n=len(keys)
    for name,pmin,xmin in [('strict',.995,4.),('broad',.98,3.)]:
        c.execute(f'CREATE OR REPLACE TABLE anchors AS SELECT sid,tid aid,p FROM scores WHERE p>={pmin} AND (xe IS NULL OR xe>={xmin}) QUALIFY row_number() OVER(PARTITION BY sid ORDER BY p DESC,tid)<=6')
        query='''SELECT k.ix,a.p,t.src ts,u.src us,t.name tn,u.name un,t.core tc,u.core uc,t.addr ta,u.addr ua,t.rawn tr,u.rawn ur
          FROM keys k JOIN anchors a ON a.sid=k.sid AND a.aid<>k.tid
          JOIN targets t ON t.rid=k.tid JOIN targets u ON u.rid=a.aid'''
        vals={'count':np.zeros(n),'same_source':np.zeros(n),'addressed':np.zeros(n),'max_p':np.zeros(n)}
        reader=c.execute(query).fetch_record_batch(50000)
        for batch in reader:
            z=batch.to_pandas();ix=z.ix.to_numpy()
            np.add.at(vals['count'],ix,1);np.add.at(vals['same_source'],ix,z.ts.eq(z.us).to_numpy().astype(float))
            np.add.at(vals['addressed'],ix,z.ua.ne('').to_numpy().astype(float));np.maximum.at(vals['max_p'],ix,z.p.to_numpy())
            for a,b,prefix in [('tn','un','name'),('tc','uc','core'),('ta','ua','addr'),('tr','ur','raw')]:
                aa=z[a].tolist();bb=z[b].tolist();nonempty=z[a].ne('').to_numpy()&z[b].ne('').to_numpy()
                for label,fun in [('ratio',fuzz.ratio),('tsort',fuzz.token_sort_ratio),('partial',fuzz.partial_ratio)]:
                    col=prefix+'_'+label
                    if col not in vals:vals[col]=np.zeros(n)
                    v=cpdist(aa,bb,scorer=fun,workers=1,dtype=np.float32);v[~nonempty]=0
                    np.maximum.at(vals[col],ix,v)
                col=prefix+'_exact'
                if col not in vals:vals[col]=np.zeros(n)
                np.maximum.at(vals[col],ix,(nonempty&z[a].eq(z[b]).to_numpy()).astype(float))
        for col,val in vals.items():feature[name+'_'+col]=val
    c.close();return pd.DataFrame(feature).astype(np.float32)

def fit(arm,X,y):
    """Fixed tuning-independent capacity; single CPU, no final-label stopping."""
    args=dict(n_jobs=1,verbosity=-1,random_state=27,deterministic=True,force_col_wise=True,learning_rate=.04,colsample_bytree=.9,min_child_samples=100)
    if arm=='context_small':args.update(n_estimators=320,num_leaves=15,max_depth=5,reg_lambda=15)
    elif arm=='context_deep':args.update(n_estimators=600,num_leaves=31,max_depth=7,reg_lambda=25)
    else:raise ValueError(arm)
    m=lgb.LGBMClassifier(**args);m.fit(X,y);return m

def pilot(out):
    start=time.monotonic();d=pd.read_parquet(PREV/'stack_v4/pairs_labels_masked.parquet')
    f=pd.read_parquet(PREV/'stack_v4/band_features.parquet');refs=pd.read_parquet(PREV/'stack_v4/tune_references.parquet')
    band=d.xe.notna().to_numpy();ix=np.flatnonzero(band)
    extra=context_features(d,band,base.NORM,out);f=pd.concat([f,extra],axis=1)
    f.to_parquet(out/'features.parquet',index=False)
    old=pd.read_parquet(PREV/'stack_v4/oof_predictions.parquet')
    if not np.array_equal(d[['sid','tid']],old[['sid','tid']]):raise ValueError('Old/new alignment')
    oldv=base.winner_values(d,old.lgb_small_tree.to_numpy(),refs,.77)
    folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
    rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8);X=f.to_numpy();results={};saved=d[['sid','tid','p','y']].copy()
    with threadpool_limits(limits=1):
        for arm in ('context_small','context_deep'):
            p=d.p.to_numpy().copy();bp=np.zeros(len(ix))
            for fold in range(3):
                train=(rf>=0)&(rf!=fold);test=rf==fold;other=rf<0
                m=fit(arm,X[train],d.y.to_numpy()[ix[train]])
                bp[test]=m.predict_proba(X[test])[:,1];bp[other]+=m.predict_proba(X[other])[:,1]/3
                joblib.dump(m,out/f'{arm}_fold{fold}.joblib')
            p[ix]=bp;t,v,curve=base.choose_cutoff(d,p,refs);diff=v-oldv;se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
            results[arm]={'cutoff':t,'tune_macro_f05':float(v.mean()),'delta_vs_v24stack':float(diff.mean()),
              'paired_95_interval':[float(diff.mean()-1.96*se),float(diff.mean()+1.96*se)],'by_country':{c:float(v[refs.country.eq(c)].mean()) for c in sorted(refs.country.unique())},'curve':curve}
            saved[arm]=p;print(arm,t,float(v.mean()),'delta',float(diff.mean()),flush=True)
    chosen=max(results,key=lambda a:results[a]['tune_macro_f05']);r=results[chosen]
    report={'protocol':'3-fold S1 tuning OOF. Sibling anchors from held-out base/neural scores only; no labels/correction probabilities in features. No final labels read.',
      'selected':chosen,'eligible_for_locked_final':r['delta_vs_v24stack']>=.0002 and r['paired_95_interval'][0]>0,
      'baseline_tune':float(oldv.mean()),'results':results,'feature_names':list(f.columns),'source_sha256':sha(__file__),
      'base_feature_source_sha256':sha(PREV/'stacker.py'),'seconds':time.monotonic()-start}
    saved.to_parquet(out/'oof_predictions.parquet',index=False)
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('ELIGIBLE FOR FINAL',report['eligible_for_locked_final'],'seconds',report['seconds'],flush=True)

def final(pilot,out):
    info=json.loads((pilot/'audit.json').read_text());arm=info['selected'];cut=info['results'][arm]['cutoff']
    if not info['eligible_for_locked_final']:raise ValueError('Predeclared tuning gate failed')
    if sha(__file__)!=info['source_sha256']:raise ValueError('Source changed after selection')
    d=pd.read_parquet(PREV/'stack_v4/pairs_labels_masked.parquet');f=pd.read_parquet(pilot/'features.parquet')
    ix=np.flatnonzero(d.xe.notna());X=f.to_numpy();train=d.y.to_numpy()[ix]>=0
    with threadpool_limits(limits=1):
        model=fit(arm,X[train],d.y.to_numpy()[ix[train]]);joblib.dump(model,out/'model.joblib')
        p=d.p.to_numpy().copy();p[ix]=model.predict_proba(X)[:,1]
    locked={'arm':arm,'cutoff':cut,'source_sha256':sha(__file__),'features':list(f.columns),'model_sha256':sha(out/'model.joblib')}
    (out/'locked.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')
    c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='300MB'");c.register('pairs',d[['sid','tid']])
    labels=c.execute(f'SELECT k.sid,k.tid,h.y FROM pairs k JOIN read_parquet({q(base.MD/"hold_preds.parquet")}) h USING(sid,tid)').df();c.close()
    d=d.drop(columns='y').merge(labels,on=['sid','tid'],validate='one_to_one');refs=pd.read_parquet(base.MD/'hold_error_rows.parquet');refs=refs[refs.half.eq('final')].sort_values('sid')
    old=pd.read_parquet(PREV/'final_v1/locked_predictions.parquet')
    if not np.array_equal(d[['sid','tid']],old[['sid','tid']]):raise ValueError('Pair alignment')
    v=base.winner_values(d,p,refs,cut);ov=base.winner_values(d,old.lgb_small_tree.to_numpy(),refs,.77);diff=v-ov;se=float(diff.std(ddof=1)/np.sqrt(len(v)))
    result={'final_macro_f05':float(v.mean()),'delta':float(diff.mean()),'paired_95_interval':[float(diff.mean()-1.96*se),float(diff.mean()+1.96*se)],'by_country':{co:float(v[refs.country.eq(co)].mean()) for co in sorted(refs.country.unique())},'locked':locked,'caveat':'Historically reused final; held-reference ownership; no French labels.'}
    d[['sid','tid']].assign(p=p).to_parquet(out/'predictions.parquet',index=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','final']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pilot',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    if a.mode=='pilot':pilot(a.out)
    else:final(a.pilot,a.out)

if __name__=='__main__':main()
