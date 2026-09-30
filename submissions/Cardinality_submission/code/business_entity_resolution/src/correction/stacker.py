"""Tune-only cross-fitted v24 third-stage models; no active run mutations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, ast, hashlib, importlib.util, json, time
from pathlib import Path
import duckdb, joblib, numpy as np, pandas as pd, psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Prefix, Postfix
from rapidfuzz.process import cpdist

ROOT=Path(__file__).resolve().parents[2]
MD=ROOT/'work/v24/kaggle/work/model'
OLD=ROOT/'work/v20/kaggle/work/model'
XE=ROOT/'work/kaggle_out/xenc_inf11/v24_hold_band_xe.parquet'
XE5=ROOT/'work/kaggle_out/xenc_inf6/v20_hold_band_xe.parquet'
NORM=ROOT/'work/v21L/run/train'
spec=importlib.util.spec_from_file_location('metric',Path(__file__).resolve().parent/'metric.py')
metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)

def q(v):
    """Quote a path or value in DuckDB."""
    return "'"+str(v).replace('\\','/').replace("'","''")+"'"

def packed(s,t):
    """Lossless pair keys used only for joins, never predictive features."""
    s=np.asarray(s);t=np.asarray(t)
    if np.any(s<0) or np.any(t<0) or np.any(s>=2**31) or np.any(t>=2**31):raise ValueError('ID range')
    return s.astype(np.int64)*2**31+t.astype(np.int64)

def load_pairs(out):
    """Mask final labels before pandas and validate band/key coverage."""
    refs=[]
    for b in pq.ParquetFile(MD/'hold_error_rows.parquet').iter_batches(use_threads=False):
        refs.append(b.filter(pc.equal(b.column('half'),'tune')).to_pandas())
    refs=pd.concat(refs).sort_values('sid').reset_index(drop=True)
    tune=pa.array(refs.sid);parts=[]
    for b in pq.ParquetFile(MD/'hold_preds.parquet').iter_batches(batch_size=65536,columns=['sid','tid','p_stage1','p','y'],use_threads=False):
        b=b.filter(pc.greater(b.column('p'),.002))
        b=b.set_column(4,'y',pc.if_else(pc.is_in(b.column('sid'),value_set=tune),pc.cast(b.column('y'),pa.int8()),pa.scalar(-1,pa.int8())))
        parts.append(b.to_pandas())
    d=pd.concat(parts,ignore_index=True)
    xe=pd.read_parquet(XE)
    d=d.merge(xe,on=['sid','tid'],how='left',validate='one_to_one')
    if not np.array_equal(d.xe.notna(),(d.p>.002)&(d.p<.9995)) or len(xe)!=int(d.xe.notna().sum()):raise ValueError('Neural coverage')
    # Avoid materializing/sorting millions of Python string IDs twice in RAM.
    a=pq.ParquetFile(MD/'split_ids.parquet');b=pq.ParquetFile(OLD/'split_ids.parquet')
    equal=a.metadata.num_rows==b.metadata.num_rows
    if equal:
        for aa,bb in zip(a.iter_batches(batch_size=65536,use_threads=False),b.iter_batches(batch_size=65536,use_threads=False)):
            if not aa.equals(bb,check_metadata=False):equal=False;break
    if not equal:
        con=duckdb.connect();con.execute("SET threads=1;SET memory_limit='250MB';SET max_temp_directory_size='1GB'")
        con.execute('SET temp_directory='+q(out/'identity_tmp'))
        x=f'SELECT entity_id,g,half FROM read_parquet({q(MD/"split_ids.parquet")})'
        y=f'SELECT entity_id,g,half FROM read_parquet({q(OLD/"split_ids.parquet")})'
        different=con.execute(f'SELECT count(*) FROM (({x} EXCEPT ALL {y}) UNION ALL ({y} EXCEPT ALL {x}))').fetchone()[0]
        con.close()
        if different:raise ValueError('Old/new identity groups differ')
    return d,refs

def old_scores(d):
    """Join old predictions without loading old labels or inventing missing rows."""
    wanted=pa.array(packed(d.sid,d.tid));parts=[]
    for b in pq.ParquetFile(OLD/'hold_preds.parquet').iter_batches(batch_size=65536,columns=['sid','tid','p_stage1','p'],use_threads=False):
        keys=pa.array(packed(b.column('sid').to_numpy(),b.column('tid').to_numpy()))
        parts.append(b.filter(pc.is_in(keys,value_set=wanted)).to_pandas())
    old=pd.concat(parts,ignore_index=True).rename(columns={'p':'p_old','p_stage1':'p1_old'})
    d=d.merge(old,on=['sid','tid'],how='left',validate='one_to_one')
    return d.merge(pd.read_parquet(XE5).rename(columns={'xe':'xe_old'}),on=['sid','tid'],how='left',validate='one_to_one')

def structural_features(d,out,norm=NORM):
    """Whole-pool frequencies and pair evidence; no IDs or country predictor."""
    src=(Path(__file__).resolve().parents[1]/'fast_er'/'model.py').read_text(encoding='utf-8')
    filler=next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILLER' for t in n.targets))
    words='['+','.join(q(w) for w in sorted(filler))+']'
    core=f"coalesce(nullif(array_to_string(list_filter(string_split(core,' '),x->NOT list_contains({words},x)),' '),''),core)"
    paths='['+','.join(q(norm/f'norm_s{i}.parquet') for i in (1,2,3))+']'
    con=duckdb.connect();con.execute("SET threads=1;SET memory_limit='500MB';SET max_temp_directory_size='2GB';SET preserve_insertion_order=false")
    con.execute('SET temp_directory='+q(out/'duck_tmp'))
    con.execute(f'CREATE VIEW rec AS SELECT rid,src,country,name,{core} core,addr,rawn FROM read_parquet({paths})')
    keys=d[['sid','tid']].assign(ix=np.arange(len(d),dtype=np.int32));con.register('keys',keys)
    con.execute('''CREATE TABLE pairtext AS SELECT k.ix,s.name sn,t.name tn,s.core sc,t.core tc,s.addr sa,t.addr ta,s.rawn sr,t.rawn tr,
      s.country scountry,t.country tcountry,t.src tsrc
      FROM keys k JOIN rec s ON s.rid=k.sid JOIN rec t ON t.rid=k.tid''')
    con.execute('CREATE TABLE wanted AS SELECT scountry country,sc core FROM pairtext UNION SELECT tcountry,tc FROM pairtext')
    con.execute('''CREATE TABLE freq AS SELECT r.country,r.core,count(*) FILTER(WHERE src=1) ns,count(*) FILTER(WHERE src>1) nt
      FROM rec r SEMI JOIN wanted w USING(country,core) GROUP BY r.country,r.core''')
    text=con.execute('''SELECT p.*,fs.ns sns,fs.nt snt,ft.ns tns,ft.nt tnt FROM pairtext p
      JOIN freq fs ON fs.country=p.scountry AND fs.core=p.sc
      JOIN freq ft ON ft.country=p.tcountry AND ft.core=p.tc ORDER BY p.ix''').df()
    con.close()
    if len(text)!=len(d) or not np.array_equal(text.ix,np.arange(len(d))):raise ValueError('Structure alignment')
    f=pd.DataFrame(index=np.arange(len(d)))
    for new,col in [('l24','p'),('l1','p_stage1'),('l20','p_old'),('l1_old','p1_old')]:
        f[new]=logit(np.clip(d[col].to_numpy(),1e-6,1-1e-6))
    f['xe4']=d.xe.to_numpy();f['xe5']=d.xe_old.to_numpy()
    f['xe5_present']=d.xe_old.notna().to_numpy().astype(float)
    f['old_candidate_present']=d.p_old.notna().to_numpy().astype(float)
    f['interaction']=f.l24*f.xe4;f['old_new_difference']=f.l24-f.l20
    for col in ('sns','snt','tns','tnt'):f['log_'+col]=np.log1p(text[col].to_numpy())
    f['source3']=text.tsrc.eq(3).to_numpy().astype(float)
    for a,b,prefix in [('sn','tn','name'),('sc','tc','core'),('sa','ta','addr'),('sr','tr','raw')]:
        aa=text[a].tolist();bb=text[b].tolist()
        for label,fun in [('ratio',fuzz.ratio),('tsort',fuzz.token_sort_ratio),('tset',fuzz.token_set_ratio),('partial',fuzz.partial_ratio),('jw',JaroWinkler.normalized_similarity)]:
            f[prefix+'_'+label]=cpdist(aa,bb,scorer=fun,workers=1,dtype=np.float32)
        f[prefix+'_exact']=(text[a].eq(text[b])&text[a].ne('')).to_numpy().astype(float)
        f[prefix+'_len_s']=text[a].str.len().to_numpy();f[prefix+'_len_t']=text[b].str.len().to_numpy()
    f['s_addr_empty']=text.sa.eq('').to_numpy().astype(float);f['t_addr_empty']=text.ta.eq('').to_numpy().astype(float)
    f['core_no_spaces']=text.sc.str.replace(' ','').eq(text.tc.str.replace(' ','')).to_numpy().astype(float)
    s_num=text.sa.str.findall(r'\d+');t_num=text.ta.str.findall(r'\d+')
    f['digit_conflict']=[float(bool(a and b and not set(a)&set(b))) for a,b in zip(s_num,t_num)]
    f['digit_overlap']=[len(set(a)&set(b)) for a,b in zip(s_num,t_num)]
    sw=text.sc.str.split();tw=text.tc.str.split()
    f['s_words']=sw.str.len().to_numpy();f['t_words']=tw.str.len().to_numpy()
    f['word_overlap']=[len(set(a)&set(b)) for a,b in zip(sw,tw)]
    for name in ('s_addr_empty','t_addr_empty','core_exact','addr_exact'):
        f[name+'_l24']=f[name]*f.l24;f[name+'_xe4']=f[name]*f.xe4
    if np.isinf(f.to_numpy()).any():raise ValueError('Infinite features')
    return f.astype(np.float32)

def winner_values(d,p,refs,cutoff):
    """Exact all-reference metric after one held-pool owner per target."""
    return metric.scores_for_threshold(d.sid.to_numpy(),d.tid.to_numpy(),p,d.y.to_numpy(),refs.sid.to_numpy(),refs.true_links.to_numpy(),cutoff)

def choose_cutoff(d,p,refs):
    """Tune only; tie-break toward the stricter cutoff."""
    order=np.lexsort((d.sid.to_numpy(),-p,d.tid.to_numpy()))
    w=order[np.r_[True,d.tid.to_numpy()[order][1:]!=d.tid.to_numpy()[order][:-1]]]
    grid=sorted(set(np.round(np.arange(.35,.991,.01),4))|{.75,.925})
    vectors={str(t):metric.evaluate_winners(d.sid.to_numpy()[w],p[w],d.y.to_numpy()[w],refs.sid.to_numpy(),refs.true_links.to_numpy(),t) for t in grid}
    curve={t:float(v.mean()) for t,v in vectors.items()}
    t=max(grid,key=lambda z:(curve[str(z)],z))
    return t,vectors[str(t)],curve

def train_one(kind,X,y,weight=None):
    """Fit a bounded fixed-complexity CPU model, without evaluation-label stopping."""
    if kind=='baseline':
        model=LogisticRegression(max_iter=2000,tol=1e-6)
    elif kind in ('small_tree','macro_small_tree'):
        model=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=12,max_depth=4,min_samples_leaf=120,learning_rate=.045,l2_regularization=15,early_stopping=False,random_state=26)
    elif kind=='lgb_small_tree':
        model=lgb.LGBMClassifier(n_estimators=320,num_leaves=15,max_depth=5,min_child_samples=100,learning_rate=.04,reg_lambda=15,colsample_bytree=.9,n_jobs=1,verbosity=-1,random_state=26,deterministic=True,force_col_wise=True)
    else:raise ValueError(kind)
    model.fit(X,y,sample_weight=weight)
    return model

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    proc=psutil.Process()
    if os.name=='nt':proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    if psutil.virtual_memory().available<2.5*2**30:raise MemoryError('Need 2.5 GiB available for isolated bounded pilot')
    start=time.monotonic()
    inputs=[MD/'hold_preds.parquet',MD/'hold_error_rows.parquet',MD/'split_ids.parquet',OLD/'hold_preds.parquet',OLD/'split_ids.parquet',XE,XE5]
    stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
    d,refs=load_pairs(args.out);d=old_scores(d)
    band=d.xe.notna().to_numpy();ix=np.flatnonzero(band)
    print(f'Pairs {len(d):,}; neural band {len(ix):,}; tune references {len(refs):,}',flush=True)
    f=structural_features(d.iloc[ix].reset_index(drop=True),args.out)
    f.to_parquet(args.out/'band_features.parquet',index=False)
    d.to_parquet(args.out/'pairs_labels_masked.parquet',index=False)
    refs.to_parquet(args.out/'tune_references.parquet',index=False)
    folds={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
    rf=d.sid.iloc[ix].map(folds).fillna(-1).to_numpy(dtype=np.int8)
    X=f.to_numpy();basecols=[f.columns.get_loc(c) for c in ('l24','xe4','interaction')]
    truth_counts=d.sid.iloc[ix].map(dict(zip(refs.sid,refs.true_links))).to_numpy()
    results={};saved=d[['sid','tid','p','y']].copy();base_v=None
    with threadpool_limits(limits=1):
        for kind in ('baseline','small_tree','macro_small_tree','lgb_small_tree'):
            features=X[:,basecols] if kind=='baseline' else X
            probability=d.p.to_numpy().copy();band_p=np.zeros(len(ix))
            for fold in range(3):
                fit=(rf>=0)&(rf!=fold);check=rf==fold;other=rf<0
                weight=1/np.maximum(truth_counts[fit],1) if kind=='macro_small_tree' else None
                model=train_one(kind,features[fit],d.y.to_numpy()[ix[fit]],weight)
                band_p[check]=model.predict_proba(features[check])[:,1]
                band_p[other]+=model.predict_proba(features[other])[:,1]/3
                joblib.dump(model,args.out/f'{kind}_fold{fold}.joblib')
            probability[ix]=band_p
            cutoff,v,curve=choose_cutoff(d,probability,refs)
            if base_v is None:base_v=v
            delta=v-base_v;se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
            results[kind]={'cutoff':cutoff,'tune_macro_f05':float(v.mean()),'delta_vs_oof_baseline':float(delta.mean()),'descriptive_paired_95_interval':[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],'better':int((delta>1e-12).sum()),'worse':int((delta < -1e-12).sum()),'by_country':{c:float(v[refs.country.eq(c)].mean()) for c in sorted(refs.country.unique())},'curve':curve}
            saved[kind]=probability
            print(kind,cutoff,float(v.mean()),'delta',float(delta.mean()),flush=True)
    chosen=max(results,key=lambda k:results[k]['tune_macro_f05'])
    if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=s for p,s in stamps.items()):raise ValueError('Input changed during experiment')
    saved.to_parquet(args.out/'oof_predictions.parquet',index=False)
    report={'protocol':'Exploratory 3-fold S1-grouped tuning OOF; final labels masked. All tune refs, exact macro F0.5, held-pool ownership. No final/test evaluation or production change.','selected_on_tune':chosen,'feature_names':list(f.columns),'results':results,'inputs':stamps,'seconds':time.monotonic()-start,'peak_working_set_MiB':(proc.memory_info().peak_wset if os.name=='nt' else proc.memory_info().rss)/2**20,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('SELECTED',chosen,'seconds',report['seconds'],'peak MiB',report['peak_working_set_MiB'],flush=True)

if __name__=='__main__':main()
