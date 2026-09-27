"""Sealed xenc7 tuning comparison; no final labels until a frozen selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/push995-20260927-0915'
PILOT=OLD/'stack_v4'
ARMS=('neural_lr','neural_tree')
LR_COLUMNS=('l24','xe4','interaction','xe7','xe7_l24')
SOURCE_SHA='0d4b05e52858e2f8da31c93767e217b60af155c6de1a0d2e7e5128c571a748cd'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def base_module():
    source=OLD/'stacker.py'
    if sha(source)!=SOURCE_SHA:raise ValueError('Saved stacker source changed')
    sp=importlib.util.spec_from_file_location('frozen_stack',source)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    return m


def exact_logits(keys,xe):
    """Reject missing/extra/duplicate/nonfinite scores; align by pair keys."""
    if list(xe.columns)!=['sid','tid','xe']:raise ValueError('Neural schema')
    if keys.duplicated(['sid','tid']).any() or xe.duplicated(['sid','tid']).any():raise ValueError('Duplicate pair keys')
    if keys.isna().any().any() or xe.isna().any().any():raise ValueError('Null keys or logits')
    if not np.isfinite(xe.xe).all():raise ValueError('Nonfinite logits')
    coverage=keys.merge(xe[['sid','tid']],on=['sid','tid'],how='outer',indicator=True,validate='one_to_one')
    if not coverage._merge.eq('both').all():raise ValueError('Neural band coverage')
    return keys.merge(xe,on=['sid','tid'],how='left',sort=False,validate='one_to_one').xe.to_numpy()


def new_features(features,xe):
    """Append evidence; neither identities nor labels enter model features."""
    if len(features)!=len(xe):raise ValueError('Feature alignment')
    f=features.reset_index(drop=True).copy()
    f['xe7']=xe
    f['xe7_l24']=xe*f.l24
    f['xe7_xe4_difference']=xe-f.xe4
    for c in ('t_addr_empty','s_addr_empty','core_exact','addr_exact'):
        f['xe7_'+c]=xe*f[c]
    if np.isinf(f.to_numpy()).any():raise ValueError('Infinite feature')
    return f.astype(np.float32)


def fold_assignments(d,refs):
    """Check labels are visible only for tuning identities before fitting."""
    if refs.sid.duplicated().any():raise ValueError('Duplicate tune references')
    tune=d.sid.isin(refs.sid).to_numpy()
    if not np.array_equal(d.y.ge(0).to_numpy(),tune):raise ValueError('Final labels unmasked or tuning labels missing')
    if not d.loc[tune,'y'].isin([0,1]).all() or not d.loc[~tune,'y'].eq(-1).all():raise ValueError('Invalid labels')
    mapping={int(s):int.from_bytes(hashlib.blake2b((str(s)+':tune-blend-audit').encode(),digest_size=8).digest(),'little')%3 for s in refs.sid}
    return d.sid.map(mapping).fillna(-1).to_numpy(dtype=np.int8)


def comparison(v,base,refs):
    delta=v-base;mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    return {'macro_f05':float(v.mean()),'delta':mean,'paired_95_interval':[mean-1.96*se,mean+1.96*se],
      'better':int((delta>1e-12).sum()),'worse':int((delta < -1e-12).sum()),
      'by_country':{str(c):{'macro_f05':float(v[refs.country.eq(c)].mean()),'delta':float(delta[refs.country.eq(c)].mean())} for c in sorted(refs.country.unique())}}


def gate(result):
    return (result['delta']>=.0001 and result['paired_95_interval'][0]>0 and
            all(r['delta']>=-.00005 for r in result['by_country'].values()))


def verified_cloud(folder):
    """Validate the whole completed run before consuming its held output."""
    audit=json.loads((folder/'xenc7_audit.json').read_text(encoding='utf-8'))
    if audit.get('complete') is not True or audit.get('held_train_overlap')!=0:raise ValueError('Cloud incomplete or held overlap')
    local_script=Path(__file__).parent/'kaggle_xenc7/script.py'
    allowed={sha(local_script),hashlib.sha256(local_script.read_bytes().replace(b'\r\n',b'\n')).hexdigest()}
    if audit.get('script_sha256') not in allowed:raise ValueError('Cloud source mismatch; inspect executed source before proceeding')
    for filename,rows in [('v24_hold_band_xe7.parquet',172740),('v24_test_band_xe7.parquet',3698863)]:
        p=folder/filename;record=audit['files'][filename]
        if record['rows']!=rows or pq.ParquetFile(p).metadata.num_rows!=rows or sha(p)!=record['sha256']:raise ValueError('Output seal/coverage: '+filename)
    source_paths={'hold_preds.parquet':ROOT/'work/v24/kaggle/work/model/hold_preds.parquet',
      'scores.parquet':ROOT/'work/v24/kaggle/out/scores.parquet',
      'train_source1.parquet':ROOT/'work/v14/pq/train_source1.parquet',
      'hard_pairs.parquet':ROOT/'work/kaggle_xenc_hard/hard_pairs.parquet'}
    for name,local in source_paths.items():
        found=[v for k,v in audit['input_sha256'].items() if Path(k).name==name]
        if len(found)!=1 or sha(local)!=found[0]:raise ValueError('Cloud input mismatch: '+name)
    return audit


def load_tune(cloud):
    audit=verified_cloud(cloud)
    d=pd.read_parquet(PILOT/'pairs_labels_masked.parquet')
    f=pd.read_parquet(PILOT/'band_features.parquet')
    refs=pd.read_parquet(PILOT/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
    saved=json.loads((PILOT/'audit.json').read_text())
    if list(f.columns)!=saved['feature_names']:raise ValueError('Saved feature schema')
    rowfold=fold_assignments(d,refs)
    band=d.xe.notna().to_numpy();ix=np.flatnonzero(band)
    if not np.array_equal(band,(d.p>.002)&(d.p<.9995)):raise ValueError('Band changed')
    xe=exact_logits(d.loc[band,['sid','tid']],pd.read_parquet(cloud/'v24_hold_band_xe7.parquet'))
    f=new_features(f,xe)
    old=pd.read_parquet(PILOT/'oof_predictions.parquet')
    if not d[['sid','tid']].equals(old[['sid','tid']]):raise ValueError('OOF alignment')
    return d,f,refs,ix,rowfold[ix],old,audit


def features_for(f,arm):
    return f[list(LR_COLUMNS)].to_numpy() if arm=='neural_lr' else f.to_numpy()


def train(m,arm,X,y):
    return m.train_one('baseline' if arm=='neural_lr' else 'lgb_small_tree',X,y)


def pilot(cloud,out):
    m=base_module();d,f,refs,ix,folds,old,cloud_audit=load_tune(cloud)
    baseline=m.winner_values(d,old.lgb_small_tree.to_numpy(),refs,.77)
    if abs(float(baseline.mean())-.9891265623499508)>1e-12:raise ValueError('Baseline score drift')
    # The user now reports v26 as best. Reproduce the current v25 correction
    # as an additional frozen control, with all held-pool competitors present.
    import lightgbm as lgb
    current_config=json.loads((ROOT/'work/comp/v25/v25_model.json').read_text())
    control=d.p.to_numpy().copy();control_band=np.zeros(len(ix));X0=f.iloc[:,:63].to_numpy()
    with threadpool_limits(limits=1):
        for fold in range(3):
            fit=(folds>=0)&(folds!=fold);check=folds==fold;other=folds<0
            cm=lgb.LGBMClassifier(**current_config['config'],n_jobs=1,verbosity=-1,random_state=26,deterministic=True,force_col_wise=True)
            cm.fit(X0[fit],d.y.to_numpy()[ix[fit]])
            control_band[check]=cm.predict_proba(X0[check])[:,1]
            control_band[other]+=cm.predict_proba(X0[other])[:,1]/3
    control[ix]=control_band
    control_v=m.winner_values(d,control,refs,.74)
    results={};pred=d[['sid','tid']].copy();pred['v25_control']=control
    with threadpool_limits(limits=1):
        for arm in ARMS:
            X=features_for(f,arm);p=d.p.to_numpy().copy();bp=np.zeros(len(ix))
            for fold in range(3):
                fit=(folds>=0)&(folds!=fold);check=folds==fold;other=folds<0
                model=train(m,arm,X[fit],d.y.to_numpy()[ix[fit]])
                bp[check]=model.predict_proba(X[check])[:,1]
                bp[other]+=model.predict_proba(X[other])[:,1]/3
                joblib.dump(model,out/f'{arm}_fold{fold}.joblib')
            p[ix]=bp
            if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid prediction')
            cutoff,v,curve=m.choose_cutoff(d,p,refs)
            results[arm]={**comparison(v,baseline,refs),'cutoff':cutoff,'curve':curve,'versus_v25':comparison(v,control_v,refs)}
            pred[arm]=p
            print(arm,cutoff,results[arm]['macro_f05'],results[arm]['delta'],flush=True)
    eligible=[a for a in ARMS if gate(results[a]) and results[a]['versus_v25']['delta']>0 and results[a]['versus_v25']['paired_95_interval'][0]>0]
    selected=max(eligible,key=lambda a:results[a]['macro_f05']) if eligible else None
    pred.to_parquet(out/'oof_predictions.parquet',index=False)
    f.to_parquet(out/'band_features.parquet',index=False)
    inputs=[PILOT/n for n in ['pairs_labels_masked.parquet','band_features.parquet','tune_references.parquet','oof_predictions.parquet']]
    inputs+=[cloud/'xenc7_audit.json',cloud/'v24_hold_band_xe7.parquet',ROOT/'work/comp/v25/v25_model.json',ROOT/'work/comp/v25/v25_lgb600.joblib']
    report={'protocol':'Predeclared two arms, S1-grouped three-fold tune OOF, final labels masked; full tune denominator; held-reference ownership only; no French labels.',
      'selected':selected,'eligible_for_final':selected is not None,'results':results,'baseline_macro_f05':float(baseline.mean()),
      'feature_names':list(f.columns),'input_sha256':{str(p):sha(p) for p in inputs},'source_sha256':sha(__file__),
      'cloud_source_sha256':cloud_audit['script_sha256'],'v25_control_macro_f05':float(control_v.mean())}
    write_json(out/'audit.json',report)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('Eligible:',selected,flush=True)


def final(pilot_dir,out):
    m=base_module();report=json.loads((pilot_dir/'audit.json').read_text())
    if not report['eligible_for_final'] or not report['selected']:raise ValueError('Tune gate failed')
    if sha(__file__)!=report['source_sha256']:raise ValueError('Pilot evaluation source changed')
    for p,value in report['input_sha256'].items():
        if sha(p)!=value:raise ValueError('Pilot input changed: '+p)
    d=pd.read_parquet(PILOT/'pairs_labels_masked.parquet')
    refs_tune=pd.read_parquet(PILOT/'tune_references.parquet')
    fold_assignments(d,refs_tune)
    f=pd.read_parquet(pilot_dir/'band_features.parquet')
    if list(f.columns)!=report['feature_names']:raise ValueError('Pilot feature schema')
    ix=np.flatnonzero(d.xe.notna());fit=d.y.to_numpy()[ix]>=0
    arm=report['selected'];cutoff=report['results'][arm]['cutoff'];X=features_for(f,arm)
    with threadpool_limits(limits=1):
        model=train(m,arm,X[fit],d.y.to_numpy()[ix[fit]])
        p=d.p.to_numpy().copy();p[ix]=model.predict_proba(X)[:,1]
    joblib.dump(model,out/'selected_model.joblib')
    pred=d[['sid','tid']].assign(selected=p)
    pred.to_parquet(out/'locked_predictions.parquet',index=False)
    manifest={'selected':arm,'cutoff':cutoff,'feature_names':list(f.columns),
      'model_sha256':sha(out/'selected_model.joblib'),'prediction_sha256':sha(out/'locked_predictions.parquet'),
      'input_sha256':report['input_sha256'],'pilot_sha256':sha(pilot_dir/'audit.json'),'source_sha256':sha(__file__),
      'protocol':'Frozen before final labels. Historically reused final split, held-pool ownership; France unlabelled.'}
    write_json(out/'locked_manifest.json',manifest)
    # This is the first point where final labels are read. No tuning follows.
    refs=pd.read_parquet(m.MD/'hold_error_rows.parquet')
    refs=refs[refs.half.eq('final')].sort_values('sid').reset_index(drop=True)
    if set(refs.sid)&set(refs_tune.sid):raise ValueError('Tune/final overlap')
    parts=[]
    for b in pq.ParquetFile(m.MD/'hold_preds.parquet').iter_batches(batch_size=65536,columns=['sid','tid','p','y'],use_threads=False):
        parts.append(b.filter(pc.greater(b.column('p'),.002)).select(['sid','tid','y']).to_pandas())
    labels=pd.concat(parts,ignore_index=True)
    pred=pred.merge(labels,on=['sid','tid'],how='left',validate='one_to_one')
    if pred.y.isna().any():raise ValueError('Missing final labels')
    old=pd.read_parquet(OLD/'final_v1/locked_predictions.parquet')
    pred=pred.merge(old[['sid','tid','lgb_small_tree']],on=['sid','tid'],how='left',validate='one_to_one')
    if pred.lgb_small_tree.isna().any():raise ValueError('Missing baseline score')
    base=m.winner_values(pred,pred.lgb_small_tree.to_numpy(),refs,.77)
    v=m.winner_values(pred,pred.selected.to_numpy(),refs,cutoff)
    current_model=joblib.load(ROOT/'work/comp/v25/v25_lgb600.joblib')
    current=d.p.to_numpy().copy()
    with threadpool_limits(limits=1):current[ix]=current_model.predict_proba(f.iloc[:,:63].to_numpy())[:,1]
    current_v=m.winner_values(pred,current,refs,.74)
    result=comparison(v,base,refs)
    result['versus_v25']=comparison(v,current_v,refs)
    result['eligible_for_test_export']=result['delta']>0 and result['paired_95_interval'][0]>0 and result['versus_v25']['delta']>0 and result['versus_v25']['paired_95_interval'][0]>0
    result.update({'baseline_macro_f05':float(base.mean()),'manifest':manifest})
    write_json(out/'audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['pilot','final']);ap.add_argument('--cloud',type=Path)
    ap.add_argument('--pilot',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if psutil.virtual_memory().available<2*2**30:raise MemoryError('Need 2 GiB available')
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    pa.set_cpu_count(1);pa.set_io_thread_count(1);start=time.monotonic()
    if a.mode=='pilot':pilot(a.cloud,a.out)
    else:final(a.pilot,a.out)
    print('Seconds',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
