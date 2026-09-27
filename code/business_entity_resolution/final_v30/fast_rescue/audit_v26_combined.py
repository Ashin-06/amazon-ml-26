"""Audit the frozen v25 plus 1.88 rescue as one pipeline, without threshold search."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib,importlib.util,json,time
from pathlib import Path
import duckdb,joblib,numpy as np,pandas as pd,pyarrow as pa,pyarrow.compute as pc,pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
OUT=HERE/'v26_combined_audit_v1';OUT.mkdir(exist_ok=False)
OLD=ROOT/'artifacts/push995-20260927-0915';NORM=ROOT/'work/v21L/run/train'
sp=importlib.util.spec_from_file_location('stack',OLD/'stacker.py');st=importlib.util.module_from_spec(sp);sp.loader.exec_module(st)
q=st.q;start=time.monotonic()
d=pd.read_parquet(OLD/'stack_v4/pairs_labels_masked.parquet');f=pd.read_parquet(OLD/'stack_v4/band_features.parquet')
refs=pd.read_parquet(st.MD/'hold_error_rows.parquet').sort_values('sid').reset_index(drop=True)
ix=np.flatnonzero(d.xe.notna());mdl=joblib.load(ROOT/'work/comp/v25/v25_lgb600.joblib')
config=json.loads((ROOT/'work/comp/v25/v25_model.json').read_text())
if list(f.columns)!=config['features'] or config['cutoff']!=.74:raise ValueError('Model contract')
p=d.p.to_numpy().copy()
with threadpool_limits(limits=1):p[ix]=mdl.predict_proba(f.to_numpy())[:,1]
order=np.lexsort((d.sid.to_numpy(),-p,d.tid.to_numpy()));w=order[np.r_[True,d.tid.to_numpy()[order][1:]!=d.tid.to_numpy()[order][:-1]]]
w=w[p[w]>=.74];base=d.iloc[w][['sid','tid']].copy()
# The model and both thresholds are fixed before any final truth is opened.
manifest={'model':'work/comp/v25/v25_lgb600.joblib','cutoff':.74,'rescue_cutoff':1.88,
 'protocol':'Audit only, no selection. Both held halves compete together; complete reference denominators. Non-held base model ownership remains unmeasured.',
 'model_sha256':hashlib.sha256((ROOT/'work/comp/v25/v25_lgb600.joblib').read_bytes()).hexdigest()}
(OUT/'fixed_contract.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='700MB';SET max_temp_directory_size='5GB';SET preserve_insertion_order=false")
c.execute('SET temp_directory='+q(OUT/'sql_tmp'));c.register('refs',refs[['sid']])
paths='['+','.join(q(NORM/f'norm_s{i}.parquet') for i in (1,2,3))+']'
c.execute(f"CREATE TABLE r AS SELECT rid,src,country,name,addr,string_split(core,' ')[1] fw,regexp_extract(addr,'[0-9]+') num FROM read_parquet({paths})")
c.execute("CREATE TABLE s AS SELECT r.* FROM r JOIN refs ON refs.sid=r.rid WHERE r.src=1 AND num<>'' AND fw<>''")
count=c.execute('SELECT count(*) FROM s JOIN r t ON t.src>1 AND t.country=s.country AND t.fw=s.fw AND t.num=s.num').fetchone()[0]
print('Raw held route join',count,flush=True)
if count>12000000:raise RuntimeError('Route join exceeds bounded audit budget')
c.execute('CREATE VIEW oldcand AS SELECT sid,tid FROM read_parquet('+q(st.MD/'hold_preds.parquet')+')')
parts=[]
for chunk in range(16):
    x=c.execute(f'''SELECT s.rid sid,t.rid tid,s.name sn,t.name tn,s.addr sa,t.addr ta
      FROM s JOIN r t ON t.src>1 AND t.country=s.country AND t.fw=s.fw AND t.num=s.num
      WHERE hash(s.rid)%16={chunk} AND NOT EXISTS(SELECT 1 FROM oldcand o WHERE o.sid=s.rid AND o.tid=t.rid)''').df()
    if len(x):
        score=(cpdist(x.sn.tolist(),x.tn.tolist(),scorer=fuzz.ratio,workers=1,dtype=np.float32)+cpdist(x.sa.tolist(),x.ta.tolist(),scorer=fuzz.token_set_ratio,workers=1,dtype=np.float32))/100
        x=x[['sid','tid']].assign(score=score);parts.append(x[x.score>=1.5])
    print('route chunk',chunk,flush=True)
x=pd.concat(parts,ignore_index=True)
if x.duplicated(['sid','tid']).any():raise ValueError('Duplicate route')
x.to_parquet(OUT/'route_pairs.parquet',index=False)
sel=x[(x.score>=1.88)&~x.tid.isin(base.tid)].sort_values(['tid','score','sid'],ascending=[True,False,True]).drop_duplicates('tid')
# Check potential rescue competitors from every S1, without labels.
c.register('targets',sel[['tid']].drop_duplicates());c.execute('CREATE TABLE target AS SELECT r.* FROM r JOIN targets ON targets.tid=r.rid')
competition=c.execute('''SELECT s.rid sid,t.rid tid,s.name sn,t.name tn,s.addr sa,t.addr ta FROM r s JOIN target t ON s.src=1 AND s.country=t.country AND s.fw=t.fw AND s.num=t.num''').df()
if len(competition):
    competition['score']=(cpdist(competition.sn.tolist(),competition.tn.tolist(),scorer=fuzz.ratio,workers=1)+cpdist(competition.sa.tolist(),competition.ta.tolist(),scorer=fuzz.token_set_ratio,workers=1))/100
    wins=competition.sort_values(['tid','score','sid'],ascending=[True,False,True]).drop_duplicates('tid')
    changed=sel[['sid','tid']].merge(wins[['sid','tid']],on='tid',suffixes=('_held','_full')).query('sid_held!=sid_full')
else:changed=pd.DataFrame()
c.close()
# Use the supplied truth derived from provided train data; verify the schema.
truth=pd.read_parquet(ROOT/'work/comp/hold_truth.parquet',columns=['sid','tid'])
if truth.duplicated(['sid','tid']).any():raise ValueError('Duplicate truth')
if int(truth.groupby('sid').size().reindex(refs.sid,fill_value=0).sum())!=int(refs.true_links.sum()):raise ValueError('Truth count mismatch')
base=base.merge(truth.assign(y=1),on=['sid','tid'],how='left',validate='one_to_one').fillna({'y':0})
sel=sel.merge(truth.assign(y=1),on=['sid','tid'],how='left',validate='one_to_one').fillna({'y':0})
combined=pd.concat([base[['sid','tid','y']],sel[['sid','tid','y']]],ignore_index=True)
if combined.tid.duplicated().any():raise ValueError('Ownership conflict')
results={}
for half in ('tune','final'):
    rr=refs[refs.half.eq(half)].reset_index(drop=True)
    bv=st.metric.evaluate_winners(base.sid.to_numpy(),np.ones(len(base)),base.y.to_numpy(),rr.sid.to_numpy(),rr.true_links.to_numpy(),.5)
    cv=st.metric.evaluate_winners(combined.sid.to_numpy(),np.ones(len(combined)),combined.y.to_numpy(),rr.sid.to_numpy(),rr.true_links.to_numpy(),.5)
    delta=cv-bv;se=float(delta.std(ddof=1)/np.sqrt(len(rr)));added=sel[sel.sid.isin(rr.sid)]
    results[half]={'references':len(rr),'v25':float(bv.mean()),'v26_combined':float(cv.mean()),'delta':float(delta.mean()),
      'paired_95_interval':[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],'added_pairs':len(added),'added_true':int(added.y.sum()),
      'added_precision':float(added.y.mean()),'by_country':{co:float(cv[rr.country.eq(co)].mean()) for co in sorted(rr.country.unique())}}
sel.to_parquet(OUT/'selected_with_labels.parquet',index=False)
report={'contract':manifest,'results':results,'route_candidates':len(x),'raw_join_pairs':count,
 'selected_rescue_pairs_with_different_full_S1_route_winner':len(changed),'seconds':time.monotonic()-start,
 'limitations':['Final labels have been reused historically; this is not a fresh holdout','Base target ownership only uses held S1; full S1 route competitor statistic does not replace missing full-base predictions','French labels unavailable']}
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
