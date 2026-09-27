"""Exact tuning-only oracle decomposition for the frozen v28 pipeline."""
from pathlib import Path
import importlib.util,json
import numpy as np,pandas as pd,duckdb
H=Path(__file__).parent;ROOT=H.parent.parent;OLD=ROOT/'artifacts/push995-20260927-0915/stack_v4'
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=module('rescue',H/'fast_rescue.py');r.st=module('stack',ROOT/'artifacts/push995-20260927-0915/stacker.py')
refs=pd.read_parquet(OLD/'tune_references.parquet').sort_values('sid').reset_index(drop=True)
d=pd.read_parquet(OLD/'pairs_labels_masked.parquet');bp=pd.read_parquet(H/'xenc7_pilot_v1/oof_predictions.parquet');assert d[['sid','tid']].equals(bp[['sid','tid']])
p=bp.v25_control.to_numpy();o=np.lexsort((d.sid.to_numpy(),-p,d.tid.to_numpy()));w=o[np.r_[True,d.tid.to_numpy()[o][1:]!=d.tid.to_numpy()[o][:-1]]];base=d.iloc[w[p[w]>=.74]][['sid','tid','y']]
x=pd.read_parquet(H/'fast_rescue_v1/pairs_masked.parquet');op=pd.read_parquet(H/'fast_rescue_v1/oof_predictions.parquet');assert x[['sid','tid']].equals(op[['sid','tid']])
add=r.choose(x,op.linear.to_numpy(),set(base.tid),.6900000000000002)
pred=pd.concat([base,add[['sid','tid','y']]],ignore_index=True);pred=pred[pred.sid.isin(refs.sid)];assert pred.y.ge(0).all()
g=pred.groupby('sid').y.agg(['sum','count']).reindex(refs.sid,fill_value=0)
tp=g['sum'].to_numpy();pc=g['count'].to_numpy();true=refs.true_links.to_numpy()
def metric(t,p):
 den=true+4*p;return np.divide(5*t,den,out=np.ones(len(t)),where=den>0)
actual=metric(tp,pc);no_fp=metric(tp,tp)
c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='500MB'");c.register('refs',refs[['sid']]);c.register('route',x.loc[x.y.eq(1),['sid','tid']])
retr=c.execute(f'''SELECT sid,count(*) n FROM (SELECT h.sid,h.tid FROM read_parquet({r.st.q(r.st.MD/'hold_preds.parquet')}) h JOIN refs r ON h.sid=r.sid WHERE h.y UNION SELECT sid,tid FROM route) GROUP BY sid''').df().set_index('sid').n.reindex(refs.sid,fill_value=0).to_numpy();c.close()
assert np.all(retr>=tp) and np.all(retr<=true)
oracle=metric(retr,retr)
report={'scope':'Tuning only, actual v28 OOF; removal of false positives then recovery of retrieved true links. Oracle is not reachable model score or test guarantee.','references':len(refs),'score':float(actual.mean()),'candidate_oracle':float(oracle.mean()),'false_positive_loss':float((no_fp-actual).mean()),'rejected_retrieved_true_loss':float((oracle-no_fp).mean()),'unretrieved_true_loss':float((1-oracle).mean()),'true_links':int(true.sum()),'false_positives':int((pc-tp).sum()),'false_negatives':int((true-tp).sum()),'missed_retrieval_links':int((true-retr).sum())}
assert abs(sum(report[k] for k in ['false_positive_loss','rejected_retrieved_true_loss','unretrieved_true_loss'])-(1-report['score']))<1e-12
(H/'v28_tuning_error_budget.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
