"""Audit rare-core rescue on the rejected/unowned cohort, not all easy positives."""
import importlib.util,json,os
from pathlib import Path
import duckdb,numpy as np,pandas as pd,psutil

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;PREV=ROOT/'artifacts/push995-20260927-0915'
spec=importlib.util.spec_from_file_location('base',PREV/'stacker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    out=HERE/'f4_rejected_v1';out.mkdir(exist_ok=False)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    d=pd.read_parquet(PREV/'stack_v4/oof_predictions.parquet');refs=pd.read_parquet(PREV/'stack_v4/tune_references.parquet')
    c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='400MB';SET max_temp_directory_size='2GB';SET preserve_insertion_order=false")
    c.execute('SET temp_directory='+m.q(out/'tmp'))
    c.execute(f'CREATE VIEW s AS SELECT rid,country,core FROM read_parquet({m.q(m.NORM/"norm_s1.parquet")})')
    c.execute('CREATE TABLE freq AS SELECT country,core,count(*) n FROM s GROUP BY country,core')
    c.register('pairs',d[['sid','tid']].assign(ix=np.arange(len(d))))
    paths='['+','.join(m.q(m.NORM/f'norm_s{i}.parquet') for i in (2,3))+']'
    st=c.execute(f"SELECT p.ix,f.n, s.core=t.core same_core,t.addr='' no_address FROM pairs p JOIN s ON s.rid=p.sid JOIN read_parquet({paths}) t ON t.rid=p.tid JOIN freq f ON f.country=s.country AND f.core=s.core ORDER BY p.ix").df();c.close()
    if not np.array_equal(st.ix,np.arange(len(d))):raise ValueError('Alignment')
    p=d.lgb_small_tree.to_numpy();tune=d.y.ge(0).to_numpy()
    eligible=(st.n.eq(1)&st.same_core&st.no_address).to_numpy()
    owned=set(d.loc[p>=.77,'tid']);unowned=~d.tid.isin(owned).to_numpy()
    selected=eligible&unowned&(d.baseline.to_numpy()>=.3)
    # Source ownership is still bounded to held references: explicit diagnostic.
    pp=p.copy();pp[selected]=.999+d.baseline.to_numpy()[selected]*.0005
    before=m.winner_values(d,p,refs,.77);after=m.winner_values(d,pp,refs,.77)
    def cohort(mask):
        v=mask&tune;n=int(v.sum());pos=int(d.y.to_numpy()[v].sum())
        return {'pairs':n,'true_links':pos,'precision':pos/n if n else None}
    result={'protocol':'Tune-only grouped OOF scores; raw normalized core as in French F4; S1 uniqueness from full provided training pool. Only unowned rejected targets and baseline probability >=.3 are added. Held-pool ownership limitation. No French labels or claims of French accuracy.',
      'all_unique_core_blank_address_pairs':cohort(eligible),
      'rejected_unique_core_blank_address_pairs':cohort(eligible&(p<.77)),
      'selected_F4_proxy_pairs':cohort(selected),
      'before_macro_f05':float(before.mean()),'after_macro_f05':float(after.mean()),'delta':float((after-before).mean()),
      'by_country':{co:{'before':float(before[refs.country.eq(co)].mean()),'after':float(after[refs.country.eq(co)].mean())} for co in sorted(refs.country.unique())}}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
