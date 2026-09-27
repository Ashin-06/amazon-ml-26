"""Read-only French overlay delta, ownership, and candidate-containment audit."""
import csv,hashlib,json,itertools,time
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent/'v27_delta_audit_v1'
OUT.mkdir(exist_ok=False)
base=ROOT/'submissions/v26_matching_results.tsv'
variants={'v27':ROOT/'submissions/v27_matching_results.tsv'}
candidate=ROOT/'work/comp/v25/candidate_pairs_v27.tsv'
fr=set()
for b in pq.ParquetFile(ROOT/'work/v21L/run/test/norm_s1.parquet').iter_batches(columns=['entity_id','country']):
    d=b.to_pandas();fr.update(d.loc[d.country.eq('france'),'entity_id'])
added={v:{} for v in variants};removed={v:{} for v in variants}
stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in [base,candidate,*variants.values()]}
stats={v:{'changed_references':0,'added_links':0,'removed_links':0,'nonfrance_changed':0} for v in variants}
files=[p.open('r',encoding='utf-8',newline='') for p in [base,*variants.values()]]
try:
    readers=[csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE) for f in files]
    for r in readers:
        if next(r)!=['source1_entity_id','matched_entity_ids']:raise ValueError('Header')
    n=0
    for rows in itertools.zip_longest(*readers):
        if any(r is None or len(r)!=2 for r in rows):raise ValueError('Row count/shape')
        orig=rows[0];n+=1
        for v,row in zip(variants,rows[1:]):
            if row[0]!=orig[0]:raise ValueError('Reference order/coverage')
            if row[1]==orig[1]:continue
            old=set(filter(None,orig[1].split(',')));new_list=list(filter(None,row[1].split(',')));new=set(new_list)
            if len(new)!=len(new_list):raise ValueError('Duplicate within list')
            plus=new-old;minus=old-new
            if not plus and not minus:continue
            stats[v]['changed_references']+=1;stats[v]['nonfrance_changed']+=row[0] not in fr
            stats[v]['added_links']+=len(plus);stats[v]['removed_links']+=len(minus)
            if plus:added[v][row[0]]=plus
            if minus:removed[v][row[0]]=minus
finally:
    for f in files:f.close()
if n!=1732544:raise ValueError('Reference count')
# A changed row contributes between -1 and +1. This is a full-test-population
# bound; the unpublished public subset can have a different changed fraction.
for v in variants:stats[v]['max_absolute_full_population_macro_change']=stats[v]['changed_references']/n
# Check newly added targets against every original owner and each other.
new_owners={v:{} for v in variants}
for v,byref in added.items():
    for s,ts in byref.items():
        for t in ts:
            if t in new_owners[v]:raise ValueError('Multiple new owners')
            new_owners[v][t]=s
with base.open(encoding='utf-8',newline='') as f:
    r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE);next(r)
    for s,targets in r:
        for t in filter(None,targets.split(',')):
            for v,owners in new_owners.items():
                if t in owners and t not in removed[v].get(s,set()):raise ValueError('Target already owned')
pending={v:{s:set(ts) for s,ts in byref.items()} for v,byref in added.items()}
with candidate.open(encoding='utf-8',newline='') as f:
    r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
    if next(r)!=['source1_entity_id','candidate_entity_ids']:raise ValueError('Candidate header')
    for s,ts in r:
        needed=[v for v in variants if s in pending[v]]
        if needed:
            candidates=set(filter(None,ts.split(',')))
            for v in needed:
                if not pending[v][s]<=candidates:raise ValueError('Added pair outside candidates')
                del pending[v][s]
if any(pending.values()) or any(x['nonfrance_changed'] for x in stats.values()):raise ValueError('Coverage/country')
if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=stamp for p,stamp in stamps.items()):raise ValueError('File changed during check')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
report={'reference_rows':n,'france_rows':len(fr),'variants':stats,'new_target_ownership':'PASS',
 'new_pair_candidate_containment':'PASS','scope':'Base previously passed original-ID/full-containment checks; this read-only audit proves the overlays preserve base reference coverage and non-French rows, have no new ownership conflict, and add only supplied candidates. It does not establish French accuracy.',
 'public_subset_caveat':'Changed-reference fractions below are for all 1,732,544 test rows; they are not upper bounds on an unknown public subset.',
 'sha256':{str(p):sha(p) for p in [base,*variants.values()]}}
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
