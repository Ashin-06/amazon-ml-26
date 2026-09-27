import csv,itertools,json
from pathlib import Path
import pyarrow.parquet as pq
root=Path('D:/Amazon ML 26');here=Path(__file__).parent
fr=set()
for b in pq.ParquetFile(root/'work/v21L/run/test/norm_s1.parquet').iter_batches(columns=['entity_id','country']):
 d=b.to_pandas();fr.update(d.loc[d.country.eq('france'),'entity_id'])
a={'rows':0,'changed_references':0,'added_links':0,'removed_links':0,'france_changed':0}
with (root/'submissions/v27_matching_results.tsv').open(encoding='utf-8',newline='') as f,(here/'fast_rescue_test_v1/matching_results.tsv').open(encoding='utf-8',newline='') as g:
 r=csv.reader(f,delimiter='\t');s=csv.reader(g,delimiter='\t');assert next(r)==next(s)
 for x,y in itertools.zip_longest(r,s):
  assert x is not None and y is not None and len(x)==len(y)==2 and x[0]==y[0]
  a['rows']+=1
  if x[0] in fr:assert x==y
  old=set(filter(None,x[1].split(',')));new=set(filter(None,y[1].split(',')))
  if old!=new:
   a['changed_references']+=1;a['added_links']+=len(new-old);a['removed_links']+=len(old-new)
assert a['rows']==1732544
a['france_rows_preserved']=len(fr)
(here/'fast_rescue_test_v1/delta_audit.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
print(json.dumps(a,indent=2))
