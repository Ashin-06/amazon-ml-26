"""Require byte-exact reproduction without modifying the submitted artifacts."""
import hashlib,json,sys
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
if __name__=='__main__':
 a,b=map(Path,sys.argv[1:3]);results={}
 for n in ['matching_results.tsv','candidate_pairs.tsv']:
  expected=digest(a/n);actual=digest(b/n);results[n]={'expected_sha256':expected,'actual_sha256':actual,'identical':expected==actual}
 (b/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 print(json.dumps(results,indent=2))
 if not all(v['identical'] for v in results.values()):raise SystemExit('Reproduction differs; inspect comparison.json. Submitted files were preserved.')
