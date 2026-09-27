from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path('D:/Amazon ML 26');out=root/'artifacts/after-0982910-20260927/fast_rescue_validation_v1'
matching=root/'artifacts/after-0982910-20260927/fast_rescue_test_v1/matching_results.tsv';candidate=root/'work/comp/v25/candidate_pairs_v27.tsv'
checker=root/'code/business_entity_resolution/kaggle/check_submission.py'
paths=[matching,candidate,checker];stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
start=time.monotonic()
with (out/'independent_full.log').open('wb') as log:
 rc=subprocess.run([sys.executable,'-u','-B','-X','utf8',str(checker),str(matching),str(root/'6ab10eb3b23ba_student_resource/student_resource/dataset/test'),'--candidate',str(candidate),'--mem','750MB'],cwd=out,stdout=log,stderr=subprocess.STDOUT).returncode
if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=stamp for p,stamp in stamps.items()):raise ValueError('File changed during validation')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
a={'independent_validator_exit':rc,'seconds':time.monotonic()-start,'files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths},'scope':'Full supplied original IDs, coverage, format, unique target ownership and all candidate IDs/containment. Unmodified independent checker, bounded DuckDB memory. This is not an official full-candidate validator success.'}
(out/'audit.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
print((out/'independent_full.log').read_text(encoding='utf-8')[-6000:]);print(json.dumps(a,indent=2));sys.exit(rc)
