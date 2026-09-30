"""Check saved model/source contracts before a costly reproduction run."""
import argparse,hashlib,importlib.metadata,json,platform
from pathlib import Path
C=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def check(environment=False):
 errors=[];manifest=json.loads((C/'models/MODELS_MANIFEST.json').read_text(encoding='utf-8'))
 compact_path=C/'models/COMPACT_MANIFEST.json'
 compact=json.loads(compact_path.read_text(encoding='utf-8')) if compact_path.is_file() else None
 omitted=set(compact['omitted_neural_weights']) if compact else set()
 if compact and omitted!={'crossencoder/xenc4_e5small/model.safetensors','crossencoder/xenc5_e5base/model.safetensors'}:
  errors.append('Unexpected compact model omission list')
 for n,v in manifest.items():
  p=C/'models'/n
  if n in omitted and not p.is_file():continue
  if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:errors.append('Model artifact mismatch: '+n)
 if compact:
  for n,v in compact['saved_scores'].items():
   p=C/'models/saved_scores'/n
   if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:
    errors.append('Compact saved-score mismatch: '+n)
  if not omitted.issubset(set(manifest)) or len(compact['saved_scores'])!=4:
   errors.append('Incomplete compact score/model manifest')
 for v,folder in [('v24','src/fast_er'),('v20','src/pipeline_v20/fast_er')]:
  m=json.loads((C/f'models/trees_{v}/model_manifest.json').read_text(encoding='utf-8'))
  for n,x in m['inputs']['code'].items():
   if sha(C/folder/n)!=x['sha256']:errors.append('Saved-source mismatch: '+v+'/'+n)
  if environment:
   want=m['inputs']['environment']
   if platform.python_version()!=want['python']:errors.append('Python must be '+want['python'])
   for n,ver in want['packages'].items():
    try:got=importlib.metadata.version(n)
    except importlib.metadata.PackageNotFoundError:got='missing'
    if got!=ver:errors.append(f'{n}: need {ver}, found {got}')
 m=json.loads((C/'models/rescue_learned/locked_manifest.json').read_text())
 if sha(C/'src/rescue/learned/frozen_fast_rescue.py')!=m['feature_source_sha256']:errors.append('Rescue feature source mismatch')
 return errors
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--environment',action='store_true');a=p.parse_args();errors=check(a.environment)
 if errors:raise SystemExit('\n'.join(errors))
 print('Saved models/scores, tree source and rescue feature source: PASS'+('; environment: PASS' if a.environment else ''))
