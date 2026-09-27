"""Fetch only the supplied official resources; retain public Q&A without names."""
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import urllib.request

out=Path(__file__).with_name('official_refresh_evening')
out.mkdir(exist_ok=False)
urls={
 'problem':'https://d8it4huxumps7.cloudfront.net/files/6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf',
 'guidelines':'https://d8it4huxumps7.cloudfront.net/files/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf',
 'clarifications':'https://docs.google.com/spreadsheets/d/1wMo3bmj5bAdkDI9FvqMnLRmANRGrXNuS6u69ce_VvR0/export?format=csv&gid=483754747'}
result={}
for name,url in urls.items():
    try:
        with urllib.request.urlopen(url,timeout=25) as r: data=r.read()
        item={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if name=='clarifications':
            rows=list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
            answers=[{'row':i+1,'question':r[4],'answer':r[6]} for i,r in enumerate(rows) if i and len(r)>6 and r[6].strip()]
            item.update(rows=len(rows),answered=len(answers))
            (out/'answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            from pypdf import PdfReader
            reader=PdfReader(io.BytesIO(data));item['pages']=len(reader.pages)
            (out/(name+'.txt')).write_text('\n\n'.join(p.extract_text() or '' for p in reader.pages),encoding='utf-8')
        result[name]=item
    except Exception as exc:
        result[name]={'url':url,'error':str(exc)}
(out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
