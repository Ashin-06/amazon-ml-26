"""Bounded read-only snapshot of a running Kaggle log; no job mutations."""
import json,queue,threading,time
from kaggle.api.kaggle_api_extended import KaggleApi
q=queue.Queue();start=time.monotonic()
def read():
    try:
        api=KaggleApi();api.authenticate()
        for event in api.kernels_logs_stream('ashinmurugan06/amlc26-xenc7-robust'):
            q.put(event.get('data',''))
    except Exception as e:q.put('Log reader error: '+type(e).__name__)
threading.Thread(target=read,daemon=True).start()
lines=[];last=None
while time.monotonic()-start<45:
    try:
        data=q.get(timeout=1);last=time.monotonic()
        if any(w in data for w in ["'step':","'scored':",'Training complete','COMPLETE','Traceback','Error:']):lines.append(data)
    except queue.Empty:
        if last and time.monotonic()-last>4:break
print(json.dumps({'seconds':round(time.monotonic()-start,2),'recent_progress':lines[-12:]},ensure_ascii=True))
