# Offline cross-encoder scoring (fine-tuned intfloat/multilingual-e5-small, MIT, from the attached training run).
# Scores every *_band.parquet pair file; writes <name>_xe.parquet (sid, tid, xe) and a truncation audit.
import glob, json, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
t0 = time.time()
mdir = os.path.dirname(glob.glob("/kaggle/input/**/xenc_model/config.json", recursive=True)[0])
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda().eval().half()
nparams = sum(p.numel() for p in model.parameters())
audit = {"model_dir": mdir, "parameters": int(nparams), "max_length": 112, "files": {}}
for f in sorted(glob.glob("/kaggle/input/**/*_band.parquet", recursive=True)):
    d = pd.read_parquet(f)
    lens = [len(x) for x in tok(list(d.a.iloc[:20000]), list(d.b.iloc[:20000]), truncation=False)["input_ids"]]
    out, t = [], time.time()
    with torch.no_grad():
        for i in range(0, len(d), 512):
            b = d.iloc[i:i + 512]
            x = tok(list(b.a), list(b.b), truncation=True, max_length=112, padding=True, return_tensors="pt").to("cuda")
            out.append(model(**x).logits.squeeze(-1).float().cpu().numpy())
    d["xe"] = np.concatenate(out)
    name = os.path.basename(f).replace(".parquet", "_xe.parquet")
    d[["sid", "tid", "xe"]].to_parquet("/kaggle/working/" + name, index=False)
    audit["files"][name] = {"pairs": len(d), "seconds": round(time.time() - t, 1),
                            "share_over_112_tokens": round(float(np.mean(np.array(lens) > 112)), 4)}
    print(name, audit["files"][name], flush=True)
json.dump(audit, open("/kaggle/working/xenc_inference_audit.json", "w"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
