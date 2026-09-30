# Kaggle GPU training (v3): fine-tuned cross-encoder intfloat/multilingual-e5-base (MIT, 278M params) on the same 2.6M
# pairs as v2 (e5-small), then scores every attached *_band.parquet (v15k holdout/test bands) for a go/no-go vs v2.
import glob, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer
t0 = time.time()
lg = lambda s: print(f"[xenc {time.time() - t0:.0f}s] {s}", flush=True)
torch.manual_seed(26)
D = os.path.dirname(glob.glob("/kaggle/input/**/train_pairs.parquet", recursive=True)[0])
tr, ev = pd.read_parquet(f"{D}/train_pairs.parquet"), pd.read_parquet(f"{D}/eval_pairs.parquet")
lg(torch.cuda.get_device_name(0)); lg(f"train pairs {len(tr):,} (pos {tr.y.mean():.3f}) | eval pairs {len(ev):,}")
name = "intfloat/multilingual-e5-base"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()


def enc(a, b):
    return tok(list(a), list(b), truncation=True, max_length=112, padding=True, return_tensors="pt").to("cuda")


B = 128
order = np.random.default_rng(26).permutation(len(tr))
steps = len(order) // B
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.06)
model.train()
for i in range(steps):
    b = tr.iloc[order[i * B:(i + 1) * B]]
    x = enc(b.a.values, b.b.values)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = lossf(model(**x).logits.squeeze(-1), torch.tensor(b.y.values, dtype=torch.float32, device="cuda"))
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if i % 1000 == 0:
        lg(f"step {i}/{steps} loss {loss.item():.4f}")
model.eval()
model.save_pretrained('/kaggle/working/xenc_model'); tok.save_pretrained('/kaggle/working/xenc_model')
lg('model saved')
scores = []
with torch.no_grad():
    for i in range(0, len(ev), 256):
        b = ev.iloc[i:i + 256]
        with torch.autocast("cuda", dtype=torch.float16):
            scores.append(model(**enc(b.a.values, b.b.values)).logits.squeeze(-1).float().cpu().numpy())
ev["xe"] = np.concatenate(scores)
lg("scored eval pairs")
h = ev.p.between(0.02, 0.98).values
lp = np.log(np.clip(ev.p, 1e-6, 1 - 1e-6) / (1 - np.clip(ev.p, 1e-6, 1 - 1e-6))).values
print(f"hard pairs: AUC v15k p {roc_auc_score(ev.y[h], ev.p[h]):.4f} | AUC cross-encoder alone {roc_auc_score(ev.y[h], ev.xe[h]):.4f}")
for label, X in (("v15k logit only", np.c_[lp]), ("v15k logit + cross-encoder", np.c_[lp, ev.xe, lp * ev.xe])):
    oof = np.zeros(len(ev))
    for a, b in GroupKFold(5).split(X, ev.y, ev.sid):
        oof[b] = LogisticRegression(max_iter=2000).fit(X[a], ev.y.values[a]).predict_proba(X[b])[:, 1]
    print(f"{label:28s}: hard AUC {roc_auc_score(ev.y[h], oof[h]):.4f} | hard logloss {log_loss(ev.y[h], oof[h]):.4f} | "
          f"errors@0.5 on hard {int(((oof[h] >= 0.5) != ev.y.values[h]).sum())} of {int(h.sum())}")
# Band scoring with the trained model (same as the offline inference kernel; fp16, max 112 tokens).
import json
audit = {"model": name, "parameters": int(sum(p.numel() for p in model.parameters())), "files": {}}
model.half()
for f in sorted(glob.glob("/kaggle/input/**/*_band.parquet", recursive=True)):
    d = pd.read_parquet(f)
    out, t = [], time.time()
    with torch.no_grad():
        for i in range(0, len(d), 512):
            b = d.iloc[i:i + 512]
            out.append(model(**enc(b.a.values, b.b.values)).logits.squeeze(-1).float().cpu().numpy())
    d["xe"] = np.concatenate(out)
    nm = os.path.basename(f).replace(".parquet", "_xe.parquet")
    d[["sid", "tid", "xe"]].to_parquet("/kaggle/working/" + nm, index=False)
    audit["files"][nm] = {"pairs": len(d), "seconds": round(time.time() - t, 1)}
    lg(f"{nm} {audit['files'][nm]}")
json.dump(audit, open("/kaggle/working/xenc_inference_audit.json", "w"), indent=1)
lg("done")
