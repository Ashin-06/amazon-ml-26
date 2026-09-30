# Offline cross-encoder scoring on the local GPU (same procedure as the Kaggle inference kernel: fp16, max 112
# tokens, batches of 512). Scores each given *_band.parquet (sid, tid, a, b) into OUT_DIR/<name>_xe.parquet (sid, tid,
# xe) and writes xenc_inference_audit.json (model hash, parameter count, truncation share, seconds).
# Usage: python work/xenc_infer_local.py MODEL_DIR OUT_DIR BAND.parquet [BAND.parquet ...]
import hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
mdir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
# Tokenizer: XE_TOKENIZER (explicit local directory) if set; else the checkpoint's own; if that was written by a newer
# transformers/tokenizers than installed here, the base model's tokenizer saved next to it (fine-tuning keeps the vocab).
if os.environ.get("XE_TOKENIZER"):
    tok = AutoTokenizer.from_pretrained(os.environ["XE_TOKENIZER"])
else:
    try:
        tok = AutoTokenizer.from_pretrained(mdir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(mdir), "tokenizer_e5small"))
model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda().eval().half()
sha = hashlib.sha256(open(os.path.join(mdir, "model.safetensors"), "rb").read()).hexdigest()
tok_dir = getattr(tok, "name_or_path", "")
tok_sha = {n: hashlib.sha256(open(os.path.join(tok_dir, n), "rb").read()).hexdigest()
           for n in ("tokenizer.json", "sentencepiece.bpe.model") if tok_dir and os.path.exists(os.path.join(tok_dir, n))}
audit = {"model_dir": mdir, "model_sha256": sha, "parameters": int(sum(p.numel() for p in model.parameters())),
         "tokenizer_dir": tok_dir, "tokenizer_sha256": tok_sha, "max_length": 112, "device": torch.cuda.get_device_name(0), "files": {}}
import pyarrow.parquet as pq
for f in sys.argv[3:]:
    keys, res, lens, t = [], [], None, time.time()
    # streamed in 204,800-row batches (= 400 GPU batches of 512, so batch composition is unchanged); bounded memory
    for rb in pq.ParquetFile(f).iter_batches(batch_size=204_800, columns=["sid", "tid", "a", "b"]):
        d = rb.to_pandas()
        if lens is None:
            lens = [len(x) for x in tok(list(d.a.iloc[:20000]), list(d.b.iloc[:20000]), truncation=False)["input_ids"]]
        with torch.no_grad():
            for i in range(0, len(d), 512):
                b = d.iloc[i:i + 512]
                x = tok(list(b.a), list(b.b), truncation=True, max_length=112, padding=True, return_tensors="pt").to("cuda")
                res.append(model(**x).logits.squeeze(-1).float().cpu().numpy())
        keys.append(d[["sid", "tid"]])
    d = pd.concat(keys, ignore_index=True) if keys else pd.DataFrame({"sid": [], "tid": []})
    d["xe"] = np.concatenate(res) if res else np.empty(0, np.float32)
    if not np.isfinite(d.xe).all():
        raise SystemExit(f"non-finite cross-encoder scores in {f}")
    name = os.path.basename(f).replace(".parquet", "_xe.parquet")
    d[["sid", "tid", "xe"]].to_parquet(os.path.join(out, name), index=False)
    audit["files"][name] = {"pairs": len(d), "seconds": round(time.time() - t, 1),
                            "share_over_112_tokens": round(float(np.mean(np.array(lens) > 112)), 4) if lens else 0.0}
    print(name, audit["files"][name], flush=True)
json.dump(audit, open(os.path.join(out, "xenc_inference_audit.json"), "w"), indent=1)
