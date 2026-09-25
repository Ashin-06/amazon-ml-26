"""Stage 1+2 driver: normalize a split, apply the learned lexicon, block it, and (if labels
given) report blocking recall.

Usage: python run_stage1.py SPLIT PQ_DIR WORK [GT_PARQUET] [--mem 20GB] [--threads N] [--workers N] ...

Every run records the code hashes and settings that produced its outputs in
WORK/SPLIT/stage1_meta.json and refuses to reuse outputs made by different code/settings
(use a fresh WORK directory instead of silently mixing versions).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.block import block, blocking_recall  # noqa: E402
from fast_er.lexicon import apply_lexicon, build_lexicon  # noqa: E402
from fast_er.prep import prepare  # noqa: E402

CODE = ("prep.py", "lexicon.py", "block.py")


def provenance(a):
    here = Path(__file__).resolve().parent
    return {"code_sha256": {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in CODE},
            "settings": {k: getattr(a, k) for k in ("cap", "cap_whole", "k", "k_field", "hold_mod")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("pq_dir"); ap.add_argument("work"); ap.add_argument("gt", nargs="?")
    ap.add_argument("--mem", default="20GB"); ap.add_argument("--threads", type=int)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--cap", type=int, default=300); ap.add_argument("--hold-mod", type=int, default=22)
    ap.add_argument("--k", type=int, default=40); ap.add_argument("--k-field", type=int, default=20)
    ap.add_argument("--cap-whole", type=int, default=1000)
    a = ap.parse_args()
    for name in ("workers", "chunks", "cap", "k", "k_field", "cap_whole"):
        if getattr(a, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1")
    if a.hold_mod < 2:
        ap.error("--hold-mod must be >= 2")
    split_dir = Path(a.work) / a.split
    meta_path = split_dir / "stage1_meta.json"
    meta = provenance(a)
    if split_dir.exists() and any(split_dir.iterdir()):
        if not meta_path.exists() or json.loads(meta_path.read_text()) != meta:
            sys.exit(f"{split_dir} holds outputs from different code/settings (or has no stage1_meta.json); "
                     "use a new WORK directory")
    split_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    t = time.time()
    prepare(a.split, pq_dir=a.pq_dir, out_dir=a.work, workers=a.workers)
    print(f"[prep {a.split}] {time.time() - t:.0f}s", flush=True)
    if a.gt and not (Path(a.work) / "lexicon.json").exists():
        build_lexicon(a.work, a.gt, a.hold_mod, split=a.split)
    if (Path(a.work) / "lexicon.json").exists():
        apply_lexicon(a.work, a.split)
    block(a.split, work=a.work, cap=a.cap, cap_whole=a.cap_whole, k=a.k, k_field=a.k_field, chunks=a.chunks,
          threads=a.threads, mem=a.mem)
    if a.gt:
        r = blocking_recall(a.split, work=a.work, gt_path=a.gt)
        print(json.dumps(r, indent=1))
        (split_dir / "blocking_recall.json").write_text(json.dumps(r, indent=1))
