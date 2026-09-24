"""Stage 1+2 driver: normalize a split, block it, and (if labels given) report blocking recall.

Usage: python run_stage1.py SPLIT PQ_DIR WORK [GT_PARQUET] [--mem 20GB] [--threads N] [--workers N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.block import block, blocking_recall  # noqa: E402
from fast_er.lexicon import apply_lexicon, build_lexicon  # noqa: E402
from fast_er.prep import prepare  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("pq_dir"); ap.add_argument("work"); ap.add_argument("gt", nargs="?")
    ap.add_argument("--mem", default="20GB"); ap.add_argument("--threads", type=int)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--cap", type=int, default=300); ap.add_argument("--hold-mod", type=int, default=22); ap.add_argument("--k", type=int, default=25); ap.add_argument("--k-field", type=int, default=15)
    a = ap.parse_args()
    t = time.time()
    prepare(a.split, pq_dir=a.pq_dir, out_dir=a.work, workers=a.workers)
    print(f"[prep {a.split}] {time.time() - t:.0f}s", flush=True)
    if a.split == "train" and a.gt and not (Path(a.work) / "lexicon.json").exists():
        build_lexicon(a.work, a.gt, a.hold_mod)
    if (Path(a.work) / "lexicon.json").exists():
        apply_lexicon(a.work, a.split)
    block(a.split, work=a.work, cap=a.cap, k=a.k, k_field=a.k_field, chunks=a.chunks, threads=a.threads, mem=a.mem)
    if a.gt:
        r = blocking_recall(a.split, work=a.work, gt_path=a.gt)
        print(json.dumps(r, indent=1))
        (Path(a.work) / a.split / "blocking_recall.json").write_text(json.dumps(r, indent=1))
