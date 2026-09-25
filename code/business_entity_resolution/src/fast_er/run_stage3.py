"""Stage 3 driver: train the matcher on train candidates, then predict a split.

Usage: python run_stage3.py WORK GT_PARQUET PRED_SPLIT OUT_DIR [--n-train N] [--hold-mod M] [--mem 20GB] [--threads N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.model import predict, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("gt"); ap.add_argument("pred_split"); ap.add_argument("out")
    ap.add_argument("--n-train", type=int, default=300_000); ap.add_argument("--hold-mod", type=int, default=22)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--mem", default="20GB"); ap.add_argument("--threads", type=int)
    ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--weight", choices=("none", "ref", "sqrt"), default="none")
    ap.add_argument("--leaves", type=int, default=127); ap.add_argument("--lr", type=float, default=0.05)
    a = ap.parse_args()
    if not a.skip_train:
        train(a.work, a.gt, a.n_train, a.hold_mod, a.mem, a.threads, rounds=a.rounds, leaves=a.leaves, lr=a.lr, weight=a.weight,
              chunks=a.chunks)
    if not a.skip_predict:
        predict(a.work, a.pred_split, a.out, a.mem, a.threads, chunks=a.chunks)
