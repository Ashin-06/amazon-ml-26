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
    ap.add_argument("--n-train", type=int, default=10_000); ap.add_argument("--hold-mod", type=int, default=22)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--mem", default="4GB"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--leaves", type=int, default=127); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight", choices=("none","ref","sqrt"), default="none")
    ap.add_argument("--backend", choices=("lgb","xgb","auto"), default="auto",
                    help="auto = XGBoost on GPU when one is usable (CPU retry on GPU OOM), else LightGBM on CPU")
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--model-dir", default="model")
    a = ap.parse_args()
    if not a.skip_train:
        train(a.work, a.gt, a.n_train, a.hold_mod, a.mem, a.threads, rounds=a.rounds, leaves=a.leaves, lr=a.lr,
              chunks=a.chunks, weight=a.weight, backend=a.backend, seed=a.seed, model_dir=a.model_dir)
    if not a.skip_predict:
        predict(a.work, a.pred_split, a.out, a.mem, a.threads, chunks=a.chunks, model_dir=a.model_dir)
