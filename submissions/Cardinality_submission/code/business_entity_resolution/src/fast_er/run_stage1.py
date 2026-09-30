"""Normalize, learn/apply the train lexicon, block, then publish a completion manifest."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_er.block import block, blocking_recall
from fast_er.integrity import seal, stage_inputs, verify, verify_stage
from fast_er.lexicon import apply_lexicon, build_lexicon
from fast_er.prep import prepare

SETTINGS = ("cap", "cap_whole", "k", "k_field", "hold_mod", "chunks", "workers", "threads", "mem")


def provenance(a):
    """Fingerprint inputs, settings, the complete source and selected lexicon."""
    lex = Path(a.work) / "lexicon.json"
    return stage_inputs(a.pq_dir, a.split, a.gt, {k: getattr(a, k) for k in SETTINGS},
                        lex if lex.exists() else None)


def run(a):
    """Run a fresh stage or verify a completed one; incomplete stages cannot resume."""
    for name in ("workers", "chunks", "cap", "k", "k_field", "cap_whole"):
        if type(getattr(a, name)) is not int or getattr(a, name) < 1:
            raise ValueError(name + " must be a positive integer")
    if a.split not in ("train", "test") or (a.split == "train") != bool(a.gt):
        raise ValueError("Train requires labels; test must never receive labels")
    if type(a.hold_mod) is not int or a.hold_mod < 2:
        raise ValueError("hold_mod must be >= 2")
    if a.threads is not None and (type(a.threads) is not int or a.threads < 1):
        raise ValueError("threads must be positive")
    split_dir = Path(a.work) / a.split
    lex = Path(a.work) / "lexicon.json"
    if split_dir.exists():
        return verify(split_dir, "stage1_meta.json", provenance(a))
    if a.split == "train" and lex.exists():
        raise ValueError("Existing lexicon without a completed train stage; use fresh WORK")
    if a.split == "test":
        train = verify_stage(a.work, "train")
        if train["inputs"]["settings"]["hold_mod"] != a.hold_mod:
            raise ValueError("Test/train hold rule differs")
    prepare(a.split, pq_dir=a.pq_dir, out_dir=a.work, workers=a.workers)
    if a.split == "train":
        build_lexicon(a.work, a.gt, a.hold_mod, split=a.split)
    apply_lexicon(a.work, a.split)
    block(a.split, work=a.work, cap=a.cap, cap_whole=a.cap_whole, k=a.k, k_field=a.k_field,
          chunks=a.chunks, threads=a.threads, mem=a.mem)
    outputs = [f"norm_s{i}.parquet" for i in (1, 2, 3)] + ["cands.parquet", "lexicon_applied"]
    if a.gt:
        result = blocking_recall(a.split, work=a.work, gt_path=a.gt)
        (split_dir / "blocking_recall.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        outputs.append("blocking_recall.json")
        print(json.dumps(result, indent=2), flush=True)
    return seal(split_dir, "stage1_meta.json", provenance(a), outputs)


def main():
    """Parse the supported stage command and execute it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "test")); ap.add_argument("pq_dir")
    ap.add_argument("work"); ap.add_argument("gt", nargs="?")
    ap.add_argument("--mem", default="4GB"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--cap", type=int, default=300); ap.add_argument("--hold-mod", type=int, default=22)
    ap.add_argument("--k", type=int, default=40); ap.add_argument("--k-field", type=int, default=20)
    ap.add_argument("--cap-whole", type=int, default=1000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
