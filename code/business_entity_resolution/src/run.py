"""Offline baseline: index, train, predict, score, validate, package and quality checks."""

import argparse
import csv
import json
import sqlite3
import sys

from business_er.validation import score_files, validate_files


def _sources(parser):
    """Add mandatory explicit source paths; never discover arbitrary workspace files."""
    for source in (1, 2, 3):
        parser.add_argument(f"--source{source}", required=True)


def main(argv=None):
    """Dispatch implemented commands; report input failures with nonzero exit codes."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="Disk-backed exact scoring, independent of row order")
    score.add_argument("--truth", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--work-dir")
    validate = sub.add_parser("validate", help="Strict disk-backed checks including target IDs")
    _sources(validate)
    validate.add_argument("--matching", required=True)
    validate.add_argument("--candidate", required=True)
    validate.add_argument("--work-dir")
    index = sub.add_parser("index", help="Build a new immutable local FTS5 index")
    _sources(index)
    index.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Train using disjoint reference groups and natural candidates")
    train.add_argument("--index", required=True)
    train.add_argument("--truth", required=True)
    train.add_argument("--model-dir", required=True)
    train.add_argument("--config")
    predict = sub.add_parser("predict", help="Predict every indexed S1 and export exact candidates")
    predict.add_argument("--index", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--output-dir", required=True)
    package = sub.add_parser("package", help="Create a checked allowlisted final ZIP")
    package.add_argument("--output-dir", required=True)
    package.add_argument("--model-dir", required=True)
    package.add_argument("--documentation", required=True)
    package.add_argument("--input-manifest", required=True)
    package.add_argument("--destination", required=True)
    quality = sub.add_parser("quality", help="Check workspace docs, sources and input integrity")
    quality.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            result = score_files(args.truth, args.predictions, args.work_dir)
        elif args.command == "validate":
            result = validate_files(args.source1, args.source2, args.source3, args.matching, args.candidate, args.work_dir)
        elif args.command == "index":
            from business_er.blocking import build_index
            result = build_index(args.source1, args.source2, args.source3, args.output)
        elif args.command == "train":
            from business_er.config import load_config
            from business_er.train import train_baseline
            result = train_baseline(args.index, args.truth, args.model_dir, load_config(args.config))
            result["metrics"] = {key: value for key, value in result["metrics"].items() if key != "selected_reference_ids"}
        elif args.command == "predict":
            from business_er.predict import predict_baseline
            result = predict_baseline(args.index, args.model_dir, args.output_dir)
        elif args.command == "package":
            from business_er.packaging import build_package
            result = build_package(args.output_dir, args.model_dir, args.documentation, args.input_manifest, args.destination)
        else:
            from business_er.quality import check_workspace
            result = check_workspace(args.workspace)
            if not result["passed"]:
                print(json.dumps(result, indent=2, allow_nan=False))
                return 1
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, UnicodeError, csv.Error, sqlite3.Error, KeyError, TypeError, ImportError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
