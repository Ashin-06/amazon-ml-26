# Business entity resolution: runnable CPU baseline

This package implements local indexing, supervised training, inference, exact scoring, strict validation and final archive creation. The detailed design preceded implementation. The workflow is integration-tested, including byte-for-byte inference reproduction from an extracted archive. Full challenge training/inference and leaderboard performance have not yet been established.

## Environment

Use Python 3.12, the tested runtime. Runtime packages are pinned in `requirements.txt`; test dependencies are in `requirements-dev.txt`. SQLite must provide FTS5 (available in the tested Python distribution). GPU/CUDA and cloud credentials are unnecessary for this baseline.

From this directory, preferably in a dedicated virtual environment:

```text
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python src/run.py --help
```

Training artifacts record package versions. Inference refuses different versions or changed source code so stale models cannot silently use a different feature implementation.

## Algorithm and bounds

The index stores original records and a local FTS5 name/address token index. Each reference searches name and address independently within each compatible country/source, allowing blank-country fallbacks. Targets are deduplicated before computing 29 numeric pair features. LightGBM learns from naturally retrieved positives and negatives.

Training uses a seeded S1 group split and bounded query samples; all target records remain available as distractors. Thresholds optimize tune-set entity-macro F0.5. Holdout labels are used only for reporting. There is no forced match, no injected evaluation positive and no one-match-per-source restriction.

This first retriever is word-based. Character n-gram retrieval, embeddings, alias learning and graph refinements are future experiments. Missing all shared tokens can cause missed candidates. Benchmark candidate recall and throughput before a full production run. Current index/inference stages restart after interruption; they do not yet resume checkpoints. Failed unpublished staging directories are retained for diagnosis.

## Reproduce from provided data

The following commands assume the supplied dataset is at `dataset/` relative to this directory. Pass the real paths when stored elsewhere. Output/index/model destinations must be new paths; existing runs are never overwritten.

```text
python src/run.py index --source1 dataset/train/train_source1.tsv --source2 dataset/train/train_source2.tsv --source3 dataset/train/train_source3.tsv --output artifacts/train.sqlite
python src/run.py train --index artifacts/train.sqlite --truth dataset/train/train_ground_truth.tsv --config configs/baseline.json --model-dir artifacts/model
python src/run.py index --source1 dataset/test/test_source1.tsv --source2 dataset/test/test_source2.tsv --source3 dataset/test/test_source3.tsv --output artifacts/test.sqlite
python src/run.py predict --index artifacts/test.sqlite --model-dir artifacts/model --output-dir output
```

The resulting `output/` contains both required TSVs and a completion manifest. Prediction processes every S1 record and performs strict disk-backed validation before publishing that directory. It verifies indexed source hashes before and after inference. An interrupted staging directory is not a completed output.

If using a final archive with a saved model already under `artifacts/model/`, skip training: build the test index and run `predict`. To retrain that exact saved configuration, pass `--config artifacts/model/config.json` and a **different** `--model-dir`. Do not overwrite the bundled model.

`metrics.json` records selected group IDs, country diagnostics, natural blocking recall, oracle score and tuning/holdout results. `manifest.json` records inputs, code hashes, environment, feature order, model hash and threshold. A threshold just above 1 represents a tune-selected all-empty policy; inspect the diagnostics instead of assuming that a trained model is useful.

## Exact scoring and strict validation

```text
python src/run.py score --truth validation_truth.tsv --predictions validation_predictions.tsv --work-dir artifacts/checks
python src/run.py validate --source1 dataset/test/test_source1.tsv --source2 dataset/test/test_source2.tsv --source3 dataset/test/test_source3.tsv --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --work-dir artifacts/checks
```

The CLI accepts arbitrary row order and rejects duplicate, missing and extra reference IDs. Validation always requires both target source files and checks target existence and match containment. Temporary SQLite files replace large Python ID sets; disk space and runtime still grow with input size.

The older library helpers `score_aligned_rows` and `validate_aligned_outputs` remain available for controlled in-memory/streaming callers. Their documented preconditions and memory limits still apply; the CLI uses the stronger disk-backed functions.

## Package the selected result

Complete team details and actual experiment results in the supplied methodology document. Remove its explicit draft marker only after finalizing it. Then:

```text
python src/run.py package --output-dir output --model-dir artifacts/model --documentation Documentation_template.md --input-manifest input_manifest.json --destination team_submission.zip
```

`input_manifest.json` must be the original audited file manifest, containing the expected SHA-256 hashes for `test_source1.tsv`, `test_source2.tsv` and `test_source3.tsv`. In the workspace this is `docs/research/input_manifest.json`. The ZIP includes both outputs, manifests, source, tests, configs, requirements, license/notices, saved model and methodology. It excludes the original datasets, tokens and unrelated workspace files. The self-contained tests work in the package; comparison with the organizer's separate validator additionally runs when that original resource bundle is present.

Package creation requires matching test-input fingerprints, a completed strict validation report, matching source/model/output hashes, and completed team fields. It cannot determine whether prose is scientifically accurate; the methodology must describe the selected run honestly.

## Workspace-only quality check

```text
python src/run.py quality --workspace PATH_TO_ORIGINAL_WORKSPACE
```

This checks original input hashes, source syntax/docstrings, required documents and local links. A final ZIP intentionally excludes the original research PDFs/data, so this workspace check is not a dependency of inference from the ZIP.
