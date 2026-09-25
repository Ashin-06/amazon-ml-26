# Runbook: verified fast pipeline

This checkout is separate from active runs in D:/Amazon ML 26. Use fresh artifact paths. Old unmanifested outputs cannot be reused. A small training-reference count does not bound full-target retrieval or global scoring cost.

## Verify this checkout

    python -m pytest code/business_entity_resolution/tests -q
    python code/business_entity_resolution/src/run.py quality --workspace . --input-root "D:\Amazon ML 26"

Install requirements-dev.txt in a dedicated virtual environment; avoid global application upgrades.

## Fresh run

The commands below use the full official dataset when DATA points to it. They were not executed by this review. Measure a bounded pilot first. Record when a pilot uses a reduced target pool, since that changes the difficulty.

    python code/business_entity_resolution/src/fast_er/to_parquet.py DATA artifacts/pq-audit-v1
    python code/business_entity_resolution/src/fast_er/run_stage1.py train artifacts/pq-audit-v1 artifacts/run-audit-v1 artifacts/pq-audit-v1/train_ground_truth.parquet --mem 4GB --threads 1 --workers 1 --chunks 24
    python code/business_entity_resolution/src/fast_er/run_stage1.py test artifacts/pq-audit-v1 artifacts/run-audit-v1 --mem 4GB --threads 1 --workers 1 --chunks 24
    python code/business_entity_resolution/src/fast_er/run_stage3.py artifacts/run-audit-v1 artifacts/pq-audit-v1/train_ground_truth.parquet test artifacts/predictions-audit-v1 --n-train 10000 --rounds 100 --mem 4GB --threads 1 --chunks 24 --weight ref --backend lgb

DATA contains train/ and test/. Default official location: D:/Amazon ML 26/6ab10eb3b23ba_student_resource/student_resource/dataset.

The memory limit applies to DuckDB, not all Python/booster allocations. Corrected evaluation includes an additional full scoring pass. Source/feature/split changes require fresh derivation and retraining. The old DuckDB-hash split is incompatible with the new stable split.

Use --skip-predict for training/evaluation only or --skip-train for matching saved models. Each experiment needs a new work/model destination. Completed blocking may be reused only after exact manifest verification; partial work does not resume. Manual-threshold rewrites of legacy scores are rejected.

The default backend is CPU LightGBM. Both fitting stages use the --n-train sample. Use --stage2-all-references only after measuring resources and deliberately choosing to fit stage 2 on all non-hold references. This does not change the full-graph context/evaluation cost. XGBoost QuantileDMatrix concatenates quantized batches in memory; it is not an external-memory guarantee. GPU allocation retries cover matrix creation/training and bounded inference transfers, and fitted devices are recorded per model.

## Candidate efficiency under the clarified rule

candidate_pairs.tsv contains every input to the FIRST scoring model. Stage-1 rejection saves stage-2 compute but never reduces exported candidates. The output score table records stage2_scored; a zero cascade score is used for an early-rejected pair. Candidate means include references with zero candidates. Routing ranks may break ties deterministically, but model rank features give tied evidence the same value.

The read-only diagnostic below fingerprints the historical completed stage and evaluates only its fixed tuning references. It does not let changed source reuse historical inputs for training and does not read final-holdout labels. PowerShell, from this checkout:

    $env:PYTHONPATH = (Resolve-Path 'code/business_entity_resolution/src').Path
    python -m fast_er.efficiency 'D:/Amazon ML 26/work/v14/run' 'D:/Amazon ML 26/work/v14/pq/train_ground_truth.parquet' artifacts/frontier-new.json --sample 50141 --mem 1GB --threads 1

Use a new report filename. Compare mean/p95/p99/max candidates, reduction ratio, candidate recall and oracle F0.5 together. This is a retrieval ceiling, not a model score. The completed [review](RULES_AND_MAXIMUM_SCORE_REVIEW_2026-09-25.md) explains why none of the smaller rank policies measured here is ready for a 0.99 target.

## Optional verified seed ensemble

Keep the same blocking run. Train a second model with --seed 27 --model-dir model_s27 and a distinct prediction destination. The saved tune/final membership is fixed across seeds. CPU pilots remain the default.

    python code/business_entity_resolution/src/fast_er/ensemble.py eval artifacts/run-audit-v1 artifacts/pq-audit-v1/train_ground_truth.parquet model model_s27
    python code/business_entity_resolution/src/fast_er/ensemble.py apply artifacts/run-audit-v1 test artifacts/predictions-audit-v1 artifacts/predictions-s27 --out artifacts/ensemble-predictions-v1

Evaluation recomputes the complete candidate graph for each model, so measure its cost first. Application verifies every member and uses the threshold saved by evaluation; arbitrary manual thresholds are not accepted. The same package command supports ensemble prediction directories and includes all selected member models. Use --model-dir for a single nondefault model when packaging.

## Validate and package

Use the existing strict validator with all three original test TSVs and both output TSVs. Also run the unchanged official validator with --check-ids when RAM is available; it can compete with active training.

    python code/business_entity_resolution/src/run.py validate --source1 DATA/test/test_source1.tsv --source2 DATA/test/test_source2.tsv --source3 DATA/test/test_source3.tsv --matching artifacts/predictions-audit-v1/matching_results.tsv --candidate artifacts/predictions-audit-v1/candidate_pairs.tsv --work-dir artifacts/checks-audit-v1
    python code/business_entity_resolution/kaggle/package_submission.py TEAM artifacts/predictions-audit-v1 --run artifacts/run-audit-v1 --doc Documentation_template.md --input-manifest docs/research/input_manifest.json --destination artifacts/TEAM_submission_audit_v1.zip

Replace DATA/TEAM with actual values. Complete real team fields and selected methodology first. Model/source/output fingerprints and original official test hashes must agree. Package creation does not upload.

## Kaggle

The builder accepts OUT OWNER SLUG STAGES, plus --s1/--s3 for measured settings. Select block_train,block_test,model or model_eval. --gpu requests hardware; choose --backend xgb/auto explicitly in stage-3 arguments to select it.

Attach only the seven converted parquet files and conversion_manifest.json. Offline execution requires compatible wheels for missing exact pins; --no-internet uses no package index. Completed prior stages need --reuse plus both blocking stages for manifest verification. Earlier unmanifested kernels are incompatible.

No local test guarantees quota, full-data disk usage or GPU capacity. Log actual public results separately. Maximum five leaderboard uploads per day; deadline 27 September 2026, 23:59 IST.

---

# Legacy SQLite baseline commands

These commands match the implemented interface. They are instructions, not a record that full-data runs have already finished. The source/Markdown checks and synthetic integration test are described in [QUALITY_REPORT.md](QUALITY_REPORT.md).

## 1. Open the correct project and verify

PowerShell, starting from the workspace:

```powershell
Set-Location -LiteralPath 'D:\Amazon ML 26'
$mlRunner = 'code\business_entity_resolution\src\run.py'
$mlDatasetRoot = '6ab10eb3b23ba_student_resource\student_resource\dataset'
python --version
python -m pytest code/business_entity_resolution/tests -q
python $mlRunner quality --workspace 'D:\Amazon ML 26'
```

The tested runtime is Python 3.12.3. Required installed versions are pinned in the package's requirements files. If setting up a new environment, create a project-local virtual environment and install the pinned requirements there; do not upgrade unrelated global packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements-dev.txt
```

Use that virtual environment's Python for **all** subsequent commands if you create it. Successful tests do not reserve cloud resources or establish model accuracy.

## 2. Choose run settings

`code/business_entity_resolution/configs/baseline.json` controls the seed, per-route/source top-K, query-token cap, query sample sizes, total pair cap, CPU threads and LightGBM settings. The default selects up to 10,000 training, 1,500 tuning and 1,500 holdout references and allows at most 750,000 pairs.

The target index always includes the complete target files passed to it. Do not build a validation pool from only known positives. If experimenting with reduced files to diagnose code, mark all scores as a reduced-pool pilot; they are not comparable with full-pool validation.

Copy configurations to versioned experiment paths before changing them. Unknown/duplicate keys, invalid types and out-of-range values fail. Use a new model directory for each experiment.

## 3. Build the training index

```powershell
python $mlRunner index --source1 "$mlDatasetRoot\train\train_source1.tsv" --source2 "$mlDatasetRoot\train\train_source2.tsv" --source3 "$mlDatasetRoot\train\train_source3.tsv" --output artifacts/train-v1.sqlite
```

This is a substantial job: training contains 12.53 million source rows. It hashes inputs, checks schemas and unique IDs, preserves raw text and builds the name/address search index. It publishes the destination only after completion. Keep adequate free disk space; the SQLite database can exceed the original TSV size. Index creation does not use labels.

The full-index elapsed time and size need measurement. The baseline does not promise a specific completion time. An index cannot be reused after changes to its ingestion/normalization/indexing code; rebuild it when those fingerprints differ.

## 4. Train and inspect the report

```powershell
python $mlRunner train --index artifacts/train-v1.sqlite --truth "$mlDatasetRoot\train\train_ground_truth.tsv" --config code/business_entity_resolution/configs/baseline.json --model-dir artifacts/model-v1
```

The training command first audits label coverage, target existence, unique ownership and country compatibility using disk-backed tables. It creates disjoint seeded S1 groups and retrieves natural candidates for the bounded sample. The tuning split chooses the threshold. Holdout is reported separately.

Inspect `artifacts/model-v1/metrics.json` and `manifest.json`. Check blocking recall, oracle macro F0.5, holdout macro F0.5, singleton accuracy, per-country results, sampled query counts and complete target-input counts. Training cannot score France because its labels are unavailable.

The baseline uses fixed boosting rounds. It does not implement automated hyperparameter search, probability calibration, checkpoint resume or neural models. These are optional later improvements. A run with insufficient retrieved positives/negatives fails rather than inventing supervision.

## 5. Build the test index and produce outputs

```powershell
python $mlRunner index --source1 "$mlDatasetRoot\test\test_source1.tsv" --source2 "$mlDatasetRoot\test\test_source2.tsv" --source3 "$mlDatasetRoot\test\test_source3.tsv" --output artifacts/test-v1.sqlite
python $mlRunner predict --index artifacts/test-v1.sqlite --model-dir artifacts/model-v1 --output-dir artifacts/predictions-v1
```

Inference scores exactly the IDs written to `candidate_pairs.tsv`, then writes thresholded matches, including empty lists. All 1,732,544 reference rows must be present, including all 259,452 French references, when these full official files are used. Neither a query sample nor a truncated source file is a valid full submission.

The command validates IDs, coverage, duplicates and candidate containment before publishing the output directory. It checks that the indexed source files have not changed. It retains failed hidden staging directories for diagnosis; do not submit their contents. Resuming a partially completed index/inference job is not implemented yet.

## 6. Validate and archive

```powershell
python $mlRunner validate --source1 "$mlDatasetRoot\test\test_source1.tsv" --source2 "$mlDatasetRoot\test\test_source2.tsv" --source3 "$mlDatasetRoot\test\test_source3.tsv" --matching artifacts/predictions-v1/matching_results.tsv --candidate artifacts/predictions-v1/candidate_pairs.tsv --work-dir artifacts/checks
python -X utf8 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py --matching artifacts/predictions-v1/matching_results.tsv --candidate artifacts/predictions-v1/candidate_pairs.tsv --test-dir "$mlDatasetRoot\test" --check-ids
```

The provided validator can need substantial RAM. Our strict validator uses disk-backed indexes. The supplied validator's warnings about missing candidates/subset mismatches are not acceptable for the final package even if it prints PASS.

Finalize the root `Documentation_template.md`: enter the actual team, members, date, selected model/threshold/results and limitations. Remove its `DRAFT: NOT FOR SUBMISSION` marker only then. The task-specific template has no stated page limit, while the general guidelines request 1–2 pages; also prepare a concise summary of the selected approach.

```powershell
python $mlRunner package --output-dir artifacts/predictions-v1 --model-dir artifacts/model-v1 --documentation Documentation_template.md --input-manifest docs/research/input_manifest.json --destination artifacts/team_submission_v1.zip
```

Package creation checks fingerprints against the original test files, refuses incomplete team details/draft documents, and includes only allowlisted files. It does not upload anything. Upload only the validated `matching_results.tsv` to the leaderboard; the final ZIP is a separate required deliverable. Record every upload in `submissions/log.csv`, including date/time, version, model, local score and returned leaderboard score. Maximum five uploads per day; deadline 27 September 2026, 23:59 IST.

## 7. Kaggle preparation, when needed

This CPU baseline needs no account token to run locally. Current Kaggle authentication, quota, memory, storage and private-data access are not verified by the local tests.

For a remote run, prepare an explicit private data/code upload containing only the required original TSVs and this package. Do not upload the whole workspace. Confirm the runtime's Python/dependency compatibility before using pinned packages. Start with a short indexing/retrieval benchmark and verify artifact download. Use mounted input paths and write outputs to a persistent/exportable working directory appropriate to that environment.

Country/source jobs must search the entire relevant target pool. Splitting queries and targets into arbitrary matching chunk numbers loses valid pairs. The current baseline has no multi-account scheduler or resumable distributed merge implementation; do not assume those features exist because accounts were discussed earlier.

## 8. Troubleshooting

| Error | Meaning and action |
|---|---|
| Destination exists | Choose a new versioned path; do not destroy a completed run. |
| Duplicate/missing/unknown ID | Inspect the supplied file or generated row. Do not suppress the exception. |
| max_pairs exceeded | Reduce query counts or top-K for a pilot, or raise the explicit cap after a memory estimate. Preserve validation denominators. |
| No retrieved positives/negatives | Inspect retrieval and the target pool. Do not inject positives into validation. |
| Code/dependency/schema fingerprint mismatch | Use the matching saved source/environment or retrain. Do not edit hashes to bypass the check. |
| No such module: fts5 | Use a Python build with SQLite FTS5 enabled. |
| Full run too slow | Measure route/query throughput; optimize retrieval before scaling. GPU allocation alone does not speed SQLite searches. |
| Package rejected | Complete the reported checks/team fields or use the correct model/output/test manifest combination. |
