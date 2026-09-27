# Runbook: local baseline and later Kaggle execution

**Follow-up experiments:** five additional variants tested; none passed adoption. v28 remains unchanged, with one upload unused. [Results and error budget](V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md).

**Latest fast-model release:** v28fast_rescue is prepared and independently validated, not submitted. Local India/US final score **0.989583** versus **0.989350**; CPU pilot 22.6 seconds. France remains identical to v27 and unlabelled. Public best stays **v26: 0.98417**. **One upload remains.** Read the [fast rescue report](FAST_RESCUE_TRAINING_2026-09-27.md) for files, checks, reproduction and limitations. Older status paragraphs below are historical.

**Latest: 27 September evening.** User-confirmed best: **v26, 0.98417**.
**One upload remains.** v27 passed a fresh full independent output/candidate
check; its French accuracy is unlabelled. The completed xenc7 experiment failed
adoption. See the [evening review](EVENING_098417_FINAL_REVIEW_2026-09-27.md) for exact results and new
organizer rules: 17 total attempts; TSV tonight, ZIP in Round 2. Earlier status
sections below are historical and superseded by this update.

**Current frozen-run instructions:** the confirmed best is
`submissions/v24stack_t077_fr0925_matching_results.tsv` at **0.982910**.
Two team leaderboard slots remain as of the afternoon update. Consult the
[post-result review](POST_0982910_BLUEPRINT_REVIEW_2026-09-27.md) for the running
private xenc7 experiment and its adoption checks, then
[v24 correction and its release record](V24_CORRECTION_AND_0995_AUDIT_2026-09-27.md),
[the public 0.982277 improvement audit](PUBLIC_0982277_IMPROVEMENT_AUDIT_2026-09-26.md),
[the 0.995 follow-up](PUSH_0995_STATUS_2026-09-26.md) and
[the measured v20x release commands](PUSH_0993_MEASURED_2026-09-26.md).
The isolated correction commands/design are in
`artifacts/push995-20260927-0915/README.md`; its model is trained on tuning
identities only and has a frozen cutoff. Baseline commands below do not reproduce
v20x/v22/v24 or that correction model. Never point an old
baseline command at their saved model/output directories. Current models require
their matching frozen source, environment, candidates and manifests.

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
