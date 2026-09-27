# Business entity resolution — Amazon ML Challenge 2026

**Follow-up experiments:** five additional variants tested; none passed adoption. v28 remains unchanged, with one upload unused. [Results and error budget](../../docs/V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md).

**Latest fast-model release:** v28fast_rescue is prepared and independently validated, not submitted. Local India/US final score **0.989583** versus **0.989350**; CPU pilot 22.6 seconds. France remains identical to v27 and unlabelled. Public best stays **v26: 0.98417**. **One upload remains.** Read the [fast rescue report](../../docs/FAST_RESCUE_TRAINING_2026-09-27.md) for files, checks, reproduction and limitations. Older status paragraphs below are historical.

**Latest: 27 September evening.** User-confirmed best: **v26, 0.98417**.
**One upload remains.** v27 passed a fresh full independent output/candidate
check; its French accuracy is unlabelled. The completed xenc7 experiment failed
adoption. See the [evening review](../../docs/EVENING_098417_FINAL_REVIEW_2026-09-27.md) for exact results and new
organizer rules: 17 total attempts; TSV tonight, ZIP in Round 2. Earlier status
sections below are historical and superseded by this update.

**Current production versions:** the user-confirmed public best is **0.982910**
for `v24stack_t077_fr0925`. Read the
[post-result review](../../docs/POST_0982910_BLUEPRINT_REVIEW_2026-09-27.md)
for current experiments and the two remaining uploads, then
[v24 correction and the 0.995 target](../../docs/V24_CORRECTION_AND_0995_AUDIT_2026-09-27.md).
The frozen v20x/v22/v24 runs use version-specific settings, GPU-trained tree models
and a neural rescorer. The historical CPU pipeline description and defaults below
do not reproduce those submissions. Use each selected run's exact source and
artifact fingerprints. v22 and v24 training are complete; the separate correction
model reaches 0.988804 on final validation. Its release status is in the linked
audit, and no 0.995 leaderboard result has been demonstrated.

Pipeline for matching every Source-1 business record to its Source-2 / Source-3
records. Everything runs on CPU and uses only the supplied data (no external lookups).
The final model is LightGBM (MIT license).

## Pipeline (`src/fast_er/`)

| Stage | File | What it does |
|---|---|---|
| 0 | `to_parquet.py` | Convert the supplied TSVs to parquet (strings verbatim). |
| 1 | `prep.py` | Normalize names/addresses: anyascii transliteration, lowercase, punctuation removal, abbreviation/state canonicalization (US, India, France), zero-padded numbers, ordinals, repeated tokens. Adds a legal-form-free `core` name. |
| 1b | `lexicon.py` | Learns a transliteration lexicon (e.g. `praivet→pvt`, `lksmi→laxmi`) from labelled train pairs whose target name was in a native script (holdout S1s excluded) and applies it to train and test. |
| 2 | `block.py` | Candidate generation in DuckDB: hashed name keys (tokens, adjacent/whole-name concatenations) and address keys (tokens, adjacent bigrams), per country; keys shared by >300 targets dropped; idf-weighted overlap score; keep top-40 by total score ∪ top-20 by name score ∪ top-20 by address score per S1 (name score includes name×address-word composite keys). Writes `cands.parquet`. |
| 3 | `model.py` | ~45 pair features (rapidfuzz similarities, digit/house-number agreement, name/address frequency, candidate-list context). Two stage-1 LightGBM models on disjoint S1 halves give out-of-fold probabilities; a stage-2 LightGBM adds competition context (rank within S1, best competing S1 for the same target). Decision: probability ≥ threshold, then each target goes to its single best S1. Threshold maximizes exact macro F0.5 on a held-out S1 set. |

Drivers: `run_stage1.py` (stages 1–2 for a split), `run_stage3.py` (train + predict + write outputs).
`kaggle/build_kernel.py` packs the same source into a Kaggle script kernel (how the full runs were executed:
4 CPU cores, ~30 GB RAM, ~2–3 h end to end).

## Reproduce

Python 3.12, `pip install -r requirements.txt`. From this directory, with the supplied data at `DATA/` (containing `train/`, `test/`):

```text
python src/fast_er/to_parquet.py DATA work/pq
python src/fast_er/run_stage1.py train work/pq work work/pq/train_ground_truth.parquet --mem 18GB
python src/fast_er/run_stage1.py test  work/pq work --mem 18GB
python src/fast_er/run_stage3.py work work/pq/train_ground_truth.parquet test output --mem 16GB
```

`output/` then holds `matching_results.tsv` (leaderboard file) and `candidate_pairs.tsv` (exactly the pairs
the matcher scored; every match is a subset of it). `work/train/blocking_recall.json` and
`work/model/metrics.json` hold blocking recall, the oracle ceiling, and holdout macro F0.5 for stage 1 and 2.
The hold references are split in two: the threshold is tuned on the `tune` half and `holdout_macro_f05`
is reported on the untouched `final` half, with competing non-hold references taking part in target
ownership exactly as in the test decision. Split membership is saved in `work/model/split_ids.parquet`.

Safety checks built into the pipeline: labels that do not resolve to known records stop training;
non-finite or out-of-range probabilities, null keys, and duplicate pairs stop output writing; saved models
whose feature list differs from the code are refused before any work starts; each stage-1 split records the
code hashes/settings that made it (`stage1_meta.json`) and refuses to mix versions.

Check format with the organizers' validator, which ships in the supplied `student_resource/utils/` folder
(not in this package):

```text
python PATH/TO/student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir DATA/test
```

Tests: `python -m pytest tests/test_fast_er.py -q`.

A submission zip in the organizers' layout (allow-listed files, optional model bundle, never overwrites):
`python kaggle/package_submission.py TEAM output --run work` from the repository root paths shown in its docstring.

## Older baseline

`src/business_er/` and `src/run.py` are the first SQLite-FTS5 baseline written before the full-scale run.
Its one-query-at-a-time retrieval is too slow for 1.7M test references, so it is not the submission pipeline.
It (and `tests/test_baseline.py`, `tests/test_failure_cases.py`, `tests/test_foundations.py`) additionally needs
`scikit-learn==1.7.2` and `scipy==1.17.1`, which the fast pipeline does not require.
