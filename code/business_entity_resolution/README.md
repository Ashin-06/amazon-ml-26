# Business entity resolution — Amazon ML Challenge 2026

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
Check format with the organizers' validator:

```text
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir DATA/test
```

## Older baseline

`src/business_er/` and `src/run.py` are the first SQLite-FTS5 baseline written before the full-scale run.
Its one-query-at-a-time retrieval is too slow for 1.7M test references, so it is not the submission pipeline.
