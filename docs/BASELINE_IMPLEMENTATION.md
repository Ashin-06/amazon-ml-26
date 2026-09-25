> Historical SQLite-baseline document. Current fast-pipeline corrections and verification scope are in [the final review](FINAL_HARDENING_REVIEW_2026-09-25.md) and [runbook](RUNBOOK.md).

# Baseline implementation contract

This addendum is written before completing the baseline code on 25 September 2026. It narrows the earlier design to one runnable CPU baseline. Optional embeddings, learned aliases and graph refinements are not required to run it.

## Changes selected during the quality review

1. Make the command-line scorer independent of row order and reject duplicate/missing/extra S1 IDs using a temporary SQLite database.
2. Make command-line submission validation use disk-backed ID membership, including both target sources, rather than large Python sets.
3. Implement a reproducible offline baseline using local SQLite FTS5 name/address retrieval, numeric/string features and LightGBM.
4. Save configuration, input/code hashes, feature order, thresholds and metrics with each trained model.
5. Provide runnable commands, tests, a completed-in-progress methodology draft and an allowlisted archive builder. Packaging must require validated real inference outputs and filled-in team details.

## Concrete module contracts

| Module | Functions/classes and behavior |
|---|---|
| `storage.py` | `temporary_database(work_dir)` creates and cleans up only its own temporary SQLite file with a bounded cache. `open_readonly(path)` refuses nonexistent files. `file_digest(path)` hashes one explicit file. `write_json(path, value)` writes JSON atomically. |
| `validation.py` | `score_files(truth, predictions, work_dir)` imports unique keyed rows to disk, checks identical reference coverage, and computes the exact metric in joined order. `validate_files(source1, source2, source3, matching, candidates, work_dir)` checks unique source IDs, exact output coverage, target existence, list uniqueness and match containment with arbitrary output order. |
| `config.py` | `BaselineConfig` contains bounded retrieval/training settings. `load_config(path)` rejects unknown keys, invalid ranges and wrong types. `config_digest(config)` provides deterministic run identity. |
| `blocking.py` | `build_index(source1, source2, source3, destination)` writes raw records plus a source/country partitioned FTS5 index and input hashes, atomically publishes the completed database, and refuses overwrite. `CandidateIndex` reads it without mutating it. `retrieve(record, k, max_query_tokens)` searches independent name/address routes for each target source, unions IDs, preserves route ranks and sorts ties deterministically. Blank countries can use a fallback. |
| `features.py` | `feature_schema()` defines the ordered numeric feature list. `pair_features(reference, candidate)` computes Unicode-preserving name/address similarities, Latin-folded similarities, number overlap, missingness, source and route evidence. No learned feature statistics or labels enter retrieval/features in this baseline. |
| `train.py` | `train_baseline(index, ground_truth, model_dir, config)` checks complete labels and ownership, creates stable S1 train/tune/holdout partitions, selects bounded deterministic query samples, retrieves natural candidates, fits LightGBM using train labels only, selects a threshold on tune macro F0.5 and reports untouched holdout performance. Training stops clearly if configured memory/pair limits are exceeded or supervision is insufficient. |
| `predict.py` | `load_model(model_dir)` checks code/feature/config/model integrity. `predict_baseline(index, model_dir, output_dir)` scores every natural candidate for every reference and writes both exact TSVs in an unpublished staging folder, validating them before marking completion. No forced match, top-one decision or synthetic output is allowed. |
| `packaging.py` | `build_package(output_dir, model_dir, documentation, destination)` verifies the completed inference manifest and hashes, checks filled-in team fields, and includes only explicit output/code/config/model/documentation paths. Original datasets, tokens, arbitrary workspace files and caches are excluded. |
| `quality.py` | `check_workspace(root)` checks required documents/code, local Markdown links, JSON manifests and input hashes. It reports incomplete modeling stages honestly. |

Helper functions are documented in their code and the [generated function reference](FUNCTION_REFERENCE.md). Keep the command-line defaults bounded. Do not label sample target-pool scores as full validation scores. The selected baseline is implemented; full-dataset performance remains to be measured.

## Retrieval details

The baseline uses FTS5 over normalized, Latin-folded word tokens, separately for name and address, with source and normalized country constraints inside the FTS query. Queries quote token literals and are parameterized; business strings cannot become SQL or FTS operators. Input Unicode text remains in the record store.

Each route/source retains a configurable K; candidates are unioned before scoring. This word-based baseline has known weaknesses on all-token typos and cross-script changes affecting both fields. It does not claim to implement the planned character n-gram or embedding routes. Measure actual recall before promotion to a final solution. FTS5 indexing/statistics over the target corpus use no label information.

Reference and target files are indexed once per train/test pool. Country/source partition keys work for arbitrary strings, including France. Target records with blank countries are eligible through the fallback rather than silently lost.

The SQLite [FTS5 documentation](https://www.sqlite.org/fts5.html) describes MATCH queries, column restrictions and BM25. The [LightGBM classifier documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) defines the classifier and saved-model interfaces. These APIs do not establish recall, runtime or prediction quality on this dataset.

## Validation and runtime boundaries

- The full target corpus stays in the retrieval index, including unlabeled distractors. Sampling affects query groups only in normal training.
- Evaluation candidates are natural retrieval outputs; missed true matches remain false negatives. No injected positives are used in this baseline.
- All labels must reference existing records, each target can have at most one labeled S1 owner, and all S1 labels must be present exactly once.
- Check country/singleton split summaries. Holdout labels never select the threshold or model settings. France remains unlabeled and has no claimed accuracy.
- Training pair matrices are bounded by configuration. Large index construction and full inference remain time-consuming; record measured throughput before running all records.
- A synthetic end-to-end run verifies code integration only. It is never a competition validation score or submit-ready result.
