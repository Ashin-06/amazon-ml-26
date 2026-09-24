# Implemented function reference

Generated from the actual Python source. These functions exist; optional future designs remain in the implementation guide.

The runbook describes command order. Functions named with an underscore are internal helpers. Legacy aligned helpers remain available, but the command line uses strict disk-backed scoring/validation.

## business_er/__init__.py

Foundations for the Amazon ML Challenge 2026 entity-resolution pipeline.

## business_er/blocking.py

Offline name/address retrieval. Word-token baseline; no external lookups.

### `_partition`

```python
_partition(source, country)
```

Represent any normalized country as an ASCII FTS token, including blank.

### `_quote`

```python
_quote(token)
```

Quote a literal FTS phrase; input text must not become a query operator.

### `build_index`

```python
build_index(source1, source2, source3, destination)
```

Stream all sources into an atomic, immutable SQLite/FTS5 baseline index.

Original text, IDs and arbitrary countries are preserved. Duplicate source
IDs fail construction. Only targets enter FTS. The finished index includes
complete source fingerprints; a partial database is never published.

### `Candidate`

One unique target record with independent route ranks; zero means absent.

### `CandidateIndex`

Read-only index with bounded per-query results and explicit lifecycle.

#### `CandidateIndex.__init__`

```python
__init__(self, path)
```

Open an existing completed index and reject unsupported formats.

#### `CandidateIndex.close`

```python
close(self)
```

Close the read-only database; callers should use a context manager.

#### `CandidateIndex.__enter__`

```python
__enter__(self)
```

Return the opened index.

#### `CandidateIndex.__exit__`

```python
__exit__(self, *args)
```

Always release file handles, including after a failed experiment.

#### `CandidateIndex.iter_references`

```python
iter_references(self)
```

Stream S1 records in original source-file order.

#### `CandidateIndex.get_record`

```python
get_record(self, entity_id)
```

Fetch one exact original record or fail on a missing ID.

#### `CandidateIndex.retrieve`

```python
retrieve(self, record, k=10, max_query_tokens=12)
```

Union independent name/address top-K routes for each source.

No labels are consulted. Literal Unicode tokens are quoted and query
values are parameterized. Known countries include blank-country targets;
a blank reference country searches the entire corresponding source.

## business_er/config.py

Strict, reproducible settings for the first offline CPU baseline.

### `BaselineConfig`

Bounded query sampling and CPU model settings; no account information.

#### `BaselineConfig.validate`

```python
validate(self)
```

Reject unknown types/ranges before allocating indexes or model matrices.

### `load_config`

```python
load_config(path=None)
```

Read a JSON object with only known setting names; validate all values.

### `_unique_keys`

```python
_unique_keys(pairs)
```

Reject duplicate JSON keys rather than silently selecting the last setting.

### `config_digest`

```python
config_digest(config)
```

Hash canonical settings so mismatched runs can be detected.

## business_er/features.py

A fixed numeric pair-feature schema with no learned statistics or labels.

### `feature_schema`

```python
feature_schema()
```

Return immutable feature order; saved models must match it exactly.

### `_similarities`

```python
_similarities(left, right)
```

Return six nonempty equality/edit/token/length signals; missing is zero.

### `pair_features`

```python
pair_features(reference, candidate)
```

Produce finite features for exactly one retrieved pair, retaining missingness.

## business_er/io.py

Strict streaming readers. Keep identifiers and missing fields as strings.

### `Record`

One original source row, without destructive cleaning or numeric IDs.

### `validate_id`

```python
validate_id(value: str, sources: tuple[str, ...])
```

Raise ValueError unless value is an exact ID in an allowed source.

### `_iter_rows`

```python
_iter_rows(path: str | Path, header: tuple[str, ...])
```

Read exact-width UTF-8 TSV rows, raising a line-specific schema error.

### `iter_records`

```python
iter_records(path: str | Path, expected_source: str | None=None)
```

Stream records; validate schema/prefix, preserve raw text and all countries.

Memory is bounded by a row. File-wide ID uniqueness belongs to the audit stage.
expected_source, when provided, is 'S1', 'S2' or 'S3'.

### `parse_id_list`

```python
parse_id_list(value: str)
```

Parse a possibly empty target list; reject duplicates and malformed IDs.

This checks syntax, not membership in a particular train/test source pool.

### `iter_id_lists`

```python
iter_id_lists(path: str | Path, column: str='matched_entity_ids')
```

Stream exact two-column ID-list rows; file-wide uniqueness is not retained.

## business_er/metrics.py

Challenge-specific entity and macro F0.5, including singleton semantics.

### `entity_f05`

```python
entity_f05(truth: Iterable[str], prediction: Iterable[str])
```

Score sets for one S1; empty/empty is 1, one-empty is 0.

Duplicates have set semantics here. Structural validation must reject them
before scoring submission files.

### `_same_keys`

```python
_same_keys(truth: Mapping, predictions: Mapping)
```

Enforce the complete nonempty reference universe before averaging.

### `macro_f05`

```python
macro_f05(truth_by_id: Mapping[str, Iterable[str]], predictions_by_id: Mapping[str, Iterable[str]])
```

Score every reference equally; require identical nonempty key sets.

### `score_aligned_rows`

```python
score_aligned_rows(truth_rows: Iterable, prediction_rows: Iterable)
```

Score same-order, previously uniqueness-validated rows in bounded memory.

Exhausts both iterators. Rejects order/length mismatch. It cannot detect
repeated aligned S1 IDs without extra storage; perform uniqueness validation
upstream. Do not use this on differently ordered truth and prediction files.

### `candidate_metrics`

```python
candidate_metrics(truth_by_id: Mapping[str, Iterable[str]], candidates_by_id: Mapping[str, Iterable[str]])
```

Report natural candidate recall and oracle score on a bounded diagnostic set.

## business_er/normalize.py

Conservative text views; raw records remain available separately.

### `normalize_text`

```python
normalize_text(text: str)
```

NFKC/casefold text, preserving Unicode alphanumerics and combining marks.

### `fold_latin_accents`

```python
fold_latin_accents(text: str)
```

Make an auxiliary normalized view without deleting non-Latin vowel marks.

Strip decomposed marks only after a Latin base character. This is accent
folding, not transliteration; retain the original normalized view as well.

### `numeric_tokens`

```python
numeric_tokens(text: str)
```

Extract digit-bearing tokens as hints, without asserting address semantics.

### `token_jaccard`

```python
token_jaccard(left: str, right: str)
```

Token overlap feature; absent text is zero evidence, even empty vs empty.

## business_er/packaging.py

Allowlisted final archives with model/output/input provenance checks.

### `build_package`

```python
build_package(output_dir, model_dir, documentation, expected_inputs_manifest, destination)
```

Package a verified run only after team details and expected test inputs agree.

The caller supplies the original audited input manifest, not a guessed row
count. ZIP contents are allowlisted source/config/model/output/doc files;
original data, cloud tokens and arbitrary workspace files are never included.

## business_er/predict.py

Full-reference inference with exact scored-candidate export and completion checks.

### `load_model`

```python
load_model(model_dir)
```

Check model/config/code/schema fingerprints before loading text-format weights.

### `predict_baseline`

```python
predict_baseline(index_path, model_dir, output_dir)
```

Score all reference groups, validate both files, then publish the output directory.

At most one reference's feature matrix is resident. This simple baseline does
not resume a partial inference stage; failed unpublished stages are retained.

## business_er/quality.py

Workspace checks and API documentation derived from actual Python sources.

### `write_function_reference`

```python
write_function_reference(workspace)
```

Document every source function/method from AST signatures and full docstrings.

### `check_workspace`

```python
check_workspace(workspace)
```

Check required docs, local links, source syntax/docstrings and original hashes.

Read only explicit project files and paths from the audited input manifest.
No model quality, resource quota or full-data runtime guarantee is implied.

## business_er/splits.py

Stable group assignment; ownership propagation belongs to preparation.

### `assign_split`

```python
assign_split(entity_id: str, seed: int=2026, train_fraction: float=0.8, tune_fraction: float=0.1)
```

Assign an S1 group deterministically without relying on row order.

This helper does not itself propagate labels to targets or stratify groups.
The preparation stage must enforce those contracts and audit split balance.

## business_er/storage.py

Bounded SQLite working storage and explicit-file integrity helpers.

### `file_digest`

```python
file_digest(path)
```

SHA-256 one explicit file without loading it into memory.

### `write_json`

```python
write_json(path, value)
```

Atomically replace a JSON file; reject NaN/Infinity and preserve UTF-8.

### `configure_database`

```python
configure_database(connection)
```

Bound SQLite's page cache to about 32 MiB; put sorting work on disk.

### `temporary_database`

```python
temporary_database(work_dir=None)
```

Yield a temporary SQLite connection; clean only its own explicit files.

### `open_readonly`

```python
open_readonly(path)
```

Open an existing local database without creating or updating it.

### `code_manifest`

```python
code_manifest()
```

Hash only the runnable package's Python sources, never workspace secrets.

### `staging_directory`

```python
staging_directory(destination)
```

Create a new unpublished sibling directory; never overwrite an existing run.

Failed stages are retained for diagnosis. Only a complete stage is renamed
to the requested destination by its caller.

## business_er/submission.py

Atomic TSV writing and strict aligned checks for foundational/pilot workflows.

### `write_id_lists`

```python
write_id_lists(path, rows, column='matched_entity_ids')
```

Write validated rows atomically; preserve order and empty list fields.

Exhausts rows with bounded row memory. File-wide S1 uniqueness must be checked
separately. A failed write leaves an existing destination unchanged.

### `validate_aligned_outputs`

```python
validate_aligned_outputs(reference_rows, matching_rows, candidate_rows, valid_targets=None)
```

Check coverage, uniqueness, syntax and candidate containment in lockstep.

Target existence is checked only when valid_targets is provided. Duplicate
reference detection uses O(number of S1 IDs) memory. A target set adds further
memory; this is not the later disk-backed full-scale validation implementation.

## business_er/train.py

Bounded CPU baseline training with entity-separated tuning and holdout.

### `_load_truth`

```python
_load_truth(connection, index, path)
```

Audit full label coverage/ownership against the index, storing labels on disk.

### `_select_queries`

```python
_select_queries(connection, config)
```

Keep a bounded, stable min-hash sample per disjoint S1 split and audit balance.

### `predictions_at_threshold`

```python
predictions_at_threshold(truth, events, scores, threshold)
```

Preserve every query, including no-candidate entities, while selecting scored IDs.

### `select_threshold`

```python
select_threshold(truth, events, scores)
```

Maximize tune-set macro F0.5, preferring the higher threshold on ties.

### `_evaluate`

```python
_evaluate(truth, events, scores, threshold, countries)
```

Compute overall/per-country macro scores and singleton diagnostics.

### `train_baseline`

```python
train_baseline(index_path, ground_truth, model_dir, config)
```

Train on natural candidates; tune threshold separately; publish complete artifacts.

Input target pools stay complete. Only S1 query groups are sampled. Matrices
are bounded by max_pairs; no positive is inserted into evaluation retrieval.
Fixed boosting rounds avoid consulting holdout for early stopping.

## business_er/validation.py

Order-independent, disk-backed scoring and complete output validation.

### `_insert_lists`

```python
_insert_lists(connection, table, path, column)
```

Load one unique S1-keyed ID-list table; table names are internal constants.

### `score_files`

```python
score_files(truth_path, prediction_path, work_dir=None)
```

Score complete TSVs in any order; reject duplicate, missing or extra S1 IDs.

### `validate_files`

```python
validate_files(source1, source2, source3, matching, candidates, work_dir=None)
```

Check both outputs against all sources using bounded cache and disk indexes.

All three sources and both outputs are mandatory. Output ordering is arbitrary.
Raises on duplicates, invalid IDs, missing/extra rows or matches not scored as
candidates. A successful return always includes target-existence checks.

## run.py

Offline baseline: index, train, predict, score, validate, package and quality checks.

### `_sources`

```python
_sources(parser)
```

Add mandatory explicit source paths; never discover arbitrary workspace files.

### `main`

```python
main(argv=None)
```

Dispatch implemented commands; report input failures with nonzero exit codes.
