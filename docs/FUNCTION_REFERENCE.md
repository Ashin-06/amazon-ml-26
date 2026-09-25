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
check_workspace(workspace, input_root=None)
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

## fast_er/__init__.py

Verified offline business-entity matching with DuckDB and LightGBM.

## fast_er/block.py

Stage 2: candidate generation (blocking) with DuckDB set joins.

Every record gets hashed keys, namespaced by country and field:
  name    : core-name tokens (len>=2), adjacent-token concatenations incl. short
            tokens ("n"+"infra" -> "ninfra", "paulita"+"wilson" -> "paulitawilson",
            catches @handles / domains), first-3 concat; whole core name (cap `cap_whole`).
  name x address word : each core-name token (len>=2) / whole-name concat joined with
            each non-numeric address word (any position) ("pioneerinfotech|pune"), so
            generic names still block within their city. Counted as name evidence.
  address : tokens (len>=3 or containing a digit) and adjacent token bigrams
            ("1550 beverly") that survive component reordering.
Keys held by more than `cap` targets are dropped (too common to discriminate).
Pair score = sum of idf over shared keys (total, name-only, address-only).
Name-rank ties: more address evidence first, then an empty address (no conflict) over a conflicting one.
Kept per S1: top-`k` by total score  UNION  top-`k_field` by name  UNION
top-`k_field` by address, so address-only (renamed DBA) and name-only
(missing address) matches both survive. Output: work/<split>/cands.parquet.

### `block`

```python
block(split, work='work', cap=300, k=25, k_field=10, chunks=8, threads=None, mem='20GB', cap_whole=1000)
```

Build fresh natural candidates; partial chunks are never treated as completed work.

### `blocking_recall`

```python
blocking_recall(split='train', work='work', gt_path='work/pq/train_ground_truth.parquet')
```

Train-only: link recall and oracle macro F0.5 (empty prediction for singletons = 1.0).

## fast_er/db.py

Shared DuckDB/SQL helpers used by candidate generation (stage 1) and the matcher (stage 3).

### `lit`

```python
lit(value)
```

SQL string literal with quotes escaped (paths may contain apostrophes).

### `normalized_sql`

```python
normalized_sql(directory)
```

Explicit three-file parquet expression; never ingest an unmanifested wildcard match.

### `configure`

```python
configure(con, mem, threads, temp_dir)
```

Validated DuckDB resource settings.

### `hold_rule`

```python
hold_rule(col, hold_mod)
```

SQL predicate selecting holdout S1s (~1/hold_mod of references).

## fast_er/ensemble.py

Verified equal-weight seed ensembles with complete-graph evaluation and immutable outputs.

### `members`

```python
members(work, model_dirs)
```

Load at least two distinct, verified models sharing hold membership and training inputs.

### `align_scores`

```python
align_scores(base, scores)
```

Align complete unique key sets, rejecting duplicates even if group totals would cancel.

### `evaluate`

```python
evaluate(work, gt_path, model_dirs, name='ensemble', mem='4GB', threads=1, chunks=24)
```

Re-score every candidate for each selected member; tune once and evaluate final once.

### `apply`

```python
apply(work, split, out_dir, score_dirs, name='ensemble', mem='4GB', threads=1)
```

Average verified member predictions on identical candidates using the selected threshold.

### `main`

```python
main()
```

Parse explicit ensemble members or prediction directories; threshold comes from evaluation.

## fast_er/integrity.py

Content fingerprints and fail-closed completion records for the fast pipeline.

### `digest`

```python
digest(path)
```

Hash a file in bounded blocks; never load a complete data file into memory.

### `files_digest`

```python
files_digest(paths)
```

Describe explicitly named files, refusing symlinks and duplicate basenames.

### `code_digest`

```python
code_digest(names=None)
```

Fingerprint every fast-pipeline module (default), or only the named ones (stage-1 outputs).

### `environment`

```python
environment()
```

Record the Python version and libraries that affect numerical or data behavior.

### `read_json`

```python
read_json(path)
```

Read UTF-8 JSON without permitting a missing completion manifest.

### `seal`

```python
seal(directory, filename, inputs, outputs)
```

Publish a completion record only after every explicitly listed output exists.

### `verify`

```python
verify(directory, filename, inputs=None)
```

Reject incomplete, changed or incompatible artifacts; never bless legacy runs.

### `stage_inputs`

```python
stage_inputs(pq_dir, split, gt, settings, lexicon=None)
```

Bind blocking to raw inputs, labels, code, environment, chunks and lexicon.

### `verify_stage`

```python
verify_stage(work, split)
```

Verify finished normalized/candidate files and the currently selected lexicon.

## fast_er/lexicon.py

Stage 1b: learned transliteration lexicon (training labels only, holdout S1s excluded, no external data).

Many India targets spell English business names in native scripts; after anyascii
"Perfect Engineering Private Limited" reads "prphekt imjiniyrimg praivet limitet".
For labelled train pairs whose target name was non-ASCII and has the same token
count as the S1 name, tokens are aligned by position and (translit -> english)
counts are kept when frequent and consistent. The lexicon is then applied to
every non-ASCII-named record (train and test) before blocking/features.

### `build_lexicon`

```python
build_lexicon(work, gt_path, hold_mod=22, min_count=3, min_share=0.5, split='train')
```

Learn deterministic token mappings from audited non-hold labels only.

### `apply_lexicon`

```python
apply_lexicon(work, split)
```

Rewrite name/core of non-ASCII-named records in work/<split>/norm_s*.parquet (idempotent via marker).

## fast_er/model.py

Stage 3: pair features, two-stage LightGBM matcher, threshold tuning, submission writing.

Stage 1 (per pair): blocking scores/ranks, rapidfuzz similarities on name / core
name / address, digit-token agreement, missingness, how common the name/address
is, and candidate-list context. Two stage-1 models are trained on disjoint S1
halves (A, B) so every training pair gets an out-of-fold probability p1.

Stage 2 (per pair, with context): stage-1 features + p1 + how p1 compares with the
S1's other candidates and with the best competing S1 for the same target.

Decision: keep pairs with p2 >= threshold, then each target goes only to its
highest-probability S1 (labels show every target belongs to at most one S1).
The threshold maximises exact macro F0.5 on the 'tune' half of the held-out S1s (with
competing non-hold S1s taking part in target ownership); the score is reported on the other
'final' half. Hold S1s are never used for fitting or for the lexicon.

### `_num_feats`

```python
_num_feats(sa, ta)
```

Digit-token agreement; hn_* compare the first number (house/plot number): typo vs neighbour.
hn_logdiff = log1p |a - b|, hn_sub1 = same length and exactly one digit differs (typo-like, while
decoys shift the number by a small amount: 353 vs 355).

### `features`

```python
features(df, threads=1)
```

Stage-1 feature matrix (float32, column order FEATS) for joined candidate rows.

### `_pairs`

```python
_pairs(con, where)
```

Candidate rows (with group features) joined to both records' normalized text.

### `scratch`

```python
scratch(work)
```

Directory for spill/temporary files: $FASTER_TMP if set (e.g. /tmp on Kaggle), else the work dir.

### `_connect`

```python
_connect(work, split, mem, threads)
```

Load a verified stage; reject incompatible schemas and candidate endpoints.

### `_context`

```python
_context(con, sid, tid, p1, anchor_where='TRUE', chunks=8, threads=1)
```

Stage-2 context columns (order CTX) aligned to (sid, tid) via a DuckDB window pass.

### `_anchors`

```python
_anchors(con, where, chunks, threads=1)
```

Cluster consistency: a reference's true matches are variants of each other, while a decoy
(another business at the same address) is not. For each plausible pair (p1 >= ANCHOR_MIN), compare
the target with the S1's other strongest candidates (top ANCHOR_K with p1 >= ANCHOR_P) and keep the
best name / core-name / address similarity and the number of such anchors. Table `anc`.

### `_ctx_matrix`

```python
_ctx_matrix(con, keys)
```

Fetch context rows for keys (DataFrame sid, tid) in keys' order.

### `check_scores`

```python
check_scores(p, thr=None, name='scores')
```

Probabilities must be finite and in [0,1]; NaN would otherwise pass a SQL `p >= thr` filter.

### `decide`

```python
decide(sid, tid, p, thr)
```

Threshold, then give each target only to its best S1 (ties: lowest sid, same rule as the SQL
writer). Returns {sid: [tid,...]}.

### `macro_f05`

```python
macro_f05(truth, pred, all_sids)
```

Exact per-reference macro F0.5, retaining empty and unretrieved references.

### `_params`

```python
_params(threads, seed, leaves=127, lr=0.05)
```

Return fixed-round CPU settings; reject invalid resources before fitting.

### `gpu_available`

```python
gpu_available()
```

True if XGBoost can train on a CUDA GPU on this machine (checked once with a tiny fit).

### `resolve_backend`

```python
resolve_backend(backend)
```

'auto' -> 'xgb' when a CUDA GPU is usable (Kaggle T4, local RTX), else 'lgb' on CPU.

### `_xgb_params`

```python
_xgb_params(P, device)
```

XGBoost settings equivalent to the LightGBM parameters in P.

### `_cupy`

```python
_cupy()
```

CuPy module when installed and a CUDA GPU is usable, else None (then XGBoost copies host data).

### `_xgb_train`

```python
_xgb_train(P, make, n, rounds)
```

Train on the GPU when usable; if the GPU runs out of memory, rebuild the matrix on the host and
train on CPU. `make(gpu)` builds the quantized matrix (~1 byte per value; built directly in GPU memory
from CuPy blocks when gpu is True). No evaluation set (it only cost GPU memory).

### `_fit`

```python
_fit(backend, P, X, y, w, names, rounds, valid=None)
```

Train one booster: XGBoost (GPU if available, CPU otherwise) or LightGBM (CPU).

### `_fit_parts`

```python
_fit_parts(backend, P, loaders, y, w, names, rounds)
```

_fit for training rows kept on disk: `loaders` return consecutive row blocks matching y / w.
XGBoost streams the blocks into its quantized matrix (~1 byte per value), so every reference fits in
memory; LightGBM (small runs and tests) concatenates them.

### `_pred`

```python
_pred(m, X, threads=None)
```

Match probability for the rows of X, from either booster type.

### `_names`

```python
_names(m)
```

Ordered feature names a saved booster was trained with.

### `_save`

```python
_save(m, base)
```

Save a booster as BASE.txt (LightGBM) or BASE.json (XGBoost).

### `_load`

```python
_load(base)
```

Load BASE.json (XGBoost, placed on the GPU when available) or BASE.txt (LightGBM).

### `_importance`

```python
_importance(m, names)
```

Gain importance per feature for either booster type.

### `pair_weights`

```python
pair_weights(sid, mode)
```

Per-pair training weights. 'none': every pair 1. 'ref': each reference's pairs sum to a
constant (the metric averages per reference); 'sqrt': halfway (1/sqrt(n_candidates)). Mean 1.

### `_predict_stage1`

```python
_predict_stage1(con, models, where, threads, chunks, tmp=None, tag='')
```

Featurize candidate rows matching `where` chunk by chunk; return keys + averaged p1.
If tmp is given, stage-1 matrices are cached there as .npy for stage 2.

### `_check_args`

```python
_check_args(n_train, hold_mod, chunks, rounds)
```

Reject nonsensical split sizes and work settings before reading data.

### `_load_truth`

```python
_load_truth(con, gt_path)
```

Compatibility entry point for the shared label-integrity gate.

### `_score_all`

```python
_score_all(con, models, m2, threads, chunks, tmp_parent)
```

Score the complete graph using the production pipeline and run-owned disk caches.
Returns only the pairs kept by the learned candidate filter (the matcher's inputs).

### `evaluate_threshold`

```python
evaluate_threshold(sid, tid, p, truth, tune_sids, final_sids, grid=None)
```

Choose with tune labels, then evaluate final labels exactly once.

### `model_path`

```python
model_path(work, name)
```

Resolve one model/ensemble subdirectory without allowing traversal or absolute paths.

### `train`

```python
train(work, gt_path, n_train=300000, hold_mod=22, mem='20GB', threads=None, seed=26, rounds=600, chunks=8, leaves=127, lr=0.05, weight='none', backend='auto', model_dir='model')
```

Fit classifiers and stacker, then evaluate disjoint tune/final reference groups.

Stacker inputs use classifier-level cross-fitting. The learned lexicon uses
non-hold references, so this is not full-pipeline cross-fitting. Every training
candidate is featurized once; the stage-1 matrices are cached in a run-owned scratch
directory (removed afterwards), streamed into the boosters and reused for the production-scoring
evaluation pass. n_train above the number of non-holdout references trains on all of them.

### `_train`

```python
_train(work, gt_path, n_train, hold_mod, mem, threads, seed, rounds, chunks, leaves, lr, weight, backend, model_dir, own)
```

Body of train(); `own` is the scratch directory for cached stage-1 matrices.

Every training candidate (all references: fit groups A/B, holdout, others) is featurized exactly once,
one sid-slice at a time; matrices live on disk and boosters read them block by block. n_train sets the
stage-1 references (A + B; above the non-holdout count means all); stage 2 always learns from every
non-holdout reference. Pairs below the learned candidate filter (CAND_P) never reach stage 2.

### `load_models`

```python
load_models(model_dir)
```

Load the three boosters and refuse any whose feature list differs from this code's.

### `predict`

```python
predict(work, split, out_dir, mem='20GB', threads=None, chunks=8, tmp=None, model_dir='model')
```

Verify model/data provenance, score every candidate and seal new output files.

### `write_outputs`

```python
write_outputs(con, scores_path, thr, out, chunks=64)
```

Decision in DuckDB (p >= thr, then each target to its best S1) and stream both TSVs.
Every S1 in `rec` gets exactly one row; lists may be empty. Aggregation runs in S1 chunks
so memory stays bounded. Returns (links, S1 rows).

### `rewrite`

```python
rewrite(work, split, scores_path, out, thr=None, mem='8GB', threads=None)
```

Refuse unverified legacy score rewrites; verified predictions use immutable destinations.

## fast_er/prep.py

Stage 1: normalize every record of one split into work/<split>/norm_s{1,2,3}.parquet.

Columns: rid (int, row id within split), src (1/2/3), entity_id, country,
name (normalized), core (name without legal/stop words), addr (normalized),
na (original name was non-ASCII, i.e. transliterated), rawn (raw name, no spaces/mapping).
Non-ASCII text (Indic scripts, French accents) is transliterated with anyascii
so cross-script pairs share a Latin view. No external lookups.

### `_tok`

```python
_tok(t)
```

Canonicalize zero padding, abbreviations and ordinal tokens.

### `norm`

```python
norm(text, states=False, france=False)
```

Lowercase, transliterate, strip punctuation, canonicalize tokens, drop repeated tokens.

### `raw_joined`

```python
raw_joined(text)
```

Transliterated, lowercase, alphanumerics only, no spaces, no abbreviation mapping.

### `_ocr`

```python
_ocr(m)
```

Map OCR digits 0/1/5/6/8 in a word to o/l/s/g/b unless the word looks genuinely numeric.

### `segment`

```python
segment(stem, vocab=None)
```

Split a glued domain stem into reference-name words, single letters as initials
("regionalcreativetowing" -> "regional creative towing", "hvidyalaya" -> "h vidyalaya").
Most probable split under a unigram model of reference-name word counts (common words win:
"cleanmarinesystems" -> "clean marine systems", not "marin esystems"; an initial counts as a word seen
once). The stem is kept unchanged when it is a known word itself or cannot be covered mostly (>= 60%)
by words of 4+ letters.

### `clean_name`

```python
clean_name(text)
```

Drop appended phone numbers; for 'Brand DBA Real Name' keep the part after the alias marker;
remove generator noise (ID tags, URL tails, honorific prefixes, OCR digits) and split domain names.

### `name_vocab`

```python
name_vocab(path)
```

Counts of lowercase transliterated words (2+ chars) over every reference name in one raw parquet.

### `_init_worker`

```python
_init_worker(vocab)
```

Pool initializer: give each worker the reference-name vocabulary used by segment().

### `_chunk`

```python
_chunk(cols)
```

Normalize one batch while preserving a separate original-name view.

### `skeleton`

```python
skeleton(text)
```

Phonetic consonant skeleton of a normalized name: similar sounds merged, vowels dropped, repeats
collapsed. Transliteration variants the lexicon has not learned still line up
("lakshmi"/"lksmi" -> "lksm", "perfect"/"prphekt" -> "prfkt").

### `core_of`

```python
core_of(name)
```

Name without legal forms / filler words (falls back to the full name).

### `prepare`

```python
prepare(split, pq_dir='work/pq', out_dir='work', workers=8)
```

Stream each source in 100k-row batches; one parquet per source, global rid offsets.

## fast_er/run_stage1.py

Normalize, learn/apply the train lexicon, block, then publish a completion manifest.

### `provenance`

```python
provenance(a)
```

Fingerprint inputs, settings, the complete source and selected lexicon.

### `run`

```python
run(a)
```

Run a fresh stage or verify a completed one; incomplete stages cannot resume.

### `main`

```python
main()
```

Parse the supported stage command and execute it.

## fast_er/run_stage3.py

Stage 3 driver: train the matcher on train candidates, then predict a split.

Usage: python run_stage3.py WORK GT_PARQUET PRED_SPLIT OUT_DIR [--n-train N] [--hold-mod M] [--mem 20GB] [--threads N]

## fast_er/to_parquet.py

Convert explicit UTF-8 TSV inputs to string-only parquet in a fresh, verified directory.

### `convert`

```python
convert(data, out)
```

Preserve every string and empty field; seal all seven original-input fingerprints.

## fast_er/validation.py

Disk-backed schema, reference coverage, label and candidate integrity gates.

### `validate_records`

```python
validate_records(con)
```

Require unique integer internal keys, valid string source IDs and country fields.

### `validate_pairs`

```python
validate_pairs(con, table)
```

Reject duplicates, unknown or reversed endpoints and cross-country candidates.

### `load_truth`

```python
load_truth(con, gt_path)
```

Require exactly one label row per S1, including empty lists, and unique target ownership.

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
