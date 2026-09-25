# Model review and improvement recommendations

Review date: 25 September 2026. Evidence snapshot: approximately 07:41–07:49 IST.

**Verdict: the approach is promising and the recorded development scores are improving, but the current project cannot yet be described as fully verified or reproducible.** The most urgent work is to preserve a compatible model/code bundle, correct evaluation differences, and make full runs reliable. Increasing model size should come after these checks.

This is a review document, not an implementation change. No training, preprocessing, cloud job, submission, dependency installation, source edit, configuration edit, or process-control action was initiated by this review. Existing documentation and artifacts were left untouched. Small checks ran in memory with no model fitting; output inspection was bounded. Recommendations below are **not applied**.

## 1. What is actually running

There are two distinct implementations:

| Implementation | Purpose and evidence |
|---|---|
| `src/business_er/` | Earlier SQLite FTS5 baseline. It has strict input/output validation, provenance checks, three-way train/tune/holdout separation, and the recorded 51-test suite. |
| `src/fast_er/` | Current submission approach: bulk DuckDB blocking, learned transliteration, and two-stage LightGBM. Real full-data runs and submission files exist for this family. The old test suite does not import this implementation. |

The package [README](<D:/Amazon ML 26/code/business_entity_resolution/README.md>) describes `fast_er`, while [IMPLEMENTATION_STATUS.md](<D:/Amazon ML 26/docs/IMPLEMENTATION_STATUS.md>), [RUNBOOK.md](<D:/Amazon ML 26/docs/RUNBOOK.md>), and the generated function reference still describe the earlier baseline. These documents should eventually be synchronized around a selected run; none were changed here.

At the beginning of inspection, local Python PID 2040 was executing the v7 **training-data preparation/blocking** driver, with `--mem 6GB --threads 14 --workers 6 --chunks 16`. Its observed working set was about 6.3 GB. This was not LightGBM fitting. By the 07:49 snapshot the process was no longer listed and its existing log contained completed blocking statistics. No stop/restart action was taken. No v7 matcher score was found at that snapshot. Saved Kaggle logs are historical observations, not a live remote-status check.

## 2. How the current model works

1. **Convert TSV to Parquet.** Read tab-separated fields as strings. Preserve entity IDs; empty fields become null and are handled later. Originals remain separate from derived files.
2. **Prepare text views.** Transliterate non-ASCII text, lowercase, normalize punctuation, shorten common address/legal terms, standardize some state names, numbers, and ordinals. Keep a full normalized name, a reduced `core` name, an address, and a concatenated name view.
3. **Learn a transliteration dictionary.** Use labelled training pairs to map frequent transliterated tokens to reference-name tokens. The designated hold group is excluded, provided the same split parameters and hash behavior are used throughout.
4. **Retrieve candidate pairs.** Build country-specific name, address, whole-name, and name×address keys. Drop overly common keys, score shared keys by inverse frequency, and take the union of the best overall, name, and address candidates. Current driver defaults are top 40 overall plus top 20 per field, with common-key caps of 300 and 1,000 for whole names. This is not exhaustive fuzzy search.
5. **Fit two first-stage LightGBM models.** Divide sampled S1 identities into A and B. Each model learns whether a candidate pair matches from 50 numeric features. A is scored by the B model and vice versa. Hold/test first-stage scores average both models.
6. **Fit a second-stage LightGBM model.** Add the first score plus information about other candidates for the reference and competing references for the target. Current source has 13 context features, giving 63 total. The saved local v4 model has 10 context features, giving 60 total.
7. **Choose matches.** Apply a probability threshold and allow each target to belong to only its best-scoring S1. Multiple S2/S3 matches per S1 remain possible. Training labels support target uniqueness; it is still an inference-time assumption.
8. **Write both required TSVs.** The current writer starts from every S1, so zero-candidate references receive empty lists. Candidates come from the scored-pair table, and accepted matches come from that same table.

No neural model is currently used in this pipeline. The labelled sample size in `metrics.json` refers to matcher training; the lexicon can learn from the larger non-hold training population.

### Function map

Paths in this table are under `code/business_entity_resolution/src/fast_er/` unless stated otherwise.

| File / functions | Role and important contract |
|---|---|
| `to_parquet.py` entry point | Converts seven supplied TSVs; skips an existing destination. It does not verify the destination against input fingerprints. |
| `prep.py`: `_tok`, `norm` | Canonicalize individual tokens and whole strings. `states=True` additionally rewrites address state terms and removes literal null-like tokens. |
| `prep.py`: `clean_name`, `raw_joined`, `core_of` | Remove phone/alias prefixes, create concatenated name text, remove legal/filler words with fallback. The current “raw” view is still cleaned and transliterated. |
| `prep.py`: `_chunk`, `prepare` | Process batches, preserve string IDs, assign internal row IDs, write normalized Parquet per source. Row IDs are used for joins/chunking; they are not direct model columns. |
| `lexicon.py`: `build_lexicon` | Align equal-length token lists in positive pairs; accept mappings with minimum count/share. Excludes the designated hold group. |
| `lexicon.py`: `apply_lexicon` | Replaces normalized name/core tokens in derived Parquet; a marker prevents repeat application. |
| `block.py`: `block` | Generate hashed keys, target postings, candidate scores/ranks, and chunked candidate files. |
| `block.py`: `blocking_recall` | Compute positive-link retrieval recall and perfect-matcher macro F0.5 ceiling. Its reported candidates-per-S1 denominator includes only references with candidates. |
| `model.py`: `_num_feats`, `features` | Compute number agreement, text similarities, missingness, frequency, and within-reference support. Feature order is fixed by `FEATS`. |
| `model.py`: `scratch`, `_connect`, `_pairs` | Choose temporary location, create record/candidate tables, and fetch joined pair data with group statistics. |
| `model.py`: `_context`, `_ctx_matrix` | Build probability ranks/competition summaries and align them back to pair keys. |
| `model.py`: `hold_rule`, `_params` | Define the hash-based hold group and LightGBM hyperparameters. |
| `model.py`: `_predict_stage1` | Score chunks and optionally cache feature matrices. Empty chunks are skipped. |
| `model.py`: `decide`, `macro_f05` | Select unique target owners and calculate per-reference F0.5. These functions are used during threshold selection. |
| `model.py`: `train` | Assemble A/B/hold data, fit three boosters, tune thresholds, and save models, metrics, and hold predictions. |
| `model.py`: `predict` | Load the three saved boosters, score candidates in two passes, save scores, then export TSVs. |
| `model.py`: `write_outputs`, `rewrite` | Export decisions/candidates; `rewrite` can regenerate TSVs from saved scores. This is not a general restart mechanism for interrupted training or feature extraction. |
| `run_stage1.py`, `run_stage3.py` | Command-line orchestration for preparation/blocking and training/prediction. |
| `kaggle/build_kernel.py` | Embed a source snapshot in a Kaggle script, attach data, optionally reuse earlier normalized data/candidates, and execute stages. |
| `kaggle/fetch_and_validate.sh` | Download selected outputs and invoke the organizer validator without target-ID checking by default. |
| `kaggle/package_submission.py` | Archive outputs, current source/docs, and a non-draft methodology. It does not currently bind those files to the selected model/run. |

## 3. Evidence of progress

These are **recorded development metrics**, not independently measured leaderboard scores or percentages of records correctly matched.

| Run | Recorded stage-1 macro F0.5 | Recorded final macro F0.5 | Full-train candidate link recall | Full-train oracle F0.5 | Evidence / outcome |
|---|---:|---:|---:|---:|---|
| v1 Kaggle | Single-stage model | 0.94218 | 0.914532 | 0.965594 | Saved log reaches full test output and `ALL DONE`. |
| v2 Kaggle | 0.95025 | 0.95345 | 0.920252 | 0.968049 | Saved log reaches full test output and `ALL DONE`. |
| v3 Kaggle | 0.95478 | 0.95781 | 0.932865 | 0.973460 | Saved log reaches full test output and `ALL DONE`. |
| v4 local | 0.9735885765 | **0.9761929438** | 0.966382 | 0.989007 | Saved models, metrics, scores, and both TSVs exist. Original stage-3 log ended with an output-writing memory error; the complete recovery was not independently revalidated here. |
| v4 / v5 Kaggle | No final score found | Not established | 0.966382 | 0.989007 | Both saved logs end with SIGKILL during the model stage. |
| v7 local blocking | Not yet found | Not established | **0.970488** | **0.990543** | Completed training-side blocking during this review; no matcher result at snapshot. |
| v8 Kaggle | Not reached | Not reached | Not established | Not established | Saved log fails because RapidFuzz is unavailable and package installation cannot resolve the network host. |

Sources: [v1 log](<D:/Amazon ML 26/work/v1_log.txt>), [v2 log](<D:/Amazon ML 26/work/v2_kaggle_log.json>), [v3 log](<D:/Amazon ML 26/work/v3_kaggle_log.json>), [v4 metrics](<D:/Amazon ML 26/work/v4/model/metrics.json>), [v4 local stage log](<D:/Amazon ML 26/work/v4/stage3.log>), [v4 Kaggle log](<D:/Amazon ML 26/work/v4_kaggle_log.json>), [v5 Kaggle log](<D:/Amazon ML 26/work/v5_kaggle_log.json>), [v7 recall](<D:/Amazon ML 26/work/v7/train/blocking_recall.json>), [v8 log](<D:/Amazon ML 26/work/v8_kaggle_log.json>).

The v1→v4 reported increase is about **0.03401**, or 3.40 percentage points of F0.5. It is encouraging but not a controlled comparison: the embedded v1 code selects a different hold group, later code changes the split rule, and local/Kaggle DuckDB versions differ. Scores also reuse their evaluation group for threshold selection.

The v4 comparison between stage 1 and stage 2 is more directly comparable within that run: **+0.00260437** at their separately selected thresholds. It supports the usefulness of competition context on that development set. Gain importance is dominated by `p1`, `p1_tdiff`, and `p1_trank`; importance alone is not causal evidence, so ablation is still needed.

v7 retrieves about **0.411 percentage points more true links** than v4 on the full training population, with approximately 1.08% more candidate pairs: 107,576,884 versus 106,427,733. That is a useful retrieval improvement. It does not yet establish a better final matcher. The lexicon learned from part of that same full training population, so full-train retrieval numbers should also be reported separately on an excluded evaluation group.

The small `lexmini2` run reports 0.98516 over 6,615 hold references. It must not be ranked above v4 using that number: the candidate universe/sample and evaluation conditions are different.

### Artifact observations

- Parquet metadata for v4 normalized files matches the previously audited source row counts, including 1,732,544 test S1 rows and 9,969,589 test targets.
- v4 test candidates and saved scores both contain **83,765,632 rows**. Equal row counts are useful evidence, but do not prove equality of pair identities.
- v4 hold artifacts contain **100,652 references** and **4,854,909 scored pairs**.
- v4 matching output is 99,773,724 bytes; candidate output is 1,101,991,224 bytes. Both have the expected header and a newline-terminated final two-field row.
- The first 2,000 corresponding v4 output rows passed ID-format and matches-subset-of-candidates checks. This is a prefix sample, not a statistical estimate or full-file validation.
- [Submission log](<D:/Amazon ML 26/submissions/log.csv>) labels v1 and v4 as ready but not yet uploaded and contains no leaderboard score.

## 4. Confirmed issues and risks, in priority order

### P0 — Current code is incompatible with the saved v4 second-stage model

The saved `work/v4/model/lgb2.txt` header contains 60 features. Current [model.py](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:55>) constructs 63, adding `p1_srank`, `p1_sn05`, and `p1_smax`. Prediction loads a model and uses current feature code without an early compatibility check. A fresh current-code prediction using that saved model will encounter LightGBM's default feature-count check at stage 2, potentially after expensive stage-1 work.

**Recommendation:** preserve the exact source, preprocessing, feature order, lexicon, dependencies, parameters, and weights for each selected run. Use the matching historical code for v4 or retrain a new version with the new features. Add compatibility validation before processing data. Do not disable shape checking. LightGBM documents this check and its risks in the [4.6.0 parameter reference](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#predict_disable_shape_check).

**Acceptance:** ordered feature names and counts match for all three boosters; a saved compatible bundle reproduces a fixed small inference fixture and its output fingerprints.

### P1 — “Holdout” is actually being used as the tuning set

[model.py:285](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:285>) searches thresholds on the hold references and reports the maximum on those same references. These labels are excluded from direct booster fitting, but they are used to choose the final decision rule. Repeated model selection against those scores further reduces their independence. There is no separate untouched final holdout in `fast_er`.

**Recommendation:** retain the existing scores as development/tuning scores. For future runs, define stable S1-level train/tune/final-hold groups before learning the lexicon, selecting features, or choosing thresholds. Tune decisions on tune only; use final hold once for the selected procedure. The old baseline's split contract is a useful design reference, but is not implemented automatically in this pipeline. See [scikit-learn cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) and [threshold selection guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

**Acceptance:** saved disjoint ID manifests; lexicon excludes tune/final-hold labels; the selected threshold is recorded before final-hold scoring. Do not claim a new split is untouched if earlier lexicon/model fits already used its labels.

### P1 — Evaluation and final prediction use different target competition

The context construction includes competitors beyond hold, but threshold evaluation calls `decide` with **only hold pairs** ([model.py:289](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:289>)). Final output resolves competition across **all test references** ([model.py:348](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:348>)). A target awarded to a hold reference in evaluation may be taken by a higher-scoring non-hold reference when the full decision policy is applied. The effect on overall F0.5 can go either way; it has not been quantified.

A two-pair in-memory check confirmed the difference: a hold reference with score 0.8 wins when considered alone, but loses the same target to another reference scoring 0.9 under the global rule.

**Recommendation:** evaluate the actual intended reference universe and target-ownership rule, then restrict the metric denominator to the designated evaluation references. Competitor predictions must also use a documented out-of-fold or otherwise leakage-controlled protocol. Simply adding in-sample training predictions would introduce another mismatch.

**Acceptance:** identical decisions between evaluation and the output writer for the same scored-pair table, including competitors outside the metric subset.

### P1 — Cache reuse is based on existence, not provenance

`prepare`, `block`, Parquet conversion, and the lexicon application marker skip existing files without checking input/code/config hashes. Blocking also skips existing numbered parts. Kaggle reuse copies earlier normalized records/candidates even if current preprocessing or blocking differs. A changed `hold_mod` or hash implementation can also make a reused lexicon inconsistent with the new hold split.

This is a concrete risk in an actively evolving workspace: embedded v4/v7 preprocessing differs from current source; v8/v9 snapshots also lack the current `clean_name` implementation. The original source change times are not sufficient to identify what a process already imported.

**Recommendation:** use immutable versioned output directories and a manifest for every stage. Include source hashes, input hashes, feature schema, lexicon hash, exact split IDs, dependency versions, and candidate parameters. Refuse incompatible reuse. Write each part atomically and validate completed-part metadata. Keep future outputs under the project-specified ignored `artifacts/` location; do not move active runs.

**Acceptance:** changing normalization, chunk count, candidate caps, or split parameters cannot silently reuse stale files. Model and output provenance is checked before packaging.

### P1 — Memory failures and environment failures are already observed

Kaggle v4/v5 logs show SIGKILL; memory exhaustion is plausible but the signal alone does not prove its cause. The local v4 log explicitly reports DuckDB out-of-memory during output writing. v8 explicitly fails to obtain RapidFuzz. Current source includes a chunked writer that may address the historical writer failure, but it was not rerun at full scale in this review.

More chunks reduce individual text batches but do not remove all whole-run arrays: `train` retains group matrices and concatenates them; prediction retains all pair keys/probabilities. For scale, 83,765,632 pairs × 50 float32 features would occupy about **16.75 GB** for feature values alone if materialized together. Current prediction caches these by chunk, but still needs adequate temporary disk plus memory for other arrays. This is arithmetic, not a measured peak-RAM result.

DuckDB's limit does not cap the whole Python process; NumPy, pandas, LightGBM, and some database allocations need additional headroom. See [DuckDB memory limits](https://duckdb.org/docs/current/operations_manual/limits) and [OOM guidance](https://www.duckdb.org/docs/current/guides/performance/oom).

**Recommendation:** measure peak process memory and temporary-disk use by stage; bound thread counts and live arrays; use memory-mapped/sharded data where appropriate. Include required pinned wheels for offline runs and fail immediately on failed installation/import. `build_kernel.py` currently installs only the discovered AnyAscii wheel locally, falls back to unpinned network installation for some packages, and uses `check=False` for installation.

**Acceptance:** a representative pilot and one complete selected run finish with measured headroom; the kernel's import/version preflight passes before dataset processing; recovered output passes complete validation.

### P1 — Final packaging does not establish reproducibility

The current [Kaggle packager](<D:/Amazon ML 26/code/business_entity_resolution/kaggle/package_submission.py:16>) includes current code and requested TSVs but omits the selected `work/.../model` weights, lexicon, metrics, split manifests, and run-specific configuration. It has no output-validation or code/model fingerprint gate. Retraining from the README defaults does not reproduce the local v4 configuration: that run used 160,000 matcher-training references and 800 rounds, whereas current defaults are 300,000 and 600.

The pinned local requirements also differ from recorded Kaggle versions, including DuckDB 1.5.5 versus 1.3.2 and several numerical/data libraries. DuckDB explicitly says its [hash function may change across versions](https://duckdb.org/docs/current/sql/functions/utility#hashvalue). Because splits use `hash`, persist split membership rather than assuming matching IDs across environments.

**Recommendation:** package an allowlisted, exact selected-run bundle and its reproduction commands; include the artifacts needed for inference or a fully specified retraining route. Validate complete S1 coverage, target existence, duplicate IDs, and matches⊆candidates, and compare exported candidates with the final scored pair identities. The official validator remains unchanged; its missing-candidate/subset cases can be warnings, and the fetch script does not enable `--check-ids`.

**Acceptance:** extract the final archive in a clean environment and reproduce an agreed fixture with its bundled model/code; complete validation and fingerprints pass for the actual final TSVs. Keep methodology draft status until genuine team details and selected-run evidence are filled in.

### P2 — Empty prediction chunks are not handled consistently

[`_predict_stage1`](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:214>) skips empty chunks and does not create their cached matrices. [`predict`](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:325>) nevertheless loads `x{k}.npy` for every chunk. A small/sparse run, or one with no candidates in a bucket, can fail with a missing file; a stale file could instead corrupt alignment. A mocked in-memory producer check confirmed the skipped-empty behavior. This was not observed as the cause of the large recorded failures.

**Recommendation:** track the exact completed chunk IDs, keys, row counts, and hashes; consume only those chunks, with explicit empty-output handling and run-unique temporary paths.

**Acceptance:** zero-candidate input, fewer references than chunks, and an empty middle chunk all produce correct outputs without loading old files.

### P2 — Tied probabilities can choose different owners

[`decide`](<D:/Amazon ML 26/code/business_entity_resolution/src/fast_er/model.py:180>) sorts by probability only; the SQL writer sorts by probability then `sid`. An in-memory tie check produced different owners: Python selected sid 2, SQL selected sid 1. Trees can produce repeated probabilities, so this is worth fixing even if its current score impact is unknown.

**Recommendation:** share one explicit deterministic ownership policy between scoring and export. Separately test whether row-order-based tie ranks influence learned features. Internal row IDs must remain bookkeeping, not predictive evidence.

**Acceptance:** Python and SQL decisions agree under ties and input-row permutations, using a documented deterministic fallback.

### P2 — Cross-fitting covers the booster, not the entire learned pipeline

First-stage A/B scores are out of fold for the corresponding booster. However, the lexicon learns from both A and B labels. Cross-reference context can also depend on competitors scored by a model trained on the focal group's labels. This means “out-of-fold” should not be interpreted as a guarantee that every upstream feature is independent of that group's labels.

This does not prove direct hold-label leakage when the same exclusion rule was used for the lexicon and fitting. It does warrant more careful stacking validation.

**Recommendation:** measure an ablation without cross-reference context; for stronger stacking guarantees, cross-fit learned preprocessing and context generation consistently, or evaluate a simpler calibrated ensemble on a truly excluded group. Compare first-stage single-fold versus averaged-score distributions, because meta-training sees the former while inference sees the latter.

### P2 — Cleaning can discard useful identity information

Current `clean_name` keeps the text after an alias marker and can discard the other legitimate name. `raw_joined` also calls that cleaner, so it cannot recover what was removed. The source itself contains a check showing a normal word containing an alias token can be reduced excessively. Broad state rewriting and number normalization can also merge distinct forms.

**Recommendation:** keep original Unicode text plus separate cleaned/transliterated/alias views; use cleaning as additional evidence rather than destructive replacement. Learn lexicon mappings using distinct S1 identity support, not only repeated variants from one identity, and preserve the original transliteration as a fallback.

**Acceptance:** controlled perturbation/alias tests and group-disjoint evaluation show benefits without worse false positives or singleton behavior.

## 5. Research-backed improvement plan

The following are proposed experiments, not claims of guaranteed gains. All retrieval experiments must use natural candidates against the real target pool, including unmatched distractors. Never inject known positives into evaluated retrieval.

| Order | Experiment | Why it is worth testing | Required evidence |
|---|---|---|---|
| 1 | Freeze compatible baseline + repair validation/provenance | Makes future gains interpretable and prevents expensive incompatible runs. | Matching feature/source/artifact fingerprints; distinct tune/final-hold groups; identical decision logic. |
| 2 | Error analysis by country, script, missing address, singleton status, name frequency, and retrieval route | Reveals whether remaining loss comes from missing candidates or incorrect acceptance. | Candidate recall and oracle per group, exact macro F0.5, false positives/reference, and singleton false-match rate. |
| 3 | Keep v7 retrieval; compare to v4 under the same evaluation protocol | Recorded training-side recall improved at modest candidate-count growth. | Same excluded evaluation identities and target universe; final matcher score plus runtime/memory. |
| 4 | Add a bounded character n-gram retrieval route for difficult queries | Exact token overlap misses typos and split/merged words. Character TF-IDF with sparse top-N products is a plausible complementary route. | Compare `char`/`char_wb` 3–5 gram pilots, candidate budget, missed-link recovery, and added false candidates; keep name and address channels independent. |
| 5 | Preserve aliases/original script and strengthen lexicon confidence | Current replacement can lose evidence; pair-count confidence can overvalue many variants of one identity. | Ablations for alias-union views, distinct-identity lexicon support, and original-versus-transliterated views. |
| 6 | Calibrate/tune thresholds on tune only; optionally compare country/source policies | F0.5 favors precision, but a good threshold depends on score distribution and final ownership logic. | Exact entity-macro scorer; minimum subgroup sizes; frozen fallback for unseen France; no test-label tuning. |
| 7 | Evaluate group-balanced training weights and hard negatives | Pairwise log loss weights candidate-rich references more, while the competition averages references. | Compare equal pair weight versus inverse-candidate-count weights and targeted confusing negatives on the same tune split; preserve natural evaluation prevalence. |
| 8 | Modest LightGBM tuning or calibrated ensemble | More leaves/rounds may help or overfit and can worsen runtime. | Vary one factor at a time, stop by tune evidence, compare paired entity-level scores; report peak memory and time. |
| 9 | Optional multilingual embedding rescue | Could complement lexical routes for cross-script/renamed cases, but similar business meanings are not proof of the same business. | Bounded pilot only after the reliable baseline; licence/size/revision recorded; actual retrieval gain, address agreement, runtime, and memory measured. |

For character retrieval, scikit-learn documents [character and word-boundary character TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). The [ING sparse_dot_topn implementation](https://github.com/ing-bank/sparse_dot_topn) supports sparse top-N multiplication and chunk merging. These APIs make a bounded pilot practical; they do not establish feasibility over ten million targets. Fit text statistics under a documented protocol and measure actual memory before expanding.

For neural exploration, the [multilingual-e5-small model card](https://huggingface.co/intfloat/multilingual-e5-small) identifies an MIT-licensed multilingual encoder with 384-dimensional embeddings. Preserve original text for that experiment and follow the model card's prefix/normalization instructions. Ten million 384-dimensional float32 target vectors alone need about 15.36 GB before an index or model; selective rescue is a more realistic initial experiment than unconditional full-data embedding. Similarity scores must be validated as features, not interpreted as calibrated match probabilities. No external business identity, registry, or geocoding lookup is needed or recommended.

### What I would prioritize before the deadline

1. Preserve the strongest compatible existing bundle and complete its full validation when it will not interfere with active work.
2. Make evaluation and final target ownership agree; establish the limits of the current scores and a clean final check.
3. Measure the completed v7 retrieval with a compatible matcher before increasing tree count/leaves again.
4. Add only one small, measurable improvement with a clear fallback. Prefer identified retrieval/normalization errors over a new unmeasured large model.
5. Finalize the allowlisted archive, documentation, and submission log with actual selected-run evidence. Preparing files is not leaderboard submission. Respect five uploads/day and the supplied 27 September 2026, 23:59 IST deadline.

The v4 threshold curve is broad: 0.675 scores 0.97619294 and 0.700 scores 0.97618401, a difference of only about 0.00000893. This is not persuasive evidence for fine threshold micro-optimization while the validation/ownership issues remain unresolved.

France is about 15% of test references and has no labelled training references. US/India validation cannot establish French accuracy. Report French coverage, zero-candidate rate, score distributions, and normalization behavior; language perturbation tests are diagnostics, not French ground-truth evaluation.

## 6. What was verified in this review

| Check | Result | Scope |
|---|---|---|
| Syntax parsing | 25 Python source/Kaggle files parsed successfully | Syntax only; no imports or execution of their entry points. |
| Current feature constants vs saved v4 header | 50 first-stage features match in count; current stage 2 is 63 versus saved 60 | Confirms incompatibility; does not validate all semantic preprocessing. |
| Exact F0.5 function | All 64 truth/prediction set combinations over a three-element universe passed | Includes empty/empty=1 and false-positive/false-negative weighting. |
| Zero-candidate metric denominator | Explicit two-reference check passed | An unmatched nonempty truth still contributes zero; an empty singleton contributes one. |
| Ownership parity | Tie mismatch and subset/global competition mismatch reproduced | Tiny in-memory fixtures; no production files changed. |
| Empty chunk behavior | Producer skip reproduced; unconditional consumer loading confirmed from source | No cache files created. |
| Saved output inspection | Headers/tails and 2,000-row prefix checks passed | Does not prove complete S1 coverage, target membership, duplicate absence, or exact candidate equality. |
| Parquet metadata | Six normalized source counts and candidate/score/hold counts read successfully | Metadata only; entire contents not revalidated. |
| Existing test evidence | Saved JUnit reports 51 passed, zero failed/errored/skipped at 01:16 IST | Historical baseline suite; no `fast_er` references in its test files. |
| Original-file integrity evidence | Saved quality report records 12 original hashes verified | Historical evidence; no fresh multi-GB fingerprint scan during active work. |

The full training pipeline, full test suite, original-input quality command, complete 1.1 GB candidate output, cloud jobs, and leaderboard were **not rerun**. This kept the review from competing materially with active work or rewriting generated files. Therefore there is no basis to certify “zero errors,” full current-source reproducibility, or a guaranteed leaderboard score.

Two inspection helpers initially encountered local access/encoding limitations: process working-directory access was denied, and an inspection-only read used Windows' default encoding. The necessary process command line was obtained read-only, and file inspection was repeated explicitly as UTF-8. These were review-tool issues, not evidence of model failures.

### Focused future regression checks

- All S1s remain in score/output denominators, including no-candidate references and singletons.
- Train/tune/final-hold identities are disjoint; lexicon training respects the same frozen exclusions.
- Changing source/input/config/lexicon invalidates derived caches and model reuse.
- Python and SQL ownership match for ties, competing references, threshold boundaries, and reordered rows.
- Empty chunks/all-empty candidates produce valid empty lists and never reuse stale matrices.
- Saved-model feature names/order and preprocessing match before any expensive inference.
- Every matched/candidate target exists in supplied S2/S3; exported candidate pairs exactly equal final matcher inputs.
- Interrupted stage recovery cannot publish mixed/incomplete artifacts; final publication requires successful validation.
- A clean offline environment can import pinned dependencies and reproduce the selected model's fixture.

## 7. Source snapshot for this review

These SHA-256 values identify the `fast_er` files inspected. They are review evidence, not a claim that current saved runs were generated by these bytes.

| File | SHA-256 |
|---|---|
| `block.py` | `f31866dcc1997439232c62746cc9cff74905a31acdc8575a15097a3590450ab9` |
| `lexicon.py` | `d517b7f5d82664523d071b5dc31941ae7a32be25672b28122f20b7e6efcfaae1` |
| `model.py` | `c7467edf0daa6c0d214d83a6fde2e1788875e797ea07c04b69df6c3f709beabd` |
| `prep.py` | `29b25442a4c6fa8453614d1739dc4c619cf7a317ba1d43be8334c37f310009e9` |
| `run_stage1.py` | `0bd964b25693bd10d16d233410eb4a21a17ebc5a007d68b37ad9d0e0b8fe86cc` |
| `run_stage3.py` | `c682517534da82a74039f4c627ce0649a32fd9b33c43b18fca15e67959cc5412` |
| `to_parquet.py` | `6ad3c77aae0a97f21a50b05b8cd8eca117c09725c94a359b5facc8eef9258a7a` |

Research was limited to primary algorithm/library/model documentation. No dataset business identities were sent to external search services. No account tokens were opened or used for this review.
