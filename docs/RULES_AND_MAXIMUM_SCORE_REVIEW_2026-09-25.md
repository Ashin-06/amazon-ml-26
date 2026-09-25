# Updated rules, corrected architecture and the 0.99 objective

## Outcome

The corrected checkout is on branch **codex/compact-pipeline-20260925**, under D:/Amazon ML 26/artifacts/compact-pipeline-20260925. It starts from experimental commit 90ead4f, including the earlier audit fixes and later normalization, feature and cached-training work. Active jobs and their saved source were preserved.

The recommended design remains multi-route blocking, two-stage boosted matching and global target ownership. The most urgent change is accurate candidate accounting under the organizer's clarification. A larger neural model has no demonstrated advantage here yet.

The objective is **macro F0.5 above 0.99**, not 99% classification accuracy. It is not achieved or guaranteed. Software checks, retrieval ceilings and actual matcher scores are separate measurements.

## Official requirements confirmed

The [updated problem statement](https://d8it4huxumps7.cloudfront.net/files/6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf) requires both TSVs in the final package and considers candidate efficiency alongside matching results. There is no published numerical weighting between these objectives in the inspected material.

The [official clarification sheet](https://docs.google.com/spreadsheets/d/1wMo3bmj5bAdkDI9FvqMnLRmANRGrXNuS6u69ce_VvR0/edit#gid=483754747), read on 25 September, resolves several details:

- Rows 133/152: a cascade exports the inputs to its **first scoring model**, including pairs rejected later.
- Rows 115/128: unsupervised statistics over the supplied test records are permitted.
- Rows 94/116/118: pure algorithms and small handwritten normalization dictionaries are allowed; external identity/geographic datasets and lookups are prohibited.
- Rows 14/300: every pretrained model must meet the MIT/Apache-2.0 and eight-billion-parameter limits and run offline. Hosted model APIs are not allowed in the solution.
- Rows 83/273: AI coding assistance is allowed. Row 338 prohibits AWS Entity Resolution.
- Row 130: the methodology has no page limit. This resolves the shorter general-guideline wording.

No query form was submitted, no dataset entity was searched online, and no leaderboard upload was made. Original PDFs, supplied code and data were preserved. Downloaded review material is held in private ignored artifacts, excluded from submission packaging.

## Concrete corrections

| Issue | Correction in this checkout |
|---|---|
| Learned-filter survivors were presented as the official candidate set | Preserve every first-model input in scores and candidate_pairs.tsv. Early rejection still saves stage-2 work; rejected pairs still count toward candidate cost |
| Different seed models could filter different pair sets | Return the same complete first-model set, with zero decision scores for rejected pairs; exact-key ensemble checks remain enabled |
| A small --n-train silently expanded stage-2 fitting to every non-holdout reference | Both stages use the requested sample by default; --stage2-all-references is explicit |
| GPU availability could be reported as GPU training after a CPU retry | Record each fitted model's actual training device and persist retry information |
| Reload/prediction could move a CPU-fallback model back to GPU | Preserve the CPU device choice; GPU inference transfers at most 65,536 rows and retries allocation failures on CPU |
| Iterator-based XGBoost was described as guaranteeing a full-data fit | Correct the claims: QuantileDMatrix retains combined quantized data in memory |
| Skeleton generation scanned records repeatedly and used fractions of corpus size as batches | Persist post-lexicon skeletons during the existing 100,000-row transformation pass |
| Blocking rank ties included target-ID order in model features | Give identical evidence identical rank features; keep deterministic routing ranks outside the feature list |
| Candidate averages omitted references with no candidates | Divide by all S1 references and test zero-candidate/singleton cases |
| No reusable candidate-cost comparison tool | Add a read-only tuning diagnostic reporting mean, percentiles, maximum, recall, oracle F0.5 and same-country reduction ratio |

Earlier safeguards remain: stable S1 splits, lexicon hold exclusion, full-graph production-consistent evaluation, tune-only threshold search, one final evaluation per run, exact F0.5, all-reference denominators, finite score checks, wider blocking ranks, strict labels/IDs/countries, completion manifests, exact candidate export and allowlisted packaging.

The scoring cascade's early-rejection threshold remains an experimental heuristic at p1=0.01. Skipped pairs have a decision score of zero, not a claimed second-stage probability. Changing or disabling the gate requires retraining and tuning-only comparison. The corrected export does not turn the previously claimed 3.9 stage-2 pairs per reference into an official candidate-efficiency result.

## Real-data measurement

I first checked 5,000 tuning references, then all **50,141 v14 tuning references**, using their natural candidates from the full target pool. The second diagnostic used one CPU thread and a 1 GB DuckDB budget and completed in **14.797 seconds**. That is diagnostic runtime, not training/inference runtime or a measured whole-process memory cap. No final-holdout labels were used to choose candidate policies.

The full tuning set contains 173,434 true links and 2,787 singleton references. Current blocking supplies 2,447,190 pairs: mean 48.806, median 49, p95 59, p99 61 and maximum 70. Its same-country exhaustive comparison baseline is 269,246,081,916 pairs, yielding a 99.999091% reduction. This does not demonstrate billion-record runtime scaling.

| Blocking policy | Mean candidates/S1 | True-link recall | Oracle macro F0.5 |
|---|---:|---:|---:|
| Existing total-40 union name/address-20 | 48.806 | 0.971292 | 0.990872 |
| Total-30 union name/address-15 | 36.567 | 0.968703 | 0.989984 |
| Total-25 union name/address-10 | 28.511 | 0.965647 | 0.989047 |
| Total-20 union name/address-10 | 24.141 | 0.963600 | 0.988435 |
| Total-25 only | 24.960 | 0.959841 | 0.986815 |
| Total-10 union name/address-5 | 11.499 | 0.948453 | 0.983986 |
| Total-10 only | 9.997 | 0.937826 | 0.979967 |

Oracle means perfectly selecting the true links already present in each candidate list. It is an upper bound for those labeled references, not an achievable model score or a hidden-test estimate. Correct singleton abstention scores one. Every reference remains in the denominator.

Total-20/field-10 dominates total-25 alone in this diagnostic: it has both fewer candidates and a higher ceiling. However, none of the smaller measured policies preserves a ceiling above 0.99. No smaller policy was silently promoted as the maximum-score choice.

| Country | Tuning references | Current candidate recall | Current oracle F0.5 |
|---|---:|---:|---:|
| India | 19,951 | 0.962353 | 0.988073 |
| US | 30,190 | 0.977210 | 0.992722 |

France has no labeled training references and therefore no measured oracle or accuracy in this analysis.

The frozen v14 model at de77cbc reports **tuning F0.5 0.979377** and **final F0.5 0.978897**. On the same tuning references, the current oracle is only 0.000872 above the requested 0.99 target. This leaves little room for matcher mistakes; stronger retrieval is a promising way to give the matcher more headroom. This is an inference from measurements, not proof that a particular new retrieval method will help.

Evidence: artifacts/v14_tuning_candidate_frontier.json and artifacts/v14_all_tune_candidate_frontier.json. They record the exact historical stage digest and diagnostic source fingerprints. V14 files were read only; the diagnostic does not bypass source checks for retraining or deployment.

## Architecture and experiment order

1. **Versioned input preparation.** Preserve original IDs/text, explicit UTF-8 and empty fields. Normalize in bounded batches; keep country open-ended. Learn supervised transformations only from allowed references.
2. **Pre-model retrieval.** Use name, address and composite routes with measured posting caps. Export their union as the first scoring model's inputs. Tune routing budgets using recall, oracle F0.5 and candidate costs together.
3. **Pair models and context.** Retain 53 pair features and 17 context features initially. Cross-fit classifiers by S1 groups. Cache matrices and reuse them; measure total RAM/VRAM/disk rather than relying on one library's limit.
4. **Cascade and ownership.** Keep all first-stage inputs in the artifact contract. Early rejection can save later compute. Use identical context and ownership rules in evaluation and inference. Allow many target IDs per reference, while assigning each target to at most one reference.
5. **Selection and export.** Select experiments/thresholds on tuning data, freeze the configuration, then evaluate final holdout. Validate both TSVs against every test reference and original data before packaging.

For further score improvement, prioritize retrieval experiments for Indian transliteration, aliases and missing addresses. Test character-based or additional raw-name routes with capped postings and measure incremental true-link recovery per added candidate. Do not simply enlarge every list. Preserve all matching target IDs even when their text is identical.

Next compare the later normalization/features against the frozen baseline on the same tuning set, with fresh compatible preprocessing and models. Inspect singleton false positives, ambiguous common names, neighbouring addresses and global ownership errors separately. Retain an ensemble only when complementary errors give a measured improvement under the same candidate contract.

A neural reranker is a later bounded experiment if residual errors justify its cost and its license fits. The literature supports comparing blocking, similarity joins and nearest-neighbour configurations empirically; it does not establish a universally best architecture for this challenge. See [Papadakis et al.](https://arxiv.org/abs/2202.12521) and [BLAST's data-derived meta-blocking approach](https://www.vldb.org/pvldb/vol9/p1173-simonini.pdf).

Batching alone does not prove bounded total training memory. XGBoost distinguishes ordinary quantile matrices from external-memory matrices, with platform/performance tradeoffs. See [its versioned documentation](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/external_memory.html).

## Verification and remaining limits

Release verification is recorded in artifacts/verification-release.xml and artifacts/verification.json. Coverage includes raw input through training, prediction, validation and packaging; repeatable outputs; malformed labels/IDs; exact candidate sets; ensembles; tied ranks; zero probabilities; CPU XGBoost and simulated GPU allocation failures. The quality command checks the 12 original manifest entries and source/document integrity.

The new version has not completed full-data retraining or hidden-test evaluation. Physical GPU fallback at full scale, full-process peak memory and billion-record throughput are not certified by tests. Complete graph keys, probabilities, anchors and quantized training data remain substantial potential allocations.

The supervised lexicon is shared across non-holdout classifier folds. This is classifier-level cross-fitting, not complete pipeline cross-fitting. A fold-specific lexicon is a further experiment if stacker optimism remains; hold references are excluded throughout. Global validation includes competitors from fitted references, whereas test references are unseen. Repeated inspection of historical final scores also limits their independence for future model selection.

The next full run uses frozen source and fresh versioned artifacts; never hot-edit active code or reuse incompatible manifests. Main-workspace jobs and existing submissions were not overwritten. Methodology remains a draft pending real team fields and the selected run. No 0.99 score or new public result is claimed.

Release verification: **108 tests passed**, zero failures/errors; **38 Python files compiled**; quality passed with all **12 original manifest fingerprints** matching. Tests completed in 22.85 seconds. This establishes software behavior, not a full retrained model score.
