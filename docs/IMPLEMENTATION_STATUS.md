# Implementation status — 25 September 2026

This status applies to branch codex/compact-pipeline-20260925. It includes source through 90ead4f plus the corrections in the [latest rules and score review](RULES_AND_MAXIMUM_SCORE_REVIEW_2026-09-25.md). Active jobs were preserved.

Implemented: explicit UTF-8 conversion, natural DuckDB blocking, supervised transliteration with hold exclusion, 53 pair and 17 context features, LightGBM/XGBoost, cached feature reuse, exact per-reference F0.5, production-consistent evaluation/global ownership, tune-only threshold selection, exact first-model candidate export and chained manifests.

Latest corrections: first-model candidate accounting after cascade rejection; ID-independent tied rank features; fixed-size post-lexicon skeleton batches; GPU prediction batching and CPU allocation retry; actual fitted-device metadata; CPU defaults and explicit expansion of stage-2 fitting; all-reference candidate averages; a tuning-only candidate-efficiency diagnostic. A smaller stage-2 workload is not a smaller official candidate set.

Frozen v14 at de77cbc reports tuning F0.5 0.979377 and final F0.5 0.978897. Its diagnostic covers all 50,141 tuning references: 48.806 candidates/reference and oracle F0.5 0.990872. Reducing to 36.567 candidates/reference lowered the ceiling to 0.989984. These are measurements of v14 retrieval, not this newly corrected source's accuracy.

Release evidence is under artifacts/verification-release.xml and artifacts/verification.json. Coverage includes raw-input-to-package integration, cascade exports, repeatability, malformed data/stale artifacts, CPU XGBoost, simulated GPU allocation retry, bounded GPU transfers, routing ties and seed ensembles. Original manifest files are checked against D:/Amazon ML 26.

Not established: a full retrained model for this version, a 0.99+ score, France accuracy, cloud capacity, full-scale peak resources or billion-record throughput. QuantileDMatrix still assembles quantized data in memory. Full-pipeline cross-fitting is not claimed: the supervised lexicon is shared across non-holdout classifier folds.

Follow the [runbook](RUNBOOK.md). Source changes require retraining or matching saved code. No partial-stage resume or multi-account scheduler is implemented. Methodology remains a draft; this review made no leaderboard upload.

Release verification: **108 tests passed**, zero failures/errors; **38 Python files compiled**; quality passed with all **12 original manifest fingerprints** matching. Tests completed in 22.85 seconds. This establishes software behavior, not a full retrained model score.

Error floor (v10 holdout): exact-name matches to empty-address targets are ambiguous when several references share the name (true-link rate 0.98 when the name is unique, 0.54 / 0.30 / 0.15 / 0.04 when 2 / 3-5 / 6-20 / >20 references share it), so part of the 0.9905 oracle is not reachable by any matcher.

GPU research (25 Sep, RTX 3060 6 GB, 14 CPU threads):
- Model training/prediction: GPU-resident XGBoost 0.18 s/round vs CPU 0.71 s/round (8M rows); used.
- XGBoost external memory (data in RAM, GPU compute): 13.4 s/round on Windows (no RMM/HMM support); unusable.
- String similarity: a custom CUDA kernel (CuPy, bit-parallel LCS) reproduces fuzz.ratio exactly on 2.69M real name pairs, but end-to-end it is not faster: 0.61 s from Python strings, 0.40 s from Arrow buffers, vs 0.30 s for rapidfuzz (SIMD C++, ~9M pairs/s). Packing and moving strings dominate; a GPU-only pipeline (RAPIDS cuDF) is not available on Windows. Kept on CPU.
- The real featurization bottleneck was Python loops; moving them into DuckDB SQL gave identical features at 1.76x speed (51k -> 90k pairs/s).

Candidate set (organizer update: a smaller candidate_pairs.tsv per reference ranks higher): candidate generation now ends with a learned filter. Key blocking keeps ~49 targets per reference (link recall 97.05%); the stage-1 classifier drops pairs with p1 < 0.01. On v12's 100,652-reference holdout this leaves 3.92 candidates per reference (48.7 before), drops 278 of 347,987 true links (0.08%) and no predicted link. The matcher trains, scores and is evaluated only on kept pairs, and candidate_pairs.tsv lists exactly those pairs (test-checked). Seed ensembles now require members with identical filtered sets.

v17 features (India error analysis on v14's holdout): look-alike decoys at the same address keep the start of the distinctive first word and change its ending ("Prananth"/"Prananeon", "Gangya"/"Gangex"), while true typos are scattered edits; targets often get an extra number prepended to the address ("#27 c/o ..."), which made first-number comparisons compare a fake number. Added ft_lev / ft_len / ft_pre / ft_suf (reference first core token vs its closest target token: edit distance, shared start, shared ending) and hn_in_t / hn_t_in_s (each side's first number anywhere in the other address), computed in SQL. Stage-1 A/B on 4.3M labeled pairs (disjoint references): log-loss 0.00799 -> 0.00713, errors at 0.5 6,050 -> 5,452. Holdout F0.5 by country (v14): India 0.9760, US 0.9812; test is 46.8% India, 38.3% US, 15.0% France.

Stage-1 outputs (normalized records, lexicon, candidates) are now fingerprinted by exactly the modules stage 1 runs (integrity.STAGE1_CODE: prep, lexicon, block, db, integrity, validation, run_stage1); a test checks that none of them imports the matcher. Matcher/feature changes therefore reuse verified candidates instead of re-blocking (~1.5 h per run); model and prediction manifests still fingerprint every module.

Final regression result: 100 tests passed; zero failures/errors.
