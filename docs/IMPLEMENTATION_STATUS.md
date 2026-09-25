# Implementation status — 25 September 2026

Update 26 September: [the score investigation](V12_SCORE_INVESTIGATION_2026-09-26.md) reconciles v12 outputs, distinguishes saved model metrics from hold-only comparisons, and measures v17 retrieval on all 50,141 tuning references (oracle 0.993748; 95.17 candidates/reference). Training in this checkout now saves exact global held decisions and per-country/additive error diagnostics. Packaging and ensemble application explicitly reject survivor-only candidate accounting. These corrections were not installed into the active frozen v17 job.


This status applies to branch codex/compact-pipeline-20260925. It includes source through 90ead4f plus the corrections in the [latest rules and score review](RULES_AND_MAXIMUM_SCORE_REVIEW_2026-09-25.md). Active jobs were preserved.

Implemented: explicit UTF-8 conversion, natural DuckDB blocking, supervised transliteration with hold exclusion, 53 pair and 17 context features, LightGBM/XGBoost, cached feature reuse, exact per-reference F0.5, production-consistent evaluation/global ownership, tune-only threshold selection, exact first-model candidate export and chained manifests.

Latest corrections: first-model candidate accounting after cascade rejection; ID-independent tied rank features; fixed-size post-lexicon skeleton batches; GPU prediction batching and CPU allocation retry; actual fitted-device metadata; CPU defaults and explicit expansion of stage-2 fitting; all-reference candidate averages; a tuning-only candidate-efficiency diagnostic. A smaller stage-2 workload is not a smaller official candidate set.

Frozen v14 at de77cbc reports tuning F0.5 0.979377 and final F0.5 0.978897. Its diagnostic covers all 50,141 tuning references: 48.806 candidates/reference and oracle F0.5 0.990872. Reducing to 36.567 candidates/reference lowered the ceiling to 0.989984. These are measurements of v14 retrieval, not this newly corrected source's accuracy.

Release evidence is under artifacts/verification-release.xml and artifacts/verification.json. Coverage includes raw-input-to-package integration, cascade exports, repeatability, malformed data/stale artifacts, CPU XGBoost, simulated GPU allocation retry, bounded GPU transfers, routing ties and seed ensembles. Original manifest files are checked against D:/Amazon ML 26.

Not established: a full retrained model for this version, a 0.99+ score, France accuracy, cloud capacity, full-scale peak resources or billion-record throughput. QuantileDMatrix still assembles quantized data in memory. Full-pipeline cross-fitting is not claimed: the supervised lexicon is shared across non-holdout classifier folds.

Follow the [runbook](RUNBOOK.md). Source changes require retraining or matching saved code. No partial-stage resume or multi-account scheduler is implemented. Methodology remains a draft; this review made no leaderboard upload.

Release verification: **108 tests passed**, zero failures/errors; **38 Python files compiled**; quality passed with all **12 original manifest fingerprints** matching. Tests completed in 22.85 seconds. This establishes software behavior, not a full retrained model score.

Verification update, 26 September: **111 tests passed**, **40 Python files compiled**, quality passed and **12 original input hashes matched**. Earlier 108-test evidence below/above remains historical for 9cb1200. Current evidence is `artifacts/v12-score-review/verification.json`.
