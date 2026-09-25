# Implementation status — 25 September 2026

This status applies to the isolated audit checkout. Main-workspace jobs and source were preserved.

Implemented: explicit UTF-8 conversion, natural DuckDB blocking, transliteration with hold exclusion, 50 pair and 17 context features, reference weighting, LightGBM/XGBoost, production-consistent global evaluation, tune-only thresholds, exact F0.5, one-owner output, full data gates and chained completion manifests.

Verified locally: synthetic raw-input-to-package integration, identical repeated TSVs across chunk counts, malformed data/stale artifact rejection, CPU XGBoost roundtrip, GPU-fallback detection and offline kernel generation. All 12 original manifest hashes match. Final evidence is in artifacts/audit-hardening-20260925; see [the review](FINAL_HARDENING_REVIEW_2026-09-25.md).

Historical v10 reports local F0.5 0.9773699030, candidate recall 0.9704723458 and oracle 0.9905325893. The release log reports validator/checker PASS, without a public score. These are earlier-code results, not measurements of this implementation.

Not established: corrected full-data accuracy/runtime, actual GPU execution of this version, current Kaggle capacity, France accuracy or superiority to 0.980473. Next is a bounded measured real-data pilot and tuning-only comparisons of weights, cluster features and boosters.

No partial-stage resume or multi-account scheduler is implemented. Methodology remains a draft. Follow the [runbook](RUNBOOK.md), upload limit and deadline.

Final regression result: 91 tests passed; 17.86 seconds; zero failures/errors.
