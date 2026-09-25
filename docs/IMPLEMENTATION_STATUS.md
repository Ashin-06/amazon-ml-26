# Implementation status — 25 September 2026

This status applies to the isolated audit checkout. Main-workspace jobs and source were preserved.

Implemented: explicit UTF-8 conversion, natural DuckDB blocking, transliteration with hold exclusion, 52 pair and 17 context features, reference weighting, LightGBM/XGBoost, production-consistent global evaluation, tune-only thresholds, exact F0.5, one-owner output, full data gates and chained completion manifests.

Verified locally: synthetic raw-input-to-package integration, identical repeated TSVs across chunk counts, malformed data/stale artifact rejection, CPU XGBoost roundtrip, GPU-fallback detection and offline kernel generation. All 12 original manifest hashes match. Final evidence is in artifacts/audit-hardening-20260925; see [the review](FINAL_HARDENING_REVIEW_2026-09-25.md).

Historical v10 reports local F0.5 0.9773699030, candidate recall 0.9704723458 and oracle 0.9905325893. The release log reports validator/checker PASS, without a public score. These are earlier-code results, not measurements of this implementation.

Not established: corrected full-data accuracy/runtime, actual GPU execution of this version, current Kaggle capacity, France accuracy or superiority to 0.980473. Next is a bounded measured real-data pilot and tuning-only comparisons of weights, cluster features and boosters.

No partial-stage resume or multi-account scheduler is implemented. Methodology remains a draft. Follow the [runbook](RUNBOOK.md), upload limit and deadline.

v15 (branch fix/v15): France region mapping scoped to country = France; phonetic skeleton built in bounded batches; target-name noise cleaning measured on the supplied data (OCR digits 0/1/5/6/8 inside words in 2.6% of target names vs ~0% of reference names, "(ID: n)" tags, "| www.x.com" tails, M/s/Sri/Shri/Smt/Mr/Dr prefixes, "fka"/"formerly known as" aliases); names that are only a glued domain (6.7% of target names end in a TLD vs 1.7% of references) are split with the split's own reference-name vocabulary; svc/ctr/partners are core-name filler; house-number delta features (log |a-b|, one-digit substitution). On 60k v10 holdout true pairs, exact core-name agreement rose 55.7% -> 66.1% (hard negatives 50.7% -> 54.6%). Training now featurizes every candidate once: stage-1 matrices are cached in a run-owned scratch directory and reused for the production-scoring evaluation pass (identical scores, verified by a regression test), removing one full re-featurization of all training pairs. Boosters read the cached matrices block by block (XGBoost quantized streaming, ~1 byte per value; tested equal to in-memory training), so --n-train above the number of non-holdout references trains on all ~2.1M of them. Full-data v15 results pending (work/v15, same settings as v14).

Error floor (v10 holdout): exact-name matches to empty-address targets are ambiguous when several references share the name (true-link rate 0.98 when the name is unique, 0.54 / 0.30 / 0.15 / 0.04 when 2 / 3-5 / 6-20 / >20 references share it), so part of the 0.9905 oracle is not reachable by any matcher.

Final regression result: 97 tests passed; zero failures/errors.
