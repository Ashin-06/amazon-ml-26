> Historical SQLite-baseline document. Current fast-pipeline corrections and verification scope are in [the final review](FINAL_HARDENING_REVIEW_2026-09-25.md) and [runbook](RUNBOOK.md).

# Amazon ML Challenge 2026 — Approach

## Problem

Resolve each Source 1 business to every matching Source 2/3 record using the supplied names, addresses and country labels. Optimize entity-macro F0.5, including correctly empty singleton predictions. Return one row per test reference and the exact scored candidate set.

## Data & validation

The complete audit found 24,229,173 source records and 7,638,365 labeled links. All training labels reference existing records; no target has multiple labeled owners. France contributes 259,452 test reference records without training labels. Preserve Unicode and open country labels.

The baseline separates train/tune/holdout by stable S1 hashes, samples query groups with an independent deterministic hash and retains all supplied target records for retrieval. Threshold selection uses tuning groups only. Full-label metadata checks use disk-backed SQLite. No external business lookup or data augmentation is performed.

## Models & features

Implemented baseline: independent country/source-aware FTS5 name and address retrieval, unioned natural candidates, 29 numeric name/address/number/missingness/rank features, and a CPU LightGBM binary classifier. Fixed boosting rounds and a tuning macro-F0.5 threshold are stored in an immutable model manifest. No forced match or one-target-per-source restriction.

Planned experiments, conditional on measured gains: character n-gram retrieval, alias/transliteration handling, multilingual embeddings and improved calibration. They are not part of the current runnable baseline.

## Experiments

Software verification includes unit/failure tests, synthetic train-to-archive integration, byte-for-byte inference reproduction from the archive and agreement with the supplied validator. The disk-backed scorer also processed every real training label against itself, obtaining the expected identity score of 1.0. These are correctness checks, not competition model scores.

Actual training runs will write `artifacts/<model-run>/metrics.json` with natural candidate recall, oracle score, tuning/holdout macro F0.5, singleton accuracy and country slices. There is no claimed full-challenge trained-model or leaderboard score yet.

## Conclusion

The offline baseline and its command-level documentation are implemented. The next operational step is a measured retrieval/indexing pilot followed by full-pool validation. Full test prediction and final methodology completion must precede any competition submission. See [implementation status](IMPLEMENTATION_STATUS.md) and [runbook](RUNBOOK.md).

