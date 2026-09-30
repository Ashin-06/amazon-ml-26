# Implementation status

**Current upload ZIP:** `submissions/Cardinality_submission.zip`, 997.4 MB, below the portal's 1,024 MB cap. It preserves the selected v30 TSVs and replaces two oversized neural weight files with checked test-score caches. Full clean-room reproduction remains unverified. [Compact archive report](FINAL_COMPACT_ZIP_2026-09-28.md). The older 2.313 GB archive report is historical.

**Language and link audit:** the rebuilt final ZIP includes verified S1–S2/S3 link statistics and native-script/French normalization details. The selected matching and candidate outputs are unchanged. [Measurements](LANGUAGE_AND_SOURCE_LINK_AUDIT_2026-09-28.md).

**28 September package audit:** selected v30 public score **0.984718**, zero leaderboard attempts remaining. All staged models and outputs match their hashes; fresh full output/candidate check passed. Fixed saved-source line endings and rescue path/integrity defects; corrected methodology. Clean end-to-end reproduction remains unverified after a failed replay. Methodology deadline in the supplied notice: **29 September, 10:00 AM**. See [audit and remaining issues](ROUND2_PACKAGE_AUDIT_2026-09-28.md). All older status banners below are historical.

**Follow-up experiments:** five additional variants tested; none passed adoption. v28 remains unchanged, with one upload unused. [Results and error budget](V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md).

**Latest fast-model release:** v28fast_rescue is prepared and independently validated, not submitted. Local India/US final score **0.989583** versus **0.989350**; CPU pilot 22.6 seconds. France remains identical to v27 and unlabelled. Public best stays **v26: 0.98417**. **One upload remains.** Read the [fast rescue report](FAST_RESCUE_TRAINING_2026-09-27.md) for files, checks, reproduction and limitations. Older status paragraphs below are historical.

**Latest: 27 September evening.** User-confirmed best: **v26, 0.98417**.
**One upload remains.** v27 passed a fresh full independent output/candidate
check; its French accuracy is unlabelled. The completed xenc7 experiment failed
adoption. See the [evening review](EVENING_098417_FINAL_REVIEW_2026-09-27.md) for exact results and new
organizer rules: 17 total attempts; TSV tonight, ZIP in Round 2. Earlier status
sections below are historical and superseded by this update.

## Current production status — 27 September, afternoon

See [v24 correction and the 0.995 target](V24_CORRECTION_AND_0995_AUDIT_2026-09-27.md).
The user confirms **0.982910** for `v24stack_t077_fr0925`, an improvement of
0.000633 over mixed-cutoff v20x. Exactly two team submissions remain.
Read the [new blueprint review](POST_0982910_BLUEPRINT_REVIEW_2026-09-27.md)
for rejected pilots and the running private Kaggle xenc7 experiment. The earlier
0.925 file's exact score remains unknown. v22 and v24 training are complete.
The separate v24 correction model scores **0.988804** on the final split,
versus 0.988313 for its frozen reference; both labeled countries improve.
The v24 final candidate oracle is 0.995824, **not a trained-model score**.
See the linked audit's release record for test-file completion and checks.
There is no verified 0.995 submission; French accuracy remains unmeasured.

The sections below are the **25 September baseline snapshot** and its command
contracts, not a claim that current full-data training has never happened.

Updated 25 September 2026. The design and baseline contracts were written before their implementations. Use the [runbook](RUNBOOK.md) for supported commands and the [quality report](QUALITY_REPORT.md) for verification evidence.

## Implemented

| Component | Status |
|---|---|
| Full input research and original-file fingerprints | Complete for all seven TSVs, both PDFs and supplied resource code/docs; sampled video slides inspected |
| Exact scorer, Unicode views, ID/TSV readers and stable S1 grouping | Implemented and tested |
| Disk-backed, order-independent score/validate commands | Implemented; duplicate/coverage/target/subset failures tested |
| Offline candidate generation | Independent FTS5 word-token name/address searches by country and source, with unknown/blank-country support |
| Features and training | 29 numeric features; LightGBM; natural retrieved positives/negatives; disjoint train/tune/holdout S1 samples |
| Threshold selection and diagnostics | Tune macro F0.5, separate holdout results, singleton and country diagnostics, natural candidate oracle |
| Full-reference inference code | Scores every exported final candidate, preserves empty outputs, validates before publication |
| Artifact integrity and packaging | Source/config/environment/model/output fingerprints, MIT license/notices, expected original test-input checks, allowlisted ZIP |
| Instructions and methodology | Current runbook, package README, generated function reference, project instructions and explicit methodology draft |

## Verified

- **51 tests passed** in the recorded local suite. Tests include failure cases and synthetic training-to-archive integration.
- Repeated synthetic training produced identical model/config/metrics files in the tested environment.
- An extracted archive reproduced both synthetic output TSVs byte for byte using its packaged source and model.
- Synthetic outputs passed the supplied, unchanged validator with target-ID checks enabled.
- The strict scorer processed all **2,206,821** real training reference labels, accounting for **7,638,365** true links. Truth-versus-itself produced the expected 1.0 identity score, using about **62.5 MiB** sampled peak process memory in **82.172 seconds** on this run. This verifies scoring, not model accuracy.
- The prior all-empty scorer check remains **0.05584820880352326**, exactly the measured training singleton proportion.

The latest test/quality artifacts are under `docs/research/`; see the quality report for their scope.

## Not yet run on the full challenge

Full training/test index construction, full-pool retrieval benchmarks, a real challenge-trained model, full-test inference and a final competition archive have not been completed. There is no claimed challenge holdout/leaderboard score and no claimed France accuracy. Synthetic models and archives are temporary QA artifacts only.

No cloud upload, account verification, GPU job, paid compute or leaderboard submission was performed. Current Kaggle allocation/authentication and unattended job behavior remain unverified.

## Optional extensions not implemented

Character n-gram retrieval, embeddings, learned aliases, probability calibration, graph/ownership refinement, resumable checkpoints, automatic hyperparameter search and distributed/multi-account scheduling. Their absence does not prevent the current CPU baseline from running. They must not be described as implemented merely because the broader design discusses them.

## Next operational milestone

Measure retrieval/indexing throughput and candidate recall against realistic target pools, then train and evaluate the first actual challenge baseline. Only promote changes supported by held-out results. Finish the methodology with actual team details and selected-run results before packaging; the draft is intentionally rejected by the package command.
