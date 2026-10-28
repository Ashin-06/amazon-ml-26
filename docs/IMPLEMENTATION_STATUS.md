# Implementation status

**Final result:** submission v30 scored **0.984718** public macro F0.5 (244th global rank); held-out India/US macro F0.5 0.9896. The complete final pipeline, with reproduction instructions, is in [`submissions/Cardinality_submission/`](../submissions/Cardinality_submission/), and the method is described in [the methodology](../Documentation_template.md). The sections below describe earlier milestones and are kept for reference.

## Current production status — 27 September, afternoon

The user confirms **0.982910** for `v24stack_t077_fr0925`, an improvement of
0.000633 over mixed-cutoff v20x. Exactly two team submissions remain.
The earlier
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
