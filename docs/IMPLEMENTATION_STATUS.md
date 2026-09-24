# Implementation status

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
