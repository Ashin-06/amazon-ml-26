# Business entity resolution — audit implementation

Latest review: [v12 public-score diagnosis and v17 measurements](../../docs/V12_SCORE_INVESTIGATION_2026-09-26.md). Training saves `hold_decisions.parquet` and `hold_error_rows.parquet`, with original tune/final membership and exact global ownership. `metrics.json.error_diagnostics` separates retrieval, later missed matches and false-positive cost by country at the selected threshold. The packager and ensemble application reject candidate manifests that hide first-model inputs.


This checkout contains the corrected fast pipeline and the older SQLite baseline. See [the review](../../docs/FINAL_HARDENING_REVIEW_2026-09-25.md), [runbook](../../docs/RUNBOOK.md) and [function reference](../../docs/FUNCTION_REFERENCE.md).

The fast model uses 53 pair features and 17 context/cluster features. Two first-stage classifiers are cross-fitted; the second stage uses their probabilities and competition context. The learned lexicon excludes held references. Thresholds use tuning macro F0.5, then final holdout is evaluated once with production scoring over the complete candidate graph.

Candidate export follows the organizer's clarified cascade rule: include every input to the first scoring model, even when stage 1 rejects it before stage 2. Rejected pairs have a cascade decision score of zero and stage2_scored=false. They still count toward candidate efficiency. Only pruning before model scoring can reduce the reported candidate set. See [the updated rules and measured frontier](../../docs/RULES_AND_MAXIMUM_SCORE_REVIEW_2026-09-25.md).

Both fitting stages use the --n-train reference sample by default; --stage2-all-references explicitly expands stage 2. Full-graph context and evaluation still process all supplied candidates. Cached XGBoost batches reduce float-matrix duplication, but QuantileDMatrix keeps the combined quantized data in memory. No full-data fit is guaranteed by the batch iterator. GPU prediction transfers at most 65,536 rows at a time and retries allocation failures on CPU.

| Module | Role |
|---|---|
| to_parquet.py | Explicit UTF-8 conversion, empty strings and original fingerprints |
| prep.py / lexicon.py | Normalized/original name views, addresses and training-only token mappings |
| block.py | Natural name/address/composite candidates |
| model.py | Features, weights, LightGBM/XGBoost, threshold selection and output |
| validation.py | Record/label/endpoint/country/ownership checks |
| integrity.py | Code, data, environment, model and output manifests |
| efficiency.py | Read-only tuning frontier: candidates/reference, percentiles, recall, oracle F0.5 and reduction ratio |
| run_stage1.py / run_stage3.py | Supported drivers |

Install requirements-dev.txt into an isolated Python 3.12 environment. CPU LightGBM is the default. XGBoost 3.0.0 is pinned; select backend xgb/auto explicitly when desired. A Kaggle GPU request alone does not select a model backend.

Test from the workspace root:

    python -m pytest code/business_entity_resolution/tests -q

Use fresh versioned directories under artifacts/. Source changes require retraining or matching saved source. Completed blocking can be verified/reused only with identical source, inputs, environment, lexicon and settings. Partial jobs do not resume. Existing model/output destinations are never overwritten.

Final packaging requires the selected run, audited original-input manifest, completed methodology and new destination. The archive includes both exact output TSVs, selected model and allowlisted source; creating it does not submit it.

The older src/run.py SQLite baseline uses a distinct artifact format. Its supported commands remain in the legacy section of the runbook. Historical full-run results are not measurements of this corrected audit version.
