# Business entity resolution — audit implementation

This checkout contains the corrected fast pipeline and the older SQLite baseline. See [the review](../../docs/FINAL_HARDENING_REVIEW_2026-09-25.md), [runbook](../../docs/RUNBOOK.md) and [function reference](../../docs/FUNCTION_REFERENCE.md).

The fast model uses 50 pair features and 17 context/cluster features. Two first-stage classifiers are cross-fitted; the second stage uses their probabilities and competition context. The learned lexicon excludes held references. Thresholds use tuning macro F0.5, then final holdout is evaluated once with production scoring over the complete candidate graph.

| Module | Role |
|---|---|
| to_parquet.py | Explicit UTF-8 conversion, empty strings and original fingerprints |
| prep.py / lexicon.py | Normalized/original name views, addresses and training-only token mappings |
| block.py | Natural name/address/composite candidates |
| model.py | Features, weights, LightGBM/XGBoost, threshold selection and output |
| validation.py | Record/label/endpoint/country/ownership checks |
| integrity.py | Code, data, environment, model and output manifests |
| run_stage1.py / run_stage3.py | Supported drivers |

Install requirements-dev.txt into an isolated Python 3.12 environment. CPU LightGBM is the default. XGBoost 3.0.0 is pinned; select backend xgb/auto explicitly when desired. A Kaggle GPU request alone does not select a model backend.

Test from the workspace root:

    python -m pytest code/business_entity_resolution/tests -q

Use fresh versioned directories under artifacts/. Source changes require retraining or matching saved source. Completed blocking can be verified/reused only with identical source, inputs, environment, lexicon and settings. Partial jobs do not resume. Existing model/output destinations are never overwritten.

Final packaging requires the selected run, audited original-input manifest, completed methodology and new destination. The archive includes both exact output TSVs, selected model and allowlisted source; creating it does not submit it.

The older src/run.py SQLite baseline uses a distinct artifact format. Its supported commands remain in the legacy section of the runbook. Historical full-run results are not measurements of this corrected audit version.
