# Quality and completeness report

Review date: 25 September 2026. This report covers the implemented offline baseline, its instructions and its verification. It does not certify that an unrun full-scale model is competitive or guarantee the absence of every possible defect.

## Findings corrected

| Finding | Correction |
|---|---|
| CLI scoring relied on prior uniqueness checks and identical row order | Disk-backed keyed scoring rejects duplicates/missing/extra IDs and accepts arbitrary order |
| Full target validation could exhaust memory using Python ID sets | Strict CLI validation uses on-disk indexes and a bounded SQLite cache |
| Retrieval/training/inference/package commands were missing | Added a runnable CPU baseline with explicit settings and artifact manifests |
| Documentation described current components as only planned | Updated start page, guide, status, function reference, code README and runbook together |
| Default JSON parsing could silently overwrite duplicate settings | Duplicate/unknown keys and invalid setting types/ranges now fail |
| Stale code/config/features/model files could be mixed | Added code, environment, schema, config, threshold and artifact-fingerprint checks |
| Incomplete writes could look like finished runs | New versioned destinations, staging directories and completion manifests prevent publication of incomplete results |
| Final archives could accidentally include unrelated workspace data | Explicit file allowlists and expected test-input fingerprints; no arbitrary workspace archiving |
| Methodology was an empty outline | Added a factual draft with all required sections, while leaving only actual team/run results for later completion |

## Verification evidence

- **51 automated tests** cover metric edge cases, full reference coverage, duplicates, arbitrary row order, unknown targets, candidate containment, missing fields, Unicode/Indic marks, French accents, country fallback, quoted search strings, configuration failures, bad labels, partial-index prevention and artifact tampering.
- A synthetic end-to-end run exercises indexing, training, tune-only threshold selection, holdout reporting, inference, strict validation and archive creation. Test artifacts are temporary and are not submitted or treated as challenge results.
- Repeating training in the same tested environment yields identical model/config/metrics files. Running packaged source and model from an extracted ZIP yields byte-identical synthetic prediction and candidate files.
- The synthetic outputs also pass the unmodified organizer validator with `--check-ids` in the original workspace.
- The strict disk-backed scorer processed **2,206,821** real training-label rows and **7,638,365** links. Scoring truth against itself gives the expected **1.0**. This is an identity check, not trained-model performance.
- The full scorer check measured **82.172 seconds** and approximately **62.5 MiB** peak process resident memory in this local run. This measurement is for scoring, not indexing/training/inference or a general hardware guarantee.
- The earlier all-empty check gives **0.05584820880352326**, agreeing with the independently measured singleton proportion.

Machine-readable evidence: [strict scorer check](research/strict_scorer_sanity_check.json), [original input fingerprints](research/input_manifest.json), [workspace quality results](research/workspace_quality_report.json), [test results](research/test-results.xml).

The workspace check verifies required files, local Markdown links, source parsing/docstrings, baseline configuration and all 12 recorded original input hashes. The current source inventory has 16 Python files and 57 documented functions/methods. The separate function reference is generated from those source definitions rather than maintained as a speculative list.

## Completeness

The current baseline has its code, configuration, pinned runtime/development dependencies, tests, MIT license, dependency notices, original-input manifest, generated function reference, implementation design, exact run instructions, methodology draft and quality/status documents.

The final model/output archive is generated from actual completed runs. It is not pre-created using fake predictions. The package command requires completed team fields, removal of the explicit draft marker, valid output provenance and exact audited test-input fingerprints.

## Remaining limits

Full challenge indexing, retrieval recall, model quality, full inference runtime, final outputs and leaderboard submission remain unverified. The initial retriever is word-token based and can miss difficult spelling/cross-script cases. Resumable jobs, neural features and cloud scheduling are not implemented. Test labels for France do not exist locally, so no France accuracy is claimed.

The runbook deliberately identifies these boundaries. Passing code tests is evidence for the tested contracts, not a promise of perfect model predictions or a finished competition entry.
