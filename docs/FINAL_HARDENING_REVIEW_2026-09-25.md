# Final hardening review and leaderboard plan — 25 September 2026

## Outcome and location

The reviewed defects have been addressed in an **isolated audit checkout**:
C:/Users/ashin/.codex/worktrees/audit-hardening/Amazon ML 26.

The active checkout at D:/Amazon ML 26, its jobs, models and submissions were preserved. This report does **not** certify that the active checkout has these changes. The corrected implementation requires fresh derived data and retraining; do not overwrite a running job's source or bypass its fingerprints.

The review began at commit 9dcf33d. The latest main-checkout snapshot inspected was ed103af. This audit version incorporates the intervening reference weighting, cluster-consistency features, French alias handling, XGBoost backend 64-chunk output writer, and seed/model-directory support. CPU LightGBM remains the default. The main checkout's later release/checker scripts are historical tools, separate from the verified packaging path here.

## Latest results and the leaderboard target

| Evidence | Value | Interpretation |
|---|---:|---|
| User's leaderboard snapshot: Grinders | 0.980473 | Target to exceed; not independently refreshed |
| User's snapshot: FitForce | 0.980341 | Second place in that snapshot |
| v10 saved local macro F0.5 | 0.9773699030 | Historical local result under earlier evaluation |
| v10 threshold | 0.700 | Earlier tuning decision |
| v10 configuration | 200,000 references, 1,000 rounds, 127 leaves, reference weights | Saved actual settings |
| v10 candidate link recall | 0.9704723458 | All-training natural blocking |
| v10 blocking oracle macro F0.5 | 0.9905325893 | Ideal selection from training candidates, not test accuracy |
| v7 candidate link recall | 0.9704883179 | Slightly higher than v10; progress is not monotonic |
| v4 candidate link recall | 0.9663820726 | Earlier retrieval baseline |

Local sources: D:/Amazon ML 26/work/{v4,v7,v10}/train/blocking_recall.json and work/v10/model/metrics.json. The submission log records v10's official-validator and checker PASS, but its public-score field is blank and its note says **not yet uploaded**. These full-file checks were observed in the log, not rerun during this task.

Do not treat local 0.97737 and leaderboard 0.980473 as measurements on identical examples. The earlier evaluator mixed stages for competing scores and exposed the final-holdout threshold grid. Those values remain historical. The new implementation has no real-data or leaderboard score yet.

## Corrections in the audit implementation

| Area | Defect or risk | Correction |
|---|---|---|
| Evaluation | Stage-2 hold predictions competed with stage-1 probabilities | Complete graph scored with the production pipeline; all ownership competitors use stage-2 scores |
| Final holdout | Every threshold received a final-half score; hold labels appeared in fitting logs | Tune-only threshold selection, one final evaluation, no hold labels passed to the learner |
| Competitor context | Outside-sample reference candidate lists were truncated | Complete candidate lists for reference support/context |
| Splits | General DuckDB hash can change across versions | Versioned MD5 reference rule, lexicon exclusion using the same rule, persisted membership |
| Rank ties | IDs broke equal-score ranks used as model features | Equal scores now have equal rank features; deterministic output decisions still break ties |
| Labels | Duplicate empty labels could hide missing references; unknown singletons and multiple owners escaped checks | Exact coverage including empty lists, valid endpoints, unique pairs/owners and compatible countries |
| Candidate export | Count equality did not establish pair equality; unknown endpoints could disappear in joins | Bidirectional scored/candidate set comparison and endpoint/source/country checks |
| Numerical edge cases | Zero-probability context divided by zero | Explicit zero handling and finite context checks |
| Names | Cleaning discarded original alias/phone views and could empty numeric names | Independent original-name view, numeric-name preservation, French alias regression |
| Empty sources | Header-only target files could fail at writer close | Explicit schema and valid empty parquet output |
| Extra files | A wildcard could include an unrecorded normalized file | Read only the three explicitly fingerprinted normalized files; regression confirms extras cannot change predictions |
| Cache reuse | Existing partial files and stale chunks were treated as completed | Completion manifest written last, exact chunk list, rejection of incomplete or incompatible stages |
| Provenance | Inputs, chunks, dependencies, weights and outputs lacked complete binding | SHA-256 chain from original TSV conversion through lexicon, candidates, model and predictions |
| Existing artifacts | Training/inference could overwrite outputs | New versioned destinations; overwrite rejected |
| Windows | Streaming lexicon reader remained open during file replacement | Close reader before replacing its derived file |
| GPU detection | Successful XGBoost fitting could mean silent CPU fallback | Check the actual fitted device; CPU fallback is not counted as GPU success |
| Packaging | Model optional; draft or mixed-version artifacts could pass | Mandatory selected model, original-test fingerprints, source/output/model binding, explicit allowlist, real team-field/draft checks |
| Ensembles | Equal total key counts could hide per-member duplicates/missing keys; hold-only validation differed from prediction | Verify each member separately, exact pair alignment, fixed tune/final groups across seeds, full-graph evaluation, selected-threshold binding and verified ensemble packaging |
| Kaggle | Unpinned failed installs were ignored; offline setup attempted network work | Exact-version import checks; checked installs; offline local wheels with no package-index access |
| Dependencies/docs | Missing SciPy/scikit-learn/XGBoost pins; status described only the older baseline | Pin tested versions and update README, runbook, status and generated function reference |

Classifier-level cross-fitting is implemented. **Full-pipeline cross-fitting is not claimed**: the supervised lexicon uses all non-hold training labels. All hold references are excluded from lexicon and classifier fitting. Fully nested lexicon learning remains an experiment if stacker overfitting is observed.

Global evaluation includes competitors from fitted reference groups. This matches the complete training candidate graph, but its population differs from entirely unseen test references. Document that limitation when interpreting scores.

## Verification

**Final result: 91 tests passed; 17.86 seconds; zero failures/errors.** Evidence: artifacts/audit-hardening-20260925/tests-release.xml.

The original 60 tests passed before changes. Expanded checks cover raw UTF-8 data through normalization, natural retrieval, training, prediction, independent strict validation and ZIP creation. Prediction with different chunk counts reproduces both TSVs byte for byte.

Failure regressions cover malformed/duplicate/missing labels, empty singleton labels, France, missing addresses, unknown or reversed endpoints, cross-country matches, multi-owner targets, missing scored candidates, zero context probabilities, stale inputs/chunks, partial work, tampered models/outputs and overwrite attempts. An end-to-end two-seed ensemble now passes training, production-consistent evaluation, inference, independent validation and archive creation. A regression explicitly rejects compensating duplicate pairs. Additional checks cover CPU XGBoost save/load, simulated GPU-to-CPU fallback, draft packaging, wrong original data and generated offline kernel syntax.

All **12 original manifest fingerprints** matched: seven TSVs, two PDFs and supplied resource code/documents. The official validator was not edited. The historical video is outside those 12 entries and was not rehashed.

Final machine-readable evidence is under artifacts/audit-hardening-20260925 in this checkout. Synthetic tests establish software behavior, not full-scale memory, remote quota, French accuracy or competitive model quality.

## Experiments most likely to help exceed 0.980473

1. **Measure a fresh real-data baseline.** Freeze code, environment, reference groups and candidate settings. Start with bounded measurements against the appropriate full target pool. The corrected evaluation adds a full production scoring pass; measure RAM, disk and time before scaling. DuckDB's memory setting does not cap pandas, NumPy or booster allocations.
2. **Separate retrieval and decision losses.** Measure candidate recall, oracle F0.5 and matcher F0.5 on the same tuning references. The historical ~0.99053 training oracle indicates selection headroom, not guaranteed test performance.
3. **Prioritize precision and singletons.** In exact F0.5, false positives have four times the denominator coefficient of false negatives. Compare missed links, incorrect links and singleton accuracy per reference, country, missing-address group and common-name frequency.
4. **Ablate current additions.** Compare none/ref/sqrt weighting, cluster features, rounds/learning rate and LightGBM versus XGBoost using identical tuning groups. A faster GPU run need not be a better model.
5. **Investigate multilingual/alias cases.** Original name information now survives cleaning. Additional alias blocking routes require recall/precision measurement. There are no French training labels: do not invent a France-tuned threshold or use external identity lookups.
6. **Measure the implemented seed ensemble.** Align predictions by exact reference/target keys, tune a small fixed blend grid, then perform global ownership once. Export the final scored candidate union.
7. **Control repeated holdout use.** Choose changes on tuning data; freeze the selected configuration before final evaluation. Previously exposed holdout results cannot become untouched again.
8. **Promote a complete result.** Recheck all 1,732,544 official test references, including 259,452 French references; finalize actual team details and selected methodology; create a new verified archive and record the actual public score separately.

No cloud training, paid compute or leaderboard upload was performed by this review. Maximum five leaderboard uploads daily; deadline 27 September 2026, 23:59 IST. Beating the supplied leaderboard remains an objective, not an established result.

## Primary research

- [DuckDB CSV reader](https://duckdb.org/docs/stable/data/csv/overview): explicit UTF-8 strings and empty-field preservation.
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html): deterministic CPU settings.
- [XGBoost GPU documentation](https://xgboost.readthedocs.io/en/release_3.0.0/gpu/index.html): device selection and CPU/GPU model behavior.
- [DuckDB utility functions](https://duckdb.org/docs/stable/sql/functions/utility.html): general hash versus stable digests.

No dataset business entities were queried externally.

Review cutoff: ed103af. Later concurrent changes are outside this completed review; no source in the active checkout was overwritten. The audit branch is the reviewable deliverable.
