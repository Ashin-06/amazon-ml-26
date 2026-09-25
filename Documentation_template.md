<!-- DRAFT: fill actual team details and selected-run measurements before submission -->
# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]

**Team Members:** [List all team members]

**Submission Date:** [Date]

## 1. Executive summary

We use country-aware name/address/composite blocking, a two-stage boosted-tree matching cascade and global target ownership. The pipeline uses only supplied records and permitted offline algorithm libraries. The selected model and final measurements are pending; this document is not ready for submission.

## 2. Data and validation

Sources are parsed as explicit UTF-8 TSVs with string IDs. Original text, empty lists, singletons and every country including France are preserved. IDs support joining, stable splits and deterministic routing, never predictive features. Tied evidence receives identical model rank features.

S1 reference groups are separated into fitting, tuning and final evaluation groups with a stable rule. The supervised lexicon excludes all hold references. Classifier cross-fitting is implemented, but the shared non-hold lexicon means this is not complete pipeline cross-fitting. Thresholds use tuning data. Production-consistent global ownership precedes one final evaluation per run. Previously inspected final scores cannot be described as permanently untouched.

The metric is exact per-reference macro F0.5, with empty/empty equal to one and every reference in the denominator. France has no labeled training data; output counts do not establish its accuracy.

## 3. Blocking and candidate contract

Country-namespaced keys include name tokens, adjacent/whole-name concatenations, address tokens/bigrams and name-address composites. Common postings are capped. The baseline union keeps total-score, name-score and address-score routes. Budgets must be selected using retrieval quality and candidate cost together.

Following the organizer's cascade clarification, candidate_pairs.tsv contains every pair fed to the FIRST scoring model. Stage-1 rejection can reduce stage-2 work but does not reduce the official candidate count. Every final match belongs to that complete set. Text-identical matching target IDs are not collapsed.

Report the selected run's total pairs, mean/p95/p99/maximum candidates per reference, zero-candidate references, same-country reduction ratio, candidate recall and oracle macro F0.5. Oracle is a labeled-data upper bound, not a model score.

## 4. Models and features

The implementation supports LightGBM (MIT) and XGBoost (Apache-2.0). CPU LightGBM is the default; the final selected backend is pending. There are 53 pair features and 17 context features covering string/phonetic similarities, numeric/address evidence, missingness, frequency, blocking evidence, within-reference support and competing-reference context.

Stage 1 uses disjoint fitting folds. Stage 2 consumes cross-fitted probabilities and context. Early-rejected pairs receive a cascade decision score of zero and remain exported. Each target goes to its highest-scoring eligible reference; a reference may receive zero or many target IDs.

Both fitting stages use the requested sample unless expansion is explicit. Cached features reduce duplicate work. Post-lexicon skeletons use 100,000-row batches; GPU prediction transfers at most 65,536 rows. QuantileDMatrix retains quantized data in memory, so full-data resources require measurement.

## 5. Results to replace with the selected run

Historical frozen v14 at de77cbc: tuning F0.5 0.979377; final F0.5 0.978897. Its 50,141 tuning references have mean 48.806 candidates and oracle F0.5 0.990872. These are not results of the latest source. A 0.99+ score has not been established.

Selected-run code/model fingerprints, full test generation, official validator results, public score and actual resource measurements: **pending**. Do not substitute synthetic tests, retrieval oracles or older scores.

## 6. Reproduction and limitations

The README/runbook define fresh-run commands. Inputs, source, dependencies, derived stages, model and outputs are fingerprinted. Outputs are not overwritten; partial-stage resume is unsupported. Both TSVs are validated for exact S1 coverage, IDs, countries, duplicates and containment. Archives use an explicit allowlist and exclude credentials/account settings.

No external business lookup, geocoding, hosted inference API or external dataset is used. Validation includes competitors from fitted references, and full-pipeline cross-fitting is not claimed. Hidden-test performance, France accuracy and new-version full-scale resources remain unmeasured. Complete this draft only after selecting and verifying the final run.
