<!-- DRAFT: NOT FOR SUBMISSION -->
# ML Challenge 2026: Business Entity Resolution

**Team Name:** [Required: actual registered team name]
**Team Members:** [Required: registered team members]
**Submission Date:** [Required: actual submission date]

This draft describes the implemented baseline and measured dataset. Replace this notice and complete the selected run's results before submission. A successful synthetic integration test is not a competition validation score.

## 1. Executive Summary

Our baseline retrieves plausible Source 2/3 records independently by business name and address, then scores the union with a supervised LightGBM classifier. A threshold selected using per-reference macro F0.5 supports multiple matches and explicit no-match predictions. The entire pipeline operates locally on supplied records, without external business lookup or geocoding.

## 2. Methodology

### 2.1 Problem Analysis

The provided files contain 24,229,173 source records, including 2,206,821 labeled training references and 1,732,544 test references. There are 7,638,365 positive links and 123,247 training singletons. France appears only in test data, contributing 259,452 reference records. Noise includes spelling/format variations, cross-script names, changed names, reordered addresses and missing target addresses.

### 2.2 Solution Strategy

Approach: blocking plus a supervised pair classifier. Original text is retained alongside Unicode-preserving normalized and Latin-accent-folded views. S1 groups are assigned deterministically to train/tune/holdout splits. Query sampling is bounded and reproducible; the complete supplied target pool remains available as distractors. No known positive is inserted into evaluation candidates.

## 3. Candidate Generation

A local SQLite FTS5 index supports independent name and address searches within compatible country/source groups, including blank-country fallback. Each route/source returns a configured top-K and their union is deduplicated before model scoring. The final candidate TSV contains exactly the pairs passed to the final matcher.

Selected-run candidate count, top-K, recall, oracle score and reduction ratio must be copied from the actual run's manifests/metrics. They have not yet been measured on a full challenge run. Word-token retrieval can miss pairs with no shared usable tokens; character/dense improvements are not claimed in this baseline.

## 4. Matching Model

The baseline uses 29 numeric features: name/address nonempty equality, edit and token similarities, Latin-folded similarities, length ratios, digit-bearing token overlap/conflict, missingness, country compatibility, source, script markers and route ranks. The model is a CPU LightGBM binary classifier trained on naturally retrieved positives and negatives. Fixed-round settings are recorded in `config.json`.

Thresholds maximize tune-set entity-macro F0.5. Every reference contributes equally, including zero-candidate entities and singletons. Holdout labels are used only to report performance. No one-to-one matching or compulsory match is imposed.

## 5. Results and Error Analysis

Full-pool trained-model validation and leaderboard results are **not yet available**. Before final submission, record the selected run ID, threshold, tuning/holdout macro F0.5, per-country results, singleton accuracy, candidate recall, runtime and common false merges/missed links from `metrics.json` and `manifest.json`.

Completed software checks cover scoring edge cases, Unicode handling, invalid/duplicate IDs, candidate containment, synthetic end-to-end training/inference, package reproduction and the supplied validator. Full-label scorer identity checks are correctness tests only. No accuracy figure is claimed for unlabeled France.

## 6. Conclusion

The implemented baseline separates scalable candidate retrieval from match decisions and makes no-match handling explicit. Final conclusions about performance must be based on the selected challenge run and its error analysis, not the synthetic test suite.

## Appendix A. Code Artifacts and Reproduction

The final ZIP contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`, source and pinned requirements under `code/business_entity_resolution/`, saved model/config/manifests, this completed document, and dependency notices. The package README gives exact `index`, `train`, `predict`, `score`, `validate` and `package` commands. Strict validation and original test-input fingerprints are required before packaging.

The original pipeline is MIT licensed. No pretrained neural weights are bundled. Model inputs, source versions, dependency versions and output fingerprints are recorded in the run artifacts.
