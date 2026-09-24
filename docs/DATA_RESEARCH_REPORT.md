# Data research report

Audit date: 25 September 2026. Dataset root: `6ab10eb3b23ba_student_resource/student_resource/dataset/` relative to the workspace.

The full scan covered every row in all seven TSV files. [data_audit.json](research/data_audit.json) contains the machine-readable measurements. [input_manifest.json](research/input_manifest.json) records SHA-256 hashes for the seven TSVs, two PDFs and three supplied code/documentation files. Original inputs were not edited.

## 1. Files and row counts

Sizes below are decimal MB. Source records across train and test total **24,229,173**. The seven TSVs total **2,520,573,701 bytes**, or approximately 2.52 GB / 2.35 GiB. Ground-truth rows are not additional business records.

| File | Rows | Size MB | US | India | France |
|---|---:|---:|---:|---:|---:|
| train_source1.tsv | 2,206,821 | 210.07 | 1,323,633 | 883,188 | 0 |
| train_source2.tsv | 5,034,616 | 489.30 | 3,016,817 | 2,017,799 | 0 |
| train_source3.tsv | 5,285,603 | 503.71 | 3,170,056 | 2,115,547 | 0 |
| train_ground_truth.tsv | 2,206,821 | 127.02 | See S1 | See S1 | 0 |
| test_source1.tsv | 1,732,544 | 175.02 | 663,106 | 809,986 | 259,452 |
| test_source2.tsv | 4,887,273 | 509.46 | 1,871,330 | 2,312,565 | 703,378 |
| test_source3.tsv | 5,082,316 | 506.00 | 1,945,701 | 2,405,000 | 731,615 |

Country distribution changes materially: US is about 59.98% of training references but only 38.27% of test references. France is 14.98% of test references. A validation score dominated by US performance can conceal a weak test result.

## 2. Schema and quality

All source files have exactly `entity_id`, `business_name`, `business_address`, `country`. Ground truth has `source1_entity_id`, `matched_entity_ids`.

The scan found:

- No malformed source-row column counts and no duplicate entity IDs within any source file.
- No duplicate ground-truth S1 rows and no repeated ID inside a ground-truth list.
- Exactly one ground-truth row for every training S1 record.
- Every labeled target exists in the corresponding training source.
- No target is linked to multiple reference businesses in the training labels.
- No training positive links records with different country labels.
- No same-source entity IDs overlap between training and test.
- No blank business names in the scanned source files.

An ID-disjoint train/test split does not prove that all underlying names/addresses are text-disjoint. Full duplicate-text and near-duplicate overlap audits have not yet been performed. IDs are retained as strings by the implemented pipeline even though compact numeric arrays were used internally for this read-only audit.

| File | Missing addresses | Missing names |
|---|---:|---:|
| train_source1.tsv | 0 | 0 |
| train_source2.tsv | 168,967 | 0 |
| train_source3.tsv | 175,916 | 0 |
| test_source1.tsv | 0 | 0 |
| test_source2.tsv | 129,408 | 0 |
| test_source3.tsv | 136,098 | 0 |

Interpretation: reference records are comparatively complete; target records are noisier. Address missingness must be an explicit feature. Name-only fallbacks remain necessary despite strong address retrieval.

## 3. Ground truth

There are **7,638,365 positive links**. The mean is approximately 3.46 links per reference, the median is 3, the 95th percentile is 6, and the 99th percentile is 8. The observed maximum is 11.

| True matches for one S1 | Number of S1 businesses |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

US has 73,896 singletons out of 1,323,633 references (5.5828%). India has 49,351 out of 883,188 (5.5878%). The similar proportions support stratified diagnostics, but do not establish the singleton proportion in unlabeled France.

Source 2 supplies 3,693,619 labeled target records and contains another 1,340,997 unmatched targets. Source 3 supplies 3,944,746 labeled targets and another 1,340,857 unmatched targets. In total, **2,681,854 training targets (25.99%)** have no link to any training S1. These are genuine retrieval distractors and should not be removed from validation indexes.

## 4. Text and multilingual noise

Non-ASCII is a character-property measurement, not a definitive language/script classifier. It includes accented Latin text, Indic scripts and punctuation.

| Source/country | Non-ASCII names | Non-ASCII addresses |
|---|---:|---:|
| train S2 / India | 562,437 | 478,396 |
| train S3 / India | 391,172 | 476,536 |
| test S2 / India | 638,614 | 551,461 |
| test S3 / India | 437,903 | 551,274 |
| test S1 / France | 40,789 | 73,335 |
| test S2 / France | 172,617 | 169,163 |
| test S3 / France | 175,185 | 177,914 |

Training S1 names are ASCII, while many true target names use Indic scripts. French test references introduce accented names even on the reference side. A destructive ASCII-only transformation would be particularly harmful.

Name lengths average approximately 24–26 characters. Address lengths average approximately 46–57 characters. The longest observed source name is 123 characters and the longest address is 269 characters. This supports a short-record representation, but tokenizer token lengths still need measurement before fixing an encoder limit.

## 5. Sampled positive-pair diagnostics

Separately from the full scan, we selected 2,165 S1 entities by a deterministic numeric-ID modulo rule solely for exploratory diagnostics. This produced **7,540 labeled positive pairs**. It is not a validation split, and numeric IDs will not be predictive features. See [positive_pair_diagnostics.json](research/positive_pair_diagnostics.json).

Normalization for this diagnostic used NFKC/casefold, punctuation separation and whitespace collapse, while preserving Unicode letters and marks. It did not apply legal-suffix removal or street-abbreviation expansion.

| Diagnostic | US positives (4,503) | India positives (3,037) |
|---|---:|---:|
| Exact normalized name | 1,185 | 493 |
| Exact normalized address | 399 | 229 |
| Both normalized fields differ | 3,003 | 2,342 |
| Target address missing | 212 | 103 |
| Median name edit similarity | 0.9000 | 0.8163 |
| 5th-percentile name edit similarity | 0.5946 | 0.0889 |
| Median address edit similarity | 0.8283 | 0.8108 |

These are distributions of true pairs only. They cannot establish precision, recall of a retrieval system, a threshold, or a model score. Address similarity includes zero for a missing address.

Observed patterns include English-to-Bengali and English-to-Devanagari name variants, state abbreviation changes, reordered address components, missing address parts and entirely different business names with matching address evidence. For example, the labeled pair `S1-622358007` / `S3-141652453` has low name similarity (0.25) but an identical normalized address. This is a local supplied-data example; no external identity verification was performed.

Consequences:

1. Retrieve by address independently of name.
2. Preserve cross-script information and test multilingual features after the lexical baseline.
3. Never require every true pair to have high name similarity.
4. Do not accept an exact address blindly: shared premises can still create false merges, which must be measured using negatives.

## 6. Scale calculations

The full train comparison space is 22,774,876,013,799 pairs. The full test comparison space is 17,272,751,604,416 pairs. Country restriction reduces test comparisons to 6,724,569,566,212, which is still infeasible for exhaustive pair classification.

At 40 candidates per test S1, there would be 69,301,760 candidate pairs. Fifty `float32` features for that many pairs alone would occupy approximately 13.86 GB decimal, excluding IDs, indexes, models and temporary copies. Feature calculation and prediction must be streamed.

Keeping one 384-dimensional FP16 embedding per test source row would consume about 8.37 GiB. This is a storage calculation, not a benchmark. GPU throughput, current Kaggle quotas, available RAM and remote authentication have not been checked in this task.

## 7. What is still unknown

- Candidate recall and speed against the full target pool.
- The best model family, thresholds, validation score or leaderboard score.
- Which normalization changes improve real positive/negative separation.
- Full text-level overlap and common-name/shared-address frequency distributions.
- Actual encoder runtime and whether its additional recall justifies GPU use.
- The private leaderboard distribution and the accuracy on France.
- Portal file-size constraints, current team/account compute availability and the team's identifying details.

These are next-stage measurements. None should be filled with guessed results.
