# Amazon ML Challenge 2026: start here

**Follow-up experiments:** five additional variants tested; none passed adoption. v28 remains unchanged, with one upload unused. [Results and error budget](docs/V28_FOLLOWUP_EXPERIMENTS_2026-09-27.md).

**Latest fast-model release:** v28fast_rescue is prepared and independently validated, not submitted. Local India/US final score **0.989583** versus **0.989350**; CPU pilot 22.6 seconds. France remains identical to v27 and unlabelled. Public best stays **v26: 0.98417**. **One upload remains.** Read the [fast rescue report](docs/FAST_RESCUE_TRAINING_2026-09-27.md) for files, checks, reproduction and limitations. Older status paragraphs below are historical.

**Latest: 27 September evening.** User-confirmed best: **v26, 0.98417**.
**One upload remains.** v27 passed a fresh full independent output/candidate
check; its French accuracy is unlabelled. The completed xenc7 experiment failed
adoption. See the [evening review](docs/EVENING_098417_FINAL_REVIEW_2026-09-27.md) for exact results and new
organizer rules: 17 total attempts; TSV tonight, ZIP in Round 2. Earlier status
sections below are historical and superseded by this update.

**Current run status (27 September, afternoon):** read
[post-0.982910 blueprint review and live experiment](docs/POST_0982910_BLUEPRINT_REVIEW_2026-09-27.md)
for the latest public result, two remaining uploads and the private Kaggle xenc7
experiment. No 0.995 score has been demonstrated. Also read
[v24 correction and the 0.995 target](docs/V24_CORRECTION_AND_0995_AUDIT_2026-09-27.md)
first. v22 and v24 training are complete. A separate v24 correction model scores
0.988804 on final validation; its release record is in the linked audit. The
last confirmed public score is **0.982910 for v24stack_t077_fr0925**. The
baseline milestone descriptions below are historical; they do not describe the
current frozen production runs. Use each run's saved source and manifests.

The task is **business entity resolution** using noisy business names and addresses. It is not the earlier product-price or image-extraction challenge.

Read in this order:

1. [Detailed implementation guide](docs/IMPLEMENTATION_GUIDE.md): rules, exact scoring, validation, retrieval, model choices, compute, milestones and final packaging.
2. [Measured dataset report](docs/DATA_RESEARCH_REPORT.md): full-file statistics, quality checks, multilingual examples and limits of what is known.
3. [Function-by-function reference](docs/FUNCTION_REFERENCE.md): actual implemented functions, signatures and docstrings, generated from source.
4. [Implementation status](docs/IMPLEMENTATION_STATUS.md): what exists, what was tested and what remains planned.
5. [Runbook](docs/RUNBOOK.md) and [code README](code/business_entity_resolution/README.md): exact commands for indexing, training, prediction, validation and packaging.
6. [Quality report](docs/QUALITY_REPORT.md): verification evidence and remaining limits.

Key facts: **24,229,173 source records**, **7,638,365 labeled matches**, **1,732,544 required test output rows**, and **259,452 French test references with no French training labels**. The score is macro F0.5 per reference, including correct no-match predictions.

The guides were written before the initial CPU baseline. Its synthetic training,
inference, validation and packaging checks are historical verification. Full-data
training, inference and cloud runs have since occurred; the current status link
above records the selected versions and their measured limitations.

The final archive requires `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Full-test prediction files now exist under `submissions/`. The latest official
candidate-scope clarifications conflict; the current audit preserves sealed
first-stage input candidates and records the conflict. Original source files,
PDFs and the supplied validator remain protected by the recorded hash manifest.

The root [methodology draft](Documentation_template.md) covers the actual approach. It is deliberately marked as a draft until team details and selected full-run results are available; packaging rejects unfinished drafts.
