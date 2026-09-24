# Amazon ML Challenge 2026: start here

The task is **business entity resolution** using noisy business names and addresses. It is not the earlier product-price or image-extraction challenge.

Read in this order:

1. [Detailed implementation guide](docs/IMPLEMENTATION_GUIDE.md): rules, exact scoring, validation, retrieval, model choices, compute, milestones and final packaging.
2. [Measured dataset report](docs/DATA_RESEARCH_REPORT.md): full-file statistics, quality checks, multilingual examples and limits of what is known.
3. [Function-by-function reference](docs/FUNCTION_REFERENCE.md): actual implemented functions, signatures and docstrings, generated from source.
4. [Implementation status](docs/IMPLEMENTATION_STATUS.md): what exists, what was tested and what remains planned.
5. [Runbook](docs/RUNBOOK.md) and [code README](code/business_entity_resolution/README.md): exact commands for indexing, training, prediction, validation and packaging.
6. [Quality report](docs/QUALITY_REPORT.md): verification evidence and remaining limits.

Key facts: **24,229,173 source records**, **7,638,365 labeled matches**, **1,732,544 required test output rows**, and **259,452 French test references with no French training labels**. The score is macro F0.5 per reference, including correct no-match predictions.

The guides were written first. A complete CPU baseline now exists and has passed synthetic training, inference, validation and packaging tests. Full challenge training, full-test inference and cloud execution have not begun. The next operational step is a measured retrieval/indexing pilot against realistic target pools.

The archive contains `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file must match the exact pairs scored by the final model. No full-challenge prediction file exists yet. Original source files, PDFs and the supplied validator are protected by a recorded hash manifest and were preserved.

The root [methodology draft](Documentation_template.md) covers the actual approach. It is deliberately marked as a draft until team details and selected full-run results are available; packaging rejects unfinished drafts.
