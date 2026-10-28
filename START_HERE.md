# Amazon ML Challenge 2026: start here

**Final result:** submission v30 scored **0.984718** public macro F0.5 (244th global rank); held-out India/US macro F0.5 0.9896. The complete final pipeline, with reproduction instructions, is in [`submissions/Cardinality_submission/`](submissions/Cardinality_submission/), and the method is described in [the methodology](Documentation_template.md). The sections below describe earlier milestones and are kept for reference.

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

The root [methodology](Documentation_template.md) describes the final approach.