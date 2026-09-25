# Amazon ML Challenge 2026: start here

The task is business entity resolution. There are 24,229,173 source records, 7,638,365 labeled links and 1,732,544 required test reference rows, including 259,452 French references without French training labels.

Read [the latest rules and maximum-score review](docs/RULES_AND_MAXIMUM_SCORE_REVIEW_2026-09-25.md), [status](docs/IMPLEMENTATION_STATUS.md) and [runbook](docs/RUNBOOK.md). The [implementation guide](docs/IMPLEMENTATION_GUIDE.md) is the broader design; the [function reference](docs/FUNCTION_REFERENCE.md) is generated from current source.

Historical full runs exist. The frozen v14 run reports local final F0.5 0.9788967. This isolated checkout incorporates later experiments and fixes the clarified first-model candidate export, resource handling and ID-independent rank features. It requires fresh derived data and retraining. It has not replaced active jobs in the main workspace.

The requested objective is macro F0.5 above 0.99 with efficient candidates. This has not been achieved. The historical v14 retrieval ceiling is 0.990872 on all 50,141 tuning references, so naive candidate cuts are unsuitable. Keep the [methodology](Documentation_template.md) as a draft until real team details and selected results are available.
