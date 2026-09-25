# Amazon ML Challenge 2026: start here

The task is business entity resolution. There are 24,229,173 source records, 7,638,365 labeled links and 1,732,544 required test reference rows, including 259,452 French references without French training labels.

Read [the final review](docs/FINAL_HARDENING_REVIEW_2026-09-25.md), [status](docs/IMPLEMENTATION_STATUS.md) and [runbook](docs/RUNBOOK.md). The [implementation guide](docs/IMPLEMENTATION_GUIDE.md) is the broader design; the [function reference](docs/FUNCTION_REFERENCE.md) is generated from current source.

Historical full runs exist. v10 reports local F0.5 0.97737 under earlier evaluation, with no public score in the inspected log. This isolated checkout fixes evaluation and integrity issues and requires fresh derived data and retraining. It has not replaced active jobs in the main workspace.

The supplied leaderboard target is greater than 0.980473. Software verification does not establish superiority. Keep the [methodology](Documentation_template.md) as a draft until real team details and selected results are available.
