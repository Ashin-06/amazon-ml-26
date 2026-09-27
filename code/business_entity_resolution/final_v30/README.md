# Final submission v30 (public 0.984718): scripts

The scripts that produced `v30_frxe2_matching_results.tsv` from the frozen v24 pipeline outputs (see ../README.md
and docs/RUNBOOK.md for v24). Candidates: `work/comp/v25/candidate_pairs_v27.tsv` (181,275,851 pairs).

| Step | Script | What it does |
|---|---|---|
| India/US correction | `india_us/v25.py` | LightGBM 600-tree correction on the uncertain band, tune-selected cutoff 0.74 |
| India/US rescue route | `india_us/rescue.py`, `india_us/rescue_apply.py` | same first core word + house number; name ratio + address token-set >= 1.88 |
| India/US learned rescue | `fast_rescue/` | logistic rescue matcher (C=0.1, cutoff 0.69) from the parallel agent (v28fast_rescue) |
| France rules | `france/fr_overlay.py` | F1-F5 French patterns with India/US analog precision 87-99.9% |
| France translation fix | `france/xe_french_forms_v2.py`, `france/fr_v2_scores.py`, `france/fr_v2_base.py` | French suffixes mapped to true-match/decoy English words by measured co-location; xenc4 rescoring; same blend |
| Merge | `france/union_additions.py` | adds new links for unowned targets only, never removes |
| Checks | `checks/check_candidates.py`, `checks/owner_check.py`, ... | streaming candidate validation, ownership audit, and the rejected experiments (reproducible) |

Evidence and numbers: `docs/FRANCE_OVERLAY_2026-09-27.md`, `docs/FAST_RESCUE_TRAINING_2026-09-27.md`,
`docs/RESEARCH_NEXT_IDEAS_2026-09-27.md`, `Documentation_template.md`. Paths inside the scripts refer to the local
workspace layout (`work/`, `artifacts/`); data, models and outputs are not in git.
