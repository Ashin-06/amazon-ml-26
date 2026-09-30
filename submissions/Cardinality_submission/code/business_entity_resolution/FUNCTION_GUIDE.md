# Function guide for compact pipeline scripts

The main v24 and v20 tree modules document their functions in source. This guide covers the remaining compact helpers and script entry points that have no function docstring. It describes the saved code as shipped; it does not change model behavior or the byte fingerprints of frozen sources. Paths below are relative to `src/`.

| Source | Function(s) | Role |
| --- | --- | --- |
| `blend/blend_eval.py` | `pick` | Sweep a decision cutoff on the tuning references and report tune/final macro F0.5 and selected links. |
| `checks/compare_outputs.py` | `digest` | Stream a file into SHA-256 for exact comparison of reproduced and submitted outputs. |
| `checks/preflight.py` | `sha`, `check` | Hash saved files; verify model, source and optionally environment versions before a costly replay. |
| `correction/metric.py` | `main` | Run the standalone exact per-reference F0.5 metric check. |
| `correction/stacker.py` | `main` | Build correction-model training/evaluation features and grouped reference splits. |
| `correction/sweep.py` | `model` | Construct a LightGBM correction-model trial for hyperparameter comparison. |
| `correction/sweep2.py` | `mk`, `oof` | Build a seeded LightGBM trial; score it out of fold on tuning identities. |
| `crossencoder/xe_french_forms.py`, `xe_french_forms_v2.py` | `norm_name` | Replace selected French legal/generic terms in the name side of a `name | address` pair; preserve the address side. The v2 table is the final French correction. |
| `france/fr_v2_scores.py` | `blend` | Combine a tree probability and cross-encoder score using the saved blend coefficients. |
| `rescue/rescue.py` | `route`, `scored`, `f05`, `pick` | Build the name/house-number route; score pairs and keep those meeting the prefilter; calculate macro F0.5 including empty references; choose the best eligible reference per target. |
| `rescue/rescue_apply.py` | `rewrite` | Stream a TSV and append accepted rescue links or exported candidate links by S1 ID. |
| `views/build_feature_views.py` | `main` | Build normalized record views for downstream correction, rescue and French stages. |
| `crossencoder/kaggle/xenc.py`, `xenc2.py`, `xenc3.py`, `xenc4.py`, `xenc5.py` | `enc` | Convert a batch of text pairs to the corresponding tokenized cross-encoder input during GPU training. These are historical training stages; saved final weights are under `models/`. |
| `rescue/learned/context.py` | `sha`, `q`, `pilot`, `final`, `main` | Hash files; quote a DuckDB path; evaluate the sibling-context experiment on tuning folds; lock/evaluate its chosen model; dispatch the requested mode. This is an experiment script, not a required saved-model replay stage. |
| `rescue/learned/export_fast_rescue.py` | `sha`, `write_rows`, `main` | Hash inputs; stream final rows while preserving French rows; load the locked rescue model, score eligible route pairs and export India/US additions. `write_rows` already has a source docstring. |
| `rescue/learned/fast_rescue.py` | `features`, `fit`, `choose`, `values`, `compare`, `main` | Compute natural-route pair features; fit a linear/tree trial; choose one owner per unowned target; evaluate full-reference macro F0.5; summarize paired score changes; run the grouped tuning pilot. |
| `rescue/learned/final_fast_rescue.py` | `module`, `sha` | Import the selected experiment module from its saved path and hash files before locking its model/source. |
| `rescue/learned/frozen_fast_rescue.py` | `features`, `fit`, `choose`, `values`, `compare`, `main` | Saved byte-identical copy of the rescue feature/selection implementation; its SHA-256 is checked by the locked model. The roles mirror `fast_rescue.py`, and this file must not be edited to add comments. |

The README gives the execution order and artifact paths. The top-level `Documentation_template.md` explains the architecture, features, validation protocol and candidate/match decisions. The full clean-workspace replay is still unverified; this guide improves navigation, not execution evidence.
