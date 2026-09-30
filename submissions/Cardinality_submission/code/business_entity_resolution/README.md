# Business Entity Resolution: Team Cardinality

This is the pipeline that produced our final leaderboard submission (public macro F0.5 **0.984718**):
`output/matching_results.tsv` and its blocking candidate set `output/candidate_pairs.tsv`. The method is described
in `Documentation_template.md` at the root of the archive.

**Size-limited final archive:** the portal accepts at most 1,024 MB. The two fine-tuned neural weight files
alone exceed that cap, so this ZIP contains four SHA-256-checked neural **test-score caches** under
`models/saved_scores/` in place of those two weight files. `reproduce.sh` regenerates the record views,
blocking candidates, tree scores and final decisions, checks that each saved neural score covers exactly
the regenerated pair band, and compares both final TSVs byte for byte with `output/`. This is a
**test-specific saved-score replay**, not independent neural inference on new records or a complete
training-from-data package. The original weight hashes remain in `models/MODELS_MANIFEST.json` for provenance;
`models/COMPACT_MANIFEST.json` records the two omissions and the four score-cache hashes. A full clean replay
of this compact procedure is still unverified.

## Contents

| Path | What it is |
|---|---|
| `src/fast_er/` | Main pipeline (v24): normalization, transliteration lexicon, DuckDB blocking and routes, pair features, two-stage boosted trees, prediction, candidate export |
| `src/pipeline_v20/fast_er/` | Earlier pipeline version (v20); its scores are input features of the correction model |
| `src/views/` | Normalized record views used by the later stages |
| `src/crossencoder/` | Cross-encoder band export, French text normalization (`xe_french_forms.py`, corrected `xe_french_forms_v2.py`), GPU scoring, French score merge. `kaggle/`: the GPU training notebooks |
| `src/blend/` | Logistic blend of tree probability and cross-encoder logit (fit: `apply_blend.py`, `blend_eval.py`; saved coefficients: `apply_saved_blend.py`); thresholding (`rethreshold.py`) |
| `src/correction/` | Correction model on the uncertain band: features (`stacker.py`), metric (`metric.py`), LightGBM model (`v25.py`), model selection (`sweep.py`, `sweep2.py`) |
| `src/rescue/` | Rescue route for matches lost at blocking (`rescue.py`, `rescue_apply.py`); `learned/`: logistic-regression rescue matcher |
| `src/france/` | French structural rules (`fr_overlay.py`), France re-decision with the corrected normalization (`fr_v2_scores.py`, `fr_v2_base.py`), additions-only merge (`union_additions.py`) |
| `src/checks/` | Output validators (format, IDs, ownership, candidate containment) |
| `models/` | Saved tree, correction and rescue models, tokenizers, blend coefficients and lexicons. The fine-tuned e5-small/base neural weights are omitted under the portal cap; four neural test-score caches replace their inference outputs. Both manifests record SHA-256 hashes |
| `reproduce.sh` | Mode A: regenerate both output files with the saved models |
| `requirements.txt` | Pinned dependencies (Python 3.12.13) |
| `FUNCTION_GUIDE.md` | Purpose of compact helper functions and script entry points without source docstrings; frozen model-linked source remains byte-identical |

## Environment

- **Python 3.12.13** with `requirements.txt`. The saved tree models record their training environment, and stage 3
  refuses to predict under different package versions. Install torch from the CUDA 11.8 wheel index (see the
  comment in `requirements.txt`).
- About 16 CPU threads and 32 GB RAM. The compact saved-score replay does not run neural inference;
  training or fresh neural inference requires a CUDA GPU (6 GB suffices for the original procedure) and the omitted weights.
- Disk: about 60 GB for intermediate files.
- Offline only. No external lookups; the pretrained encoder weights are included in `models/`.

## Data

Place the organisers' dataset in `dataset/` inside this folder (or set `DATASET_DIR`):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

All commands run from this folder. Intermediate files go to `work/`, `artifacts/` and `submissions/` under it.
The stage scripts read and write those fixed relative paths.

## Mode A: size-limited saved-score replay (full replay not yet verified)

```bash
python src/checks/preflight.py --environment
bash reproduce.sh
```

| Step | What it does |
|---|---|
| 0 | TSV → string-only parquet (`fast_er.to_parquet`) |
| 1 | **v24 pipeline**: normalization, lexicon (learned from the training labels; checked against `models/lexicon.json`) and blocking for train and test, then **prediction with the saved v24 trees** (`--skip-train`). Writes `scores.parquet` and `candidate_pairs.tsv`, the blocking candidates scored by the models |
| 2 | Same for the **v20 pipeline**, whose scores feed the correction model |
| 3 | Normalized record views with `models/lexicon_feature_views.json`, the lexicon the final submission's later stages used |
| 4 | Regenerate the uncertain bands (0.002 < p < 0.9995), verify their pair IDs against the included e5-small/e5-base score caches, and use those saved scores. The original models were fine-tuned on the supplied pairs |
| 5 | **Blend** with the saved coefficients; France decision base at cutoff 0.925 |
| 6 | **French rules** F1 (same address, name minus a word plus a French generic suffix), F2 (invented rebrand token at a lone reference's address), F4 (name-only target with a unique core name) |
| 7 | **India/US correction model**: LightGBM, 600 trees, cutoff 0.74, on 63 features |
| 8 | **India/US rescue route**: same first core word and house number; accepted at name ratio + address token-set ≥ 1.88 |
| 9 | **French rules** F3/F5 (same content words or subset at another address, legal form not swapped) and the French rescue route. The final candidate set is complete here |
| 10 | **Learned rescue matcher** (India/US), logistic regression at probability ≥ 0.69 |
| 11 | **Corrected French cross-encoder normalization** (`xe_french_forms_v2.py`). French generic suffixes map by measured behaviour: & Fils, Frères, Groupe and Développement → Services; Associés → Partners; Holding, Participations and Distribution → Group. The v2 pair band is checked against its saved neural scores, then re-decision and the French rules run again; only links for unassigned targets are added |
| 12 | Write `../../reproduced_output/`, validate, and compare against the untouched `../../output/` files |

## Mode B: retraining outline (additional prerequisites required)

The commands are the same as in Mode A, with training enabled:

1. **Trees (v24, then v20 with `PYTHONPATH=src/pipeline_v20`)**, after stage 1 on train and test:
   ```bash
   PYTHONPATH=src python src/fast_er/run_stage3.py work/v24/kaggle/work work/v14/pq/train_ground_truth.parquet test work/v24/kaggle/out \
       --n-train 3000000 --stage1-hard-rank 15 --stage1-neg-keep 0.12 --stage2-all-references --hold-mod 22 \
       --rounds 2500 --leaves 63 --lr 0.03 --weight sqrt --backend auto --threads 4 --chunks 48 --mem 10GB
   ```
   `--backend auto` uses XGBoost on a CUDA GPU when available, otherwise LightGBM with equivalent settings. The run
   also writes holdout predictions (`model/hold_preds.parquet`, `split_ids.parquet`). The holdout is the reference
   identities with md5(entity_id + salt) mod 22 = 0; it is split into tune and final halves.
2. **Cross-encoders** (`src/crossencoder/kaggle/*.py`, one GPU each). Hard training pairs come from the uncertain
   band of the stage-1 out-of-fold scores of non-holdout references (`xenc_hard_export.py`).
   - e5-small chain: `xenc.py` → `xenc2.py` → `xenc4.py`.
   - e5-base chain: `xenc3.py` → `xenc5.py`.
   - Score the holdout and test bands with `xenc_infer_local.py`.
3. **Blend**: `src/blend/apply_blend.py` fits the logistic blend on the tune half of the holdout band;
   `blend_eval.py` reports tune/final scores.
4. **Correction model**:
   ```bash
   python src/correction/stacker.py --out artifacts/push995-20260927-0915/stack_v4   # band features, grouped-CV comparison
   python src/correction/v25.py fit work/comp/v25                                     # final LightGBM (config chosen by sweep.py)
   ```
5. **Rescue route threshold**:
   ```bash
   python src/rescue/make_hold_truth.py dataset work/v21L/run/train work/v24/kaggle/work/model/hold_preds.parquet work/comp/hold_truth.parquet
   python src/rescue/rescue.py eval
   ```
   This selects 1.88 on the tune half.
6. **Learned rescue matcher**: `src/rescue/learned/fast_rescue.py` (grouped-CV selection), then `final_fast_rescue.py`
   (fit and lock). It uses the correction model's out-of-fold decisions, shipped as
   `models/rescue_learned/training_inputs/v25_oof_control.parquet`; copy that file to
   `artifacts/after-0982910-20260927/xenc7_pilot_v1/oof_predictions.parquet`.
7. **France**: no labels exist. The rules and normalization tables are fixed in code and their thresholds are
   documented. Nothing is trained on French data.

The source describes the GPU training method, but the compact ZIP does **not** include the two fine-tuned neural
weights and cannot independently retrain or infer them offline. Bit-identical neural weights are not expected
from retraining because GPU training is not deterministic.

## Verification status for this package

- The 33 included entries from `models/MODELS_MANIFEST.json` and all four score caches in `models/COMPACT_MANIFEST.json` match their saved hashes; the two omitted neural weights remain listed in the original manifest for provenance. The two output TSVs match the selected v30/candidate files byte for byte.
- Frozen v24/v20 source files have been restored to the exact byte hashes expected by the saved tree manifests. No fingerprint check was bypassed. The rescue exporter also verifies the exact saved `frozen_fast_rescue.py` feature source and model hash.
- The verification chain in the development workspace **failed** with out-of-memory errors and missing intermediate files. It cannot be treated as a successful replay. French normalization text pairs matched, but the subsequent neural-score comparison reported 13,786 differing pairs (27,572 rows in the two-sided comparison); follow-up comparison found no changes at the 0.925 threshold or in selected owners at that step. Complete final-pipeline replay remains unverified.
- Python syntax checks pass. Two rescue-training path variables accidentally included in comments were repaired and checked without starting training.
- **Not verified:** clean-workspace Mode A execution, end-to-end bit-identical outputs, the exact environment across both neural models, and complete Mode B retraining. Mode B needs naturally retrieved rescue training pairs, base out-of-fold predictions and held-band features in addition to the commands shown; not all prerequisites are generated by this outline. The supplied old out-of-fold control file must not be mixed with newly retrained base predictions without rebuilding and validating the corresponding folds and inputs.
- Rerun verification serially on adequate memory, fail immediately on a failed stage, use a new output directory, and compare every expected artifact. Do not overwrite the submitted files or change manifests to accept a mismatch.

The deliverable is `Cardinality_submission.zip`, containing `output/`, `code/`, and `Documentation_template.md` directly at the ZIP root. The models are included for offline saved-model execution. Archive construction and validation do not establish full numerical reproduction or constitute a portal upload.

## Compliance

- Only the supplied data is used. There are no external business registries, geocoding or hosted inference APIs.
- Entity IDs and row order are never used as predictive features; IDs only join records and break score ties.
- Original pretrained model: intfloat/multilingual-e5 (small and base), MIT license, fine-tuned on supplied
  training pairs. This compact ZIP ships their test-score caches, not their fine-tuned weights.

## Figures and artifact integrity

The methodology references PNG figures under `docs/figures/`; editable SVG copies and plotted values are included there. Diagram arrows describe logical data flow and validation inputs, not a claim that the rescue routes execute simultaneously. The public-score graph excludes local validation and explicitly labels its truncated vertical axis.

`reproduce.sh` uses LF line endings, stops on errors, forces offline model loading, checks saved source/model/environment contracts, and refuses existing run artifacts. It preserves the submitted `output/` files and writes `reproduced_output/` separately. A hash mismatch is reported as a failure rather than hidden by copying the submitted output over a regenerated file. Run it in a fresh extraction under the pinned environment, with sufficient memory and working disk space.

The normalization-view helper has a guarded entry point and uses one worker for portability. Frozen tree source bytes and rescue feature-source hashes remain unchanged. `LICENSE` and `THIRD_PARTY_NOTICES.md` record code/model provenance; `licenses/UNILM_MIT_LICENSE.txt` preserves the upstream notice.
