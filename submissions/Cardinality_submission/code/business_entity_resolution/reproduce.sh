#!/bin/bash
# Mode A: regenerate output/matching_results.tsv and output/candidate_pairs.tsv from the supplied data with the
# saved trained models in models/. Run from this folder (code/business_entity_resolution) after installing
# requirements.txt (Python 3.12.13). Put the organisers' dataset under DATASET_DIR (default: dataset/, holding
# train/ and test/ with the original TSV files). Intermediate files keep the relative paths the stages expect.
# Hardware used: 16 CPU threads, 32 GB RAM, one CUDA GPU (6 GB is enough) for the cross-encoder steps.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python src/checks/preflight.py --environment
if [ -e work/v24/kaggle/work ] || [ -e artifacts/after-0982910-20260927/fast_rescue_test_v1 ]; then
    echo "Use a fresh extracted package for reproduction; existing run artifacts found." >&2
    exit 2
fi
export DATASET_DIR="${DATASET_DIR:-dataset}"
PY=python
M=models

echo "== 0. Parquet copies of the supplied TSV files"
PYTHONPATH=src $PY -m fast_er.to_parquet "$DATASET_DIR" work/v14/pq

echo "== 1. Main pipeline v24 (normalization, lexicon, blocking, two-stage trees) with the saved tree models"
# Stage 1 learns the transliteration lexicon from the training labels (deterministic); it must equal models/lexicon.json,
# which the saved tree models were trained with (stage 3 checks the lexicon, code and environment fingerprints).
PYTHONPATH=src $PY src/fast_er/run_stage1.py train work/v14/pq work/v24/kaggle/work work/v14/pq/train_ground_truth.parquet --mem 18GB --threads 4 --workers 4 --chunks 16 --k 80 --k-field 40
PYTHONPATH=src $PY src/fast_er/run_stage1.py test work/v14/pq work/v24/kaggle/work --mem 18GB --threads 4 --workers 4 --chunks 16 --k 80 --k-field 40
cmp work/v24/kaggle/work/lexicon.json $M/lexicon.json
mkdir -p work/v24/kaggle/work/model work/v24/kaggle/out && cp $M/trees_v24/* work/v24/kaggle/work/model/
PYTHONPATH=src $PY src/fast_er/run_stage3.py work/v24/kaggle/work work/v14/pq/train_ground_truth.parquet test work/v24/kaggle/out     --skip-train --mem 10GB --threads 4 --chunks 48
# -> work/v24/kaggle/out/scores.parquet and candidate_pairs.tsv (the blocking candidate set scored by the models)

echo "== 2. Earlier pipeline v20 (its scores are features of the correction model)"
PYTHONPATH=src/pipeline_v20 $PY src/pipeline_v20/fast_er/run_stage1.py train work/v14/pq work/v20/kaggle/work work/v14/pq/train_ground_truth.parquet --mem 18GB --threads 4 --workers 4 --chunks 16 --k 80 --k-field 40
PYTHONPATH=src/pipeline_v20 $PY src/pipeline_v20/fast_er/run_stage1.py test work/v14/pq work/v20/kaggle/work --mem 18GB --threads 4 --workers 4 --chunks 16 --k 80 --k-field 40
cmp work/v20/kaggle/work/lexicon.json $M/lexicon.json
mkdir -p work/v20/kaggle/work/model work/v20/kaggle/out && cp $M/trees_v20/* work/v20/kaggle/work/model/
PYTHONPATH=src/pipeline_v20 $PY src/pipeline_v20/fast_er/run_stage3.py work/v20/kaggle/work work/v14/pq/train_ground_truth.parquet test work/v20/kaggle/out     --skip-train --mem 10GB --threads 4 --chunks 48

echo "== 3. Normalized text views read by the correction, rescue and France stages"
$PY src/views/build_feature_views.py work/v14/pq work/v21L/run $M/lexicon_feature_views.json test

echo "== 4. Cross-encoder scores on the uncertain band (0.002 < p < 0.9995)"
mkdir -p work/kaggle_xenc_inf11 work/kaggle_out/xenc_inf11 work/kaggle_out/xenc_inf6 work/kaggle_xenc_inf6
$PY src/crossencoder/xenc_band_export.py work/v24/kaggle/out/scores.parquet work/v21L/run/test work/v14/pq work/kaggle_xenc_inf11/v24_test_band.parquet test 0.002 0.9995
$PY src/crossencoder/xe_french_forms.py work/kaggle_xenc_inf11/v24_test_band.parquet work/v21L/run/test work/kaggle_xenc_inf11
if [ -f "$M/COMPACT_MANIFEST.json" ]; then
    cp "$M/saved_scores/v24_test_band_frnorm_xe.parquet" work/kaggle_out/xenc_inf11/
    cp "$M/saved_scores/v24_test_fr_band_xe.parquet" work/kaggle_out/xenc_inf11/
    $PY src/checks/verify_saved_scores.py work/kaggle_xenc_inf11/v24_test_band.parquet work/kaggle_out/xenc_inf11/v24_test_band_frnorm_xe.parquet
    $PY src/checks/verify_saved_scores.py work/kaggle_xenc_inf11/v24_test_fr_band.parquet work/kaggle_out/xenc_inf11/v24_test_fr_band_xe.parquet
else
    XE_TOKENIZER=$M/crossencoder/tokenizer_e5small $PY src/crossencoder/xenc_infer_local.py $M/crossencoder/xenc4_e5small work/kaggle_out/xenc_inf11 \
        work/kaggle_xenc_inf11/v24_test_band.parquet work/kaggle_xenc_inf11/v24_test_fr_band.parquet
    $PY src/crossencoder/merge_french_xe.py work/kaggle_out/xenc_inf11/v24_test_band_xe.parquet work/kaggle_out/xenc_inf11/v24_test_fr_band_xe.parquet \
        work/kaggle_out/xenc_inf11/v24_test_band_frnorm_xe.parquet
fi
$PY src/crossencoder/xenc_band_export.py work/v20/kaggle/out/scores.parquet work/v21L/run/test work/v14/pq work/kaggle_xenc_inf6/v20_test_band.parquet test 0.002 0.9995
if [ -f "$M/COMPACT_MANIFEST.json" ]; then
    cp "$M/saved_scores/v20_test_band_xe.parquet" work/kaggle_out/xenc_inf6/
    $PY src/checks/verify_saved_scores.py work/kaggle_xenc_inf6/v20_test_band.parquet work/kaggle_out/xenc_inf6/v20_test_band_xe.parquet
else
    XE_TOKENIZER=$M/crossencoder/tokenizer_e5base $PY src/crossencoder/xenc_infer_local.py $M/crossencoder/xenc5_e5base work/kaggle_out/xenc_inf6 \
        work/kaggle_xenc_inf6/v20_test_band.parquet
fi

echo "== 5. Blend (France decisions) and the France base file"
mkdir -p work/v24x4f && cp $M/blend/blend.json work/v24x4f/blend.json
$PY src/blend/apply_saved_blend.py $M/blend/blend.json work/v24/kaggle/out/scores.parquet work/kaggle_out/xenc_inf11/v24_test_band_frnorm_xe.parquet work/v24x4f/scores_blend.parquet
$PY src/blend/rethreshold.py work/v24x4f/scores_blend.parquet 0.75 work/v24x4f/v24x4f_t075_fr0925.tsv work/v14/pq 0.925

echo "== 6. French structural rules F1/F2/F4"
mkdir -p submissions work/comp/v25
$PY src/france/fr_overlay.py work/v24x4f/v24x4f_t075_fr0925.tsv work/v24x4f/scores_blend.parquet work/v21L/run/test work/comp/v25/frA.tsv 0.3 F1,F2,F4

echo "== 7. India/US correction model (LightGBM, cutoff 0.74)"
cp $M/correction/v25_lgb600.joblib $M/correction/v25_model.json work/comp/v25/
$PY src/correction/v25.py bandinputs
$PY src/correction/v25.py score work/comp/v25
$PY src/correction/v25.py export work/comp/v25 work/v24x4f/v24x4f_t075_fr0925.tsv submissions/v25stack_t074_fr0925_matching_results.tsv
$PY src/correction/v25.py export work/comp/v25 work/comp/v25/frA.tsv work/comp/v25/v25FRa.tsv

echo "== 8. India/US rescue route (first core word + house number)"
$PY src/rescue/rescue.py test work/comp/v25/v25FRa.tsv work/comp/v25/rescue_test_pairs.parquet 1.88
$PY src/rescue/rescue_apply.py work/comp/v25/v25FRa.tsv work/v24/kaggle/out/candidate_pairs.tsv work/comp/v25/rescue_test_pairs.parquet 1.88 \
    work/comp/v25/v26.tsv work/comp/v25/candidate_pairs_v26.tsv

echo "== 9. French rules F3/F5 and French rescue route"
$PY src/france/fr_overlay.py work/comp/v25/v26.tsv work/v24x4f/scores_blend.parquet work/v21L/run/test work/comp/v25/v27a.tsv 0.3 F3,F5
RESCUE_COUNTRIES="'france'" $PY src/rescue/rescue.py test work/comp/v25/v27a.tsv work/comp/v25/rescue_fr_pairs.parquet 1.88
$PY src/rescue/rescue_apply.py work/comp/v25/v27a.tsv work/comp/v25/candidate_pairs_v26.tsv work/comp/v25/rescue_fr_pairs.parquet 1.88 \
    submissions/v27_matching_results.tsv work/comp/v25/candidate_pairs_v27.tsv

echo "== 10. Learned rescue matcher (India/US)"
mkdir -p artifacts/after-0982910-20260927
$PY src/rescue/learned/export_fast_rescue.py
cp artifacts/after-0982910-20260927/fast_rescue_test_v1/matching_results.tsv work/comp/v25/v28.tsv

echo "== 11. Corrected French cross-encoder normalization, French re-decision (additions only)"
mkdir -p work/kaggle_out/xenc_inf12 work/v24x4f2
$PY src/crossencoder/xe_french_forms_v2.py work/kaggle_xenc_inf11/v24_test_band.parquet work/v21L/run/test work/kaggle_xenc_inf11/v24_test_fr2_band.parquet
if [ -f "$M/COMPACT_MANIFEST.json" ]; then
    cp "$M/saved_scores/v24_test_fr2_band_xe.parquet" work/kaggle_out/xenc_inf12/
    $PY src/checks/verify_saved_scores.py work/kaggle_xenc_inf11/v24_test_fr2_band.parquet work/kaggle_out/xenc_inf12/v24_test_fr2_band_xe.parquet
else
    XE_TOKENIZER=$M/crossencoder/tokenizer_e5small $PY src/crossencoder/xenc_infer_local.py $M/crossencoder/xenc4_e5small work/kaggle_out/xenc_inf12 \
        work/kaggle_xenc_inf11/v24_test_fr2_band.parquet
fi
$PY src/france/fr_v2_scores.py
$PY src/france/fr_v2_base.py work/comp/v25/v28.tsv work/v24x4f2/scores_blend_fr.parquet work/comp/v25/v29_base.tsv
$PY src/france/fr_overlay.py work/comp/v25/v29_base.tsv work/v24x4f2/scores_blend_fr.parquet work/v21L/run/test work/comp/v25/v29_a.tsv 0.3 F1,F2,F4
$PY src/france/fr_overlay.py work/comp/v25/v29_a.tsv work/v24x4f2/scores_blend_fr.parquet work/v21L/run/test work/comp/v25/v29_b.tsv 0.3 F3,F5
RESCUE_COUNTRIES="'france'" $PY src/rescue/rescue.py test work/comp/v25/v29_b.tsv work/comp/v25/v29_frrescue.parquet 1.88
$PY src/rescue/rescue_apply.py work/comp/v25/v29_b.tsv work/comp/v25/candidate_pairs_v27.tsv work/comp/v25/v29_frrescue.parquet 1.88 \
    work/comp/v25/v29.tsv work/comp/v25/candidate_pairs_v29_check.tsv
cmp work/comp/v25/candidate_pairs_v29_check.tsv work/comp/v25/candidate_pairs_v27.tsv   # French rescue adds no new candidates here
$PY src/france/union_additions.py work/comp/v25/v28.tsv work/comp/v25/v29.tsv work/comp/v25/v30.tsv

echo "== 12. Outputs and checks"
mkdir -p ../../reproduced_output
cp work/comp/v25/v30.tsv ../../reproduced_output/matching_results.tsv
cp work/comp/v25/candidate_pairs_v27.tsv ../../reproduced_output/candidate_pairs.tsv
$PY src/checks/check_candidates.py ../../reproduced_output/candidate_pairs.tsv ../../reproduced_output/matching_results.tsv "$DATASET_DIR/test/test_source1.tsv"
$PY src/checks/check_submission.py ../../reproduced_output/matching_results.tsv "$DATASET_DIR/test" --candidate ../../reproduced_output/candidate_pairs.tsv --mem 1.5GB
$PY src/checks/compare_outputs.py ../../output ../../reproduced_output
echo "DONE: outputs regenerated, validated and compared"
