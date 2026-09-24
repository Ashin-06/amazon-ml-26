#!/bin/bash
# Download a finished Kaggle run's small outputs and validate the submission files.
# Usage (from repo root): bash code/business_entity_resolution/kaggle/fetch_and_validate.sh ashinmurugan06/amlc26-v3 v3
set -e
KERNEL=$1; NAME=$2; DEST="work/kout/$NAME"
mkdir -p "$DEST"
kaggle kernels output "$KERNEL" -p "$DEST" -o -q \
  --file-pattern '(metrics\.json|blocking_recall\.json|lexicon\.json|matching_results\.tsv|candidate_pairs\.tsv|hold_preds\.parquet|hold_sids\.parquet|\.log)$'
find "$DEST" -type f | sed 's/^/  /'
M=$(find "$DEST" -name matching_results.tsv | head -1); C=$(find "$DEST" -name candidate_pairs.tsv | head -1)
if [ -n "$M" ]; then
  python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching "$M" --candidate "$C" --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
fi
