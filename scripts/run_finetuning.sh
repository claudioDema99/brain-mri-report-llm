#!/usr/bin/env bash
# QLoRA pipeline: dataset preparation, the 10 LoRA configurations, evaluation on the test split.
# Usage: scripts/run_finetuning.sh REPORTS_JSON [WORK_DIR]
# The split is rebuilt with the same seed and the same forced IDs as in the paper
# (data/split_index.json), so it reproduces the original train/val/test partition
# when run on the same export.
set -eu
REPORTS=${1:?Label Studio export}; WORK=${2:-./work}
mkdir -p "$WORK"

# Brain reports not labelled "Negativo", unchanged (Label Studio format)
python finetuning/filter_reports.py --input "$REPORTS" --output "$WORK/filtered_reports.json"
# Label combinations and dataset-level weights (logged by training_data.py for comparison)
python finetuning/label_powerset.py --input "$REPORTS" --out_dir "$WORK/powerset"

ids() { python -c "import json,sys; print(' '.join(map(str, json.load(open('data/split_index.json'))[sys.argv[1]])))" "$1"; }
python finetuning/training_data.py \
    --input "$WORK/filtered_reports.json" \
    --powerset "$WORK/powerset/label_powerset.json" \
    --out_dir "$WORK/dataset" \
    --seed 42 --train_ratio 0.8 --val_ratio 0.1 \
    --force_ids_train $(ids forced_train) \
    --force_ids_val $(ids forced_val) \
    --force_ids_test $(ids forced_test)

for i in $(seq 1 10); do
    python finetuning/qwen_finetune.py \
        --config "$i" \
        --dataset_dir "$WORK/dataset/hf_dataset" \
        --weights_file "$WORK/dataset/class_weights.json" \
        --output_dir "$WORK/output/config_$(printf '%02d' "$i")"
done

python finetuning/qwen_evaluation.py \
    --configs $(seq 1 10) \
    --output_base_dir "$WORK/output" \
    --dataset_dir "$WORK/dataset/hf_dataset" \
    --reports_file "$WORK/filtered_reports.json" \
    --prompts dataset
