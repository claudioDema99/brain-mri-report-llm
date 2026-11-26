#!/usr/bin/env bash
# Metrics, figures and statistical tests from the run logs.
# Run from the repository root after the prompting and fine-tuning runs.
set -eu
python analysis/binary_analysis.py --root_dir ./results_binary --output_dir ./figures/binary
python analysis/multiclass_analysis.py --root_dir ./results_multiclass --output_dir ./figures/multiclass
python analysis/peft_analysis.py --peft_dir ./results_multiclass/qwen_finetuned \
    --multiclass_dir ./results_multiclass --output_dir ./figures/peft
