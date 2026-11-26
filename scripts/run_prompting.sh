#!/usr/bin/env bash
# Prompting experiments: every prompt (A-D) for one model and one task, in parallel.
# Usage: scripts/run_prompting.sh REPORTS_JSON TASK MODEL
#   TASK  = binary | multiclass
#   MODEL = gpt | mistral | gemma | qwen
# The paper used the full export for the binary task and the filtered
# (brain, pathological) reports for the multiclass task.
# Endpoints: VLLM_BASE_URL or VLLM_BASE_URL_<MODEL> (see README).
set -u
DATA=${1:?reports JSON}; TASK=${2:?binary|multiclass}; MODEL=${3:?model}
mkdir -p logs
declare -A PIDS
for P in A B C D; do
    python -m prompting.main "$DATA" "$TASK" "$MODEL" "$P" > "logs/${TASK}_${MODEL}_prompt_${P}.log" 2>&1 &
    PIDS[$P]=$!
    echo "prompt $P started (PID ${PIDS[$P]})"
done
FAILED=0
for P in A B C D; do
    if wait "${PIDS[$P]}"; then echo "prompt $P: ok"; else echo "prompt $P: failed"; FAILED=1; fi
done
exit $FAILED
