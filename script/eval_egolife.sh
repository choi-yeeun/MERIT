#!/usr/bin/env bash
# MERIT evaluation on EgoLifeQA. Unrecognised flags are forwarded to
# eval/evaluate_egolife.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBJECT="A1_JAKE"
RESPOND_MODEL="gpt-5"
TOP_K=10
MAX_ROUNDS=5
DATA_DIR="data/EgoLife"
NUM_WORKERS=4
NUM_THREADS=1
GPU_IDS="0,1,2,3"
IMAGE_MAX_TOKENS=512
IMAGE_MAX_SIZE=2048
EXP_NAME=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)          SUBJECT="$2";          shift 2 ;;
        --respond-model)    RESPOND_MODEL="$2";    shift 2 ;;
        --top-k)   TOP_K="$2";   shift 2 ;;
        --max-rounds)       MAX_ROUNDS="$2";       shift 2 ;;
        --data-dir)         DATA_DIR="$2";         shift 2 ;;
        --num-workers)      NUM_WORKERS="$2";      shift 2 ;;
        --num-threads)      NUM_THREADS="$2";      shift 2 ;;
        --gpu-ids)          GPU_IDS="$2";          shift 2 ;;
        --image-max-tokens) IMAGE_MAX_TOKENS="$2"; shift 2 ;;
        --image-max-size)   IMAGE_MAX_SIZE="$2";   shift 2 ;;
        --exp-name)         EXP_NAME="$2";         shift 2 ;;
        # Anything else is passed straight through to evaluate_egolife.py.
        *)                  EXTRA+=("$1");         shift   ;;
    esac
done

CMD=(python eval/evaluate_egolife.py
     --subject "$SUBJECT"
     --respond-model "$RESPOND_MODEL"
     --top-k "$TOP_K"
     --max-rounds "$MAX_ROUNDS"
     --data-dir "$DATA_DIR"
     --num-workers "$NUM_WORKERS"
     --num-threads "$NUM_THREADS"
     --gpu-ids "$GPU_IDS"
     --image-max-tokens "$IMAGE_MAX_TOKENS"
     --image-max-size "$IMAGE_MAX_SIZE")

if [[ -n "$EXP_NAME" ]]; then
    CMD+=(--exp-name "$EXP_NAME")
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    CMD+=("${EXTRA[@]}")
fi

printf 'Running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
