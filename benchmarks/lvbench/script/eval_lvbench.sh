#!/usr/bin/env bash
# MERIT evaluation on LVBench. Unrecognised flags are forwarded to
# eval/evaluate_lvbench.py.
set -e

# MERIT evaluation on LVBench.
# Usage:
#   bash script/eval_lvbench.sh \
#       --exp-name merit_gpt5_top10 \
#       --num-workers 4 --gpu-ids 0,1,2,3
#
#   bash script/eval_lvbench.sh \

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
EXP_NAME=""
RESPOND_MODEL="gpt-5"
MAX_ROUNDS=5
NUM_WORKERS=4
GPU_IDS="0,1,2,3"
GPUS_PER_WORKER=1
IMAGE_MAX_TOKENS=512
IMAGE_MAX_SIZE=2048
TOP_K=10
DEBUG_LOG_COUNT=5
SERVICE_TIER=""
TIMEOUT=""
CONCURRENCY=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --respond-model) RESPOND_MODEL="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --image-max-tokens) IMAGE_MAX_TOKENS="$2"; shift 2 ;;
        --image-max-size) IMAGE_MAX_SIZE="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --debug-log-count) DEBUG_LOG_COUNT="$2"; shift 2 ;;
        --service-tier) SERVICE_TIER="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    EXP_NAME="merit_${RESPOND_MODEL}_top${TOP_K}"
fi

OPTIONAL_ARGS=""
[[ -n "$SERVICE_TIER" ]] && OPTIONAL_ARGS="$OPTIONAL_ARGS --service-tier $SERVICE_TIER"
[[ -n "$TIMEOUT" ]]      && OPTIONAL_ARGS="$OPTIONAL_ARGS --timeout $TIMEOUT"
[[ -n "$CONCURRENCY" ]]  && OPTIONAL_ARGS="$OPTIONAL_ARGS --concurrency $CONCURRENCY"

echo "============================================"
echo "MERIT on LVBench"
echo "============================================"
echo "Exp name:         $EXP_NAME"
echo "Respond model:    $RESPOND_MODEL"
echo "Num workers:      $NUM_WORKERS"
echo "GPU IDs:          $GPU_IDS"
echo "GPUs/Worker:      $GPUS_PER_WORKER"
echo "Image max tokens: $IMAGE_MAX_TOKENS"
echo "Image max size:   $IMAGE_MAX_SIZE"
echo "Episodic top-k:   $TOP_K"
echo "Extra args:       $EXTRA_ARGS"
[[ -n "$SERVICE_TIER" ]] && echo "Service Tier:     $SERVICE_TIER"
[[ -n "$TIMEOUT" ]] && echo "Timeout:          $TIMEOUT"
[[ -n "$CONCURRENCY" ]] && echo "Concurrency:      $CONCURRENCY"
echo "============================================"

cd "$PROJECT_ROOT"

python eval/evaluate_lvbench.py \
    --exp-name "$EXP_NAME" \
    --respond-model "$RESPOND_MODEL" \
    --max-rounds "$MAX_ROUNDS" \
    --num-workers "$NUM_WORKERS" \
    --gpu-ids "$GPU_IDS" \
    --gpus-per-worker "$GPUS_PER_WORKER" \
    --debug-log-count "$DEBUG_LOG_COUNT" \
    --image-max-tokens "$IMAGE_MAX_TOKENS" \
    --image-max-size "$IMAGE_MAX_SIZE" \
    --top-k "$TOP_K" \
    $OPTIONAL_ARGS \
    $EXTRA_ARGS
