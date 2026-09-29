#!/usr/bin/env bash
# MERIT evaluation on the Video-MME long split. Unrecognised flags are
# forwarded to eval/evaluate_videomme.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Defaults
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
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --respond-model) RESPOND_MODEL="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --gpus-per-worker) GPUS_PER_WORKER="$2"; shift 2 ;;
        --image-max-tokens) IMAGE_MAX_TOKENS="$2"; shift 2 ;;
        --image-max-size) IMAGE_MAX_SIZE="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --debug-log-count) DEBUG_LOG_COUNT="$2"; shift 2 ;;
        --service-tier) SERVICE_TIER="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="merit_${RESPOND_MODEL}_top${TOP_K}"
    echo "Auto-generated exp-name: $EXP_NAME"
fi

echo "============================================"
echo "MERIT on Video-MME"
echo "============================================"
echo "Experiment:       $EXP_NAME"
echo "Model:            $RESPOND_MODEL"
echo "Max Rounds:       $MAX_ROUNDS"
echo "Workers:          $NUM_WORKERS"
echo "GPUs:             $GPU_IDS"
echo "GPUs/Worker:      $GPUS_PER_WORKER"
echo "Image Max Tokens: $IMAGE_MAX_TOKENS"
echo "Image Max Size:   $IMAGE_MAX_SIZE"
echo "Episodic Top-K:   $TOP_K"
[[ -n "$SERVICE_TIER" ]] && echo "Service Tier:     $SERVICE_TIER"
[[ -n "$TIMEOUT" ]] && echo "Timeout:          $TIMEOUT"
[[ -n "$CONCURRENCY" ]] && echo "Concurrency:      $CONCURRENCY"
echo "============================================"

CMD=(python eval/evaluate_videomme.py
    --exp-name "$EXP_NAME"
    --respond-model "$RESPOND_MODEL"
    --max-rounds "$MAX_ROUNDS"
    --num-workers "$NUM_WORKERS"
    --gpu-ids "$GPU_IDS"
    --gpus-per-worker "$GPUS_PER_WORKER"
    --image-max-tokens "$IMAGE_MAX_TOKENS"
    --image-max-size "$IMAGE_MAX_SIZE"
    --top-k "$TOP_K"
    --debug-log-count "$DEBUG_LOG_COUNT"
)
[[ -n "$SERVICE_TIER" ]] && CMD+=(--service-tier "$SERVICE_TIER")
[[ -n "$TIMEOUT" ]] && CMD+=(--timeout "$TIMEOUT")
[[ -n "$CONCURRENCY" ]] && CMD+=(--concurrency "$CONCURRENCY")
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
