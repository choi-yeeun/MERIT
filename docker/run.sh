#!/usr/bin/env bash
# Run a MERIT container.
#
#
#   MERIT_IMAGE    image tag                   (default merit:1.0)
#   MERIT_GPUS     --gpus value                (default all)
#   MERIT_HF_CACHE Hugging Face cache          (default ~/.cache/huggingface)
#   MERIT_SHM      shared memory               (default 32g)
#   MERIT_MOUNTS   host dirs, colon-separated, mounted at the same path
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${MERIT_IMAGE:-merit:1.0}"
GPUS="${MERIT_GPUS:-all}"
HF_CACHE="${MERIT_HF_CACHE:-$HOME/.cache/huggingface}"
SHM="${MERIT_SHM:-32g}"

mkdir -p "$HF_CACHE"

MOUNTS=()
if [[ -n "${MERIT_MOUNTS:-}" ]]; then
    IFS=: read -ra paths <<< "$MERIT_MOUNTS"
    for path in "${paths[@]}"; do
        [[ -z "$path" ]] && continue
        if [[ ! -e "$path" ]]; then
            echo "MERIT_MOUNTS: $path does not exist" >&2
            exit 1
        fi
        abs="$(cd "$path" && pwd)"
        MOUNTS+=(-v "$abs:$abs")
        echo "mount $abs"
    done
fi

ENV_ARGS=()
if [[ -f "$REPO_ROOT/.env" ]]; then
    ENV_ARGS+=(--env-file "$REPO_ROOT/.env")
else
    echo "note: $REPO_ROOT/.env not found; forwarding keys from the shell instead." >&2
    for var in OPENAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY VERTEX_API_KEY HF_TOKEN; do
        [[ -n "${!var:-}" ]] && ENV_ARGS+=(-e "$var")
    done
fi

exec docker run --rm -it \
    --gpus "$GPUS" \
    --shm-size "$SHM" \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_ROOT:/workspace" \
    -v "$HF_CACHE:/workspace/.hf_cache" \
    "${MOUNTS[@]}" \
    -w /workspace \
    "${ENV_ARGS[@]}" \
    "$IMAGE" \
    "${@:-bash}"
