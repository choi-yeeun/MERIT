#!/usr/bin/env bash
# Bare-metal setup with micromamba + uv. Prefer docker/build.sh when Docker is
# available. MERIT_CUDA selects the torch wheel index (default cu124).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${MERIT_ENV:-merit}"
CUDA_TAG="${MERIT_CUDA:-cu124}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

if ! command -v micromamba >/dev/null 2>&1; then
    echo "micromamba not found on PATH." >&2
    echo "Install: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html" >&2
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"

# micromamba create deletes the prefix before rebuilding it, so re-running this
# script would destroy a working environment. Reuse it instead; steps 2-4 are
# idempotent. Start clean with: micromamba env remove -n "$ENV_NAME"
if micromamba run -n "$ENV_NAME" true >/dev/null 2>&1; then
    echo "=== 1/4  Reusing the existing '${ENV_NAME}' environment ==="
else
    echo "=== 1/4  Create the '${ENV_NAME}' environment ==="
    micromamba create -y -n "$ENV_NAME" -f environment.yml
fi
micromamba activate "$ENV_NAME"

echo
echo "=== 2/4  Install torch 2.6.0 (${CUDA_TAG}) ==="
uv pip install --python "$CONDA_PREFIX/bin/python" \
    torch==2.6.0 torchvision==0.21.0 --index-url "$TORCH_INDEX"

echo
echo "=== 3/4  Install flash-attn 2.7.4.post1 ==="
# Required: the Qwen3 embedding model requests attn_implementation="flash_attention_2".
# The prebuilt wheel avoids a 30-60 minute source build.
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation "$FLASH_WHEEL" \
    || uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation "flash-attn==2.7.4.post1"

echo
echo "=== 4/4  Install merit ==="
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[benchmarks]"

echo
python - <<'PY'
import torch, torchvision, transformers
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("transformers", transformers.__version__)
print("cuda avail  ", torch.cuda.is_available(),
      f"({torch.cuda.device_count()} device(s))" if torch.cuda.is_available() else "")
PY

cat <<EOF

=== Setup complete ===
Activate with:  micromamba activate ${ENV_NAME}
Then:           cp .env.example .env   # and fill in OPENAI_API_KEY
EOF
