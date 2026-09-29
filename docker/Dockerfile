# MERIT runtime image. Python 3.11 + torch 2.6.0 + CUDA 12.4.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf_cache \
    TOKENIZERS_PARALLELISM=false

# The flash-attn wheel and decord 0.6.0 are cp311-only.
RUN PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" \
    && echo "python $PYVER" \
    && if [ "$PYVER" != "3.11" ]; then \
         echo "ERROR: this image expects Python 3.11, base image has $PYVER." >&2; \
         exit 1; \
       fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Prebuilt wheel first; the fallback compiles from source.
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-build-isolation "${FLASH_ATTN_WHEEL}" \
    || pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"

# Manifest first so the slow layer is cached independently of code edits.
COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
RUN pip install -e ".[benchmarks]"

# flash-attn by metadata, not by import: the build has no GPU.
RUN python -c "import importlib.metadata as md, torch, torchvision, transformers; \
print('torch', torch.__version__, 'torchvision', torchvision.__version__); \
print('transformers', transformers.__version__, 'flash-attn', md.version('flash_attn')); \
assert torch.__version__.startswith('2.6.0'), torch.__version__"

# Datasets and the API key are mounted / passed at run time, never baked in.
COPY . /workspace

RUN test -f data/EgoLife/captions/A1_JAKE/A1_JAKE_30sec_4_key.json \
    && python tools/selftest.py

CMD ["/bin/bash"]
