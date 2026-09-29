#!/usr/bin/env bash
# Build the MERIT image. Override the tag with MERIT_IMAGE.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MERIT_IMAGE:-merit:1.0}"

cd "$REPO_ROOT"
echo "Building $IMAGE from $REPO_ROOT"
docker build -f docker/Dockerfile -t "$IMAGE" .
echo
echo "Done. Next:"
echo "  bash docker/run.sh"
