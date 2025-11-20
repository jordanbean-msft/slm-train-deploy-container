#!/usr/bin/env bash
set -euo pipefail

# Build multi-architecture image (amd64 + arm64) using docker buildx.
# Env vars:
#   IMAGE_NAME (default: slm-inference)
#   IMAGE_TAG (default: latest)
#   DOCKERFILE (default: docker/Dockerfile)

IMAGE_NAME=${IMAGE_NAME:-slm-inference}
IMAGE_TAG=${IMAGE_TAG:-latest}
DOCKERFILE=${DOCKERFILE:-docker/Dockerfile}

echo "[INFO] Ensuring buildx builder"
docker buildx inspect multiarch-builder >/dev/null 2>&1 || docker buildx create --name multiarch-builder --use
docker buildx use multiarch-builder

echo "[INFO] Building ${IMAGE_NAME}:${IMAGE_TAG} for linux/amd64,linux/arm64"
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${DOCKERFILE}" . \
  --load

echo "[INFO] Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
docker images | grep "${IMAGE_NAME}" || true
