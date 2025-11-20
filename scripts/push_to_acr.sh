#!/usr/bin/env bash
set -euo pipefail

# Push built image to Azure Container Registry.
# Requires: az CLI authenticated (Managed Identity or az login), docker installed.
# Env vars:
#   ACR_NAME (required) - name of ACR (without .azurecr.io)
#   IMAGE_NAME (default: slm-inference)
#   IMAGE_TAG (default: latest)
#   SOURCE_IMAGE (default: slm-inference:latest)

if [[ -z "${ACR_NAME:-}" ]]; then
  echo "ERROR: ACR_NAME not set" >&2
  exit 1
fi

IMAGE_NAME=${IMAGE_NAME:-slm-inference}
IMAGE_TAG=${IMAGE_TAG:-latest}
SOURCE_IMAGE=${SOURCE_IMAGE:-${IMAGE_NAME}:latest}
TARGET="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[INFO] Logging into ACR: ${ACR_NAME}"
az acr login --name "${ACR_NAME}" >/dev/null

echo "[INFO] Tagging image ${SOURCE_IMAGE} -> ${TARGET}"
docker tag "${SOURCE_IMAGE}" "${TARGET}"

echo "[INFO] Pushing ${TARGET}"
docker push "${TARGET}"

echo "[INFO] Push complete: ${TARGET}"
echo "[INFO] List repository tags:"
az acr repository show-tags --name "${ACR_NAME}" --repository "${IMAGE_NAME}" || true
