#!/usr/bin/env bash
set -euo pipefail

URL=${URL:-http://localhost:8080/health}

start=$(date +%s%3N)
code=$(curl -s -o /tmp/hc.out -w '%{http_code}' "$URL" || true)
end=$(date +%s%3N)
latency_ms=$((end-start))

echo "status_code=$code latency_ms=$latency_ms"
if [[ "$code" != "200" ]]; then
  echo "Health check failed" >&2
  cat /tmp/hc.out || true
  exit 1
fi
cat /tmp/hc.out
