# Multi-stage minimal Dockerfile for SLM inference

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (optional minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libstdc++6 git && rm -rf /var/lib/apt/lists/*

FROM base AS builder
COPY pyproject.toml uv.lock* requirements.txt* ./
RUN pip install --no-cache-dir fastapi uvicorn transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy model + code
COPY src/ ./src/
COPY models/optimized/ ./models/optimized/

FROM base AS runtime
WORKDIR /app
ENV MODEL_DIR=/app/models/optimized/sshleifer_tiny-gpt2-quant \
  ONNX_PATH=/app/models/optimized/sshleifer_tiny-gpt2.onnx \
  USE_ONNX=false PORT=8080

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models/optimized /app/models/optimized

EXPOSE 8080
ENTRYPOINT ["python", "-m", "src.inference.server"]
