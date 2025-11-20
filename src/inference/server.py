"""Minimal FastAPI inference server wrapping scoring module."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from . import scoring


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None


app = FastAPI(title="SLM Inference Server")


@app.on_event("startup")
def _startup():  # pragma: no cover - server lifecycle
    model_dir = os.getenv(
        "MODEL_DIR",
        "models/optimized/sshleifer_tiny-gpt2-quant",
    )
    onnx_path = os.getenv(
        "ONNX_PATH",
        "models/optimized/sshleifer_tiny-gpt2.onnx",
    )
    use_onnx = os.getenv("USE_ONNX", "false").lower() == "true"
    scoring.init(model_dir=model_dir, onnx_path=onnx_path, use_onnx=use_onnx)


@app.get("/health")
def health():
    return {"status": "ok", "model": scoring.metadata()}


@app.post("/generate")
def generate(req: GenerateRequest):
    return scoring.run({
        "prompt": req.prompt,
        "max_new_tokens": req.max_new_tokens,
    })


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
