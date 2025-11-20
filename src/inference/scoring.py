"""Optimized scoring script for embedded deployment.

Provides init() and run(data) interface expected by container entrypoints.
Supports quantized HF model directories or ONNX model file fallback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # optional
    import onnxruntime as ort  # noqa: F401
    _ONNX_AVAILABLE = True
except Exception:  # pragma: no cover
    _ONNX_AVAILABLE = False

MODEL: Optional[AutoModelForCausalLM] = None
TOKENIZER: Optional[AutoTokenizer] = None
ONNX_SESSION: Optional[Any] = None
MODEL_INFO: Dict[str, Any] = {}


def init(
    model_dir: str = "models/optimized/sshleifer_tiny-gpt2-quant",
    onnx_path: str = "models/optimized/sshleifer_tiny-gpt2.onnx",
    use_onnx: bool = False,
    max_new_tokens: int = 32,
):
    """Initialize model/tokenizer lazily.

    Loading time targeted <5s for small models; we record latency.
    """
    global MODEL, TOKENIZER, ONNX_SESSION, MODEL_INFO
    t0 = time.time()
    m_dir = Path(model_dir)

    if use_onnx and _ONNX_AVAILABLE and Path(onnx_path).exists():
        ONNX_SESSION = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        # Need tokenizer assets from model_dir
        TOKENIZER = AutoTokenizer.from_pretrained(m_dir)
        MODEL_INFO = {
            "backend": "onnx",
            "path": onnx_path,
        }
    else:
        TOKENIZER = AutoTokenizer.from_pretrained(m_dir)
        MODEL = AutoModelForCausalLM.from_pretrained(m_dir)
        MODEL.eval()
        torch.set_grad_enabled(False)
        MODEL_INFO = {
            "backend": "hf",
            "path": model_dir,
        }

    MODEL_INFO["load_latency_sec"] = round(time.time() - t0, 4)
    MODEL_INFO["onnx_available"] = _ONNX_AVAILABLE
    MODEL_INFO["max_new_tokens_default"] = max_new_tokens
    return MODEL_INFO


def _generate_hf(prompt: str, max_new_tokens: int) -> str:
    assert MODEL is not None and TOKENIZER is not None
    inputs = TOKENIZER(prompt, return_tensors="pt")
    gen_ids = MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
    return TOKENIZER.decode(gen_ids[0], skip_special_tokens=True)


def _generate_onnx(prompt: str, max_new_tokens: int) -> str:
    # simple greedy loop using ONNX logits
    assert ONNX_SESSION is not None and TOKENIZER is not None
    tokens = TOKENIZER(prompt, return_tensors="pt")["input_ids"].tolist()[0]
    for _ in range(max_new_tokens):
        inp = {"input_ids": [tokens]}
        out = ONNX_SESSION.run(["logits"], inp)[0]
        next_id = int(out[0, -1].argmax())
        tokens.append(next_id)
        if next_id == TOKENIZER.eos_token_id:
            break
    return TOKENIZER.decode(tokens, skip_special_tokens=True)


def run(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform inference.

    Expects {'prompt': str, 'max_new_tokens': int?}. Returns timing & output.
    """
    if TOKENIZER is None and MODEL is None and ONNX_SESSION is None:
        init()  # default lazy init

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return {"error": "empty prompt", "output": ""}

    max_new_tokens = int(
        data.get(
            "max_new_tokens",
            MODEL_INFO.get("max_new_tokens_default", 32),
        )
    )
    t0 = time.time()
    if ONNX_SESSION is not None:
        output = _generate_onnx(prompt, max_new_tokens)
    else:
        output = _generate_hf(prompt, max_new_tokens)
    latency = time.time() - t0
    return {
        "prompt": prompt,
        "output": output,
        "latency_sec": round(latency, 4),
        "backend": MODEL_INFO.get("backend"),
    }


def metadata() -> Dict[str, Any]:
    return MODEL_INFO.copy()


if __name__ == "__main__":  # simple manual test
    info = init()
    print("Loaded:", json.dumps(info, indent=2))
    res = run({"prompt": "Hello"})
    print(json.dumps(res, indent=2))
