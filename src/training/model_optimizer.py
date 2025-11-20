"""Model optimization utilities: quantization, ONNX export, size/latency.

Defensive design: optional libs (bitsandbytes, onnxruntime) may be absent.
We degrade gracefully and annotate results with availability metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional deps
    import bitsandbytes  # noqa: F401
    _BITSANDBYTES_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _BITSANDBYTES_AVAILABLE = False

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    _ONNXRUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _ONNXRUNTIME_AVAILABLE = False


@dataclass
class OptimizationResult:
    artifact_path: Path
    method: str
    size_bytes: int
    load_latency_sec: float
    notes: str
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "method": self.method,
            "size_bytes": self.size_bytes,
            "load_latency_sec": self.load_latency_sec,
            "notes": self.notes,
            "extra": self.extra,
        }


def _measure_directory_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def merge_lora_if_present(model) -> None:
    """Merge LoRA adapters into base weights if supported.

    Safe to call on non-LoRA models.
    """
    if hasattr(model, "merge_and_unload"):
        try:
            model.merge_and_unload()
        except Exception:  # pragma: no cover - best effort
            pass


def quantize_model(
    base_model_id: str,
    output_dir: Path,
    method: str = "int8",
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
) -> OptimizationResult:
    """Quantize a model using bitsandbytes 8-bit or 4-bit loading.

    method: "int8" | "int4" | "none"
    Returns OptimizationResult with metadata. If bitsandbytes not available,
    falls back to standard fp16/fp32 load depending on dtype.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}

    if dtype == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif dtype == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16

    if method == "int8" and _BITSANDBYTES_AVAILABLE:
        load_kwargs["load_in_8bit"] = True
    elif method == "int4" and _BITSANDBYTES_AVAILABLE:
        load_kwargs["load_in_4bit"] = True

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=trust_remote_code,
    )
    merge_lora_if_present(model)
    load_latency = time.time() - t0

    # Save to output_dir (HF format)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    size_bytes = _measure_directory_size(output_dir)
    quantized_flag = (
        "load_in_8bit" in load_kwargs or "load_in_4bit" in load_kwargs
    )
    notes = "quantized" if quantized_flag else "standard-precision"
    if (method in {"int8", "int4"}) and not _BITSANDBYTES_AVAILABLE:
        notes += " (bitsandbytes missing, fallback)"

    return OptimizationResult(
        artifact_path=output_dir,
        method=method,
        size_bytes=size_bytes,
        load_latency_sec=load_latency,
        notes=notes,
        extra={
            "dtype": str(getattr(model, "dtype", "unknown")),
            "device": str(next(model.parameters()).device),
            "bitsandbytes_available": _BITSANDBYTES_AVAILABLE,
        },
    )


def export_to_onnx(
    base_model_id: str,
    output_path: Path,
    opset: int = 17,
    sequence_length: int = 128,
    trust_remote_code: bool = False,
) -> OptimizationResult:
    """Export model to ONNX. Falls back gracefully if onnxruntime unavailable.

    Writes model to output_path (file). Returns metadata result.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=trust_remote_code,
    )
    merge_lora_if_present(model)
    model.eval()

    # Dummy input
    dummy_inputs = tokenizer("Hello", return_tensors="pt")
    input_ids = dummy_inputs["input_ids"][:, :sequence_length]

    if _ONNXRUNTIME_AVAILABLE:
        torch.onnx.export(
            model,
            (input_ids,),
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
        )
        notes = "onnx-export"
    else:  # pragma: no cover - environment dependent
        # Create placeholder to indicate export skipped
        with open(output_path, "w") as f:
            f.write("ONNX export skipped: onnxruntime not available")
        notes = "onnx-export-skipped"

    load_latency = time.time() - t0
    size_bytes = output_path.stat().st_size if output_path.exists() else 0

    return OptimizationResult(
        artifact_path=output_path,
        method="onnx",
        size_bytes=size_bytes,
        load_latency_sec=load_latency,
        notes=notes,
        extra={
            "opset": opset,
            "sequence_length": sequence_length,
            "onnxruntime_available": _ONNXRUNTIME_AVAILABLE,
        },
    )


def summarize_artifact(
    result: OptimizationResult,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return dict summary; optionally persist JSON metadata."""
    data = result.to_dict()
    if save_path is None:
        if result.artifact_path.is_dir():
            save_path = result.artifact_path / "optimization_metadata.json"
        else:
            save_path = (
                result.artifact_path.parent
                / f"{result.artifact_path.name}.metadata.json"
            )
    try:
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:  # pragma: no cover - best effort
        pass
    return data


def benchmark_inference(
    model_dir: Path,
    prompt: str = "Hello",
    repetitions: int = 3,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """Simple CPU inference benchmark for latency estimation."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    torch.set_grad_enabled(False)
    latencies = []
    for _ in range(repetitions):
        inputs = tokenizer(prompt, return_tensors="pt")
        t0 = time.time()
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        latencies.append(time.time() - t0)
    return {
        "prompt": prompt,
        "repetitions": repetitions,
        "p50": float(torch.tensor(latencies).median().item()),
        "p95": float(torch.quantile(torch.tensor(latencies), 0.95).item()),
        "samples": latencies,
    }


def load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    meta_file = (
        path / "optimization_metadata.json"
        if path.is_dir()
        else path.parent / f"{path.name}.metadata.json"
    )
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:  # pragma: no cover
            return None
    return None


__all__ = [
    "OptimizationResult",
    "quantize_model",
    "export_to_onnx",
    "summarize_artifact",
    "benchmark_inference",
    "merge_lora_if_present",
    "load_metadata",
]
