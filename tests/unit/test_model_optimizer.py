from pathlib import Path

import pytest

from src.training.model_optimizer import (OptimizationResult,
                                          benchmark_inference, export_to_onnx,
                                          quantize_model, summarize_artifact)


@pytest.mark.unit
def test_quantize_model_returns_result(tmp_path: Path):
    out_dir = tmp_path / "quantized"
    result = quantize_model("sshleifer/tiny-gpt2", out_dir, method="int8")
    assert isinstance(result, OptimizationResult)
    assert out_dir.exists()
    meta = summarize_artifact(result)
    assert meta["method"] == "int8"


@pytest.mark.unit
def test_export_to_onnx(tmp_path: Path):
    onnx_path = tmp_path / "model.onnx"
    result = export_to_onnx(
        "sshleifer/tiny-gpt2", onnx_path, sequence_length=8
    )
    assert isinstance(result, OptimizationResult)
    # Placeholder possible if onnxruntime missing; ensure file exists
    assert onnx_path.exists()
    meta = summarize_artifact(result)
    assert meta["method"] == "onnx"


@pytest.mark.unit
def test_benchmark_inference(tmp_path: Path):
    # Reuse quantized artifact for a quick benchmark
    out_dir = tmp_path / "quantized"
    quantize_model("sshleifer/tiny-gpt2", out_dir, method="none")
    stats = benchmark_inference(out_dir, repetitions=2, max_new_tokens=4)
    assert "p50" in stats and "p95" in stats
    assert stats["repetitions"] == 2
