"""CLI entry point for post-training optimization.

Usage:
    uv run python -m src.training.optimize \
        --config configs/optimization_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .model_optimizer import (benchmark_inference, export_to_onnx,
                              quantize_model, summarize_artifact)


def run(cfg_path: Path) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text())
    base_model_id = cfg["base_model_id"]
    output_root = Path(cfg["output_root"]) / base_model_id.replace("/", "_")
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Quantization
    q_cfg = cfg.get("quantization", {})
    if q_cfg.get("enabled", True):
        q_result = quantize_model(
            base_model_id,
            output_root / "quantized",
            method=q_cfg.get("method", "int8"),
            dtype=q_cfg.get("dtype"),
        )
        artifacts["quantization"] = summarize_artifact(q_result)
    else:
        artifacts["quantization"] = {"enabled": False}

    # ONNX
    onnx_cfg = cfg.get("onnx", {})
    if onnx_cfg.get("enabled", True):
        onnx_result = export_to_onnx(
            base_model_id,
            output_root / f"{base_model_id.replace('/', '_')}.onnx",
            opset=onnx_cfg.get("opset", 17),
            sequence_length=onnx_cfg.get("sequence_length", 128),
        )
        artifacts["onnx"] = summarize_artifact(onnx_result)
    else:
        artifacts["onnx"] = {"enabled": False}

    # Benchmark
    bench_cfg = cfg.get("benchmark", {})
    if artifacts.get("quantization", {}).get("enabled", True):
        quant_dir = output_root / "quantized"
        if quant_dir.exists():
            bench_stats = benchmark_inference(
                quant_dir,
                prompt=bench_cfg.get("prompt", "Hello"),
                repetitions=bench_cfg.get("repetitions", 3),
                max_new_tokens=bench_cfg.get("max_new_tokens", 16),
            )
            artifacts["benchmark"] = bench_stats
    else:
        artifacts["benchmark"] = {"skipped": True}

    # Persist summary
    summary_path = output_root / "optimization_summary.json"
    summary_path.write_text(json.dumps(artifacts, indent=2))
    print(json.dumps(artifacts, indent=2))
    return summary_path


def main():  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        description="Optimize fine-tuned model artifacts"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to optimization config YAML",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":  # pragma: no cover
    main()
