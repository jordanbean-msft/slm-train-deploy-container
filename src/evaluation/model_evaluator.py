"""Model evaluation utilities: perplexity and simple generation metrics.

Supports HF models in standard or quantized directories. For ONNX evaluation,
we fallback to basic token-level perplexity using logits produced by HF model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvaluationConfig:
    model_path: Path
    dataset_path: Path
    max_samples: int = 32
    max_new_tokens: int = 32
    compute_perplexity: bool = True
    compute_length_stats: bool = True


@dataclass
class EvaluationResult:
    perplexity: Optional[float]
    avg_input_length: Optional[float]
    avg_output_length: Optional[float]
    samples: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perplexity": self.perplexity,
            "avg_input_length": self.avg_input_length,
            "avg_output_length": self.avg_output_length,
            "samples": self.samples,
        }


def load_dataset(path: Path, limit: int) -> List[str]:
    data: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data.append(line)
        if len(data) >= limit:
            break
    if not data:
        raise ValueError("Dataset is empty or unreadable")
    return data


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
) -> float:
    model.eval()
    torch.set_grad_enabled(False)
    total_log_likelihood = 0.0
    total_tokens = 0
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt")
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = model(input_ids=input_ids)
        # shift for causal LM
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        total_log_likelihood += -loss.item()
        total_tokens += labels.numel()
    avg_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-avg_log_likelihood)
    return float(perplexity)


def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens)
        text_out = tokenizer.decode(gen[0], skip_special_tokens=True)
        results.append({"prompt": p, "output": text_out})
    return results


def evaluate(cfg: EvaluationConfig) -> EvaluationResult:
    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {cfg.model_path}")
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {cfg.dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    texts = load_dataset(cfg.dataset_path, cfg.max_samples)
    perplexity_val = (
        compute_perplexity(model, tokenizer, texts)
        if cfg.compute_perplexity
        else None
    )
    gen_samples = generate_samples(
        model,
        tokenizer,
        texts[: min(5, len(texts))],
        cfg.max_new_tokens,
    )

    avg_in = (
        sum(len(t) for t in texts) / len(texts)
        if cfg.compute_length_stats
        else None
    )
    avg_out = (
        sum(len(s["output"]) for s in gen_samples) / len(gen_samples)
        if cfg.compute_length_stats
        else None
    )

    return EvaluationResult(
        perplexity=perplexity_val,
        avg_input_length=avg_in,
        avg_output_length=avg_out,
        samples=gen_samples,
    )


__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "evaluate",
]
