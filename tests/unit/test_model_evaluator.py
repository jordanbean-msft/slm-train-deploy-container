from pathlib import Path

import pytest

from src.evaluation.model_evaluator import EvaluationConfig, evaluate


@pytest.mark.unit
def test_evaluate_perplexity_and_lengths(tmp_path: Path):
    # Prepare tiny dataset
    data_path = tmp_path / "prompts.txt"
    data_path.write_text("Hello world\nHow are you\nTest prompt")

    # Use tiny model for speed
    model_dir = tmp_path / "model"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    t = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    m.save_pretrained(model_dir)
    t.save_pretrained(model_dir)

    cfg = EvaluationConfig(
        model_path=model_dir,
        dataset_path=data_path,
        max_samples=3,
        max_new_tokens=4,
    )
    result = evaluate(cfg)
    d = result.to_dict()
    assert d["perplexity"] is not None
    assert d["avg_input_length"] > 0
    assert len(d["samples"]) > 0
