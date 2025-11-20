"""Unit tests for data preprocessing."""

from pathlib import Path

import pytest

from src.data.preprocessing import (DataValidationError,
                                    calculate_data_statistics, load_jsonl_data,
                                    save_jsonl_data, split_train_validation,
                                    validate_jsonl_format)


def test_validate_jsonl_format_success(sample_jsonl_path: Path) -> None:
    """Test successful validation of JSONL file."""
    is_valid, errors = validate_jsonl_format(sample_jsonl_path)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_jsonl_format_missing_file(tmp_path: Path) -> None:
    """Test validation with missing file."""
    missing_file = tmp_path / "nonexistent.jsonl"
    is_valid, errors = validate_jsonl_format(missing_file)
    assert is_valid is False
    assert len(errors) > 0
    assert "not found" in errors[0].lower()


def test_validate_jsonl_format_wrong_extension(tmp_path: Path) -> None:
    """Test validation with wrong file extension."""
    wrong_file = tmp_path / "data.json"
    wrong_file.write_text("{}")
    is_valid, errors = validate_jsonl_format(wrong_file)
    assert is_valid is False
    assert any(".jsonl extension" in e for e in errors)


def test_load_jsonl_data_success(sample_jsonl_path: Path) -> None:
    """Test successful loading of JSONL data."""
    data = load_jsonl_data(sample_jsonl_path)
    assert len(data) == 5
    assert all("prompt" in item for item in data)
    assert all("completion" in item for item in data)


def test_load_jsonl_data_invalid_file(tmp_path: Path) -> None:
    """Test loading invalid JSONL file raises error."""
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text("not valid json\n")

    with pytest.raises(DataValidationError):
        load_jsonl_data(invalid_file)


def test_split_train_validation_basic() -> None:
    """Test basic train/validation split."""
    data = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(10)]
    train, val = split_train_validation(data, validation_split=0.2, seed=42)

    assert len(train) == 8
    assert len(val) == 2
    assert len(train) + len(val) == len(data)


def test_split_train_validation_reproducible() -> None:
    """Test that splits are reproducible with same seed."""
    data = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(10)]

    train1, val1 = split_train_validation(data, validation_split=0.2, seed=42)
    train2, val2 = split_train_validation(data, validation_split=0.2, seed=42)

    assert train1 == train2
    assert val1 == val2


def test_split_train_validation_invalid_split() -> None:
    """Test invalid validation split raises error."""
    data = [{"prompt": "p", "completion": "c"}]

    with pytest.raises(ValueError):
        split_train_validation(data, validation_split=1.5)

    with pytest.raises(ValueError):
        split_train_validation(data, validation_split=-0.1)


def test_split_train_validation_empty_data() -> None:
    """Test splitting empty data returns empty lists."""
    train, val = split_train_validation([], validation_split=0.2)
    assert train == []
    assert val == []


def test_calculate_data_statistics(sample_jsonl_path: Path) -> None:
    """Test dataset statistics calculation."""
    data = load_jsonl_data(sample_jsonl_path)
    stats = calculate_data_statistics(data)

    assert stats["num_samples"] == 5
    assert stats["avg_prompt_length"] > 0
    assert stats["avg_completion_length"] > 0
    assert stats["min_prompt_length"] > 0
    assert stats["max_prompt_length"] >= stats["min_prompt_length"]
    assert stats["total_tokens_estimate"] > 0


def test_calculate_data_statistics_empty() -> None:
    """Test statistics with empty data."""
    stats = calculate_data_statistics([])
    assert stats["num_samples"] == 0
    assert stats["avg_prompt_length"] == 0


def test_save_jsonl_data(tmp_path: Path) -> None:
    """Test saving data to JSONL file."""
    data = [
        {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
        {"prompt": "What is ML?", "completion": "Machine Learning"},
    ]

    output_file = tmp_path / "output" / "test.jsonl"
    save_jsonl_data(data, output_file)

    assert output_file.exists()

    # Verify content
    loaded_data = load_jsonl_data(output_file)
    assert loaded_data == data
