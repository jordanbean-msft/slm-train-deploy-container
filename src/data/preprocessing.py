"""Data preprocessing and validation utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    pass


def validate_jsonl_format(file_path: Path) -> Tuple[bool, List[str]]:
    """Validate JSONL file format and structure.

    Args:
        file_path: Path to JSONL file

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if not file_path.exists():
        errors.append(f"File not found: {file_path}")
        return False, errors

    if file_path.suffix != ".jsonl":
        errors.append(f"File must have .jsonl extension, got: {file_path.suffix}")
        return False, errors

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Validate required fields
                if not isinstance(data, dict):
                    errors.append(f"Line {line_num}: Expected dict, got {type(data)}")
                    continue

                if "prompt" not in data:
                    errors.append(f"Line {line_num}: Missing required field 'prompt'")

                if "completion" not in data:
                    errors.append(
                        f"Line {line_num}: Missing required field 'completion'"
                    )

                # Validate field types
                if "prompt" in data and not isinstance(data["prompt"], str):
                    errors.append(
                        f"Line {line_num}: 'prompt' must be string, got {type(data['prompt'])}"
                    )

                if "completion" in data and not isinstance(data["completion"], str):
                    errors.append(
                        f"Line {line_num}: 'completion' must be string, got {type(data['completion'])}"
                    )

    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors

    is_valid = len(errors) == 0
    if is_valid:
        logger.info(f"Validation successful for {file_path}")
    else:
        logger.error(f"Validation failed for {file_path}: {len(errors)} errors")

    return is_valid, errors


def load_jsonl_data(file_path: Path) -> List[Dict[str, str]]:
    """Load and parse JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries with prompt/completion pairs

    Raises:
        DataValidationError: If validation fails
    """
    is_valid, errors = validate_jsonl_format(file_path)

    if not is_valid:
        error_msg = "\n".join(errors)
        raise DataValidationError(f"Data validation failed:\n{error_msg}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def split_train_validation(
    data: List[Dict[str, str]], validation_split: float = 0.2, seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split data into training and validation sets.

    Args:
        data: List of data samples
        validation_split: Fraction of data for validation (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, validation_data)

    Raises:
        ValueError: If validation_split is not in valid range
    """
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0 (exclusive)")

    if len(data) == 0:
        return [], []

    # Use simple deterministic split based on seed
    import random

    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    split_idx = int(len(data) * (1 - validation_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    logger.info(
        f"Split data: {len(train_data)} training, {len(val_data)} validation samples"
    )

    return train_data, val_data


def calculate_data_statistics(data: List[Dict[str, str]]) -> Dict[str, Any]:
    """Calculate statistics for the dataset.

    Args:
        data: List of data samples

    Returns:
        Dictionary with statistics
    """
    if not data:
        return {
            "num_samples": 0,
            "avg_prompt_length": 0,
            "avg_completion_length": 0,
            "min_prompt_length": 0,
            "max_prompt_length": 0,
            "min_completion_length": 0,
            "max_completion_length": 0,
            "total_tokens_estimate": 0,
        }

    prompt_lengths = [len(sample["prompt"]) for sample in data]
    completion_lengths = [len(sample["completion"]) for sample in data]

    stats = {
        "num_samples": len(data),
        "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
        "avg_completion_length": sum(completion_lengths) / len(completion_lengths),
        "min_prompt_length": min(prompt_lengths),
        "max_prompt_length": max(prompt_lengths),
        "min_completion_length": min(completion_lengths),
        "max_completion_length": max(completion_lengths),
        # Rough estimate: 1 token ≈ 4 characters
        "total_tokens_estimate": (sum(prompt_lengths) + sum(completion_lengths)) // 4,
    }

    logger.info(f"Dataset statistics: {stats}")
    return stats


def save_jsonl_data(data: List[Dict[str, str]], output_path: Path) -> None:
    """Save data to JSONL file.

    Args:
        data: List of data samples
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(data)} samples to {output_path}")
