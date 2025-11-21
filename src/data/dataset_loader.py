"""
PyTorch dataset classes for training data loading.

This module provides dataset classes for loading and processing
JSONL training data for fine-tuning language models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ConversationDataset(Dataset):
    """
    PyTorch dataset for conversation-style training data.

    Expects JSONL files with 'prompt' and 'completion' fields.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 2048,
        prompt_column: str = "prompt",
        completion_column: str = "completion",
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length (default: 2048)
            prompt_column: Name of prompt field in JSONL
            completion_column: Name of completion field in JSONL
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_column = prompt_column
        self.completion_column = completion_column

        # Load data
        self.examples = self._load_data()

        logger.info(
            f"Loaded {len(self.examples)} examples from {data_path}",
            extra={
                "num_examples": len(self.examples),
                "max_length": max_length,
            },
        )

    def _load_data(self) -> List[Dict[str, str]]:
        """Load JSONL data from file."""
        import json

        examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if self.prompt_column in data and self.completion_column in data:
                        examples.append(
                            {
                                "prompt": data[self.prompt_column],
                                "completion": data[self.completion_column],
                            }
                        )
                    else:
                        logger.warning(f"Line {line_num} missing required fields")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        return examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with tokenized inputs and labels
        """
        example = self.examples[idx]

        # Combine prompt and completion
        text = example["prompt"] + example["completion"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare labels (same as input_ids for language modeling)
        labels = encoding["input_ids"].clone()

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Calculate prompt length to mask prompt tokens in labels
        prompt_encoding = self.tokenizer(
            example["prompt"],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]

        # Mask prompt tokens (we only train on completion)
        labels[0, :prompt_length] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class InstructionDataset(Dataset):
    """
    PyTorch dataset for instruction-following data.

    Expects JSONL with 'instruction', 'input' (optional), and 'output'.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 2048,
        instruction_template: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length
            instruction_template: Template for formatting instructions
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Default instruction template
        if instruction_template is None:
            self.instruction_template = (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{output}"
            )
        else:
            self.instruction_template = instruction_template

        # Load data
        self.examples = self._load_data()

        logger.info(
            f"Loaded {len(self.examples)} instruction examples",
            extra={"num_examples": len(self.examples)},
        )

    def _load_data(self) -> List[Dict[str, str]]:
        """Load JSONL data from file."""
        import json

        examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if "instruction" in data and "output" in data:
                        examples.append(
                            {
                                "instruction": data["instruction"],
                                "input": data.get("input", ""),
                                "output": data["output"],
                            }
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        return examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.examples[idx]

        # Format text using template
        text = self.instruction_template.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare labels
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Calculate instruction+input length to mask
        instruction_text = self.instruction_template.format(
            instruction=example["instruction"],
            input=example["input"],
            output="",
        )
        instruction_encoding = self.tokenizer(
            instruction_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        instruction_length = instruction_encoding["input_ids"].shape[1]

        # Mask instruction tokens (only train on output)
        labels[0, :instruction_length] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def create_data_collator(tokenizer, padding: str = "longest"):
    """
    Create a data collator for batching.

    Args:
        tokenizer: HuggingFace tokenizer
        padding: Padding strategy ('longest' or 'max_length')

    Returns:
        Data collator function
    """
    from transformers import DataCollatorForLanguageModeling

    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)


def load_tokenizer(
    model_name_or_path: str, padding_side: str = "right"
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Load and configure tokenizer.

    Args:
        model_name_or_path: Model name or path (HuggingFace ID or local path)
        padding_side: Side to pad on ('right' or 'left')

    Returns:
        Configured tokenizer
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    logger.info(
        f"Loaded tokenizer for {model_name_or_path}",
        extra={
            "vocab_size": tokenizer.vocab_size,
            "pad_token": tokenizer.pad_token,
        },
    )

    return tokenizer
