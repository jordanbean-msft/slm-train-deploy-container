"""
Unit tests for dataset_loader module.

Tests PyTorch dataset classes and tokenization.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def sample_jsonl_file():
    """Create temporary JSONL file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = [
            {
                "prompt": "What is Python?",
                "completion": "Python is a programming language.",
            },
            {
                "prompt": "Explain machine learning.",
                "completion": "Machine learning is a type of AI.",
            },
            {"prompt": "What is Azure?", "completion": "Azure is Microsoft's cloud platform."},
        ]
        for item in data:
            f.write(json.dumps(item) + "\n")

        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def sample_instruction_file():
    """Create temporary JSONL file with instruction data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = [
            {
                "instruction": "Translate to French",
                "input": "Hello world",
                "output": "Bonjour le monde",
            },
            {
                "instruction": "Summarize text",
                "input": "Long text here...",
                "output": "Short summary",
            },
        ]
        for item in data:
            f.write(json.dumps(item) + "\n")

        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50000
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "[EOS]"

    # Mock tokenization
    def mock_call(*args, **kwargs):
        text = args[0] if args else kwargs.get("text", "")
        max_length = kwargs.get("max_length", 2048)

        # Simple mock: create fixed-length tensors
        input_ids = torch.randint(1, 50000, (1, max_length))
        attention_mask = torch.ones(1, max_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    tokenizer.__call__ = mock_call
    return tokenizer


class TestConversationDataset:
    """Tests for ConversationDataset class."""

    def test_dataset_initialization(self, sample_jsonl_file, mock_tokenizer):
        """Test dataset initialization."""
        from src.data.dataset_loader import ConversationDataset

        dataset = ConversationDataset(
            data_path=sample_jsonl_file,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 3
        assert dataset.max_length == 512

    def test_dataset_getitem(self, sample_jsonl_file, mock_tokenizer):
        """Test getting single item from dataset."""
        from src.data.dataset_loader import ConversationDataset

        dataset = ConversationDataset(
            data_path=sample_jsonl_file,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_dataset_length(self, sample_jsonl_file, mock_tokenizer):
        """Test dataset length."""
        from src.data.dataset_loader import ConversationDataset

        dataset = ConversationDataset(
            data_path=sample_jsonl_file,
            tokenizer=mock_tokenizer,
        )

        assert len(dataset) == 3

    def test_dataset_custom_columns(self, mock_tokenizer):
        """Test dataset with custom column names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data = [
                {"question": "What?", "answer": "Response"},
            ]
            f.write(json.dumps(data[0]) + "\n")
            temp_path = f.name

        try:
            from src.data.dataset_loader import ConversationDataset

            dataset = ConversationDataset(
                data_path=temp_path,
                tokenizer=mock_tokenizer,
                prompt_column="question",
                completion_column="answer",
            )

            assert len(dataset) == 1
        finally:
            Path(temp_path).unlink()


class TestInstructionDataset:
    """Tests for InstructionDataset class."""

    def test_instruction_dataset_initialization(self, sample_instruction_file, mock_tokenizer):
        """Test instruction dataset initialization."""
        from src.data.dataset_loader import InstructionDataset

        dataset = InstructionDataset(
            data_path=sample_instruction_file,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 2

    def test_instruction_dataset_getitem(self, sample_instruction_file, mock_tokenizer):
        """Test getting item from instruction dataset."""
        from src.data.dataset_loader import InstructionDataset

        dataset = InstructionDataset(
            data_path=sample_instruction_file,
            tokenizer=mock_tokenizer,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestTokenizerLoading:
    """Tests for tokenizer loading utilities."""

    @patch("src.data.dataset_loader.AutoTokenizer")
    def test_load_tokenizer(self, mock_auto_tokenizer):
        """Test loading tokenizer."""
        from src.data.dataset_loader import load_tokenizer

        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.vocab_size = 50000
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test-model")

        assert tokenizer.pad_token == "[EOS]"
        assert tokenizer.padding_side == "right"
        mock_auto_tokenizer.from_pretrained.assert_called_once()

    @patch("src.data.dataset_loader.AutoTokenizer")
    def test_load_tokenizer_with_pad_token(self, mock_auto_tokenizer):
        """Test loading tokenizer that already has pad token."""
        from src.data.dataset_loader import load_tokenizer

        # Setup mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.vocab_size = 50000
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test-model")

        assert tokenizer.pad_token == "[PAD]"


class TestDataCollator:
    """Tests for data collator."""

    def test_create_data_collator(self, mock_tokenizer):
        """Test creating data collator."""
        from src.data.dataset_loader import create_data_collator

        collator = create_data_collator(mock_tokenizer)

        assert collator is not None
        assert collator.mlm is False
