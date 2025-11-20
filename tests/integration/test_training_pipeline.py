"""
Integration tests for training pipeline.

Tests end-to-end training workflow integration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def training_config():
    """Create test training configuration."""
    return {
        "model": {"name_or_path": "test-model", "trust_remote_code": True},
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "num_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "max_grad_norm": 1.0,
            "fp16": False,
            "bf16": False,
            "use_8bit": False,
            "use_4bit": False,
            "optim": "adamw_torch",
        },
        "data": {
            "train_file": "train.jsonl",
            "validation_file": None,
            "max_seq_length": 512,
            "prompt_column": "prompt",
            "completion_column": "completion",
        },
        "checkpointing": {
            "output_dir": "outputs",
            "save_steps": 100,
            "save_total_limit": 2,
        },
        "logging": {
            "logging_steps": 10,
            "mlflow": {"enabled": False},
        },
        "evaluation": {"eval_steps": 100},
        "hardware": {"seed": 42},
    }


@pytest.fixture
def sample_training_data():
    """Create sample training data file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = [
            {
                "prompt": "Question 1?",
                "completion": "Answer 1",
            },
            {
                "prompt": "Question 2?",
                "completion": "Answer 2",
            },
        ] * 5  # 10 examples

        for item in data:
            f.write(json.dumps(item) + "\n")

        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


class TestTrainingPipelineIntegration:
    """Integration tests for full training pipeline."""

    def test_config_loading(self, tmp_path, training_config):
        """Test loading training configuration."""
        import yaml

        config_path = tmp_path / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(training_config, f)

        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["model"]["name_or_path"] == "test-model"
        assert loaded_config["training"]["num_epochs"] == 1

    @patch("src.training.train.load_tokenizer")
    @patch("src.training.train.setup_lora_model")
    def test_dataset_and_dataloader_creation(
        self,
        mock_setup_model,
        mock_load_tokenizer,
        sample_training_data,
        training_config,
    ):
        """Test dataset and dataloader creation."""
        from torch.utils.data import DataLoader

        from src.data.dataset_loader import (ConversationDataset,
                                             create_data_collator)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 50000
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.pad_token_id = 0

        def mock_tokenize(*args, **kwargs):
            import torch

            max_length = kwargs.get("max_length", 512)
            return {
                "input_ids": torch.randint(1, 50000, (1, max_length)),
                "attention_mask": torch.ones(1, max_length),
            }

        mock_tokenizer.__call__ = mock_tokenize
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create dataset
        dataset = ConversationDataset(
            data_path=sample_training_data,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 10

        # Create dataloader
        collator = create_data_collator(mock_tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        assert len(dataloader) == 5

    def test_checkpoint_manager_workflow(self, tmp_path):
        """Test checkpoint manager workflow."""
        from src.training.checkpointing import CheckpointManager

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.save_pretrained.return_value = None
        mock_tokenizer = MagicMock()
        mock_tokenizer.save_pretrained.return_value = None

        manager = CheckpointManager(
            output_dir=str(tmp_path),
            max_checkpoints=2,
        )

        # Save multiple checkpoints
        checkpoint1 = manager.save_checkpoint(
            mock_model, mock_tokenizer, step=100, epoch=1, loss=0.5
        )
        checkpoint2 = manager.save_checkpoint(
            mock_model, mock_tokenizer, step=200, epoch=2, loss=0.3
        )

        assert checkpoint1.exists()
        assert checkpoint2.exists()
        assert len(manager.checkpoints) == 2

        # Get latest
        latest = manager.get_latest_checkpoint()
        assert "step-200" in str(latest)

    def test_training_config_creation(self, training_config):
        """Test creating TrainingConfig from dict."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig(
            model_name_or_path=training_config["model"]["name_or_path"],
            use_lora=training_config["lora"]["enabled"],
            lora_r=training_config["lora"]["r"],
            lora_alpha=training_config["lora"]["alpha"],
            num_epochs=training_config["training"]["num_epochs"],
            batch_size=training_config["training"]["per_device_train_batch_size"],
        )

        assert config.model_name_or_path == "test-model"
        assert config.use_lora is True
        assert config.lora_r == 8
        assert config.num_epochs == 1
