"""
Unit tests for trainer module.

Tests training configuration and trainer setup.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = MagicMock()
    model.to.return_value = model
    model.train.return_value = None
    model.eval.return_value = None
    model.save_pretrained.return_value = None

    # Mock named_parameters
    params = [
        ("layer.weight", torch.randn(10, 10, requires_grad=True)),
        ("layer.bias", torch.randn(10, requires_grad=True)),
    ]
    model.named_parameters.return_value = params

    # Mock forward pass
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5)
    model.return_value = mock_output

    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 50000
    tokenizer.pad_token = "[PAD]"
    tokenizer.save_pretrained.return_value = None
    return tokenizer


@pytest.fixture
def mock_dataloader():
    """Create mock dataloader."""
    # Create sample batch
    batch = {
        "input_ids": torch.randint(0, 50000, (2, 512)),
        "attention_mask": torch.ones(2, 512),
        "labels": torch.randint(0, 50000, (2, 512)),
    }

    dataloader = [batch] * 10  # 10 batches
    return dataloader


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig(model_name_or_path="test-model")

        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig(
            model_name_or_path="test-model",
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-4,
        )

        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self, mock_model, mock_tokenizer, mock_dataloader):
        """Test trainer initialization."""
        from src.training.trainer import Trainer, TrainingConfig

        config = TrainingConfig(
            model_name_or_path="test-model",
            num_epochs=1,
            batch_size=2,
        )

        trainer = Trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_dataloader,
            config=config,
        )

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def test_optimizer_creation(self, mock_model, mock_tokenizer, mock_dataloader):
        """Test optimizer creation."""
        from src.training.trainer import Trainer, TrainingConfig

        config = TrainingConfig(
            model_name_or_path="test-model",
            learning_rate=1e-4,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_dataloader,
            config=config,
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_scheduler_creation(self, mock_model, mock_tokenizer, mock_dataloader):
        """Test learning rate scheduler creation."""
        from src.training.trainer import Trainer, TrainingConfig

        config = TrainingConfig(
            model_name_or_path="test-model",
            warmup_steps=100,
        )

        trainer = Trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_dataloader,
            config=config,
        )

        assert trainer.scheduler is not None

    @patch("src.training.trainer.tqdm")
    def test_training_step(self, mock_tqdm, mock_model, mock_tokenizer, mock_dataloader):
        """Test single training step."""
        from src.training.trainer import Trainer, TrainingConfig

        config = TrainingConfig(
            model_name_or_path="test-model",
            gradient_accumulation_steps=1,
        )

        trainer = Trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_dataloader,
            config=config,
        )

        batch = mock_dataloader[0]
        loss = trainer._training_step(batch)

        assert isinstance(loss, float)
        assert loss >= 0

    @patch("src.training.trainer.tqdm")
    def test_evaluate(self, mock_tqdm, mock_model, mock_tokenizer, mock_dataloader):
        """Test evaluation."""
        from src.training.trainer import Trainer, TrainingConfig

        config = TrainingConfig(model_name_or_path="test-model")

        trainer = Trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_dataloader,
            eval_dataloader=mock_dataloader,
            config=config,
        )

        eval_loss = trainer.evaluate()

        assert isinstance(eval_loss, float)
        assert eval_loss >= 0

    def test_save_checkpoint(self, mock_model, mock_tokenizer, mock_dataloader, tmp_path):
        """Test checkpoint saving."""
        from src.training.trainer import Trainer, TrainingConfig

        config = TrainingConfig(
            model_name_or_path="test-model",
            output_dir=str(tmp_path),
        )

        trainer = Trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_dataloader,
            config=config,
        )

        trainer.save_checkpoint("test-checkpoint")

        mock_model.save_pretrained.assert_called()
        mock_tokenizer.save_pretrained.assert_called()


class TestLoRASetup:
    """Tests for LoRA model setup."""

    @patch("src.training.trainer.AutoModelForCausalLM")
    @patch("src.training.trainer.get_peft_model")
    def test_setup_lora_model(self, mock_get_peft, mock_auto_model):
        """Test LoRA model setup."""
        from src.training.trainer import TrainingConfig, setup_lora_model

        # Setup mocks
        mock_model = MagicMock()
        mock_model.print_trainable_parameters.return_value = None
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_get_peft.return_value = mock_model

        config = TrainingConfig(
            model_name_or_path="test-model",
            use_lora=True,
            lora_r=16,
        )

        model = setup_lora_model("test-model", config)

        assert model is not None
        mock_get_peft.assert_called_once()

    @patch("src.training.trainer.AutoModelForCausalLM")
    def test_setup_model_without_lora(self, mock_auto_model):
        """Test model setup without LoRA."""
        from src.training.trainer import TrainingConfig, setup_lora_model

        # Setup mock
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        config = TrainingConfig(
            model_name_or_path="test-model",
            use_lora=False,
        )

        model = setup_lora_model("test-model", config)

        assert model is not None
