"""
Unit tests for checkpointing module.

Tests checkpoint management and state persistence.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = MagicMock()
    model.save_pretrained.return_value = None
    model.from_pretrained.return_value = model
    model.load_state_dict.return_value = None
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.save_pretrained.return_value = None
    return tokenizer


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_manager_initialization(self, tmp_path):
        """Test checkpoint manager initialization."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(
            output_dir=str(tmp_path),
            max_checkpoints=3,
        )

        assert manager.output_dir == tmp_path
        assert manager.max_checkpoints == 3
        assert len(manager.checkpoints) == 0

    def test_save_checkpoint(self, tmp_path, mock_model, mock_tokenizer):
        """Test saving checkpoint."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path))

        checkpoint_path = manager.save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            step=100,
            epoch=1,
            loss=0.5,
        )

        assert checkpoint_path.exists()
        assert len(manager.checkpoints) == 1
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_save_checkpoint_metadata(self, tmp_path, mock_model, mock_tokenizer):
        """Test checkpoint metadata saving."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path))

        checkpoint_path = manager.save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            step=100,
            epoch=1,
            loss=0.5,
            metrics={"accuracy": 0.9},
        )

        metadata_file = checkpoint_path / "checkpoint_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["step"] == 100
        assert metadata["epoch"] == 1
        assert metadata["loss"] == 0.5
        assert metadata["metrics"]["accuracy"] == 0.9

    def test_best_checkpoint_tracking(self, tmp_path, mock_model, mock_tokenizer):
        """Test best checkpoint tracking."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path))

        # Save checkpoints with different losses
        manager.save_checkpoint(mock_model, mock_tokenizer, 100, 1, loss=0.5)
        manager.save_checkpoint(mock_model, mock_tokenizer, 200, 2, loss=0.3, is_best=True)
        manager.save_checkpoint(mock_model, mock_tokenizer, 300, 3, loss=0.4)

        best_checkpoint = manager.get_best_checkpoint()
        assert best_checkpoint is not None
        assert "step-200" in str(best_checkpoint)

    def test_checkpoint_rotation(self, tmp_path, mock_model, mock_tokenizer):
        """Test checkpoint rotation."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(
            output_dir=str(tmp_path),
            max_checkpoints=2,
        )

        # Save 3 checkpoints (should keep only 2)
        manager.save_checkpoint(mock_model, mock_tokenizer, 100, 1, loss=0.5)
        manager.save_checkpoint(mock_model, mock_tokenizer, 200, 2, loss=0.4)
        manager.save_checkpoint(mock_model, mock_tokenizer, 300, 3, loss=0.3)

        assert len(manager.checkpoints) == 2

    def test_get_latest_checkpoint(self, tmp_path, mock_model, mock_tokenizer):
        """Test getting latest checkpoint."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path))

        manager.save_checkpoint(mock_model, mock_tokenizer, 100, 1, loss=0.5)
        manager.save_checkpoint(mock_model, mock_tokenizer, 200, 2, loss=0.4)

        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "step-200" in str(latest)

    def test_list_checkpoints(self, tmp_path, mock_model, mock_tokenizer):
        """Test listing checkpoints."""
        from src.training.checkpointing import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path))

        manager.save_checkpoint(mock_model, mock_tokenizer, 100, 1, loss=0.5)
        manager.save_checkpoint(mock_model, mock_tokenizer, 200, 2, loss=0.4)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2


class TestTrainingState:
    """Tests for training state persistence."""

    def test_save_training_state(self, tmp_path):
        """Test saving training state."""
        from src.training.checkpointing import save_training_state

        optimizer_state = {"param_groups": [{"lr": 1e-4}]}
        scheduler_state = {"last_epoch": 10}

        save_training_state(
            output_dir=str(tmp_path),
            step=100,
            epoch=1,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            loss=0.5,
        )

        state_file = tmp_path / "training_state.pt"
        assert state_file.exists()

    def test_load_training_state(self, tmp_path):
        """Test loading training state."""
        from src.training.checkpointing import (load_training_state,
                                                save_training_state)

        optimizer_state = {"param_groups": [{"lr": 1e-4}]}
        scheduler_state = {"last_epoch": 10}

        save_training_state(
            output_dir=str(tmp_path),
            step=100,
            epoch=1,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            loss=0.5,
        )

        loaded_state = load_training_state(str(tmp_path))

        assert loaded_state is not None
        assert loaded_state["step"] == 100
        assert loaded_state["epoch"] == 1
        assert loaded_state["loss"] == 0.5

    def test_load_nonexistent_state(self, tmp_path):
        """Test loading nonexistent training state."""
        from src.training.checkpointing import load_training_state

        state = load_training_state(str(tmp_path))
        assert state is None
