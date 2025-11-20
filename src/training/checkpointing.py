"""
Checkpoint management utilities for training.

This module provides utilities for saving, loading, and managing
training checkpoints.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages training checkpoints with rotation."""

    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 3,
        keep_best: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory for saving checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: Whether to always keep best checkpoint
        """
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoint metadata
        self.checkpoints: List[Dict] = []
        self.best_checkpoint: Optional[Path] = None
        self.best_metric: float = float("inf")

        logger.info(
            f"Checkpoint manager initialized: {output_dir}",
            extra={"max_checkpoints": max_checkpoints},
        )

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
        epoch: int,
        loss: float,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            step: Global training step
            epoch: Current epoch
            loss: Training loss
            metrics: Optional additional metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint-step-{step}"
        checkpoint_dir = self.output_dir / checkpoint_name

        # Save model and tokenizer
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics or {},
            "is_best": is_best,
        }

        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Track checkpoint
        self.checkpoints.append({"path": checkpoint_dir, "step": step, "loss": loss})

        # Handle best checkpoint
        if is_best or loss < self.best_metric:
            self.best_metric = loss
            self.best_checkpoint = checkpoint_dir
            self._save_best_checkpoint(checkpoint_dir)

        # Rotate checkpoints
        self._rotate_checkpoints()

        logger.info(
            f"Checkpoint saved: {checkpoint_dir}",
            extra={"step": step, "epoch": epoch, "loss": loss},
        )

        return checkpoint_dir

    def _save_best_checkpoint(self, checkpoint_dir: Path) -> None:
        """Save/update best checkpoint."""
        best_dir = self.output_dir / "best_model"

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(checkpoint_dir, best_dir)
        logger.info(f"Best checkpoint updated: {best_dir}")

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints if limit exceeded."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by step
        self.checkpoints.sort(key=lambda x: x["step"])

        # Remove oldest checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            checkpoint_path = old_checkpoint["path"]

            # Don't delete best checkpoint
            if self.keep_best and checkpoint_path == self.best_checkpoint:
                continue

            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, model: PreTrainedModel) -> Dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load weights into

        Returns:
            Checkpoint metadata
        """
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_dir}")

        logger.info(f"Loading checkpoint: {checkpoint_dir}")

        # Load model weights
        model_path = checkpoint_dir / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            # Try loading with from_pretrained
            model.from_pretrained(checkpoint_dir)

        # Load metadata
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        logger.info(
            "Checkpoint loaded successfully",
            extra={"metadata": metadata},
        )

        return metadata

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoints:
            return None

        latest = max(self.checkpoints, key=lambda x: x["step"])
        return latest["path"]

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None
        """
        return self.best_checkpoint

    def list_checkpoints(self) -> List[Dict]:
        """
        List all checkpoints.

        Returns:
            List of checkpoint metadata
        """
        return self.checkpoints.copy()


def save_training_state(
    output_dir: str,
    step: int,
    epoch: int,
    optimizer_state: Dict,
    scheduler_state: Dict,
    loss: float,
) -> None:
    """
    Save training state for resumption.

    Args:
        output_dir: Output directory
        step: Current step
        epoch: Current epoch
        optimizer_state: Optimizer state dict
        scheduler_state: Scheduler state dict
        loss: Current loss
    """
    state_path = Path(output_dir) / "training_state.pt"

    state = {
        "step": step,
        "epoch": epoch,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "loss": loss,
    }

    torch.save(state, state_path)
    logger.info(f"Training state saved: {state_path}")


def load_training_state(output_dir: str) -> Optional[Dict]:
    """
    Load training state for resumption.

    Args:
        output_dir: Output directory

    Returns:
        Training state dict or None
    """
    state_path = Path(output_dir) / "training_state.pt"

    if not state_path.exists():
        logger.warning(f"Training state not found: {state_path}")
        return None

    state = torch.load(state_path, map_location="cpu")
    logger.info(f"Training state loaded: {state_path}")

    return state
