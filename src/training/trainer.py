"""
Model training utilities and trainer class.

This module implements the training loop for fine-tuning language models
with LoRA/QLoRA.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, PreTrainedModel,
                          PreTrainedTokenizer, get_linear_schedule_with_warmup)

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model settings
    model_name_or_path: str
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Hardware settings
    fp16: bool = False
    bf16: bool = True
    use_8bit: bool = False
    use_4bit: bool = False

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "outputs"

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500

    # Data settings
    max_seq_length: int = 2048
    seed: int = 42


class Trainer:
    """Trainer for fine-tuning language models with LoRA."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Pre-trained model
            tokenizer: Tokenizer instance
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        logger.info(
            f"Trainer initialized on device: {self.device}",
            extra={
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = (
            len(self.train_dataloader)
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self) -> Dict[str, float]:
        """
        Run training loop.

        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting training...")
        self.model.train()

        total_loss = 0
        best_eval_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch()
            total_loss += epoch_loss

            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} " f"- Loss: {epoch_loss:.4f}")

            # Evaluate if eval dataloader provided
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                logger.info(f"Evaluation Loss: {eval_loss:.4f}")

                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.save_checkpoint("best_model")

            # Save checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

        avg_loss = total_loss / self.config.num_epochs

        logger.info(
            f"Training completed - Avg Loss: {avg_loss:.4f}",
            extra={"total_steps": self.global_step},
        )

        return {"avg_loss": avg_loss, "best_eval_loss": best_eval_loss}

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        epoch_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")

        for step, batch in enumerate(progress_bar):
            loss = self._training_step(batch)
            epoch_loss += loss

            # Logging
            if self.global_step % self.config.logging_steps == 0 and step > 0:
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return epoch_loss / len(self.train_dataloader)

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            Average evaluation loss
        """
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided")
            return 0.0

        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.eval_dataloader)
        self.model.train()

        return avg_loss

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir) / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(
            f"Checkpoint saved to {output_dir}",
            extra={"step": self.global_step, "epoch": self.current_epoch},
        )


def setup_lora_model(
    model_name_or_path: str,
    config: TrainingConfig,
) -> PreTrainedModel:
    """
    Load model and apply LoRA configuration.

    Args:
        model_name_or_path: Model name or path
        config: Training configuration

    Returns:
        Model with LoRA adapters
    """
    logger.info(f"Loading model: {model_name_or_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    if config.use_lora:
        logger.info("Applying LoRA configuration")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info("Model setup complete")
    return model
