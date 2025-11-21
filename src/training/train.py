"""Main training script for fine-tuning language models.

Handles full pipeline: data loading, model setup, training loop, checkpointing,
and optional MLflow tracking.
"""
from src.data.dataset_loader import create_data_collator, load_tokenizer
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader

# Insert repository root (parent of 'src') into sys.path before importing src.*
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    # ensure repository root added for src.* imports
    sys.path.insert(0, str(project_root))  # noqa: E402

from src.data.dataset_loader import ConversationDataset  # noqa: E402
from src.training.checkpointing import CheckpointManager  # noqa: E402
from src.training.trainer import (Trainer, TrainingConfig,  # noqa: E402
                                  setup_lora_model)
from src.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

mlflow: Any = None  # will be set if import succeeds
try:
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available, tracking disabled")


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config: dict) -> Optional[str]:
    """Setup MLflow tracking if enabled; return run id or None.

    Note: When running in Azure ML, skip all MLflow setup - Azure ML handles
    tracking automatically. Only set up MLflow for local development.
    """
    mlflow_cfg = config.get("logging", {}).get("mlflow", {})
    if not (MLFLOW_AVAILABLE and mlflow_cfg.get("enabled", False)):
        return None

    # Detect Azure ML environment - check for AZUREML_RUN_ID env var
    is_azure_ml = os.environ.get("AZUREML_RUN_ID") is not None

    if is_azure_ml:
        # Azure ML environment - do nothing, platform handles everything
        logger.info("Running in Azure ML - using platform-managed tracking")
        active = mlflow.active_run()  # type: ignore[attr-defined]
        return active.info.run_id if active else None
    else:
        # Local environment only - set up tracking manually
        mlflow_config = mlflow_cfg
        if mlflow_config.get("tracking_uri"):
            mlflow.set_tracking_uri(  # type: ignore[attr-defined]
                mlflow_config["tracking_uri"]
            )

        experiment_name = mlflow_config.get(
            "experiment_name", "model-training"
        )
        mlflow.set_experiment(experiment_name)  # type: ignore[attr-defined]

        run_name = mlflow_config.get("run_name")
        mlflow.start_run(run_name=run_name)  # type: ignore[attr-defined]
        logger.info(f"MLflow tracking started: {experiment_name}")
        active = mlflow.active_run()  # type: ignore[attr-defined]

        # Log parameters
        if active:
            mlflow.log_params(  # type: ignore[attr-defined]
                {
                    "model": config["model"]["name_or_path"],
                    "num_epochs": config["training"]["num_epochs"],
                    "batch_size": config["training"][
                        "per_device_train_batch_size"
                    ],
                    "learning_rate": config["training"]["learning_rate"],
                    "lora_r": config["lora"]["r"],
                    "lora_alpha": config["lora"]["alpha"],
                }
            )
            return active.info.run_id if active else None

    return None


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting training script...")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Setup MLflow
    setup_mlflow(config)

    # Set random seed
    seed = config["hardware"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load tokenizer
    model_path = config["model"]["name_or_path"]
    tokenizer = load_tokenizer(model_path)

    # Load datasets
    logger.info("Loading training data...")
    train_dataset = ConversationDataset(
        data_path=config["data"]["train_file"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_seq_length"],
        prompt_column=config["data"]["prompt_column"],
        completion_column=config["data"]["completion_column"],
    )

    eval_dataset = None
    if config["data"].get("validation_file"):
        logger.info("Loading validation data...")
        eval_dataset = ConversationDataset(
            data_path=config["data"]["validation_file"],
            tokenizer=tokenizer,
            max_length=config["data"]["max_seq_length"],
            prompt_column=config["data"]["prompt_column"],
            completion_column=config["data"]["completion_column"],
        )

    # Create data loaders
    data_collator = create_data_collator(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config["data"].get("dataloader_num_workers", 0),
        pin_memory=config["data"].get("dataloader_pin_memory", False),
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config["training"]["per_device_eval_batch_size"],
            shuffle=False,
            collate_fn=data_collator,
            num_workers=config["data"].get("dataloader_num_workers", 0),
            pin_memory=config["data"].get("dataloader_pin_memory", False),
        )

    # Setup model
    logger.info("Setting up model...")
    training_config = TrainingConfig(
        model_name_or_path=model_path,
        use_lora=config["lora"]["enabled"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        lora_target_modules=config["lora"]["target_modules"],
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"][
            "gradient_accumulation_steps"
        ],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        use_8bit=config["training"]["use_8bit"],
        use_4bit=config["training"]["use_4bit"],
        save_steps=config["checkpointing"]["save_steps"],
        save_total_limit=config["checkpointing"]["save_total_limit"],
        output_dir=config["checkpointing"]["output_dir"],
        logging_steps=config["logging"]["logging_steps"],
        eval_steps=config["evaluation"]["eval_steps"],
        max_seq_length=config["data"]["max_seq_length"],
        seed=seed,
    )

    model = setup_lora_model(model_path, training_config)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
    )

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=config["checkpointing"]["output_dir"],
        max_checkpoints=config["checkpointing"]["save_total_limit"],
    )

    # Resume from checkpoint if specified
    if config["checkpointing"].get("resume_from_checkpoint"):
        checkpoint_path = config["checkpointing"]["resume_from_checkpoint"]
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint_manager.load_checkpoint(checkpoint_path, model)

    # Train model
    logger.info("Starting training loop...")
    try:
        metrics = trainer.train()

        # Log final metrics
        if (
            MLFLOW_AVAILABLE
            and mlflow.active_run()  # type: ignore[attr-defined]
        ):
            mlflow.log_metrics(metrics)  # type: ignore[attr-defined]

        logger.info(
            "Training completed successfully",
            extra={"metrics": metrics},
        )

        # Save final model
        final_checkpoint = checkpoint_manager.save_checkpoint(
            model=model,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            loss=metrics["avg_loss"],
            is_best=True,
        )

        logger.info(f"Final model saved: {final_checkpoint}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        # End MLflow run only if we started it (not in Azure ML)
        if MLFLOW_AVAILABLE:
            active = mlflow.active_run()  # type: ignore[attr-defined]
            # Only end run if it was started locally (has run_name we set)
            if active and active.data.tags.get("mlflow.runName"):
                mlflow.end_run()  # type: ignore[attr-defined]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train language model with LoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Training script failed: {e}", exc_info=True)
        sys.exit(1)
