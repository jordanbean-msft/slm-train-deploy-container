"""
Utilities for downloading trained models from Azure ML.

This module extends model_loader.py with functions for downloading
trained model artifacts from Azure ML jobs and model registry.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelDownloader:
    """Handles downloading trained models from Azure ML."""

    def __init__(
        self,
        ml_client: Optional[MLClient] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        workspace_name: Optional[str] = None,
    ):
        """
        Initialize model downloader.

        Args:
            ml_client: Azure ML client (created if not provided)
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            workspace_name: Workspace name
        """
        if ml_client is None:
            credential = DefaultAzureCredential()
            if all([subscription_id, resource_group, workspace_name]):
                self.ml_client = MLClient(
                    credential=credential,
                    subscription_id=subscription_id,
                    resource_group_name=resource_group,
                    workspace_name=workspace_name,
                )
            else:
                self.ml_client = MLClient.from_config(credential=credential)
        else:
            self.ml_client = ml_client

        logger.info(f"Model downloader initialized for: " f"{self.ml_client.workspace_name}")

    def download_from_job(
        self,
        job_name: str,
        output_path: str,
        checkpoint: str = "best_model",
    ) -> Path:
        """
        Download model from training job outputs.

        Args:
            job_name: Training job name/ID
            output_path: Local directory for download
            checkpoint: Checkpoint to download (e.g., 'best_model',
                       'checkpoint-step-1000')

        Returns:
            Path to downloaded model
        """
        logger.info(f"Downloading model from job: {job_name}, " f"checkpoint: {checkpoint}")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download job outputs
        self.ml_client.jobs.download(
            name=job_name,
            download_path=str(output_dir),
            output_name="outputs",
        )

        # Find checkpoint directory
        checkpoint_path = output_dir / "named-outputs" / "outputs" / checkpoint

        if not checkpoint_path.exists():
            # Try alternative path structure
            checkpoint_path = output_dir / "outputs" / checkpoint
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        logger.info(f"Model downloaded to: {checkpoint_path}")
        return checkpoint_path

    def download_from_registry(
        self,
        model_name: str,
        version: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        Download model from Azure ML model registry.

        Args:
            model_name: Name of registered model
            version: Model version (latest if not specified)
            output_path: Local directory for download

        Returns:
            Path to downloaded model
        """
        logger.info(f"Downloading model from registry: {model_name}")

        # Get model from registry
        if version:
            model = self.ml_client.models.get(name=model_name, version=version)
        else:
            # Get latest version
            models = list(self.ml_client.models.list(name=model_name, latest=True))
            if not models:
                raise ValueError(f"Model not found: {model_name}")
            model = models[0]

        logger.info(f"Found model: {model.name} version {model.version}")

        # Download model
        if output_path is None:
            output_path = f"models/{model_name}/v{model.version}"

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download using model path
        self.ml_client.models.download(
            name=model.name,
            version=model.version,
            download_path=str(output_dir),
        )

        logger.info(f"Model downloaded to: {output_dir}")
        return output_dir

    def list_job_checkpoints(self, job_name: str) -> List[str]:
        """
        List available checkpoints in job outputs.

        Args:
            job_name: Training job name/ID

        Returns:
            List of checkpoint names
        """
        logger.info(f"Listing checkpoints for job: {job_name}")

        # Download outputs metadata to temp location
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            self.ml_client.jobs.download(
                name=job_name,
                download_path=temp_dir,
                output_name="outputs",
            )

            # Look for checkpoint directories
            checkpoints = []
            outputs_dir = Path(temp_dir) / "named-outputs" / "outputs"

            if not outputs_dir.exists():
                outputs_dir = Path(temp_dir) / "outputs"

            if outputs_dir.exists():
                for item in outputs_dir.iterdir():
                    if item.is_dir():
                        checkpoints.append(item.name)

        logger.info(f"Found {len(checkpoints)} checkpoints")
        return sorted(checkpoints)

    def get_best_checkpoint(self, job_name: str, metric: str = "eval_loss") -> str:
        """
        Identify best checkpoint based on metric.

        Args:
            job_name: Training job name/ID
            metric: Metric to compare (default: eval_loss)

        Returns:
            Name of best checkpoint
        """
        logger.info(f"Finding best checkpoint for job: {job_name} " f"(metric: {metric})")

        checkpoints = self.list_job_checkpoints(job_name)

        # If best_model exists, return it
        if "best_model" in checkpoints:
            return "best_model"

        # Otherwise, need to check checkpoint metadata
        # This requires downloading and parsing metadata files
        logger.warning("No 'best_model' found. Returning latest checkpoint.")

        # Return checkpoint with highest step number
        step_checkpoints = [c for c in checkpoints if c.startswith("checkpoint-step-")]
        if step_checkpoints:
            return max(
                step_checkpoints,
                key=lambda x: int(x.split("-")[-1]),
            )

        return checkpoints[-1] if checkpoints else "best_model"

    def validate_model_files(self, model_path: Path) -> Dict[str, bool]:
        """
        Validate downloaded model files.

        Args:
            model_path: Path to downloaded model

        Returns:
            Dictionary of validation results
        """
        logger.info(f"Validating model files in: {model_path}")

        required_files = {
            "config.json": False,
            "tokenizer_config.json": False,
            "pytorch_model.bin or model.safetensors": False,
        }

        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return required_files

        # Check for config files
        config_file = model_path / "config.json"
        required_files["config.json"] = config_file.exists()

        tokenizer_config = model_path / "tokenizer_config.json"
        required_files["tokenizer_config.json"] = tokenizer_config.exists()

        # Check for model weights
        pytorch_model = model_path / "pytorch_model.bin"
        safetensors = model_path / "model.safetensors"
        adapter_model = model_path / "adapter_model.bin"

        required_files["pytorch_model.bin or model.safetensors"] = (
            pytorch_model.exists() or safetensors.exists() or adapter_model.exists()
        )

        # Log results
        all_valid = all(required_files.values())
        if all_valid:
            logger.info("✓ All required files present")
        else:
            logger.warning("⚠️  Some required files missing")
            for file, present in required_files.items():
                status = "✓" if present else "✗"
                logger.info(f"  {status} {file}")

        return required_files

    def extract_model_metadata(self, model_path: Path) -> Dict:
        """
        Extract metadata from downloaded model.

        Args:
            model_path: Path to downloaded model

        Returns:
            Dictionary of model metadata
        """
        logger.info(f"Extracting metadata from: {model_path}")

        metadata = {
            "path": str(model_path),
            "size_mb": 0,
            "config": {},
            "tokenizer_config": {},
        }

        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        metadata["size_mb"] = total_size / (1024 * 1024)

        # Load config.json
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                metadata["config"] = json.load(f)

        # Load tokenizer_config.json
        tokenizer_config = model_path / "tokenizer_config.json"
        if tokenizer_config.exists():
            with open(tokenizer_config) as f:
                metadata["tokenizer_config"] = json.load(f)

        # Extract key information
        config = metadata["config"]
        metadata["model_type"] = config.get("model_type", "unknown")
        metadata["vocab_size"] = config.get("vocab_size", 0)
        metadata["hidden_size"] = config.get("hidden_size", 0)
        metadata["num_layers"] = config.get("num_hidden_layers", config.get("n_layer", 0))

        logger.info(f"Model metadata: {metadata['model_type']}, " f"{metadata['size_mb']:.1f} MB")

        return metadata

    def list_registered_models(self, name_filter: Optional[str] = None) -> List[Dict]:
        """
        List models in registry.

        Args:
            name_filter: Optional name filter

        Returns:
            List of model information dictionaries
        """
        logger.info("Listing registered models")

        models = []
        for model in self.ml_client.models.list():
            if name_filter and name_filter not in model.name:
                continue

            models.append(
                {
                    "name": model.name,
                    "version": model.version,
                    "description": model.description,
                    "tags": model.tags,
                    "created": str(model.creation_context.created_at),
                }
            )

        logger.info(f"Found {len(models)} registered models")
        return models


def download_trained_model(
    job_name: str,
    output_path: str,
    checkpoint: str = "best_model",
    validate: bool = True,
) -> Path:
    """
    Convenience function to download and validate trained model.

    Args:
        job_name: Training job name/ID
        output_path: Local directory for download
        checkpoint: Checkpoint to download
        validate: Whether to validate downloaded files

    Returns:
        Path to downloaded model
    """
    downloader = ModelDownloader()

    # Download model
    model_path = downloader.download_from_job(
        job_name=job_name,
        output_path=output_path,
        checkpoint=checkpoint,
    )

    # Validate if requested
    if validate:
        validation_results = downloader.validate_model_files(model_path)
        if not all(validation_results.values()):
            logger.warning("Model validation found missing files. " "Model may not work correctly.")

    # Extract metadata
    metadata = downloader.extract_model_metadata(model_path)
    logger.info(f"Model ready: {metadata['model_type']}")

    return model_path
