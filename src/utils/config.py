"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AzureConfig(BaseModel):
    """Azure configuration settings."""

    subscription_id: str = Field(..., description="Azure subscription ID")
    tenant_id: str = Field(..., description="Azure tenant ID")
    resource_group: str = Field(..., description="Resource group name")
    location: str = Field(default="eastus", description="Azure region")
    workspace_name: str = Field(..., description="Azure ML workspace name")
    compute_name: str = Field(default="gpu-cluster", description="Compute cluster name")


class StorageConfig(BaseModel):
    """Storage configuration settings."""

    account_name: str = Field(..., description="Storage account name")
    container_name: str = Field(default="training-data", description="Blob container name")


class ModelConfig(BaseModel):
    """Model configuration settings."""

    name: str = Field(default="microsoft/phi-4", description="Model name or path")
    version: str = Field(default="latest", description="Model version")
    base_path: Path = Field(default=Path("./models/base"), description="Base model path")
    trained_path: Path = Field(default=Path("./models/trained"), description="Trained model path")
    optimized_path: Path = Field(
        default=Path("./models/optimized"), description="Optimized model path"
    )


class TrainingConfig(BaseModel):
    """Training configuration settings."""

    batch_size: int = Field(default=4, ge=1, description="Training batch size")
    learning_rate: float = Field(default=2e-5, gt=0, description="Learning rate")
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    max_seq_length: int = Field(default=512, ge=1, description="Maximum sequence length")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint every N steps")


class InferenceConfig(BaseModel):
    """Inference configuration settings."""

    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    model_format: str = Field(default="onnx", description="Model format (onnx, pytorch)")
    quantization_level: str = Field(default="int8", description="Quantization level")
    timeout: int = Field(default=10, ge=1, description="Request timeout in seconds")
    max_memory_gb: int = Field(default=4, ge=1, description="Max memory usage in GB")


class Config(BaseModel):
    """Main configuration container."""

    azure: AzureConfig
    storage: StorageConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig

    @classmethod
    def from_env(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables and optional YAML file.

        Args:
            config_path: Optional path to YAML configuration file

        Returns:
            Config instance
        """
        # Load environment variables
        load_dotenv()

        # Start with environment-based config
        config_dict: Dict[str, Any] = {
            "azure": {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
                "tenant_id": os.getenv("AZURE_TENANT_ID", ""),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP", ""),
                "location": os.getenv("AZURE_LOCATION", "eastus"),
                "workspace_name": os.getenv("AZUREML_WORKSPACE_NAME", ""),
                "compute_name": os.getenv("AZUREML_COMPUTE_NAME", "gpu-cluster"),
            },
            "storage": {
                "account_name": os.getenv("AZURE_STORAGE_ACCOUNT_NAME", ""),
                "container_name": os.getenv("AZURE_STORAGE_CONTAINER_NAME", "training-data"),
            },
            "model": {
                "name": os.getenv("MODEL_NAME", "microsoft/phi-4"),
                "version": os.getenv("MODEL_VERSION", "latest"),
                "base_path": os.getenv("BASE_MODEL_PATH", "./models/base"),
                "trained_path": os.getenv("TRAINED_MODEL_PATH", "./models/trained"),
                "optimized_path": os.getenv("OPTIMIZED_MODEL_PATH", "./models/optimized"),
            },
            "training": {
                "batch_size": int(os.getenv("BATCH_SIZE", "4")),
                "learning_rate": float(os.getenv("LEARNING_RATE", "2e-5")),
                "num_epochs": int(os.getenv("NUM_EPOCHS", "3")),
                "max_seq_length": int(os.getenv("MAX_SEQ_LENGTH", "512")),
                "warmup_steps": int(os.getenv("WARMUP_STEPS", "100")),
                "save_steps": int(os.getenv("SAVE_STEPS", "500")),
            },
            "inference": {
                "port": int(os.getenv("INFERENCE_PORT", "8000")),
                "model_format": os.getenv("MODEL_FORMAT", "onnx"),
                "quantization_level": os.getenv("QUANTIZATION_LEVEL", "int8"),
                "timeout": int(os.getenv("INFERENCE_TIMEOUT", "10")),
                "max_memory_gb": int(os.getenv("MAX_MEMORY_GB", "4")),
            },
        }

        # Override with YAML config if provided
        if config_path and config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    _deep_update(config_dict, yaml_config)

        return cls(**config_dict)


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Deep update base_dict with update_dict.

    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load application configuration.

    Args:
        config_path: Optional path to YAML configuration file

    Returns:
        Config instance
    """
    return Config.from_env(config_path)
