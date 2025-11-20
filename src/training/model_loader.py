"""
Model download utilities for Azure AI Foundry.

This module handles downloading models from Azure AI Foundry catalog
and registering them in Azure ML workspace.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceExistsError

from .azure_auth import get_ml_client
from .config import AzureConfig
from .logging_config import get_logger

logger = get_logger(__name__)


def download_model_from_foundry(
    model_name: str,
    output_dir: Path,
    ml_client: Optional[MLClient] = None,
    config: Optional[AzureConfig] = None,
) -> Path:
    """
    Download a model from Azure AI Foundry catalog.

    Args:
        model_name: Name of the model in AI Foundry (e.g., 'microsoft/phi-4')
        output_dir: Directory to save downloaded model
        ml_client: Optional Azure ML client (created if not provided)
        config: Optional Azure configuration

    Returns:
        Path to downloaded model directory

    Raises:
        Exception: If download fails
    """
    if ml_client is None:
        if config is None:
            from .config import load_config

            config = load_config().azure
        ml_client = get_ml_client(config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading model '{model_name}' from Azure AI Foundry",
        extra={"model_name": model_name, "output_dir": str(output_dir)},
    )

    try:
        # Azure AI Foundry models are accessed through the registry
        # Format: azureml://registries/<registry>/models/<model>/versions/<version>

        # Parse model name
        if "/" in model_name:
            parts = model_name.split("/")
            registry_model = parts[-1]  # e.g., phi-4
        else:
            registry_model = model_name

        logger.info(f"Searching for model: {registry_model}")

        # Try to get model from Azure AI Foundry registry
        # The model path for Foundry is typically azureml://registries/azureml/models/<model>
        foundry_registry = "azureml"

        try:
            # List available versions
            model_versions = ml_client.models.list(
                name=registry_model,
            )

            latest_version = None
            for model_version in model_versions:
                if latest_version is None or int(model_version.version) > int(
                    latest_version.version
                ):
                    latest_version = model_version

            if latest_version is None:
                raise Exception(f"No versions found for model {registry_model}")

            logger.info(
                f"Found model version: {latest_version.version}",
                extra={"version": latest_version.version},
            )

            # Download model files
            model_path = ml_client.models.download(
                name=registry_model,
                version=latest_version.version,
                download_path=output_dir,
            )

            logger.info(
                f"Successfully downloaded model to {model_path}",
                extra={"model_path": str(model_path)},
            )

            return Path(model_path)

        except Exception as e:
            logger.warning(
                f"Could not download from workspace models: {e}. " f"Attempting registry access..."
            )

            # Alternative: Access through registry client
            # This requires the model to be in Azure AI Foundry registry
            registry_client = (
                ml_client._registry_client if hasattr(ml_client, "_registry_client") else None
            )

            if registry_client is None:
                raise Exception(
                    f"Model '{model_name}' not found in workspace. "
                    f"Ensure the model is available in Azure AI Foundry "
                    f"and you have access permissions."
                )

            raise

    except Exception as e:
        logger.error(
            f"Failed to download model '{model_name}': {e}",
            exc_info=True,
        )
        raise


def register_model_in_workspace(
    model_name: str,
    model_path: Path,
    model_type: str = "custom_model",
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    ml_client: Optional[MLClient] = None,
    config: Optional[AzureConfig] = None,
) -> Model:
    """
    Register a model in Azure ML workspace.

    Args:
        model_name: Name for the registered model
        model_path: Path to model files
        model_type: Type of model (default: custom_model)
        description: Optional model description
        tags: Optional tags for the model
        ml_client: Optional Azure ML client
        config: Optional Azure configuration

    Returns:
        Registered Model object

    Raises:
        Exception: If registration fails
    """
    if ml_client is None:
        if config is None:
            from .config import load_config

            config = load_config().azure
        ml_client = get_ml_client(config)

    if tags is None:
        tags = {}

    if description is None:
        description = f"Model downloaded from Azure AI Foundry: {model_name}"

    logger.info(
        f"Registering model '{model_name}' in workspace",
        extra={"model_path": str(model_path), "type": model_type},
    )

    try:
        model = Model(
            path=str(model_path),
            name=model_name,
            description=description,
            type=model_type,
            tags=tags,
        )

        registered_model = ml_client.models.create_or_update(model)

        logger.info(
            f"Successfully registered model: {registered_model.name} v{registered_model.version}",
            extra={
                "model_id": registered_model.id,
                "version": registered_model.version,
            },
        )

        return registered_model

    except ResourceExistsError:
        logger.warning(f"Model '{model_name}' already exists, retrieving...")
        existing_model = ml_client.models.get(name=model_name, label="latest")
        return existing_model
    except Exception as e:
        logger.error(
            f"Failed to register model '{model_name}': {e}",
            exc_info=True,
        )
        raise


def verify_model_files(model_path: Path) -> Dict[str, any]:
    """
    Verify model files exist and are valid.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary with validation results
    """
    logger.info(f"Verifying model files in {model_path}")

    if not model_path.exists():
        return {
            "valid": False,
            "error": "Model path does not exist",
            "path": str(model_path),
        }

    if not model_path.is_dir():
        return {
            "valid": False,
            "error": "Model path is not a directory",
            "path": str(model_path),
        }

    # Check for common model files
    files_found = {
        "config": (model_path / "config.json").exists(),
        "pytorch_model": any(
            (model_path / f).exists() for f in ["pytorch_model.bin", "model.safetensors"]
        ),
        "tokenizer": (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists(),
    }

    all_files = list(model_path.rglob("*"))
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())

    result = {
        "valid": any(files_found.values()),
        "path": str(model_path),
        "files_found": files_found,
        "total_files": len([f for f in all_files if f.is_file()]),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

    if result["valid"]:
        logger.info(
            f"Model validation successful: {result['total_files']} files, "
            f"{result['total_size_mb']} MB"
        )
    else:
        logger.warning(f"Model validation failed: no expected files found")

    return result


def download_and_register_model(
    model_name: str,
    output_dir: Path,
    registered_name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    ml_client: Optional[MLClient] = None,
    config: Optional[AzureConfig] = None,
    verify: bool = True,
) -> tuple[Path, Model]:
    """
    Download model from AI Foundry and register in workspace.

    This is a convenience function that combines download, verification,
    and registration.

    Args:
        model_name: Name of model in AI Foundry
        output_dir: Directory to save model
        registered_name: Name for registered model (defaults to model_name)
        description: Model description
        tags: Model tags
        ml_client: Optional Azure ML client
        config: Optional Azure configuration
        verify: Whether to verify model files after download

    Returns:
        Tuple of (model_path, registered_model)
    """
    if registered_name is None:
        # Clean up model name for registration
        registered_name = model_name.replace("/", "-").replace("_", "-")

    logger.info(
        f"Starting download and registration for '{model_name}'",
        extra={"registered_name": registered_name},
    )

    # Download model
    start_time = time.time()
    model_path = download_model_from_foundry(
        model_name=model_name,
        output_dir=output_dir,
        ml_client=ml_client,
        config=config,
    )
    download_time = time.time() - start_time

    logger.info(f"Download completed in {download_time:.2f} seconds")

    # Verify model files
    if verify:
        validation = verify_model_files(model_path)
        if not validation["valid"]:
            raise Exception(f"Model validation failed: {validation.get('error', 'Unknown error')}")

    # Register model
    registered_model = register_model_in_workspace(
        model_name=registered_name,
        model_path=model_path,
        description=description,
        tags=tags,
        ml_client=ml_client,
        config=config,
    )

    logger.info(
        f"Model '{model_name}' successfully downloaded and registered",
        extra={
            "download_time": download_time,
            "model_version": registered_model.version,
        },
    )

    return model_path, registered_model


def list_available_foundry_models(
    ml_client: Optional[MLClient] = None,
    config: Optional[AzureConfig] = None,
) -> list[str]:
    """
    List available models in Azure AI Foundry.

    Note: This is a placeholder. Actual implementation would query
    the Foundry registry.

    Args:
        ml_client: Optional Azure ML client
        config: Optional Azure configuration

    Returns:
        List of available model names
    """
    # Common models available in Azure AI Foundry
    common_models = [
        "microsoft/phi-4",
        "microsoft/phi-3-mini-4k-instruct",
        "microsoft/phi-3-small-8k-instruct",
        "microsoft/phi-3-medium-4k-instruct",
        "mistralai/mistral-7b-v0.1",
        "meta-llama/llama-2-7b",
        "meta-llama/llama-2-13b",
    ]

    logger.info(f"Returning {len(common_models)} common models")
    return common_models
