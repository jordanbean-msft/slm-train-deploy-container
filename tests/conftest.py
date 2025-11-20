"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import (AzureConfig, Config, InferenceConfig,
                              ModelConfig, StorageConfig, TrainingConfig)


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_jsonl_path(test_data_dir: Path) -> Path:
    """Path to sample JSONL file."""
    return test_data_dir / "sample_data.jsonl"


@pytest.fixture
def mock_azure_config() -> AzureConfig:
    """Mock Azure configuration."""
    return AzureConfig(
        subscription_id="test-subscription-id",
        tenant_id="test-tenant-id",
        resource_group="test-rg",
        location="eastus",
        workspace_name="test-workspace",
        compute_name="test-compute",
    )


@pytest.fixture
def mock_storage_config() -> StorageConfig:
    """Mock storage configuration."""
    return StorageConfig(
        account_name="teststorage",
        container_name="training-data",
    )


@pytest.fixture
def mock_model_config(tmp_path: Path) -> ModelConfig:
    """Mock model configuration."""
    return ModelConfig(
        name="microsoft/phi-4",
        version="latest",
        base_path=tmp_path / "models" / "base",
        trained_path=tmp_path / "models" / "trained",
        optimized_path=tmp_path / "models" / "optimized",
    )


@pytest.fixture
def mock_training_config() -> TrainingConfig:
    """Mock training configuration."""
    return TrainingConfig(
        batch_size=2,
        learning_rate=1e-5,
        num_epochs=1,
        max_seq_length=128,
        warmup_steps=10,
        save_steps=50,
    )


@pytest.fixture
def mock_inference_config() -> InferenceConfig:
    """Mock inference configuration."""
    return InferenceConfig(
        port=8000,
        model_format="onnx",
        quantization_level="int8",
        timeout=10,
        max_memory_gb=4,
    )


@pytest.fixture
def mock_config(
    mock_azure_config: AzureConfig,
    mock_storage_config: StorageConfig,
    mock_model_config: ModelConfig,
    mock_training_config: TrainingConfig,
    mock_inference_config: InferenceConfig,
) -> Config:
    """Mock complete configuration."""
    return Config(
        azure=mock_azure_config,
        storage=mock_storage_config,
        model=mock_model_config,
        training=mock_training_config,
        inference=mock_inference_config,
    )


@pytest.fixture
def mock_ml_client() -> Generator[MagicMock, None, None]:
    """Mock Azure ML client."""
    with patch("azure.ai.ml.MLClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_blob_service_client() -> Generator[MagicMock, None, None]:
    """Mock Azure Blob Service client."""
    with patch("azure.storage.blob.BlobServiceClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path: Path) -> Generator[None, None, None]:
    """Setup test environment variables."""
    os.environ["AZURE_SUBSCRIPTION_ID"] = "test-sub"
    os.environ["AZURE_TENANT_ID"] = "test-tenant"
    os.environ["AZURE_RESOURCE_GROUP"] = "test-rg"
    os.environ["AZUREML_WORKSPACE_NAME"] = "test-workspace"
    os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "teststorage"

    yield

    # Cleanup
    for key in [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_TENANT_ID",
        "AZURE_RESOURCE_GROUP",
        "AZUREML_WORKSPACE_NAME",
        "AZURE_STORAGE_ACCOUNT_NAME",
    ]:
        os.environ.pop(key, None)
