"""Unit tests for configuration management."""

from pathlib import Path

import pytest

from src.utils.config import AzureConfig, Config, load_config


def test_azure_config_creation(mock_azure_config: AzureConfig) -> None:
    """Test Azure configuration creation."""
    assert mock_azure_config.subscription_id == "test-subscription-id"
    assert mock_azure_config.tenant_id == "test-tenant-id"
    assert mock_azure_config.resource_group == "test-rg"
    assert mock_azure_config.location == "eastus"
    assert mock_azure_config.workspace_name == "test-workspace"


def test_config_from_env(mock_config: Config) -> None:
    """Test configuration loading from environment."""
    assert mock_config.azure.subscription_id == "test-subscription-id"
    assert mock_config.storage.account_name == "teststorage"
    assert mock_config.model.name == "microsoft/phi-4"
    assert mock_config.training.batch_size == 2
    assert mock_config.inference.port == 8000


def test_load_config() -> None:
    """Test loading configuration."""
    config = load_config()
    assert config is not None
    assert isinstance(config, Config)
    assert config.azure.subscription_id == "test-sub"


def test_training_config_validation(mock_training_config) -> None:
    """Test training configuration validation."""
    assert mock_training_config.batch_size >= 1
    assert mock_training_config.learning_rate > 0
    assert mock_training_config.num_epochs >= 1
    assert mock_training_config.save_steps >= 1


def test_inference_config_validation(mock_inference_config) -> None:
    """Test inference configuration validation."""
    assert 1024 <= mock_inference_config.port <= 65535
    assert mock_inference_config.timeout >= 1
    assert mock_inference_config.max_memory_gb >= 1
