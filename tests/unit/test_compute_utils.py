"""
Tests for compute cluster management utilities.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

from src.utils.compute_utils import (create_compute_cluster,
                                     delete_compute_cluster,
                                     estimate_compute_cost,
                                     get_available_vm_sizes,
                                     get_compute_cluster, get_compute_status,
                                     list_compute_clusters,
                                     wait_for_compute_ready)


@pytest.fixture
def mock_ml_client():
    """Create a mock Azure ML client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_compute():
    """Create a mock compute cluster."""
    compute = Mock(spec=AmlCompute)
    compute.name = "test-cluster"
    compute.type = "amlcompute"
    compute.provisioning_state = "Succeeded"
    compute.size = "Standard_NC6s_v3"
    compute.min_instances = 0
    compute.max_instances = 4
    compute.idle_time_before_scale_down = 300
    compute.tier = "dedicated"
    compute.id = "/subscriptions/test/resourceGroups/test/providers/Microsoft.MachineLearningServices/workspaces/test/computes/test-cluster"
    return compute


def test_get_compute_cluster_found(mock_ml_client, mock_compute):
    """Test getting an existing compute cluster."""
    mock_ml_client.compute.get.return_value = mock_compute

    result = get_compute_cluster(mock_ml_client, "test-cluster")

    assert result is not None
    assert result.name == "test-cluster"
    mock_ml_client.compute.get.assert_called_once_with("test-cluster")


def test_get_compute_cluster_not_found(mock_ml_client):
    """Test getting a non-existent compute cluster."""
    mock_ml_client.compute.get.side_effect = ResourceNotFoundError()

    result = get_compute_cluster(mock_ml_client, "nonexistent")

    assert result is None


def test_list_compute_clusters(mock_ml_client, mock_compute):
    """Test listing all compute clusters."""
    mock_ml_client.compute.list.return_value = [mock_compute]

    result = list_compute_clusters(mock_ml_client)

    assert len(result) == 1
    assert result[0].name == "test-cluster"


def test_create_compute_cluster_new(mock_ml_client, mock_compute):
    """Test creating a new compute cluster."""
    mock_ml_client.compute.get.side_effect = ResourceNotFoundError()
    mock_operation = MagicMock()
    mock_operation.result.return_value = mock_compute
    mock_ml_client.compute.begin_create_or_update.return_value = mock_operation

    result = create_compute_cluster(mock_ml_client, "test-cluster", vm_size="Standard_NC6s_v3")

    assert result.name == "test-cluster"
    mock_ml_client.compute.begin_create_or_update.assert_called_once()


def test_create_compute_cluster_existing(mock_ml_client, mock_compute):
    """Test creating a compute cluster that already exists."""
    mock_ml_client.compute.get.return_value = mock_compute

    result = create_compute_cluster(mock_ml_client, "test-cluster")

    assert result.name == "test-cluster"
    # Should not attempt to create
    mock_ml_client.compute.begin_create_or_update.assert_not_called()


def test_get_compute_status(mock_ml_client, mock_compute):
    """Test getting compute cluster status."""
    mock_ml_client.compute.get.return_value = mock_compute

    status = get_compute_status(mock_ml_client, "test-cluster")

    assert status["exists"] is True
    assert status["name"] == "test-cluster"
    assert status["state"] == "Succeeded"
    assert status["vm_size"] == "Standard_NC6s_v3"
    assert status["min_instances"] == 0
    assert status["max_instances"] == 4


def test_get_compute_status_not_found(mock_ml_client):
    """Test getting status of non-existent cluster."""
    mock_ml_client.compute.get.side_effect = ResourceNotFoundError()

    status = get_compute_status(mock_ml_client, "nonexistent-cluster")

    assert status["exists"] is False
    assert status["state"] == "NotFound"


@patch("src.utils.compute_utils.time.sleep")
def test_wait_for_compute_ready_success(mock_sleep, mock_ml_client, mock_compute):
    """Test waiting for compute to be ready (success)."""
    mock_ml_client.compute.get.return_value = mock_compute

    result = wait_for_compute_ready(mock_ml_client, "test-cluster", timeout_seconds=30)

    assert result is True


@patch("src.utils.compute_utils.time.sleep")
@patch("src.utils.compute_utils.time.time")
def test_wait_for_compute_ready_timeout(mock_time, mock_sleep, mock_ml_client, mock_compute):
    """Test waiting for compute with timeout."""
    mock_compute.provisioning_state = "Creating"
    mock_ml_client.compute.get.return_value = mock_compute

    # Simulate time passing
    mock_time.side_effect = [0, 10, 20, 30, 40]

    result = wait_for_compute_ready(mock_ml_client, "test-cluster", timeout_seconds=30)

    assert result is False


@patch("src.utils.compute_utils.time.sleep")
def test_wait_for_compute_ready_failed(mock_sleep, mock_ml_client, mock_compute):
    """Test waiting for compute that fails."""
    mock_compute.provisioning_state = "Failed"
    mock_ml_client.compute.get.return_value = mock_compute

    with pytest.raises(Exception, match="Compute provisioning failed"):
        wait_for_compute_ready(mock_ml_client, "test-cluster")


def test_delete_compute_cluster(mock_ml_client, mock_compute):
    """Test deleting a compute cluster."""
    mock_ml_client.compute.get.return_value = mock_compute
    mock_operation = MagicMock()
    mock_ml_client.compute.begin_delete.return_value = mock_operation

    delete_compute_cluster(mock_ml_client, "test-cluster")

    mock_ml_client.compute.begin_delete.assert_called_once_with("test-cluster")
    mock_operation.wait.assert_called_once()


def test_delete_compute_cluster_not_found(mock_ml_client):
    """Test deleting a non-existent cluster."""
    mock_ml_client.compute.get.side_effect = ResourceNotFoundError()

    # Should not raise exception
    delete_compute_cluster(mock_ml_client, "nonexistent")

    mock_ml_client.compute.begin_delete.assert_not_called()


def test_get_available_vm_sizes(mock_ml_client):
    """Test getting available VM sizes."""
    vm_sizes = get_available_vm_sizes(mock_ml_client)

    assert len(vm_sizes) > 0
    assert all("name" in vm for vm in vm_sizes)
    assert all("gpus" in vm for vm in vm_sizes)
    assert all("gpu_type" in vm for vm in vm_sizes)


def test_estimate_compute_cost():
    """Test compute cost estimation."""
    cost = estimate_compute_cost(
        vm_size="Standard_NC6s_v3",
        max_instances=4,
        hours_per_day=8.0,
    )

    assert cost["vm_size"] == "Standard_NC6s_v3"
    assert cost["max_instances"] == 4
    assert cost["hours_per_day"] == 8.0
    assert cost["estimated_daily_cost"] > 0
    assert cost["estimated_monthly_cost"] > 0
    assert "note" in cost


def test_estimate_compute_cost_unknown_vm():
    """Test cost estimation for unknown VM size."""
    cost = estimate_compute_cost(vm_size="Unknown_VM_Size", max_instances=1, hours_per_day=1.0)

    assert cost["estimated_daily_cost"] == 0.0
    assert cost["estimated_monthly_cost"] == 0.0
