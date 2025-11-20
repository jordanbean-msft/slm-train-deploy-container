"""
Utilities for managing Azure ML compute clusters.

This module provides functions for:
- Creating and provisioning compute clusters
- Checking cluster status and health
- Managing cluster lifecycle (start, stop, delete)
- Monitoring cluster metrics
"""

import time
from typing import Dict, List, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

from .logging_config import get_logger

logger = get_logger(__name__)


def get_compute_cluster(ml_client: MLClient, compute_name: str) -> Optional[AmlCompute]:
    """
    Get compute cluster by name.

    Args:
        ml_client: Azure ML client instance
        compute_name: Name of the compute cluster

    Returns:
        AmlCompute object if found, None otherwise
    """
    try:
        compute = ml_client.compute.get(compute_name)
        logger.info(
            f"Found compute cluster: {compute_name}",
            extra={
                "compute_name": compute_name,
                "compute_type": compute.type,
                "state": compute.provisioning_state,
            },
        )
        return compute
    except ResourceNotFoundError:
        logger.warning(f"Compute cluster not found: {compute_name}")
        return None
    except Exception as e:
        logger.error(
            f"Error retrieving compute cluster: {e}",
            exc_info=True,
        )
        raise


def list_compute_clusters(ml_client: MLClient) -> List[AmlCompute]:
    """
    List all compute clusters in the workspace.

    Args:
        ml_client: Azure ML client instance

    Returns:
        List of AmlCompute objects
    """
    try:
        computes = list(ml_client.compute.list())
        logger.info(f"Found {len(computes)} compute resources")
        return computes
    except Exception as e:
        logger.error(f"Error listing compute clusters: {e}", exc_info=True)
        raise


def create_compute_cluster(
    ml_client: MLClient,
    compute_name: str,
    vm_size: str = "Standard_NC6s_v3",
    min_instances: int = 0,
    max_instances: int = 4,
    idle_time_before_scale_down: int = 300,
    tier: str = "dedicated",
) -> AmlCompute:
    """
    Create or update a compute cluster.

    Args:
        ml_client: Azure ML client instance
        compute_name: Name for the compute cluster
        vm_size: Azure VM size (default: Standard_NC6s_v3 with V100 GPU)
        min_instances: Minimum number of nodes (default: 0)
        max_instances: Maximum number of nodes (default: 4)
        idle_time_before_scale_down: Seconds before scaling down idle
            nodes (default: 300)
        tier: Priority tier - 'dedicated' or 'low_priority'
            (default: dedicated)

    Returns:
        Created or updated AmlCompute object
    """
    # Check if compute already exists
    existing_compute = get_compute_cluster(ml_client, compute_name)
    if existing_compute:
        logger.info(
            f"Compute cluster {compute_name} already exists",
            extra={
                "vm_size": existing_compute.size,
                "min_nodes": existing_compute.min_instances,
                "max_nodes": existing_compute.max_instances,
            },
        )
        return existing_compute

    logger.info(
        f"Creating compute cluster: {compute_name}",
        extra={
            "vm_size": vm_size,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "tier": tier,
        },
    )

    compute_config = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size=vm_size,
        min_instances=min_instances,
        max_instances=max_instances,
        idle_time_before_scale_down=idle_time_before_scale_down,
        tier=tier,
    )

    try:
        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        logger.info(
            f"Successfully created compute cluster: {compute_name}",
            extra={"compute_id": compute.id},
        )
        return compute
    except Exception as e:
        logger.error(f"Failed to create compute cluster: {e}", exc_info=True)
        raise


def get_compute_status(ml_client: MLClient, compute_name: str) -> Dict[str, any]:
    """
    Get detailed status of a compute cluster.

    Args:
        ml_client: Azure ML client instance
        compute_name: Name of the compute cluster

    Returns:
        Dictionary with status information
    """
    compute = get_compute_cluster(ml_client, compute_name)
    if not compute:
        return {
            "exists": False,
            "name": compute_name,
            "state": "NotFound",
        }

    status = {
        "exists": True,
        "name": compute.name,
        "type": compute.type,
        "state": compute.provisioning_state,
        "vm_size": compute.size,
        "min_instances": compute.min_instances,
        "max_instances": compute.max_instances,
        "idle_time": compute.idle_time_before_scale_down,
        "tier": compute.tier,
    }

    # Add current node counts if available
    if hasattr(compute, "scale_settings"):
        status["current_node_count"] = getattr(compute, "current_node_count", 0)

    logger.info(f"Compute status for {compute_name}: {status['state']}")
    return status


def wait_for_compute_ready(
    ml_client: MLClient,
    compute_name: str,
    timeout_seconds: int = 600,
    poll_interval: int = 10,
) -> bool:
    """
    Wait for compute cluster to be in a ready state.

    Args:
        ml_client: Azure ML client instance
        compute_name: Name of the compute cluster
        timeout_seconds: Maximum time to wait (default: 600 seconds)
        poll_interval: Seconds between status checks (default: 10)

    Returns:
        True if compute is ready, False if timeout

    Raises:
        Exception if compute creation fails
    """
    logger.info(f"Waiting for compute {compute_name} to be ready " f"(timeout: {timeout_seconds}s)")

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        status = get_compute_status(ml_client, compute_name)

        if not status["exists"]:
            raise Exception(f"Compute cluster {compute_name} does not exist")

        state = status["state"]
        logger.debug(f"Current state: {state}")

        if state == "Succeeded":
            logger.info(
                f"Compute {compute_name} is ready",
                extra={"elapsed_seconds": time.time() - start_time},
            )
            return True

        if state in ["Failed", "Canceled"]:
            raise Exception(f"Compute provisioning {state.lower()} " f"for {compute_name}")

        time.sleep(poll_interval)

    logger.warning(f"Timeout waiting for compute {compute_name} to be ready")
    return False


def delete_compute_cluster(ml_client: MLClient, compute_name: str) -> None:
    """
    Delete a compute cluster.

    Args:
        ml_client: Azure ML client instance
        compute_name: Name of the compute cluster to delete
    """
    compute = get_compute_cluster(ml_client, compute_name)
    if not compute:
        logger.warning(f"Cannot delete - compute {compute_name} not found")
        return

    logger.info(f"Deleting compute cluster: {compute_name}")

    try:
        ml_client.compute.begin_delete(compute_name).wait()
        logger.info(f"Successfully deleted compute cluster: {compute_name}")
    except Exception as e:
        logger.error(f"Failed to delete compute cluster: {e}", exc_info=True)
        raise


def get_available_vm_sizes(ml_client: MLClient) -> List[Dict]:
    """
    Get list of available VM sizes in the workspace location.

    Args:
        ml_client: Azure ML client instance

    Returns:
        List of dictionaries with VM size information
    """
    try:
        # Note: This requires additional Azure SDK functionality
        # For now, return common GPU VM sizes
        common_gpu_sizes = [
            {
                "name": "Standard_NC6s_v3",
                "gpus": 1,
                "gpu_type": "V100",
                "ram_gb": 112,
                "vcpus": 6,
            },
            {
                "name": "Standard_NC12s_v3",
                "gpus": 2,
                "gpu_type": "V100",
                "ram_gb": 224,
                "vcpus": 12,
            },
            {
                "name": "Standard_NC24s_v3",
                "gpus": 4,
                "gpu_type": "V100",
                "ram_gb": 448,
                "vcpus": 24,
            },
            {
                "name": "Standard_NC24ads_A100_v4",
                "gpus": 1,
                "gpu_type": "A100",
                "ram_gb": 220,
                "vcpus": 24,
            },
            {
                "name": "Standard_NC48ads_A100_v4",
                "gpus": 2,
                "gpu_type": "A100",
                "ram_gb": 440,
                "vcpus": 48,
            },
        ]
        logger.info(f"Returning {len(common_gpu_sizes)} common GPU VM sizes")
        return common_gpu_sizes
    except Exception as e:
        logger.error(f"Error getting available VM sizes: {e}", exc_info=True)
        return []


def estimate_compute_cost(
    vm_size: str, max_instances: int, hours_per_day: float = 8.0
) -> Dict[str, float]:
    """
    Estimate monthly compute costs.

    Note: These are rough estimates. Check Azure pricing for accurate
    costs.

    Args:
        vm_size: Azure VM size
        max_instances: Maximum number of nodes
        hours_per_day: Average hours of usage per day

    Returns:
        Dictionary with cost estimates
    """
    # Approximate hourly costs (USD) - update with current pricing
    pricing = {
        "Standard_NC6s_v3": 3.06,  # V100
        "Standard_NC12s_v3": 6.12,  # 2x V100
        "Standard_NC24s_v3": 12.24,  # 4x V100
        "Standard_NC24ads_A100_v4": 3.67,  # A100
        "Standard_NC48ads_A100_v4": 7.35,  # 2x A100
    }

    hourly_cost = pricing.get(vm_size, 0.0)
    if hourly_cost == 0.0:
        logger.warning(f"Unknown VM size for cost estimation: {vm_size}")

    daily_cost = hourly_cost * max_instances * hours_per_day
    monthly_cost = daily_cost * 30

    return {
        "vm_size": vm_size,
        "hourly_rate_per_node": hourly_cost,
        "max_instances": max_instances,
        "hours_per_day": hours_per_day,
        "estimated_daily_cost": round(daily_cost, 2),
        "estimated_monthly_cost": round(monthly_cost, 2),
        "note": "Estimates only - check Azure pricing for accuracy",
    }
