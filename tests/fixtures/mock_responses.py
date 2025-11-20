"""Mock responses for Azure API calls."""

from typing import Any, Dict


def mock_workspace_response() -> Dict[str, Any]:
    """Mock Azure ML workspace response."""
    return {
        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.MachineLearningServices/workspaces/test-workspace",
        "name": "test-workspace",
        "location": "eastus",
        "type": "Microsoft.MachineLearningServices/workspaces",
        "properties": {
            "friendlyName": "test-workspace",
            "description": "Test workspace",
            "storageAccount": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Storage/storageAccounts/teststorage",
        },
    }


def mock_compute_response() -> Dict[str, Any]:
    """Mock compute cluster response."""
    return {
        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.MachineLearningServices/workspaces/test-workspace/computes/test-compute",
        "name": "test-compute",
        "type": "Microsoft.MachineLearningServices/workspaces/computes",
        "properties": {
            "computeType": "AmlCompute",
            "provisioningState": "Succeeded",
            "scaleSettings": {
                "minNodeCount": 0,
                "maxNodeCount": 4,
            },
        },
    }


def mock_job_response() -> Dict[str, Any]:
    """Mock training job response."""
    return {
        "id": "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.MachineLearningServices/workspaces/test-workspace/jobs/test-job",
        "name": "test-job",
        "type": "Microsoft.MachineLearningServices/workspaces/jobs",
        "properties": {
            "jobType": "Command",
            "status": "Completed",
            "displayName": "test-training-job",
        },
    }


def mock_blob_list_response() -> list:
    """Mock blob list response."""
    return [
        {"name": "train.jsonl", "size": 1024},
        {"name": "val.jsonl", "size": 512},
    ]
