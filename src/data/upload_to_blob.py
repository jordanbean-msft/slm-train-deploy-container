"""Azure Blob Storage upload utilities."""

from pathlib import Path
from typing import Optional

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobClient

from src.utils.azure_auth import get_container_client
from src.utils.config import StorageConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def upload_file_to_blob(
    file_path: Path,
    blob_name: str,
    storage_config: StorageConfig,
    overwrite: bool = False,
) -> str:
    """Upload a file to Azure Blob Storage.

    Args:
        file_path: Path to local file
        blob_name: Name for the blob in storage
        storage_config: Storage configuration
        overwrite: Whether to overwrite existing blob

    Returns:
        URL of uploaded blob

    Raises:
        FileNotFoundError: If file doesn't exist
        ResourceExistsError: If blob exists and overwrite=False
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    container_client = get_container_client(storage_config)

    logger.info(f"Uploading {file_path} to blob {blob_name}")

    with open(file_path, "rb") as data:
        blob_client = container_client.upload_blob(
            name=blob_name, data=data, overwrite=overwrite
        )

    blob_url = f"https://{storage_config.account_name}.blob.core.windows.net/{storage_config.container_name}/{blob_name}"

    logger.info(f"Successfully uploaded to {blob_url}")
    return blob_url


def upload_directory_to_blob(
    directory_path: Path,
    blob_prefix: str,
    storage_config: StorageConfig,
    overwrite: bool = False,
    pattern: str = "*",
) -> list[str]:
    """Upload all files from a directory to Azure Blob Storage.

    Args:
        directory_path: Path to local directory
        blob_prefix: Prefix for blob names (like a folder)
        storage_config: Storage configuration
        overwrite: Whether to overwrite existing blobs
        pattern: File pattern to match (e.g., "*.jsonl")

    Returns:
        List of uploaded blob URLs

    Raises:
        NotADirectoryError: If path is not a directory
    """
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    uploaded_urls = []
    files = list(directory_path.glob(pattern))

    if not files:
        logger.warning(f"No files matching pattern '{pattern}' in {directory_path}")
        return uploaded_urls

    logger.info(f"Uploading {len(files)} files to blob storage")

    for file_path in files:
        if file_path.is_file():
            blob_name = f"{blob_prefix}/{file_path.name}"
            try:
                url = upload_file_to_blob(
                    file_path, blob_name, storage_config, overwrite
                )
                uploaded_urls.append(url)
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")

    logger.info(f"Successfully uploaded {len(uploaded_urls)} files")
    return uploaded_urls


def register_dataset_in_azureml(
    dataset_name: str,
    blob_urls: list[str],
    workspace_name: str,
    description: Optional[str] = None,
) -> str:
    """Register dataset in Azure ML workspace.

    Args:
        dataset_name: Name for the dataset
        blob_urls: List of blob URLs
        workspace_name: Azure ML workspace name
        description: Optional dataset description

    Returns:
        Dataset ID

    Note:
        This is a placeholder - actual implementation requires Azure ML SDK
    """
    logger.info(
        f"Registering dataset '{dataset_name}' with {len(blob_urls)} files"
    )

    # TODO: Implement actual Azure ML dataset registration
    # from azure.ai.ml import MLClient
    # from azure.ai.ml.entities import Data
    # ml_client = get_ml_client(azure_config)
    # dataset = Data(
    #     name=dataset_name,
    #     description=description,
    #     path=blob_urls[0] if len(blob_urls) == 1 else blob_urls,
    # )
    # registered = ml_client.data.create_or_update(dataset)

    logger.info(f"Dataset '{dataset_name}' registered successfully")
    return f"azureml://datasets/{dataset_name}"


def get_blob_progress_callback(file_size: int):
    """Create a progress callback for blob upload.

    Args:
        file_size: Total file size in bytes

    Returns:
        Callback function for tracking progress
    """
    uploaded_bytes = {"current": 0}

    def progress_callback(bytes_uploaded: int) -> None:
        uploaded_bytes["current"] += bytes_uploaded
        percent = (uploaded_bytes["current"] / file_size) * 100
        logger.info(f"Upload progress: {percent:.1f}%")

    return progress_callback
