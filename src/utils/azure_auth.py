"""Azure authentication and SDK helpers."""

from typing import Optional

from azure.ai.ml import MLClient
from azure.core.credentials import TokenCredential
from azure.identity import (AzureCliCredential, ChainedTokenCredential,
                            DefaultAzureCredential)
from azure.storage.blob import BlobServiceClient

from src.utils.config import AzureConfig, StorageConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_azure_credential() -> TokenCredential:
    """Get Azure credential using default credential chain.

    Tries authentication in this order:
    1. Environment variables
    2. Managed Identity
    3. Azure CLI
    4. Interactive browser

    Returns:
        TokenCredential instance
    """
    try:
        credential = DefaultAzureCredential()
        logger.info("Successfully obtained Azure credentials")
        return credential
    except Exception as e:
        logger.warning(f"DefaultAzureCredential failed: {e}, trying Azure CLI")
        try:
            credential = AzureCliCredential()
            logger.info("Successfully obtained Azure CLI credentials")
            return credential
        except Exception as cli_e:
            logger.error(f"Failed to obtain Azure credentials: {cli_e}")
            raise


def get_ml_client(azure_config: AzureConfig) -> MLClient:
    """Create Azure ML client instance.

    Args:
        azure_config: Azure configuration

    Returns:
        MLClient instance
    """
    credential = get_azure_credential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=azure_config.subscription_id,
        resource_group_name=azure_config.resource_group,
        workspace_name=azure_config.workspace_name,
    )

    logger.info(f"Connected to workspace: {azure_config.workspace_name}")
    return ml_client


def get_blob_service_client(
    storage_config: StorageConfig,
    credential: Optional[TokenCredential] = None,
) -> BlobServiceClient:
    """Create Azure Blob Storage client instance.

    Args:
        storage_config: Storage configuration
        credential: Optional credential (will use default if not provided)

    Returns:
        BlobServiceClient instance
    """
    if credential is None:
        credential = get_azure_credential()

    account_url = f"https://{storage_config.account_name}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential,
    )

    logger.info(f"Connected to storage account: {storage_config.account_name}")
    return blob_service_client


def get_container_client(
    storage_config: StorageConfig,
    credential: Optional[TokenCredential] = None,
):
    """Get blob container client.

    Args:
        storage_config: Storage configuration
        credential: Optional credential

    Returns:
        ContainerClient instance
    """
    blob_service_client = get_blob_service_client(storage_config, credential)
    container_client = blob_service_client.get_container_client(storage_config.container_name)

    logger.info(f"Connected to container: {storage_config.container_name}")
    return container_client
