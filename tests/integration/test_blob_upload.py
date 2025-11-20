"""Integration tests for blob upload (with mocking)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.upload_to_blob import (upload_directory_to_blob,
                                     upload_file_to_blob)
from src.utils.config import StorageConfig


@pytest.fixture
def mock_storage_config() -> StorageConfig:
    """Mock storage configuration."""
    return StorageConfig(
        account_name="teststorage",
        container_name="training-data",
    )


@patch("src.data.upload_to_blob.get_container_client")
def test_upload_file_to_blob_success(
    mock_get_container: MagicMock,
    mock_storage_config: StorageConfig,
    sample_jsonl_path: Path,
) -> None:
    """Test successful file upload to blob storage."""
    # Setup mock
    mock_container_client = MagicMock()
    mock_get_container.return_value = mock_container_client

    # Upload file
    url = upload_file_to_blob(
        sample_jsonl_path,
        "test/sample.jsonl",
        mock_storage_config,
        overwrite=True,
    )

    # Verify
    assert "teststorage" in url
    assert "training-data" in url
    assert "test/sample.jsonl" in url
    mock_container_client.upload_blob.assert_called_once()


@patch("src.data.upload_to_blob.get_container_client")
def test_upload_file_to_blob_missing_file(
    mock_get_container: MagicMock,
    mock_storage_config: StorageConfig,
    tmp_path: Path,
) -> None:
    """Test upload with missing file raises error."""
    missing_file = tmp_path / "nonexistent.jsonl"

    with pytest.raises(FileNotFoundError):
        upload_file_to_blob(
            missing_file,
            "test/missing.jsonl",
            mock_storage_config,
        )


@patch("src.data.upload_to_blob.get_container_client")
def test_upload_directory_to_blob_success(
    mock_get_container: MagicMock,
    mock_storage_config: StorageConfig,
    tmp_path: Path,
) -> None:
    """Test successful directory upload."""
    # Setup mock
    mock_container_client = MagicMock()
    mock_get_container.return_value = mock_container_client

    # Create test directory with files
    test_dir = tmp_path / "data"
    test_dir.mkdir()
    (test_dir / "file1.jsonl").write_text('{"test":1}\n')
    (test_dir / "file2.jsonl").write_text('{"test":2}\n')
    (test_dir / "readme.txt").write_text("readme")

    # Upload only .jsonl files
    urls = upload_directory_to_blob(
        test_dir,
        "test-data",
        mock_storage_config,
        pattern="*.jsonl",
    )

    # Verify
    assert len(urls) == 2
    assert mock_container_client.upload_blob.call_count == 2


@patch("src.data.upload_to_blob.get_container_client")
def test_upload_directory_to_blob_empty_directory(
    mock_get_container: MagicMock,
    mock_storage_config: StorageConfig,
    tmp_path: Path,
) -> None:
    """Test upload of empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    urls = upload_directory_to_blob(
        empty_dir,
        "test-data",
        mock_storage_config,
    )

    assert len(urls) == 0


def test_upload_directory_not_a_directory(
    mock_storage_config: StorageConfig,
    sample_jsonl_path: Path,
) -> None:
    """Test upload with file instead of directory raises error."""
    with pytest.raises(NotADirectoryError):
        upload_directory_to_blob(
            sample_jsonl_path,  # This is a file, not a directory
            "test-data",
            mock_storage_config,
        )
