"""
Integration tests for Azure ML job management.

Tests job submission, monitoring, and output retrieval.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_ml_client():
    """Create mock Azure ML client."""
    client = MagicMock()
    client.workspace_name = "test-workspace"
    client.subscription_id = "test-subscription"
    client.resource_group_name = "test-rg"
    return client


@pytest.fixture
def mock_job():
    """Create mock Azure ML job."""
    job = MagicMock()
    job.name = "test_job_12345"
    job.status = "Running"
    job.studio_url = "https://ml.azure.com/..."
    job.experiment_name = "test-experiment"
    job.display_name = "Test Training Job"

    # Mock creation context
    from datetime import datetime

    job.creation_context.created_at = datetime(2024, 1, 1, 10, 0, 0)
    job.creation_context.last_modified_at = datetime(2024, 1, 1, 11, 0, 0)

    return job


class TestAzureMLJobManager:
    """Tests for AzureMLJobManager class."""

    @patch("src.training.job_manager.MLClient")
    def test_job_manager_initialization(self, mock_client_class):
        """Test job manager initialization."""
        from src.training.job_manager import AzureMLJobManager

        mock_client = MagicMock()
        mock_client.workspace_name = "test-workspace"
        mock_client_class.from_config.return_value = mock_client

        manager = AzureMLJobManager()

        assert manager.ml_client is not None
        mock_client_class.from_config.assert_called_once()

    @patch("src.training.job_manager.MLClient")
    def test_create_environment(self, mock_client_class, mock_ml_client):
        """Test creating Azure ML environment."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        # Mock environment creation
        mock_env = MagicMock()
        mock_env.name = "test-env"
        mock_ml_client.environments.create_or_update.return_value = mock_env

        manager = AzureMLJobManager()
        env = manager.create_environment(
            name="test-env",
            description="Test environment",
        )

        assert env.name == "test-env"
        mock_ml_client.environments.create_or_update.assert_called_once()

    @patch("src.training.job_manager.MLClient")
    @patch("src.training.job_manager.command")
    def test_submit_training_job(self, mock_command, mock_client_class, mock_ml_client, mock_job):
        """Test submitting training job."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        # Mock job creation
        mock_command.return_value = MagicMock()
        mock_ml_client.jobs.create_or_update.return_value = mock_job

        manager = AzureMLJobManager()
        job = manager.submit_training_job(
            experiment_name="test-experiment",
            display_name="Test Job",
            code_path=".",
            command_str="python train.py",
            environment_name="test-env",
            compute_name="test-compute",
        )

        assert job.name == "test_job_12345"
        assert job.status == "Running"
        mock_ml_client.jobs.create_or_update.assert_called_once()

    @patch("src.training.job_manager.MLClient")
    def test_get_job_status(self, mock_client_class, mock_ml_client, mock_job):
        """Test getting job status."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client
        mock_ml_client.jobs.get.return_value = mock_job

        manager = AzureMLJobManager()
        status = manager.get_job_status("test_job_12345")

        assert status["name"] == "test_job_12345"
        assert status["status"] == "Running"
        assert "studio_url" in status

    @patch("src.training.job_manager.MLClient")
    @patch("src.training.job_manager.time.sleep")
    def test_wait_for_completion(self, mock_sleep, mock_client_class, mock_ml_client, mock_job):
        """Test waiting for job completion."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        # Simulate job completing after 2 checks
        mock_job_running = MagicMock()
        mock_job_running.status = "Running"

        mock_job_completed = MagicMock()
        mock_job_completed.status = "Completed"

        mock_ml_client.jobs.get.side_effect = [
            mock_job_running,
            mock_job_completed,
        ]

        manager = AzureMLJobManager()
        final_status = manager.wait_for_completion(
            job_name="test_job",
            timeout_seconds=300,
            check_interval=10,
        )

        assert final_status == "Completed"

    @patch("src.training.job_manager.MLClient")
    def test_wait_for_completion_timeout(self, mock_client_class, mock_ml_client, mock_job):
        """Test timeout during wait for completion."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client
        mock_ml_client.jobs.get.return_value = mock_job  # Always running

        manager = AzureMLJobManager()

        with pytest.raises(TimeoutError):
            manager.wait_for_completion(
                job_name="test_job",
                timeout_seconds=1,  # Very short timeout
                check_interval=1,
            )

    @patch("src.training.job_manager.MLClient")
    def test_cancel_job(self, mock_client_class, mock_ml_client):
        """Test canceling a job."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        manager = AzureMLJobManager()
        manager.cancel_job("test_job")

        mock_ml_client.jobs.begin_cancel.assert_called_once_with("test_job")

    @patch("src.training.job_manager.MLClient")
    def test_list_jobs(self, mock_client_class, mock_ml_client, mock_job):
        """Test listing jobs."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        # Mock job list
        mock_ml_client.jobs.list.return_value = [mock_job]

        manager = AzureMLJobManager()
        jobs = manager.list_jobs(max_results=5)

        assert len(jobs) == 1
        assert jobs[0].name == "test_job_12345"

    @patch("src.training.job_manager.MLClient")
    def test_download_job_outputs(self, mock_client_class, mock_ml_client, tmp_path):
        """Test downloading job outputs."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        manager = AzureMLJobManager()
        output_path = manager.download_job_outputs(
            job_name="test_job",
            output_path=str(tmp_path),
        )

        assert output_path.exists()
        mock_ml_client.jobs.download.assert_called_once()

    @patch("src.training.job_manager.MLClient")
    def test_register_model_from_job(self, mock_client_class, mock_ml_client):
        """Test registering model from job outputs."""
        from src.training.job_manager import AzureMLJobManager

        mock_client_class.from_config.return_value = mock_ml_client

        # Mock model registration
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.version = "1"
        mock_model.id = "test-model:1"
        mock_ml_client.models.create_or_update.return_value = mock_model

        manager = AzureMLJobManager()
        model = manager.register_model_from_job(
            job_name="test_job",
            model_name="test-model",
            description="Test model",
            tags={"framework": "pytorch"},
        )

        assert model.name == "test-model"
        assert model.version == "1"
        mock_ml_client.models.create_or_update.assert_called_once()


class TestEnvironmentCreation:
    """Tests for environment creation utilities."""

    @patch("src.training.job_manager.MLClient")
    def test_create_training_environment(self, mock_client_class):
        """Test creating standard training environment."""
        from src.training.job_manager import create_training_environment

        mock_client = MagicMock()
        mock_env = MagicMock()
        mock_env.name = "training-env"
        mock_client.environments.create_or_update.return_value = mock_env

        env = create_training_environment(
            ml_client=mock_client,
            environment_name="training-env",
        )

        assert env.name == "training-env"
        mock_client.environments.create_or_update.assert_called_once()
