"""
Azure ML job submission and orchestration utilities.

This module provides utilities for submitting training jobs to Azure ML,
monitoring their progress, and retrieving outputs.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, Environment, Job, Model
from azure.identity import DefaultAzureCredential

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class AzureMLJobManager:
    """Manager for Azure ML training jobs."""

    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        workspace_name: Optional[str] = None,
    ):
        """
        Initialize Azure ML job manager.

        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            workspace_name: Workspace name
        """
        credential = DefaultAzureCredential()

        if all([subscription_id, resource_group, workspace_name]):
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )
        else:
            # Use config.json in workspace
            self.ml_client = MLClient.from_config(credential=credential)

        logger.info(
            f"Connected to workspace: {self.ml_client.workspace_name}",
            extra={
                "subscription": self.ml_client.subscription_id,
                "resource_group": self.ml_client.resource_group_name,
            },
        )

    def create_environment(
        self,
        name: str,
        conda_file: Optional[str] = None,
        docker_image: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Environment:
        """
        Create or update Azure ML environment.

        Args:
            name: Environment name
            conda_file: Path to conda.yaml file
            docker_image: Base Docker image
            description: Environment description

        Returns:
            Azure ML Environment
        """
        if docker_image is None:
            docker_image = (
                "mcr.microsoft.com/azureml/" "openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
            )

        env = Environment(
            name=name,
            description=description or "Training environment",
            image=docker_image,
            conda_file=conda_file,
        )

        env = self.ml_client.environments.create_or_update(env)
        logger.info(f"Environment created/updated: {name}")

        return env

    def submit_training_job(
        self,
        experiment_name: str,
        display_name: str,
        code_path: str,
        command_str: str,
        environment_name: str,
        compute_name: str,
        environment_variables: Optional[Dict[str, str]] = None,
        inputs: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
    ) -> Job:
        """
        Submit training job to Azure ML.

        Args:
            experiment_name: Experiment name for grouping runs
            display_name: Display name for this job
            code_path: Path to code directory
            command_str: Command to execute
            environment_name: Environment name to use
            compute_name: Compute cluster name
            environment_variables: Environment variables
            inputs: Input data references
            outputs: Output data references

        Returns:
            Submitted job object
        """
        logger.info(f"Submitting training job: {display_name}")

        job = command(
            code=code_path,
            command=command_str,
            environment=f"{environment_name}@latest",
            compute=compute_name,
            experiment_name=experiment_name,
            display_name=display_name,
            environment_variables=environment_variables or {},
            inputs=inputs or {},
            outputs=outputs or {},
        )

        submitted_job = self.ml_client.jobs.create_or_update(job)

        logger.info(
            f"Job submitted: {submitted_job.name}",
            extra={
                "job_id": submitted_job.name,
                "studio_url": submitted_job.studio_url,
            },
        )

        return submitted_job

    def get_job_status(self, job_name: str) -> Dict[str, str]:
        """
        Get job status information.

        Args:
            job_name: Job name/ID

        Returns:
            Dictionary with job status details
        """
        job = self.ml_client.jobs.get(job_name)

        return {
            "name": job.name,
            "status": job.status,
            "creation_time": str(job.creation_context.created_at),
            "duration": str(
                job.creation_context.last_modified_at - job.creation_context.created_at
            ),
            "studio_url": job.studio_url,
        }

    def wait_for_completion(
        self,
        job_name: str,
        timeout_seconds: int = 7200,
        check_interval: int = 30,
        show_output: bool = False,
    ) -> str:
        """
        Wait for job to complete.

        Args:
            job_name: Job name/ID
            timeout_seconds: Maximum wait time in seconds
            check_interval: Seconds between status checks
            show_output: Whether to stream job output

        Returns:
            Final job status
        """
        logger.info(f"Waiting for job completion: {job_name}")

        start_time = time.time()
        last_status = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Job did not complete within {timeout_seconds}s")

            job = self.ml_client.jobs.get(job_name)
            current_status = job.status

            if current_status != last_status:
                logger.info(
                    f"Job status: {current_status}",
                    extra={"elapsed": f"{elapsed:.0f}s"},
                )
                last_status = current_status

            # Check if job is in terminal state
            if current_status in [
                "Completed",
                "Failed",
                "Canceled",
                "NotResponding",
            ]:
                logger.info(
                    f"Job finished: {current_status}",
                    extra={"total_time": f"{elapsed:.0f}s"},
                )
                return current_status

            if show_output:
                self._stream_job_logs(job_name)

            time.sleep(check_interval)

    def _stream_job_logs(self, job_name: str) -> None:
        """Stream job logs (basic implementation)."""
        try:
            job = self.ml_client.jobs.get(job_name)
            # Note: Full log streaming requires additional setup
            logger.debug(f"Job {job_name} is running...")
        except Exception as e:
            logger.warning(f"Could not retrieve logs: {e}")

    def cancel_job(self, job_name: str) -> None:
        """
        Cancel a running job.

        Args:
            job_name: Job name/ID
        """
        logger.info(f"Canceling job: {job_name}")
        self.ml_client.jobs.begin_cancel(job_name)

    def list_jobs(
        self,
        experiment_name: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Job]:
        """
        List recent jobs.

        Args:
            experiment_name: Filter by experiment name
            max_results: Maximum number of results

        Returns:
            List of job objects
        """
        jobs = []
        job_list = self.ml_client.jobs.list(max_results=max_results)

        for job in job_list:
            if experiment_name and job.experiment_name != experiment_name:
                continue
            jobs.append(job)

        return jobs

    def download_job_outputs(self, job_name: str, output_path: str) -> Path:
        """
        Download job outputs to local directory.

        Args:
            job_name: Job name/ID
            output_path: Local directory for outputs

        Returns:
            Path to downloaded outputs
        """
        logger.info(f"Downloading outputs for job: {job_name}")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download outputs
        self.ml_client.jobs.download(
            name=job_name,
            download_path=str(output_dir),
            output_name="outputs",
        )

        logger.info(f"Outputs downloaded to: {output_dir}")
        return output_dir

    def register_model_from_job(
        self,
        job_name: str,
        model_name: str,
        model_path: str = "outputs/best_model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Model:
        """
        Register model from job outputs.

        Args:
            job_name: Job name/ID
            model_name: Name for registered model
            model_path: Path to model in job outputs
            description: Model description
            tags: Model tags

        Returns:
            Registered model
        """
        logger.info(f"Registering model from job: {job_name}")

        model = Model(
            name=model_name,
            path=f"azureml://jobs/{job_name}/outputs/{model_path}",
            description=description or f"Model from training job {job_name}",
            tags=tags or {},
        )

        registered_model = self.ml_client.models.create_or_update(model)

        logger.info(
            f"Model registered: {model_name}",
            extra={
                "version": registered_model.version,
                "id": registered_model.id,
            },
        )

        return registered_model

    def get_job_metrics(self, job_name: str) -> Dict:
        """
        Get metrics from completed job.

        Args:
            job_name: Job name/ID

        Returns:
            Dictionary of metrics
        """
        job = self.ml_client.jobs.get(job_name)

        # Get metrics from job properties
        metrics = {}
        if hasattr(job, "properties") and job.properties:
            for key, value in job.properties.items():
                if key.startswith("metric_"):
                    metrics[key.replace("metric_", "")] = value

        return metrics


def create_training_environment(
    ml_client: MLClient,
    environment_name: str = "training-env",
) -> Environment:
    """
    Create standard training environment.

    Args:
        ml_client: Azure ML client
        environment_name: Name for environment

    Returns:
        Created environment
    """
    # Use pre-built PyTorch environment with GPU support
    base_image = "mcr.microsoft.com/azureml/" "openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"

    env = Environment(
        name=environment_name,
        description="Training environment with PyTorch and transformers",
        image=base_image,
        conda_file="configs/conda.yaml",
    )

    env = ml_client.environments.create_or_update(env)
    logger.info(f"Training environment created: {environment_name}")

    return env
