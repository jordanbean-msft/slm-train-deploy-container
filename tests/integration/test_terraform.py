"""
Integration tests for Terraform infrastructure deployment.

These tests validate:
- Terraform configuration syntax
- Module structure and inputs/outputs
- Variable validation
"""

import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def terraform_dir():
    """Get the Terraform directory path."""
    return Path(__file__).parent.parent.parent / "infra" / "terraform"


@pytest.fixture
def terraform_initialized(terraform_dir):
    """Ensure Terraform is initialized before tests."""
    # Check if already initialized
    if not (terraform_dir / ".terraform").exists():
        result = subprocess.run(
            ["terraform", "init"],
            cwd=terraform_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"Terraform init failed: {result.stderr}")
    return terraform_dir


def test_terraform_validate(terraform_initialized):
    """Test that Terraform configuration is valid."""
    result = subprocess.run(
        ["terraform", "validate", "-json"],
        cwd=terraform_initialized,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert output["valid"] is True, f"Validation errors: {output}"


def test_terraform_format_check(terraform_dir):
    """Test that Terraform files are properly formatted."""
    result = subprocess.run(
        ["terraform", "fmt", "-check", "-recursive"],
        cwd=terraform_dir,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Unformatted files: {result.stdout}"


@pytest.mark.parametrize(
    "environment",
    ["dev", "prod"],
)
def test_terraform_plan(terraform_initialized, environment):
    """
    Test that Terraform can create a plan for each environment.

    Note: This requires Azure credentials and may be skipped in CI.
    """
    tfvars_file = terraform_initialized / "environments" / f"{environment}.tfvars"

    # Skip if no Azure credentials
    try:
        subprocess.run(
            ["az", "account", "show"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Azure CLI not authenticated")

    result = subprocess.run(
        [
            "terraform",
            "plan",
            f"-var-file={tfvars_file}",
            "-out=tfplan-test",
        ],
        cwd=terraform_initialized,
        capture_output=True,
        text=True,
    )

    # Clean up plan file
    plan_file = terraform_initialized / "tfplan-test"
    if plan_file.exists():
        plan_file.unlink()

    assert result.returncode == 0, f"Plan failed: {result.stderr}"


def test_required_modules_exist(terraform_dir):
    """Test that all required Terraform modules exist."""
    modules_dir = terraform_dir / "modules"

    required_modules = [
        "storage",
        "container-registry",
        "azureml-workspace",
        "compute-cluster",
    ]

    for module in required_modules:
        module_path = modules_dir / module / "main.tf"
        assert module_path.exists(), f"Module {module} main.tf not found"


def test_module_has_required_files(terraform_dir):
    """Test that each module has required Terraform files."""
    modules_dir = terraform_dir / "modules"

    for module_dir in modules_dir.iterdir():
        if not module_dir.is_dir():
            continue

        # Only main.tf is strictly required; some modules may not need variables/outputs
        main_tf = module_dir / "main.tf"
        assert main_tf.exists(), f"Module {module_dir.name} missing main.tf"


def test_environment_tfvars_exist(terraform_dir):
    """Test that environment-specific tfvars files exist."""
    environments = ["dev", "prod"]

    for env in environments:
        tfvars_path = terraform_dir / "environments" / f"{env}.tfvars"
        assert tfvars_path.exists(), f"Missing {env}.tfvars file"


def test_terraform_outputs_defined(terraform_dir):
    """Test that outputs.tf defines all expected outputs."""
    outputs_file = terraform_dir / "outputs.tf"
    content = outputs_file.read_text()

    # Core outputs that should be present
    expected_outputs = [
        "resource_group_name",
        "workspace_name",
        "workspace_id",
        "storage_account_name",
        "acr_name",
        "acr_login_server",
    ]

    for output in expected_outputs:
        assert f'output "{output}"' in content, f"Missing output: {output}"


def test_terraform_variables_defined(terraform_dir):
    """Test that variables.tf defines all required variables."""
    variables_file = terraform_dir / "variables.tf"
    content = variables_file.read_text()

    # Core variables that should be present
    required_variables = [
        "location",
        "resource_group_name",
        "storage_account_name",
        "acr_name",
        "workspace_name",
    ]

    for variable in required_variables:
        assert f'variable "{variable}"' in content, f"Missing variable: {variable}"


@pytest.mark.parametrize(
    "script_name",
    ["deploy.sh", "destroy.sh", "validate-deployment.sh"],
)
def test_scripts_exist_and_executable(script_name):
    """Test that deployment scripts exist and are executable."""
    script_path = Path(__file__).parent.parent.parent / "infra" / "scripts" / script_name

    assert script_path.exists(), f"Script {script_name} not found"
    assert script_path.stat().st_mode & 0o111, f"Script {script_name} not executable"


def test_deploy_script_syntax():
    """Test that deploy.sh has valid bash syntax."""
    script_path = Path(__file__).parent.parent.parent / "infra" / "scripts" / "deploy.sh"

    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Bash syntax errors: {result.stderr}"


def test_destroy_script_syntax():
    """Test that destroy.sh has valid bash syntax."""
    script_path = Path(__file__).parent.parent.parent / "infra" / "scripts" / "destroy.sh"

    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Bash syntax errors: {result.stderr}"


def test_validate_script_syntax():
    """Test that validate-deployment.sh has valid bash syntax."""
    script_path = (
        Path(__file__).parent.parent.parent / "infra" / "scripts" / "validate-deployment.sh"
    )

    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Bash syntax errors: {result.stderr}"
