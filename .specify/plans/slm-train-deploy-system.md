# Implementation Plan: SLM Training and Deployment System

**Branch**: `001-slm-train-deploy-system` | **Date**: November 20, 2025 | **Spec**: [slm-train-deploy-system.md](../specs/slm-train-deploy-system.md)

**Input**: Feature specification from `.specify/specs/slm-train-deploy-system.md`

## Summary

Build a complete system for fine-tuning Microsoft Phi-4 mini language model (sourced from Azure AI Foundry model catalog) using Azure ML infrastructure and deploying the trained model as a highly optimized containerized inference service for embedded hardware. The solution uses Python Jupyter notebooks for orchestration, Terraform for infrastructure provisioning, and Azure Blob Storage for data/model artifacts. The container image is aggressively optimized for minimal size (<500MB target) and fast inference (<50ms latency) to run on resource-constrained embedded devices. The end-to-end workflow covers data preparation, Azure ML workspace setup, GPU compute provisioning, model training with quantization, evaluation, containerization with model optimization, and registry upload.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**:

- uv (ultra-fast Python package manager and environment manager)
- Azure ML SDK v2 (azure-ai-ml)
- Transformers 4.36+ (Hugging Face)
- PyTorch 2.1+ with CUDA support
- MLflow for experiment tracking
- Docker for containerization
- Terraform 1.6+ for infrastructure

**Storage**:

- Azure Blob Storage (training data, model artifacts)
- Azure ML Datastore (managed dataset versioning)
- Azure Container Registry (container images)

**Testing**:

- pytest for unit/integration tests (managed via uv)
- Azure ML pipeline validation
- Container smoke tests with test inference requests
- Local container deployment testing
- Embedded system deployment validation

**Target Platform**:

- Azure ML Compute (GPU training - Standard_NC6s_v3 or NC24ads_A100_v4)
- Local Docker deployment (testing and development)
- Embedded Linux devices (ARM64/x86_64) - primary deployment target
- Azure Container Instances/Apps (optional cloud deployment)
- Linux containers (Alpine/Ubuntu minimal base)

**Project Type**: ML training/deployment pipeline with notebook-driven orchestration

**Performance Goals**:

- Training: Complete fine-tuning of Phi-4 mini on 10K samples in <2 hours
- Inference: P95 latency <50ms for single requests on CPU (embedded hardware)
- Container startup: <10 seconds from cold start on embedded device
- Model loading: <5 seconds into memory

**Constraints**:

- Container image size: <500MB (critical for embedded deployment)
- Optimized model size: <2GB (quantized to int8/fp16)
- Memory footprint during inference: <4GB RAM
- CPU-only inference (no GPU on embedded hardware)
- GPU utilization during training: >80%
- Model checkpoint frequency: every 500 steps
- Training data format: JSONL with prompt/completion pairs

**Scale/Scope**:

- Training datasets: 1K-100K samples
- Model size: Phi-4 mini (~14B parameters, quantized to ~4GB)
- Concurrent inference requests: 10-50 on embedded hardware
- Deployment target: Embedded Linux devices (ARM64/x86_64) with 8GB+ RAM
- Azure resources: 1 workspace, 1 compute cluster, 1 ACR, 1 storage account
- Model source: Azure AI Foundry model catalog

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

✅ **Code Quality Principles**

- Type hints required for all Python functions
- Docstrings following Google style for public APIs
- Maximum function length 50 lines
- Environment variables for secrets (no hardcoded credentials)

✅ **Testing Standards**

- 80% minimum code coverage target
- Unit tests for data processing, training logic, inference
- Integration tests for Azure ML job submission
- Container smoke tests for deployment validation

✅ **User Experience Consistency**

- Jupyter notebooks provide clear step-by-step guidance
- Progress bars for long-running operations (training, upload)
- Structured logging with clear error messages
- Configuration via YAML files with validation

✅ **Performance Requirements**

- GPU utilization monitoring during training
- Checkpointing every 500 steps for recovery
- Multi-stage Docker builds for image optimization
- Batch inference support for throughput

## Project Structure

### Documentation (this feature)

```text
.specify/
├── specs/
│   └── slm-train-deploy-system.md    # Feature specification
├── plans/
│   └── slm-train-deploy-system.md    # This file
└── templates/                         # Speckit templates
```

### Source Code (repository root)

```text
slm-train-deploy-container/
├── infra/                             # Infrastructure as Code
│   ├── terraform/
│   │   ├── main.tf                   # Root module
│   │   ├── variables.tf              # Input variables
│   │   ├── outputs.tf                # Output values
│   │   ├── modules/
│   │   │   ├── azureml-workspace/    # AML workspace module
│   │   │   ├── compute-cluster/      # Compute provisioning
│   │   │   ├── storage/              # Blob storage module
│   │   │   └── container-registry/   # ACR module
│   │   └── environments/
│   │       ├── dev.tfvars            # Dev environment config
│   │       └── prod.tfvars           # Prod environment config
│   └── scripts/
│       └── validate-deployment.sh    # Post-deployment validation
│
├── notebooks/                         # Orchestration notebooks
│   ├── 01-setup-infrastructure.ipynb  # Provision Azure resources
│   ├── 02-prepare-data.ipynb         # Data prep and upload
│   ├── 03-download-base-model.ipynb  # Get Phi-4 mini model
│   ├── 04-train-model.ipynb          # Submit training job
│   ├── 05-evaluate-model.ipynb       # Run evaluation
│   ├── 06-build-container.ipynb      # Create Docker image
│   ├── 07-push-to-acr.ipynb          # Upload to registry
│   └── 08-deploy-inference.ipynb     # Deploy to ACI/ACA
│
├── src/                               # Source code modules
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # Data validation and formatting
│   │   ├── dataset_loader.py         # PyTorch dataset classes
│   │   └── upload_to_blob.py         # Azure Blob upload utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                  # Main training script for AML
│   │   ├── model_loader.py           # Load Phi-4 mini model
│   │   ├── trainer.py                # Training loop implementation
│   │   └── checkpointing.py          # Checkpoint management
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py               # Evaluation script
│   │   └── metrics.py                # Metric computation
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── score.py                  # Scoring script for container
│   │   ├── model_wrapper.py          # Model loading and caching
│   │   └── api.py                    # FastAPI inference server
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration loading
│       ├── logging_config.py         # Structured logging setup
│       └── azure_auth.py             # Azure authentication helpers
│
├── configs/                           # Configuration files
│   ├── training_config.yaml          # Training hyperparameters
│   ├── model_config.yaml             # Model architecture settings
│   ├── deployment_config.yaml        # Container deployment config
│   └── data_config.yaml              # Data processing settings
│
├── docker/                            # Container definitions
│   ├── Dockerfile                    # Multi-stage inference image
│   ├── Dockerfile.training           # Training environment image
│   ├── requirements.txt              # Python dependencies
│   └── .dockerignore                 # Build exclusions
│
├── tests/                             # Test suite
│   ├── unit/
│   │   ├── test_preprocessing.py     # Data preprocessing tests
│   │   ├── test_trainer.py           # Training logic tests
│   │   ├── test_inference.py         # Inference tests
│   │   └── test_metrics.py           # Evaluation tests
│   │
│   ├── integration/
│   │   ├── test_azureml_job.py       # AML job submission tests
│   │   ├── test_blob_upload.py       # Storage integration tests
│   │   └── test_acr_push.py          # Registry integration tests
│   │
│   └── fixtures/
│       ├── sample_data.jsonl         # Test dataset samples
│       └── mock_responses.py         # Mock Azure API responses
│
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh          # Local dev environment setup
│   ├── run_tests.sh                  # Test execution wrapper
│   └── cleanup_resources.py          # Azure resource cleanup
│
├── .github/                           # GitHub configuration
│   ├── workflows/
│   │   ├── ci.yml                    # CI pipeline (test, lint)
│   │   └── cd.yml                    # CD pipeline (deploy)
│   └── prompts/
│       └── constitution.md           # Project principles
│
├── pyproject.toml                    # Project metadata and dependencies (uv)
├── uv.lock                           # Locked dependencies (uv)
├── .python-version                   # Python version specification (uv)
├── pytest.ini                        # Pytest configuration
├── .env.template                     # Environment variable template
├── README.md                         # Project documentation
└── .gitignore                        # Git exclusions
```

**Structure Decision**:

- **Notebooks for orchestration**: Non-technical users can follow step-by-step workflow
- **src/ for reusable modules**: Separates business logic from orchestration
- **infra/ for IaC**: Terraform modules enable reproducible infrastructure
- **configs/ for settings**: YAML files externalize configuration from code
- **docker/ for containers**: Separate training and inference Dockerfiles
- **tests/ mirroring src/**: Clear test organization following source structure

## Complexity Tracking

> All complexity is justified by legitimate requirements. No violations of constitution detected.

| Decision                                         | Justification                                                                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Jupyter notebooks + Python modules               | Notebooks provide accessibility for data scientists while modules ensure code reusability and testability                     |
| Terraform for IaC                                | Azure ML workspace requires complex resource relationships; Terraform provides declarative, version-controlled infrastructure |
| Separate training/inference containers           | Training requires GPU drivers and large dependencies; inference needs minimal footprint for fast startup                      |
| Multi-stage project (infra, training, inference) | Each stage has distinct concerns, dependencies, and deployment targets; separation improves maintainability                   |

## Phase 0: Research & Technical Validation

### Azure ML Integration Research

**Goal**: Validate Azure ML SDK v2 capabilities for Phi-4 model training and AI Foundry model access

**Tasks**:

1. Research Azure ML Python SDK v2 job submission patterns

   - Command jobs vs pipeline jobs
   - Environment configuration (Docker vs curated environments)
   - Compute target attachment and scaling

2. Investigate Phi-4 mini model access from Azure AI Foundry

   - AI Foundry model catalog API and SDK
   - Authentication requirements (workspace connection, API keys)
   - Model download patterns from AI Foundry (not HuggingFace)
   - Model registry integration
   - Version pinning and model metadata access

3. Validate PyTorch + Transformers compatibility

   - Phi-4 mini model class requirements
   - Tokenizer configuration from AI Foundry models
   - Fine-tuning API (LoRA preferred for smaller footprint)
   - Model quantization during/after training

3a. Research MLflow integration with Azure ML jobs ✅ **COMPLETED**

- Azure ML automatically starts MLflow runs for all training jobs
- AZUREML_RUN_ID environment variable indicates Azure ML execution context
- Training scripts MUST NOT call mlflow.start_run() or mlflow.set_experiment() in Azure ML
- Only call MLflow setup functions when AZUREML_RUN_ID is not set (local execution)
- Azure ML platform manages run lifecycle, experiment tracking, and metrics logging
- Training scripts can log metrics via mlflow.log_params/log_metrics to existing run
- Reference: Azure ML SDK documentation on automatic MLflow tracking

4. Research Azure Blob Storage integration
   - Datastore mounting in compute jobs
   - Large file upload optimization
   - Dataset versioning patterns

4a. Research Python module execution patterns in Azure ML ✅ **COMPLETED**

- Azure ML jobs should use module execution: `python -m src.training.train`
- Module execution sets sys.path[0] to package root, not script directory
- Import order critical: stdlib → sys.path setup → project imports
- Use Path(**file**).resolve().parents[2] for reliable project root location
- Add noqa: E402 comments for imports after sys.path modification
- Prevents ModuleNotFoundError when scripts are in nested directories

**Output**: Document recommended patterns in `research.md`

### Terraform Azure Provider Research

**Goal**: Determine Terraform resource configurations for Azure ML

**Tasks**:

1. Research azurerm provider resources

   - `azurerm_machine_learning_workspace`
   - `azurerm_machine_learning_compute_cluster`
   - `azurerm_storage_account` with ML integration
   - `azurerm_container_registry`

2. Validate dependency ordering

   - Workspace dependencies (Key Vault, Storage, App Insights)
   - Compute cluster attachment timing
   - Network configuration requirements

3. Research authentication patterns
   - Service principal vs managed identity
   - Terraform state storage (backend configuration)
   - RBAC role assignments

**Output**: Document Terraform module design in `research.md`

### Container Optimization Research

**Goal**: Identify aggressive optimization strategies for embedded hardware deployment

**Tasks**:

1. Research PyTorch inference optimization for CPU-only embedded devices

   - Model quantization options (int8, int4 for extreme size reduction)
   - Post-training quantization vs quantization-aware training
   - TorchScript compilation for CPU optimization
   - ONNX Runtime for embedded deployment (vs PyTorch)
   - Model pruning and distillation techniques
   - ARM64 optimization considerations

2. Investigate minimal Docker base image options for embedded

   - python:3.11-slim-bullseye (Debian) vs Alpine Linux
   - Distroless images for security and size
   - Multi-stage build with aggressive dependency pruning
   - Static linking and dependency reduction
   - Remove build tools and unnecessary packages
   - Target both ARM64 and x86_64 architectures

3. Research lightweight inference server patterns

   - FastAPI vs Flask vs pure Python HTTP server
   - Uvicorn alternatives (lightweight ASGI servers)
   - Minimize Python dependencies
   - Single-worker synchronous pattern for embedded
   - Health check with minimal overhead

4. Research embedded deployment patterns
   - Container runtime options (Docker, containerd, Podman)
   - Resource limits and memory management
   - Cold start optimization techniques
   - Model loading and caching strategies

**Output**: Document embedded container strategy in `research.md`

## Phase 1: Design & Contracts

### Training Data Workflow

**Goal**: Define the end-to-end flow from user's local JSONL file to Azure ML training job

**Workflow Steps**:

1. **User Preparation** (Local)

   - User creates `data/` directory in repository root
   - User places training data at `data/training_data.jsonl`
   - Format: Each line is JSON with `{"prompt": "input text", "completion": "expected output"}`
   - Example:
     ```jsonl
     {"prompt": "Translate to French: Hello", "completion": "Bonjour"}
     {"prompt": "Summarize: Long text here...", "completion": "Short summary"}
     ```

2. **Validation** (Local)

   - Script: `src/data/preprocessing.py`
   - Notebook: `02-prepare-data.ipynb`
   - Actions:
     - Verify file exists and has `.jsonl` extension
     - Parse each line as JSON
     - Check required fields: `prompt` and `completion`
     - Validate field types (non-empty strings)
     - Check token counts against max limits
     - Compute dataset statistics (sample count, avg tokens)
     - Report errors with line numbers and examples

3. **Splitting** (Local)

   - Script: `src/data/preprocessing.py`
   - Actions:
     - Split data into train (80%) and validation (20%) sets
     - Shuffle with fixed random seed for reproducibility
     - Write split files to `data/processed/train.jsonl` and `data/processed/val.jsonl`
     - Generate manifest file with split metadata

4. **Upload to Azure Blob** (Cloud)

   - Script: `src/data/upload_to_blob.py`
   - Notebook: `02-prepare-data.ipynb`
   - Actions:
     - Authenticate to Azure Storage (managed identity or connection string from `.env`)
     - Create blob container if needed (name from config: `training-data`)
     - Upload train/val files with progress bars
     - Store under organized prefix: `training-data/phi4-finetune/v1/`
     - Return blob URIs for dataset registration

5. **Register as Azure ML Dataset** (Cloud)

   - Script: `src/data/upload_to_blob.py`
   - Notebook: `02-prepare-data.ipynb`
   - Actions:
     - Create Azure ML FileDataset pointing to uploaded blobs
     - Add metadata: sample counts, schema version, creation timestamp
     - Register with name (e.g., `phi4-finetune-dataset`) and version (e.g., `v1`)
     - Enable dataset mounting for compute jobs
     - Verify dataset appears in Azure ML Studio

6. **Use in Training** (Cloud)
   - Script: `src/training/train.py`
   - Notebook: `05-train-model.ipynb` or `06-submit-training-job.ipynb`
   - Actions:
     - Azure ML compute job mounts registered dataset
     - Training script reads from mounted path (e.g., `/mnt/data/train.jsonl`)
     - Data is already validated and split—ready for immediate use
     - MLflow tracks dataset version used in experiment

**Configuration Files**:

- `configs/data_config.yaml`: Paths, split ratios, validation thresholds
- `.env`: Azure Storage credentials, container names
- `.env.template`: Template showing required variables

**Error Handling**:

- Clear error messages with line numbers for validation failures
- Retry logic with exponential backoff for blob uploads
- Validation checkpoint before expensive upload operation
- Rollback dataset registration if upload incomplete

### Data Model Design

**Goal**: Define schemas for training data, configurations, and artifacts

**Entities**:

1. **TrainingDataset**

```python
@dataclass
class TrainingDataset:
    """Represents a versioned training dataset in Azure ML"""
    name: str
    version: str
    local_path: str  # e.g., "data/training_data.jsonl"
    datastore_path: str  # Azure Blob path after upload
    format: Literal["jsonl", "parquet"]
    sample_count: int
    train_split_ratio: float
    validation_split_ratio: float
    created_at: datetime
    schema_version: str
    validation_status: Literal["pending", "valid", "invalid"]
    validation_errors: List[str]
```

2. **TrainingConfig**

```python
@dataclass
class TrainingConfig:
    """Hyperparameters and training settings"""
    model_source: str = "azureml://registries/azureml/models/Phi-4/versions/1"  # AI Foundry model path
    model_name: str = "Phi-4"
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 2048
    checkpoint_frequency: int = 500
    logging_steps: int = 10
    use_lora: bool = True  # Use LoRA for smaller adapter weights
    lora_r: int = 16
    lora_alpha: int = 32
    quantization_config: Optional[str] = "int8"  # Quantize during training for smaller models
```

3. **ModelArtifact**

```python
@dataclass
class ModelArtifact:
    """Trained model metadata"""
    model_id: str
    base_model: str
    training_job_id: str
    checkpoint_path: str
    final_loss: float
    validation_perplexity: float
    training_duration_minutes: int
    total_steps: int
    created_at: datetime
```

4. **InferenceConfig**

```python
@dataclass
class InferenceConfig:
    """Container deployment configuration for embedded hardware"""
    container_name: str
    registry_name: str
    image_tag: str
    target_architecture: Literal["arm64", "amd64", "multi"] = "multi"
    cpu_cores: float = 2.0  # Typical embedded device capability
    memory_gb: float = 4.0  # Conservative for embedded devices
    max_batch_size: int = 8  # Reduced for embedded constraints
    inference_timeout_seconds: int = 10  # Faster timeout for embedded
    model_format: Literal["pytorch", "onnx", "torchscript"] = "onnx"  # ONNX for embedded
    quantization: Literal["int8", "int4", "fp16", "none"] = "int8"
    enable_model_caching: bool = True
    environment_variables: Dict[str, str] = field(default_factory=dict)

@dataclass
class ModelOptimizationConfig:
    """Configuration for model optimization and quantization"""
    quantization_method: Literal["dynamic", "static", "qat"] = "static"
    target_format: Literal["pytorch", "onnx", "torchscript"] = "onnx"
    optimization_level: int = 3  # ONNX optimization level
    target_device: Literal["cpu", "arm"] = "cpu"
    calibration_dataset_size: int = 100  # Samples for static quantization
    preserve_accuracy_threshold: float = 0.95  # Minimum accuracy retention
```

**Output**: Full data model documentation in `data-model.md`

### API Contracts

**Goal**: Define interfaces between components

1. **Data Preparation Contract**

```python
def validate_jsonl_format(
    file_path: Path
) -> Tuple[bool, List[str]]:
    """
    Validates JSONL file format and required fields.

    Args:
        file_path: Path to JSONL file (e.g., data/training_data.jsonl)

    Returns:
        Tuple of (is_valid, list_of_errors)
        - is_valid: True if all checks pass
        - list_of_errors: Line-specific error messages

    Validation Rules:
        - File must have .jsonl extension
        - Each line must be valid JSON
        - Each JSON object must have "prompt" and "completion" fields
        - Fields must be non-empty strings
        - Token counts must be within configured limits
    """

def prepare_training_data(
    input_path: Path,
    output_path: Path,
    train_ratio: float = 0.8
) -> TrainingDataset:
    """
    Validates and splits raw training data for Azure ML.

    Workflow:
        1. Load JSONL from input_path (e.g., data/training_data.jsonl)
        2. Validate format and content
        3. Compute statistics (sample count, token distributions)
        4. Create train/validation split
        5. Write processed files to output_path
        6. Return TrainingDataset metadata object

    Args:
        input_path: Path to raw JSONL file in repo (data/training_data.jsonl)
        output_path: Path to write processed data (data/processed/)
        train_ratio: Ratio of data for training set (default 0.8)

    Returns:
        TrainingDataset metadata object with paths and statistics

    Raises:
        ValidationError: If data format is invalid (with line numbers)
        FileNotFoundError: If input_path doesn't exist
    """

def upload_to_blob_storage(
    local_path: Path,
    storage_account: str,
    container_name: str,
    blob_prefix: str,
    show_progress: bool = True
) -> str:
    """
    Uploads training data to Azure Blob Storage.

    Workflow:
        1. Authenticate to Azure Storage (managed identity or connection string)
        2. Create container if it doesn't exist
        3. Upload file with progress tracking
        4. Return blob URI for Azure ML dataset registration

    Args:
        local_path: Path to local file (e.g., data/training_data.jsonl)
        storage_account: Azure Storage account name
        container_name: Blob container name (e.g., "training-data")
        blob_prefix: Prefix for blob path (e.g., "phi4-finetune/v1/")
        show_progress: Display upload progress bar

    Returns:
        Full blob URI (e.g., "https://<account>.blob.core.windows.net/<container>/<prefix>/training_data.jsonl")

    Raises:
        AuthenticationError: If Azure credentials are invalid
        StorageError: If upload fails
    """

def register_azureml_dataset(
    workspace: Workspace,
    blob_uri: str,
    dataset_name: str,
    dataset_version: str,
    description: str,
    metadata: Dict[str, Any]
) -> Dataset:
    """
    Registers blob data as versioned Azure ML dataset.

    Workflow:
        1. Create FileDataset pointing to blob URI
        2. Add metadata (sample count, schema, creation date)
        3. Register in workspace with version
        4. Enable dataset mounting for compute jobs

    Args:
        workspace: Azure ML workspace object
        blob_uri: URI of uploaded blob
        dataset_name: Name for dataset (e.g., "phi4-finetune-dataset")
        dataset_version: Version string (e.g., "v1", "2024-11-20")
        description: Human-readable description
        metadata: Additional metadata (sample_count, train_ratio, etc.)

    Returns:
        Registered Dataset object

    Raises:
        RegistrationError: If dataset registration fails
        WorkspaceError: If workspace is invalid
    """
```

2. **Training Job Submission Contract**

```python
def download_model_from_foundry(
    workspace: Workspace,
    model_path: str,
    output_dir: Path
) -> Path:
    """
    Downloads model from Azure AI Foundry model catalog.

    Args:
        workspace: Azure ML workspace object
        model_path: AI Foundry model path (e.g., azureml://registries/azureml/models/Phi-4/versions/1)
        output_dir: Local directory to save model

    Returns:
        Path to downloaded model directory

    Raises:
        ModelNotFoundError: If model doesn't exist in catalog
        AuthenticationError: If workspace lacks permissions
    """

def submit_training_job(
    workspace: Workspace,
    compute_target: str,
    dataset: TrainingDataset,
    config: TrainingConfig
) -> Job:
    """
    Submits fine-tuning job to Azure ML.

    Args:
        workspace: Azure ML workspace object
        compute_target: Name of compute cluster
        dataset: Training dataset reference
        config: Training hyperparameters with AI Foundry model source

    Returns:
        Azure ML Job object for monitoring

    Raises:
        ComputeNotFoundError: If compute target doesn't exist
        QuotaExceededError: If GPU quota insufficient
        ModelAccessError: If AI Foundry model unavailable
    """
```

3. **Inference API Contract**

```python
class InferenceRequest(BaseModel):
    """Request schema for model inference"""
    text: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

class InferenceResponse(BaseModel):
    """Response schema for model inference"""
    request_id: str
    prediction: str
    tokens_generated: int
    latency_ms: int
    model_version: str
```

4. **Model Optimization Contract**

```python
def optimize_model_for_embedded(
    model_path: Path,
    output_path: Path,
    config: ModelOptimizationConfig,
    calibration_data: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Optimizes trained model for embedded deployment.

    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save optimized model
        config: Optimization configuration (quantization, format)
        calibration_data: Sample inputs for static quantization

    Returns:
        Dict with optimization metrics:
        - original_size_mb: Original model size
        - optimized_size_mb: Optimized model size
        - compression_ratio: Size reduction ratio
        - inference_speedup: Latency improvement factor
        - accuracy_delta: Accuracy change (negative = degradation)

    Raises:
        OptimizationError: If optimization fails
        AccuracyThresholdError: If accuracy drops below threshold
    """
```

**Output**: Complete API contracts in `contracts/` directory

### Quickstart Guide

**Goal**: Step-by-step guide for first-time users

**Contents**:

1. Prerequisites checklist

   - Azure subscription with quota
   - Azure CLI installed and configured
   - Terraform installed
   - Python 3.11+ with uv installed (ultra-fast package manager)
   - Docker installed

2. Infrastructure setup (5 minutes)

   ```bash
   cd infra/terraform
   terraform init
   terraform apply -var-file=environments/dev.tfvars
   ```

3. First training run (30 minutes)

   - Prepare your training data:
     ```bash
     mkdir -p data
     # Place your training data at data/training_data.jsonl
     # Format: {"prompt": "...", "completion": "..."} per line
     ```
   - Open `notebooks/02-prepare-data.ipynb`
     - Validates JSONL format
     - Creates train/validation split
     - Uploads to Azure Blob Storage
     - Registers as Azure ML dataset
   - Open `notebooks/03-download-base-model.ipynb`
     - Downloads Phi-4 model from Azure AI Foundry
   - Open `notebooks/04-train-model.ipynb` or `notebooks/05-submit-training-job.ipynb`
     - Submits training job to Azure ML compute
     - Monitors training progress and metrics
   - Verify outputs at each step

4. Optimize model for embedded (15 minutes)

   - Execute `notebooks/05-optimize-model.ipynb`
   - Quantize and convert to ONNX format
   - Validate accuracy retention

5. Deploy inference container (10 minutes)
   - Execute `notebooks/06-build-container.ipynb` (optimized for <500MB)
   - Push to ACR with `notebooks/07-push-to-acr.ipynb`
   - Test on embedded hardware target

**Output**: Complete quickstart in `quickstart.md`

## Phase 2: Implementation Tasks

_Tasks will be generated by `/speckit.tasks` command based on this plan and the specification._

**Implementation Status Summary**:

- ✅ Infrastructure (Terraform + azd): 100% complete
- ✅ Data pipeline: 100% complete
- ✅ Training job submission: 95% complete
- ⚠️ Model download: 70% complete
- ⚠️ Model optimization: 30% complete
- ❌ Model evaluation: 5% complete (placeholder only)
- ❌ Container build/push: Scripts exist, notebook integration 0%
- ❌ Deployment: 0% complete (documentation only)

**High-level task breakdown with status**:

1. **Infrastructure Tasks** (Priority: P1) ✅ **COMPLETED** ✅ **COMPLETED**

   - ✅ Created modular Terraform structure with 8 modules:
     - log-analytics (monitoring foundation)
     - application-insights (telemetry)
     - key-vault (secrets management with RBAC)
     - storage (blob storage with ML integration)
     - container-registry (ACR for container images)
     - azureml-workspace (ML workspace with managed identity)
     - compute-cluster (GPU cluster with auto-scaling)
     - user-assigned-identity (shared across resources)
   - ✅ Implemented azd integration with azure.yaml and setup script
   - ✅ RBAC configuration with proper role assignments and propagation delays
   - ✅ Random suffix generation for globally unique names
   - ✅ Comprehensive outputs for .env file generation
   - ✅ uv configured in pyproject.toml with locked dependencies

2. **Data Pipeline Tasks** (Priority: P1) ✅ **COMPLETED**

   - ✅ Implemented JSONL validation and preprocessing in notebook 01-prepare-data.ipynb
     - User places training data at `data/training_data.jsonl` in repo root
     - Validate JSONL format (each line: `{"prompt": "...", "completion": "..."}`)
     - Check field presence, encoding, max token lengths
     - Report line-specific errors with examples
   - Create blob upload utilities
     - Authenticate to Azure Storage using managed identity or connection string
     - Upload validated JSONL to Azure Blob Storage container
     - Show progress bars for large file uploads
     - Support resumable uploads for reliability
   - Build dataset registration script
     - Register uploaded blob as Azure ML Dataset with versioning
     - Add metadata (sample count, creation date, schema version)
     - Enable dataset mounting in Azure ML compute jobs
   - Create notebook for data preparation workflow
     - Step 1: Load local JSONL from `data/training_data.jsonl`
     - Step 2: Run validation and display statistics
     - Step 3: Create train/validation split (80/20 default)
     - Step 4: Upload to blob storage with progress tracking
     - Step 5: Register as Azure ML dataset
     - Step 6: Verify dataset is accessible from workspace

3. **Training Tasks** (Priority: P2) ✅ **COMPLETED**

   - ✅ Training script (src/training/train.py) with full pipeline
   - ✅ LoRA fine-tuning via PEFT library
   - ✅ Checkpointing system with resumption support
   - ✅ MLflow integration with Azure ML compatibility:
     - Detects Azure ML via AZUREML_RUN_ID environment variable
     - Skips MLflow setup in Azure ML (platform manages automatically)
     - Full MLflow tracking for local execution
   - ✅ Module execution pattern: `python -m src.training.train`
   - ✅ Proper import ordering for Azure ML compatibility
   - ✅ AzureMLJobManager abstraction layer wrapping SDK complexity
   - ✅ Comprehensive monitoring in notebook 02-submit-training-job.ipynb:
     - Environment build tracking (`prepare_image` experiment)
     - Job status monitoring across lifecycle phases
     - Recent jobs listing with detailed status
     - Cost estimation and duration guidance
   - ✅ Excellent documentation with troubleshooting section

4. **Model Download Tasks** (Priority: P2) ⚠️ **PARTIALLY COMPLETED**

   - ✅ Notebook 03-download-trained-model.ipynb with ModelDownloader class
   - ✅ Job listing and checkpoint discovery
   - ✅ Artifact download with validation
   - ✅ Model registry integration with versioning
   - ✅ Metadata extraction and size calculation
   - ❌ Missing: Model loading test with base model
   - ❌ Missing: Inference validation
   - ❌ Missing: LoRA adapter merge for standalone deployment

5. **Evaluation Tasks** (Priority: P2) ❌ **NOT IMPLEMENTED**

   - ❌ Notebook 05-evaluate-model.ipynb is placeholder only
   - ❌ No metric computation (perplexity, BLEU, ROUGE, accuracy)
   - ❌ No test dataset loading
   - ❌ No inference execution on evaluation set
   - ❌ No comparison with base/optimized models
   - ❌ No evaluation report generation
   - ❌ No quality gates implementation

6. **Model Optimization Tasks** (Priority: P2) ⚠️ **MINIMAL IMPLEMENTATION**

   - ⚠️ Notebook 04-optimize-model.ipynb has function scaffolding
   - ⚠️ Calls to quantize_model(), export_to_onnx(), benchmark_inference()
   - ❌ Missing: LoRA merge with base model (critical for deployment)
   - ❌ Missing: Quantization validation and quality checks
   - ❌ Missing: Performance comparison (before/after optimization)
   - ❌ Missing: Accuracy preservation verification (>95% target)
   - ❌ Missing: Memory profiling (<4GB requirement)
   - ❌ Missing: Calibration dataset generation for static quantization
   - ✅ Inference scripts exist in src/inference/ (score.py, server.py)

7. **Inference Tasks** (Priority: P2) ✅ **COMPLETED**

   - ✅ Scoring script (src/inference/score.py) with lazy loading
   - ✅ FastAPI server (src/inference/server.py) with /health and /generate endpoints
   - ✅ Supports both PyTorch and ONNX backends
   - ✅ Environment variable configuration
   - ✅ Single-request optimization for embedded
   - ✅ Memory-efficient model caching

8. **Container Tasks** (Priority: P2) ⚠️ **SCRIPTS EXIST, NOTEBOOK INCOMPLETE**

   - ✅ Multi-stage Dockerfile (docker/Dockerfile) for minimal image size
   - ✅ Build script for multi-architecture (scripts/build_multiarch.sh)
   - ✅ Push script for ACR (scripts/push_to_acr.sh)
   - ✅ Health check script (scripts/health_check.sh)
   - ❌ Notebook 06-push-to-acr.ipynb is documentation-only
   - ❌ No notebook integration of build/push workflow
   - ❌ No size validation (<500MB target)
   - ❌ No automated testing of built containers

   - Create multi-stage Dockerfile with aggressive optimization (<500MB target)
   - Implement minimal base image selection (Alpine/distroless)
   - Build multi-architecture support (ARM64 + x86_64)
   - Remove all unnecessary dependencies and build tools
   - Optimize Python package installation with uv (no cache, minimal deps)
   - Implement container build script with size validation
   - Create ACR push automation with architecture tags
   - Build container orchestration notebooks
   - Add embedded hardware deployment guide

9. **Container Deployment Tasks** (Priority: P2) ❌ **NOT IMPLEMENTED**

   - ❌ Notebook 07-deploy-inference.ipynb is documentation-only
   - ❌ No Azure Container Apps deployment code
   - ❌ No local deployment automation
   - ❌ No Docker Compose configuration
   - ❌ No embedded system deployment scripts
   - ❌ No systemd service files
   - ❌ No health monitoring setup
   - ❌ No rollback/update procedures
   - ✅ Excellent documentation explaining deployment options

10. **Testing Tasks** (Priority: P3)

    - Write unit tests for data processing
    - Create training logic tests with mocks
    - Implement integration tests for Azure ML and AI Foundry
    - Build model optimization validation tests
    - Build container smoke tests on embedded hardware
    - Test multi-architecture container builds
    - Setup CI pipeline with pytest

11. **Documentation Tasks** (Priority: P3)

- Complete README with architecture diagram
- Document configuration options
- Create troubleshooting guide
- Create embedded deployment guide
- Write API documentation
- Add inline code documentation

## Success Metrics

_Derived from specification success criteria_

**Phase 0 Complete**: All research documented with clear recommendations
**Phase 1 Complete**: Data models, contracts, and quickstart validated by team
**Phase 2 Complete**: All tasks implemented and passing tests

**System Acceptance**:

- ✅ Infrastructure provisions in <10 minutes via Terraform
- ✅ Training completes on 10K samples in <2 hours with GPU utilization >80%
- ✅ Model optimization reduces size by >50% with <2% accuracy loss
- ✅ Container image builds in <10 minutes and is <500MB
- ✅ Container supports both ARM64 and x86_64 architectures
- ✅ Inference P95 latency <50ms for single requests on CPU
- ✅ Container starts in <10 seconds on embedded hardware
- ✅ Memory footprint during inference <4GB
- ✅ Container deploys successfully locally for testing
- ✅ Container deploys successfully to embedded Linux device
- ✅ Systemd service runs reliably on embedded system
- ✅ End-to-end notebook workflow completes successfully
- ✅ 80% code coverage achieved
- ✅ All integration tests pass against live Azure resources
- ✅ Embedded deployment validated on target hardware
- ✅ Python environment managed consistently via uv

## Dependencies & Risks

**External Dependencies**:

- Azure subscription with GPU quota (Standard_NC6s_v3 or better)
- Access to Azure AI model catalog for Phi-4 mini
- Azure Container Registry with sufficient storage
- Terraform cloud or Azure storage for state

**Technical Risks**:
| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU quota unavailable | HIGH - Training blocked | Request quota early; provide CPU fallback with reduced performance |
| AI Foundry model access issues | HIGH - Cannot download base model | Validate workspace permissions early; cache model locally |
| Container exceeds 500MB target | HIGH - Won't fit on embedded devices | Use aggressive quantization (int4); minimal base image; ONNX format |
| Quantization degrades accuracy >5% | HIGH - Model unusable | Use quantization-aware training; adjust optimization settings |
| Inference latency >50ms on embedded | MEDIUM - Poor user experience | Profile and optimize; consider smaller model variant |
| ARM64 architecture incompatibilities | MEDIUM - Can't deploy to edge | Test early on ARM hardware; use multi-arch builds |
| Azure ML SDK breaking changes | MEDIUM - Job submission fails | Pin SDK versions; test against staging workspace |
| ONNX conversion loses functionality | LOW - Fallback to PyTorch | Validate ONNX output; maintain PyTorch option |
| Terraform state conflicts | LOW - Deployment issues | Use remote backend with state locking |

**Timeline Estimate**:

- Phase 0 (Research): 3-5 days
- Phase 1 (Design): 3-5 days
- Phase 2 (Implementation): 15-20 days
- **Total**: 21-30 days for complete implementation

## Next Steps

1. **Review this plan** with team for feedback and adjustments
2. **Execute Phase 0**: Begin research on Azure ML SDK patterns and Phi-4 access
3. **Update constitution** if new patterns require policy additions
4. **Generate tasks**: Run `/speckit.tasks` command after Phase 1 completes
5. **Setup branch**: Create `001-slm-train-deploy-system` feature branch
6. **Begin infrastructure**: Start with Terraform module development as foundation
