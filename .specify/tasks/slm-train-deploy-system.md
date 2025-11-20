---
description: "Task list for SLM Training and Deployment System"
---

# Tasks: SLM Training and Deployment System

**Input**: Design documents from `.specify/specs/` and `.specify/plans/`
**Prerequisites**: plan.md, spec.md, constitution.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Training Data Workflow**: User places JSONL at `data/training_data.jsonl` → validation → split → upload to blob → register as Azure ML dataset → use in training. See Phase 3 (User Story 1) for detailed tasks.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan: notebooks/, src/, infra/, configs/, docker/, tests/
- [ ] T002 Initialize Python project with pyproject.toml using uv (uv init)
- [ ] T003 [P] Create pyproject.toml with Azure ML SDK, PyTorch, Transformers, MLflow dependencies using uv
- [ ] T004 [P] Add dev dependencies with uv add --dev pytest black mypy ruff
- [ ] T005 [P] Configure .gitignore for Python, Jupyter, Terraform, Docker artifacts
- [ ] T006 [P] Create .env.template with required environment variables
- [ ] T007 [P] Setup pytest.ini configuration with coverage settings
- [ ] T008 [P] Create GitHub Actions workflow stubs in .github/workflows/
- [ ] T009 Create README.md with project overview and quickstart instructions
- [ ] T010 [P] Setup pre-commit hooks for code formatting and linting

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Infrastructure Foundation

- [ ] T011 Create Terraform root module in infra/terraform/main.tf
- [ ] T012 [P] Define Terraform variables in infra/terraform/variables.tf
- [ ] T013 [P] Define Terraform outputs in infra/terraform/outputs.tf
- [ ] T014 [P] Create dev environment config in infra/terraform/environments/dev.tfvars
- [ ] T015 [P] Create prod environment config in infra/terraform/environments/prod.tfvars

### Python Infrastructure

- [ ] T016 Setup uv environment with uv sync to install all dependencies from pyproject.toml
- [ ] T017 Create configuration loader in src/utils/config.py
- [ ] T018 [P] Create structured logging setup in src/utils/logging_config.py
- [ ] T019 [P] Create Azure authentication helpers in src/utils/azure_auth.py
- [ ] T020 Create base configuration YAML schemas in configs/
- [ ] T021 [P] Create data validation utilities in src/data/**init**.py
- [ ] T022 [P] Create common model utilities in src/training/**init**.py

### Testing Infrastructure

- [ ] T023 Create test fixtures directory structure in tests/fixtures/
- [ ] T024 [P] Create sample test data in tests/fixtures/sample_data.jsonl
- [ ] T025 [P] Create mock Azure API responses in tests/fixtures/mock_responses.py
- [ ] T026 Setup conftest.py with pytest fixtures for Azure ML mocking

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Setup and Upload Training Data (Priority: P1) 🎯 MVP

**Goal**: Enable data scientists to prepare and upload training datasets from local JSONL file to Azure ML datastore

**User Workflow**: User places `data/training_data.jsonl` in repo → validation → train/val split → upload to blob storage → register as Azure ML dataset

**Independent Test**: Place sample JSONL at `data/training_data.jsonl`, run data prep script, upload to Azure ML, verify dataset in Studio

### Implementation for User Story 1

- [ ] T026 [P] [US1] Create data validation function `validate_jsonl_format()` in src/data/preprocessing.py

  - Check file exists at expected path (data/training_data.jsonl)
  - Verify .jsonl extension
  - Parse each line as valid JSON with line number tracking
  - Validate required fields: "prompt" and "completion" in each object
  - Check fields are non-empty strings
  - Validate token counts against max_prompt_length and max_completion_length from config
  - Return tuple: (is_valid: bool, errors: List[str] with line numbers)

- [ ] T027 [P] [US1] Implement `prepare_training_data()` function in src/data/preprocessing.py

  - Load JSONL from input_path (default: data/training_data.jsonl)
  - Run validation via validate_jsonl_format()
  - Compute dataset statistics: sample count, avg/max token lengths, field distributions
  - Create train/validation split with configurable ratio (default 80/20)
  - Shuffle with fixed random seed for reproducibility
  - Write split files to output_path (default: data/processed/train.jsonl, data/processed/val.jsonl)
  - Generate manifest JSON with split metadata and statistics
  - Return TrainingDataset metadata object

- [ ] T028 [P] [US1] Create `compute_dataset_statistics()` function in src/data/preprocessing.py

  - Count total samples
  - Compute token length distributions (mean, median, max, percentiles)
  - Detect potential issues: duplicates, very short/long examples, encoding problems
  - Generate summary dict for logging and metadata

- [ ] T029 [US1] Implement `upload_to_blob_storage()` function in src/data/upload_to_blob.py

  - Authenticate to Azure Storage using DefaultAzureCredential or connection string from .env
  - Retrieve storage_account_name and container_name from config/env
  - Create blob container if it doesn't exist
  - Upload train.jsonl and val.jsonl with blob_prefix from config (e.g., "training-data/phi4-v1/")
  - Show progress bar using tqdm for large files
  - Support resumable uploads for reliability
  - Return blob URIs for both files

- [ ] T030 [US1] Create `register_azureml_dataset()` function in src/data/upload_to_blob.py

  - Accept workspace, blob_uris, dataset_name, dataset_version, metadata
  - Create Azure ML FileDataset from blob URIs
  - Add metadata: sample_count, train_ratio, schema_version, creation_timestamp
  - Register in workspace with versioning
  - Enable dataset mounting for compute jobs
  - Return registered Dataset object

- [ ] T031 [US1] Add comprehensive progress tracking in src/data/upload_to_blob.py

  - File size calculation before upload
  - Progress bar with percentage, speed, ETA
  - Success/failure logging with timestamps
  - Retry logic with exponential backoff for network errors

- [ ] T032 [US1] Create configs/data_config.yaml with clear documentation

  - input_file: "data/training_data.jsonl" (where user places data)
  - output_dir: "data/processed" (split files destination)
  - validation_split: 0.2 (train/val ratio)
  - random_seed: 42 (reproducibility)
  - blob_prefix: "training-data/phi4-v1/" (Azure blob path)
  - dataset_name: "phi4-finetune-dataset" (Azure ML dataset name)
  - dataset_version: "v1" (version tracking)
  - max_prompt_length: 2048 (validation threshold)
  - max_completion_length: 1024 (validation threshold)
  - Add inline YAML comments explaining each field

- [ ] T033 [US1] Create notebooks/02-prepare-data.ipynb with comprehensive workflow

  - Cell 1: Introduction and prerequisites
    - Explain user must place data/training_data.jsonl first
    - Show expected JSONL format with example
    - List required .env variables (AZURE_STORAGE_ACCOUNT_NAME, etc.)
  - Cell 2: Load configuration from configs/data_config.yaml
  - Cell 3: Validate local JSONL file
    - Run validate_jsonl_format()
    - Display validation results: pass/fail, error count, examples
  - Cell 4: Prepare and split data
    - Run prepare_training_data()
    - Display statistics: sample counts, token distributions
    - Show split file paths
  - Cell 5: Upload to Azure Blob Storage
    - Run upload_to_blob_storage()
    - Show progress and blob URIs
  - Cell 6: Register as Azure ML Dataset
    - Run register_azureml_dataset()
    - Display registration confirmation and dataset ID
  - Cell 7: Verify in Azure ML Studio
    - Provide link to dataset in Studio
    - Show how to view dataset metadata
  - Cell 8: Next steps guidance (proceed to model download or training)

- [ ] T034 [US1] Add comprehensive error handling in src/data/preprocessing.py and src/data/upload_to_blob.py
  - FileNotFoundError: Clear message if data/training_data.jsonl missing
  - JSONDecodeError: Report line number and invalid JSON snippet
  - ValidationError: List all validation failures with examples
  - AuthenticationError: Guide user to check .env credentials
  - StorageError: Suggest quota/permissions troubleshooting
  - NetworkError: Implement retry with user notification
  - All errors include actionable remediation steps

### Tests for User Story 1

- [ ] T035 [P] [US1] Unit test for `validate_jsonl_format()` in tests/unit/test_preprocessing.py

  - Test valid JSONL with correct format passes validation
  - Test missing prompt field triggers error with line number
  - Test missing completion field triggers error
  - Test empty string fields trigger errors
  - Test non-JSON lines trigger parsing errors
  - Test file with wrong extension (.json) triggers error
  - Test missing file triggers FileNotFoundError
  - Test very long prompts exceed max_prompt_length threshold
  - Verify error messages include line numbers and examples

- [ ] T036 [P] [US1] Unit test for `prepare_training_data()` in tests/unit/test_preprocessing.py

  - Test train/validation split ratios are correct (80/20)
  - Test random seed produces reproducible splits
  - Test output files written to correct paths
  - Test manifest file contains accurate metadata
  - Test statistics calculation: sample counts, token distributions
  - Test error handling when validation fails
  - Verify TrainingDataset object contains all required fields

- [ ] T037 [P] [US1] Integration test for blob upload in tests/integration/test_blob_upload.py

  - Test successful upload to Azure Blob Storage
  - Test progress tracking during upload
  - Test retry logic with network simulation
  - Test authentication with valid credentials
  - Test error handling with invalid credentials
  - Test container creation if it doesn't exist
  - Test blob URI format is correct
  - Verify uploaded files are accessible and complete

- [ ] T038 [P] [US1] Integration test for dataset registration in tests/integration/test_blob_upload.py
  - Test successful Azure ML dataset registration
  - Test metadata is correctly attached (sample_count, version, etc.)
  - Test versioning works correctly
  - Test dataset is mountable in compute jobs
  - Test dataset appears in Azure ML Studio
  - Test error handling when registration fails
  - Verify registered dataset contains correct file references

**Checkpoint**: Can prepare and upload training data to Azure ML independently

---

## Phase 4: User Story 2 - Setup Azure ML Workspace and Resources (Priority: P1) 🎯 MVP

**Goal**: Provision complete Azure ML environment with all required resources

**Independent Test**: Run Terraform apply, verify workspace exists in Azure Portal

### Implementation for User Story 2

- [ ] T039 [P] [US2] Create Azure ML workspace Terraform module in infra/terraform/modules/azureml-workspace/main.tf
- [ ] T040 [P] [US2] Create storage account Terraform module in infra/terraform/modules/storage/main.tf
- [ ] T041 [P] [US2] Create container registry Terraform module in infra/terraform/modules/container-registry/main.tf
- [ ] T042 [P] [US2] Create Key Vault Terraform module (optional) in infra/terraform/modules/azureml-workspace/
- [ ] T043 [US2] Configure workspace dependencies in infra/terraform/modules/azureml-workspace/main.tf
- [ ] T044 [US2] Add RBAC role assignments in infra/terraform/modules/azureml-workspace/rbac.tf
- [ ] T045 [US2] Create deployment validation script in infra/scripts/validate-deployment.sh
- [ ] T046 [US2] Create Jupyter notebook 01-setup-infrastructure.ipynb for infrastructure provisioning
- [ ] T047 [US2] Add error handling for quota limits and permission issues

### Tests for User Story 2

- [ ] T048 [P] [US2] Create Terraform validation test in tests/integration/test_terraform_deploy.py
- [ ] T049 [P] [US2] Test workspace connectivity in tests/integration/test_azureml_workspace.py

**Checkpoint**: Azure ML workspace and supporting resources are provisioned and accessible

---

## Phase 5: User Story 3 - Setup Compute Cluster (Priority: P1) 🎯 MVP

**Goal**: Provision GPU-enabled compute resources for training

**Independent Test**: Create compute cluster, verify it's running and accessible from workspace

### Implementation for User Story 3

- [ ] T050 [P] [US3] Create compute cluster Terraform module in infra/terraform/modules/compute-cluster/main.tf
- [ ] T051 [P] [US3] Add compute cluster variables in infra/terraform/modules/compute-cluster/variables.tf
- [ ] T052 [P] [US3] Create AKS attachment module (optional) in infra/terraform/modules/compute-cluster/aks.tf
- [ ] T053 [US3] Configure auto-scaling policies in infra/terraform/modules/compute-cluster/main.tf
- [ ] T054 [US3] Add quota validation checks in infra/terraform/modules/compute-cluster/
- [ ] T055 [US3] Update notebook 01-setup-infrastructure.ipynb with compute provisioning steps
- [ ] T056 [US3] Add compute status monitoring utilities in src/utils/azure_auth.py

### Tests for User Story 3

- [ ] T057 [P] [US3] Integration test for compute cluster creation in tests/integration/test_compute_cluster.py
- [ ] T058 [P] [US3] Test compute cluster scaling in tests/integration/test_compute_cluster.py

**Checkpoint**: GPU compute cluster is ready for training jobs

---

## Phase 6: User Story 4 - Download and Register Phi-4 Model from AI Foundry (Priority: P2)

**Goal**: Download Phi-4 mini from AI Foundry and register in Azure ML

**Independent Test**: Run download script, verify model files exist and are registered in Azure ML

### Implementation for User Story 4

- [ ] T059 [P] [US4] Create AI Foundry authentication helper in src/training/model_loader.py
- [ ] T060 [P] [US4] Implement model download from AI Foundry catalog in src/training/model_loader.py
- [ ] T061 [US4] Add progress tracking for model download in src/training/model_loader.py
- [ ] T062 [US4] Implement model registration in Azure ML in src/training/model_loader.py
- [ ] T063 [US4] Add retry logic with exponential backoff in src/training/model_loader.py
- [ ] T064 [US4] Create model_config.yaml with AI Foundry model path in configs/model_config.yaml
- [ ] T065 [US4] Create Jupyter notebook 03-download-base-model.ipynb for model download workflow
- [ ] T066 [US4] Add validation to verify model integrity after download

### Tests for User Story 4

- [ ] T067 [P] [US4] Unit test for AI Foundry authentication in tests/unit/test_model_loader.py
- [ ] T068 [P] [US4] Integration test for model download (with mock) in tests/integration/test_model_download.py
- [ ] T069 [P] [US4] Test model registration in Azure ML in tests/integration/test_model_download.py

**Checkpoint**: Phi-4 base model is available locally and registered in Azure ML

---

## Phase 7: User Story 5 - Create Data Loading, Training, and Validation Code (Priority: P2)

**Goal**: Implement core training pipeline code

**Independent Test**: Run training script locally with small dataset, verify checkpoints and metrics

### Implementation for User Story 5

- [ ] T070 [P] [US5] Create PyTorch dataset class in src/data/dataset_loader.py
- [ ] T071 [P] [US5] Implement tokenization and padding logic in src/data/dataset_loader.py
- [ ] T072 [P] [US5] Create training configuration dataclass in src/training/trainer.py
- [ ] T073 [US5] Implement LoRA fine-tuning setup in src/training/trainer.py
- [ ] T074 [US5] Create training loop with gradient accumulation in src/training/trainer.py
- [ ] T075 [US5] Implement validation loop in src/training/trainer.py
- [ ] T076 [US5] Add checkpoint saving logic in src/training/checkpointing.py
- [ ] T077 [US5] Add checkpoint loading/resume logic in src/training/checkpointing.py
- [ ] T078 [US5] Integrate MLflow tracking in src/training/train.py
- [ ] T079 [US5] Create main training script for Azure ML in src/training/train.py
- [ ] T080 [US5] Create training_config.yaml with hyperparameters in configs/training_config.yaml
- [ ] T081 [US5] Add GPU utilization monitoring in src/training/trainer.py

### Tests for User Story 5

- [ ] T082 [P] [US5] Unit test for dataset loader in tests/unit/test_dataset_loader.py
- [ ] T083 [P] [US5] Unit test for tokenization in tests/unit/test_dataset_loader.py
- [ ] T084 [P] [US5] Unit test for training loop in tests/unit/test_trainer.py
- [ ] T085 [P] [US5] Unit test for checkpointing in tests/unit/test_trainer.py
- [ ] T086 [P] [US5] Integration test for local training run in tests/integration/test_training.py

**Checkpoint**: Training code runs successfully locally with sample data

---

## Phase 8: User Story 6 - Run Training Job in Azure ML (Priority: P2)

**Goal**: Submit and execute training job on Azure ML compute cluster

**Independent Test**: Submit job to Azure ML, monitor execution, verify outputs

### Implementation for User Story 6

- [ ] T087 [P] [US6] Create Azure ML environment definition in src/training/train.py
- [ ] T088 [P] [US6] Implement job submission logic in src/training/train.py
- [ ] T089 [US6] Add job monitoring utilities in src/training/train.py
- [ ] T090 [US6] Create experiment tracking setup in src/training/train.py
- [ ] T091 [US6] Add job output retrieval logic in src/training/train.py
- [ ] T092 [US6] Create Jupyter notebook 04-train-model.ipynb for training job workflow
- [ ] T093 [US6] Add error handling for common training failures (OOM, quota)
- [ ] T094 [US6] Implement training progress visualization in notebook

### Tests for User Story 6

- [ ] T095 [P] [US6] Integration test for job submission in tests/integration/test_azureml_job.py
- [ ] T096 [P] [US6] Test job monitoring in tests/integration/test_azureml_job.py

**Checkpoint**: Training job runs successfully on Azure ML and produces trained model

---

## Phase 9: User Story 7 - Download Trained Model Weights and Artifacts (Priority: P2)

**Goal**: Download trained model from Azure ML to local environment

**Independent Test**: Run download with job ID, verify all model files are present locally

### Implementation for User Story 7

- [ ] T097 [P] [US7] Create model download utilities in src/training/model_loader.py
- [ ] T098 [P] [US7] Implement checkpoint selection logic (best/latest/specific) in src/training/model_loader.py
- [ ] T099 [US7] Add download progress tracking in src/training/model_loader.py
- [ ] T100 [US7] Create model artifact validation in src/training/model_loader.py
- [ ] T101 [US7] Implement model registry registration in src/training/model_loader.py
- [ ] T102 [US7] Add model metadata extraction in src/training/model_loader.py
- [ ] T103 [US7] Create script for downloading models (can integrate into notebook)

### Tests for User Story 7

- [ ] T104 [P] [US7] Integration test for model download in tests/integration/test_model_download.py
- [ ] T105 [P] [US7] Test artifact validation in tests/unit/test_model_loader.py

**Checkpoint**: Trained model is downloaded and ready for optimization

---

## Phase 10: User Story 7.5 - Optimize Model for Embedded Deployment (Priority: P2)

**Goal**: Quantize and optimize model for embedded hardware constraints

**Independent Test**: Run optimization pipeline, verify size reduction and accuracy retention

### Implementation for User Story 7.5

- [ ] T106 [P] [US7.5] Create model optimization utilities in src/inference/model_optimizer.py
- [ ] T107 [P] [US7.5] Implement post-training quantization (int8) in src/inference/model_optimizer.py
- [ ] T108 [P] [US7.5] Implement aggressive quantization (int4) in src/inference/model_optimizer.py
- [ ] T109 [US7.5] Create ONNX export pipeline in src/inference/model_optimizer.py
- [ ] T110 [US7.5] Add ONNX optimization (level 3) in src/inference/model_optimizer.py
- [ ] T111 [US7.5] Implement TorchScript compilation option in src/inference/model_optimizer.py
- [ ] T112 [US7.5] Create calibration dataset generator in src/inference/model_optimizer.py
- [ ] T113 [US7.5] Implement accuracy validation after optimization in src/inference/model_optimizer.py
- [ ] T114 [US7.5] Add optimization metrics reporting (size, speedup, accuracy) in src/inference/model_optimizer.py
- [ ] T115 [US7.5] Create optimization_config.yaml in configs/
- [ ] T116 [US7.5] Create Jupyter notebook 05-optimize-model.ipynb for optimization workflow
- [ ] T117 [US7.5] Add memory profiling during inference in src/inference/model_optimizer.py

### Tests for User Story 7.5

- [ ] T118 [P] [US7.5] Unit test for quantization in tests/unit/test_model_optimizer.py
- [ ] T119 [P] [US7.5] Unit test for ONNX export in tests/unit/test_model_optimizer.py
- [ ] T120 [P] [US7.5] Integration test for optimization pipeline in tests/integration/test_optimization.py
- [ ] T121 [P] [US7.5] Test accuracy validation in tests/unit/test_model_optimizer.py

**Checkpoint**: Optimized model is <2GB, maintains >95% accuracy, runs on CPU efficiently

---

## Phase 11: User Story 8 - Create Optimized Scoring Script for Embedded Deployment (Priority: P2)

**Goal**: Create inference script optimized for embedded hardware

**Independent Test**: Load optimized model with scoring script, send test requests on CPU

### Implementation for User Story 8

- [ ] T122 [P] [US8] Create model wrapper for ONNX Runtime in src/inference/model_wrapper.py
- [ ] T123 [P] [US8] Implement efficient model loading (<5s) in src/inference/model_wrapper.py
- [ ] T124 [P] [US8] Create scoring script init() function in src/inference/score.py
- [ ] T125 [US8] Implement scoring script run() function in src/inference/score.py
- [ ] T126 [US8] Add input validation and sanitization in src/inference/score.py
- [ ] T127 [US8] Implement edge case handling in src/inference/score.py
- [ ] T128 [US8] Add memory management and cleanup in src/inference/score.py
- [ ] T129 [US8] Create FastAPI server with minimal dependencies in src/inference/api.py
- [ ] T130 [US8] Implement /health endpoint in src/inference/api.py
- [ ] T131 [US8] Implement /predict endpoint in src/inference/api.py
- [ ] T132 [US8] Add request/response validation with Pydantic in src/inference/api.py
- [ ] T133 [US8] Add lightweight logging for inference in src/inference/api.py

### Tests for User Story 8

- [ ] T134 [P] [US8] Unit test for model wrapper in tests/unit/test_inference.py
- [ ] T135 [P] [US8] Unit test for scoring script in tests/unit/test_inference.py
- [ ] T136 [P] [US8] Test edge case handling in tests/unit/test_inference.py
- [ ] T137 [P] [US8] Integration test for inference API in tests/integration/test_inference_api.py
- [ ] T138 [P] [US8] Performance test for latency (<50ms) in tests/integration/test_inference_api.py
- [ ] T139 [P] [US8] Memory profiling test (<4GB) in tests/integration/test_inference_api.py

**Checkpoint**: Scoring script loads model <5s, infers <50ms P95 on CPU

---

## Phase 12: User Story 9 - Create Highly Optimized Container for Embedded Hardware (Priority: P2)

**Goal**: Build container image <500MB for embedded deployment

**Independent Test**: Build container, run locally on CPU, verify size and startup time

### Implementation for User Story 9

- [ ] T140 [P] [US9] Create multi-stage Dockerfile in docker/Dockerfile with minimal base (Alpine/distroless) and uv for fast package installation
- [ ] T141 [P] [US9] Configure Docker to use uv for dependency installation (COPY pyproject.toml, uv sync --no-dev)
- [ ] T142 [P] [US9] Create .dockerignore in docker/.dockerignore
- [ ] T143 [US9] Implement aggressive layer optimization in docker/Dockerfile
- [ ] T144 [US9] Add ONNX Runtime for CPU in docker/Dockerfile using uv
- [ ] T145 [US9] Remove build tools and unnecessary packages in docker/Dockerfile (uv caches nothing by default)
- [ ] T146 [US9] Create multi-architecture build support (ARM64/x86_64) in docker/Dockerfile
- [ ] T147 [US9] Add health check configuration in docker/Dockerfile
- [ ] T148 [US9] Create container build script in scripts/build_container.sh
- [ ] T149 [US9] Add image size validation (<500MB) in scripts/build_container.sh
- [ ] T150 [US9] Create deployment_config.yaml in configs/deployment_config.yaml
- [ ] T151 [US9] Create Jupyter notebook 06-build-container.ipynb for container build workflow
- [ ] T152 [US9] Add container startup time testing in notebook

### Tests for User Story 9

- [ ] T153 [P] [US9] Container build test in tests/integration/test_container_build.py
- [ ] T154 [P] [US9] Container size validation test in tests/integration/test_container_build.py
- [ ] T155 [P] [US9] Container startup time test (<10s) in tests/integration/test_container_build.py
- [ ] T156 [P] [US9] Container inference smoke test in tests/integration/test_container_build.py
- [ ] T157 [P] [US9] Multi-architecture build test in tests/integration/test_container_build.py

**Checkpoint**: Container <500MB, starts <10s, runs on both ARM64 and x86_64

---

## Phase 13: User Story 10 - Upload Container to Azure Container Registry (Priority: P2)

**Goal**: Push optimized container to ACR for deployment

**Independent Test**: Push image to ACR, verify it appears in registry with correct tags

### Implementation for User Story 10

- [ ] T158 [P] [US10] Create ACR authentication utilities in src/utils/azure_auth.py
- [ ] T159 [P] [US10] Implement container tagging logic in scripts/push_to_acr.sh
- [ ] T160 [US10] Create ACR push script in scripts/push_to_acr.sh
- [ ] T161 [US10] Add multi-architecture manifest creation in scripts/push_to_acr.sh
- [ ] T162 [US10] Implement push progress tracking in scripts/push_to_acr.sh
- [ ] T163 [US10] Add vulnerability scanning integration (Defender for Containers) in scripts/push_to_acr.sh
- [ ] T164 [US10] Create Jupyter notebook 07-push-to-acr.ipynb for ACR upload workflow
- [ ] T165 [US10] Add image verification after push in notebook

### Tests for User Story 10

- [ ] T166 [P] [US10] Integration test for ACR push in tests/integration/test_acr_push.py
- [ ] T167 [P] [US10] Test multi-architecture manifest in tests/integration/test_acr_push.py
- [ ] T168 [P] [US10] Test image pull from ACR in tests/integration/test_acr_push.py

**Checkpoint**: Container is in ACR and pullable from Azure services and local/embedded devices

---

## Phase 13.5: User Story 11 - Deploy Container Locally and to Embedded Hardware (Priority: P2)

**Goal**: Deploy and run containerized model on local systems and embedded Linux devices

**Independent Test**: Deploy locally with Docker, test inference, then deploy to embedded device (or VM simulating embedded)

### Implementation for User Story 11

- [ ] T169 [P] [US11] Create local deployment script in scripts/deploy_local.sh for Docker run
- [ ] T170 [P] [US11] Create Docker Compose configuration in docker/docker-compose.yml for local testing
- [ ] T171 [P] [US11] Implement health check and readiness validation in scripts/deploy_local.sh
- [ ] T172 [US11] Create embedded deployment script in scripts/deploy_embedded.sh with ACR pull
- [ ] T173 [US11] Generate systemd service file template in configs/slm-inference.service
- [ ] T174 [US11] Implement automatic restart configuration for systemd service
- [ ] T175 [US11] Create rollback script in scripts/rollback_deployment.sh for version switching
- [ ] T176 [US11] Implement update script in scripts/update_deployment.sh for graceful updates
- [ ] T177 [US11] Add deployment health monitoring script in scripts/check_deployment_health.sh
- [ ] T178 [US11] Create deployment configuration guide in docs/deployment.md
- [ ] T179 [US11] Create Jupyter notebook 08-deploy-inference.ipynb for deployment workflow
- [ ] T180 [US11] Add local and embedded deployment examples in notebook

### Tests for User Story 11

- [ ] T181 [P] [US11] Local deployment test in tests/integration/test_local_deployment.py
- [ ] T182 [P] [US11] Docker Compose deployment test in tests/integration/test_local_deployment.py
- [ ] T183 [P] [US11] Embedded deployment simulation test in tests/integration/test_embedded_deployment.py
- [ ] T184 [P] [US11] Systemd service configuration test in tests/integration/test_embedded_deployment.py
- [ ] T185 [P] [US11] Deployment rollback test in tests/integration/test_embedded_deployment.py
- [ ] T186 [P] [US11] Health monitoring test in tests/integration/test_embedded_deployment.py

**Checkpoint**: Container deployed and running on both local and embedded systems with automated management

---

## Phase 14: Evaluation & Documentation (Cross-Cutting)

**Purpose**: Model evaluation, documentation, and deployment guides

### Evaluation Implementation

- [ ] T187 [P] Create evaluation metrics module in src/evaluation/metrics.py
- [ ] T188 [P] Implement evaluation script in src/evaluation/evaluate.py
- [ ] T189 Create evaluation notebook for both original and optimized models
- [ ] T190 Add evaluation to CI/CD pipeline

### Documentation

- [ ] T191 [P] Complete README.md with architecture diagram and full setup
- [ ] T192 [P] Document all configuration options in configs/README.md
- [ ] T193 [P] Create troubleshooting guide in docs/troubleshooting.md
- [ ] T194 [P] Document embedded deployment process in docs/embedded-deployment.md
- [ ] T195 [P] Add API documentation for inference endpoints in docs/api.md
- [ ] T196 [P] Create quickstart guide matching notebooks in docs/quickstart.md
- [ ] T197 Update all Jupyter notebooks with clear markdown explanations
- [ ] T198 [P] Add inline docstrings to all public functions

### Deployment and Polish

- [ ] T199 Create cleanup script in scripts/cleanup_resources.py
- [ ] T200 [P] Setup CI/CD pipeline in .github/workflows/ci.yml
- [ ] T201 [P] Setup CD pipeline in .github/workflows/cd.yml
- [ ] T202 Add performance benchmarking script
- [ ] T203 Create embedded hardware testing guide

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-13)**: All depend on Foundational phase completion
  - US1, US2, US3 are P1 and should be completed first (can be parallel if staffed)
  - US4-US10 are P2 and can proceed after P1 stories complete
- **Evaluation & Documentation (Phase 14)**: Depends on all core user stories being complete

### User Story Dependencies

- **US1 (Data)**: Can start after Foundational - No dependencies on other stories
- **US2 (Infrastructure)**: Can start after Foundational - No dependencies on other stories
- **US3 (Compute)**: Depends on US2 (needs workspace)
- **US4 (Model Download)**: Can start after Foundational - No dependencies
- **US5 (Training Code)**: Can start after Foundational - No hard dependencies (can use mock data/model)
- **US6 (Training Job)**: Depends on US1, US2, US3, US4, US5
- **US7 (Download Model)**: Depends on US6
- **US7.5 (Optimize)**: Depends on US7
- **US8 (Scoring Script)**: Depends on US7.5
- **US9 (Container)**: Depends on US7.5, US8
- **US10 (ACR Push)**: Depends on US2 (needs ACR), US9
- **US11 (Deployment)**: Depends on US9 (for local), US10 (for embedded with ACR)

### Critical Path

The fastest path to a working system:

1. Phase 1: Setup (T001-T010)
2. Phase 2: Foundational (T011-T025)
3. Phase 4: US2 Infrastructure (T039-T049) - Creates workspace and ACR
4. Phase 5: US3 Compute (T050-T058) - Creates compute cluster
5. Phase 3: US1 Data (T026-T038) - Prepares training data
6. Phase 6: US4 Model (T059-T069) - Downloads Phi-4
7. Phase 7: US5 Training Code (T070-T086) - Implements training
8. Phase 8: US6 Run Training (T087-T096) - Trains model
9. Phase 9: US7 Download (T097-T105) - Gets trained model
10. Phase 10: US7.5 Optimize (T106-T121) - Optimizes for embedded
11. Phase 11: US8 Scoring (T122-T139) - Creates inference script
12. Phase 12: US9 Container (T140-T157) - Builds optimized container
13. Phase 13: US10 ACR (T158-T168) - Pushes to registry

### Parallel Opportunities

**After Foundational Phase completes, these can run in parallel:**

- Team A: US1 (Data prep)
- Team B: US2 (Infrastructure) → US3 (Compute)
- Team C: US4 (Model download)
- Team D: US5 (Training code)

**After US6 (Training) completes:**

- Team A: US7 (Download) → US7.5 (Optimize)
- Team B: US8 (Scoring script)
- Team C: US9 (Container setup)

**Within each user story**, tasks marked [P] can run in parallel

---

## Implementation Strategy

### MVP First (P1 Stories Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete US2: Infrastructure
4. Complete US3: Compute
5. Complete US1: Data
6. **STOP and VALIDATE**: Can upload data to Azure ML
7. This is a minimal but functional starting point

### Incremental Delivery

1. **Foundation** → Setup + Foundational complete
2. **Infrastructure** → Add US2 + US3 → Can provision Azure resources
3. **Data Pipeline** → Add US1 → Can prepare and upload data
4. **Model Access** → Add US4 → Can download Phi-4 from AI Foundry
5. **Training** → Add US5 + US6 → Can train models
6. **Deployment** → Add US7 + US7.5 + US8 + US9 + US10 → Can deploy optimized model

### Testing Strategy

- Write tests (marked with ⚠️) FIRST before implementation
- Ensure tests FAIL before implementing features
- Run tests after each task completion
- Achieve 80% code coverage target
- Run integration tests against live Azure resources in staging

---

## Notes

- [P] tasks work on different files and can run in parallel
- [Story] label maps task to specific user story
- Each user story should be independently completable and testable
- Commit after each task or logical group of tasks
- **Use uv for all Python package management** (faster than pip, better than venv)
- Target: <500MB container, <50ms P95 latency, <10s startup
- All notebooks should execute successfully in sequence
- Maintain 80% code coverage throughout development
- Use constitution.md principles for code quality standards
