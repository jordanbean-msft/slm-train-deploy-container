# Feature Specification: SLM Training and Deployment System

**Feature Branch**: `001-slm-train-deploy-system`
**Created**: November 20, 2025
**Status**: Draft
**Input**: Build an example system for training a small language model (SLM) using training data, then deploying this trained SLM as a container

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Setup and Upload Training Data (Priority: P1)

As a data scientist, I want to prepare my training dataset and upload it to Azure ML datastore so that it's accessible for training jobs in the cloud.

**Why this priority**: Foundation for the entire workflow. Without properly configured training data in Azure ML, no subsequent steps can proceed. This is the critical first step that enables all other functionality.

**Independent Test**: Can be fully tested by preparing a local dataset, running the upload script, and verifying the data appears in Azure ML datastore with correct structure. Delivers a reusable data asset.

**Acceptance Scenarios**:

1. **Given** raw training data in JSONL format with prompt-completion pairs, **When** I run the data preparation script, **Then** the system validates format, creates train/validation splits, and generates a data manifest file
2. **Given** prepared data files locally, **When** I execute the upload command with Azure ML workspace credentials, **Then** the data is uploaded to Azure ML datastore and registered as a versioned dataset
3. **Given** data upload is in progress, **When** I monitor the process, **Then** I see upload progress, file counts, and total size being transferred
4. **Given** data is uploaded successfully, **When** I check Azure ML Studio, **Then** I see the registered dataset with metadata (row count, file size, schema, creation timestamp)
5. **Given** data validation fails, **When** errors are detected, **Then** the system logs specific issues (missing fields, encoding errors, schema mismatches) with line numbers and examples

---

### User Story 2 - Setup Azure ML Workspace and Resources (Priority: P1)

As a ML engineer, I want to provision and configure an Azure ML workspace with necessary resources within an existing resource group using Azure Developer CLI (azd) so that I have a complete environment for model training with streamlined deployment workflow.

**Why this priority**: Essential infrastructure that must exist before any training can occur. Without the workspace, no Azure ML operations are possible. This establishes the foundation for all subsequent work. Using azd provides a unified deployment experience with environment management.

**Independent Test**: Can be tested by running `azd up` with an existing resource group and verifying Azure ML workspace exists with correct configuration. Delivers a working Azure ML environment with azd integration.

**Acceptance Scenarios**:

1. **Given** Azure subscription credentials, existing resource group name, and azd configuration (azure.yaml), **When** I run `azd up` or `azd provision`, **Then** the system deploys Terraform infrastructure creating Azure ML workspace, storage account, and container registry within the existing resource group
2. **Given** azd deployment is complete, **When** I verify the resources, **Then** I see all components properly linked (workspace connected to storage and container registry) and azd stores deployment state
3. **Given** I need to configure authentication, **When** I run azd commands, **Then** the system uses Azure CLI authentication and the provided resource group without attempting to create it
4. **Given** workspace is provisioned via azd, **When** I access Azure ML Studio, **Then** I can navigate the workspace and see all configured resources
5. **Given** I want to use different environments, **When** I run `azd env new <env-name>` and `azd up`, **Then** azd manages environment-specific configurations and deploys to the specified environment
6. **Given** deployment fails due to missing resource group, invalid names, or permissions, **When** the error occurs, **Then** azd and Terraform provide specific error messages with remediation steps (create resource group, check naming constraints, required roles)
7. **Given** I want to tear down resources, **When** I run `azd down`, **Then** azd destroys all provisioned infrastructure while preserving the resource group

---

### User Story 3 - Setup Compute Cluster (Azure ML or Attached AKS) (Priority: P1)

As a ML engineer, I want to provision GPU-enabled compute resources (Azure ML compute cluster or attached AKS) so that I can run training jobs with appropriate hardware acceleration.

**Why this priority**: Training requires compute resources. Without properly configured GPU clusters, training cannot proceed or will be prohibitively slow. This is a blocker for the core training workflow.

**Independent Test**: Can be tested by running compute provisioning script and verifying the cluster is created, running, and accessible from Azure ML workspace. Delivers training-ready compute infrastructure.

**Acceptance Scenarios**:

1. **Given** Azure ML workspace and compute configuration (VM size, min/max nodes), **When** I run the compute creation script, **Then** the system provisions an Azure ML compute cluster with specified GPU VM SKU
2. **Given** I prefer using existing AKS cluster, **When** I provide AKS cluster details and run attach script, **Then** the system attaches the AKS cluster to Azure ML workspace as compute target
3. **Given** compute cluster is provisioned, **When** I check the cluster status, **Then** I see the cluster state (running/stopped), node count, and available GPU resources
4. **Given** compute cluster is idle, **When** scaling is configured to scale-to-zero, **Then** the cluster automatically scales down to save costs and scales up when jobs are submitted
5. **Given** GPU quota is insufficient, **When** provisioning fails, **Then** the system provides specific quota information (current/required) and instructions to request increase

---

### User Story 4 - Download and Register Phi-4 Model from AI Foundry (Priority: P2)

As a data scientist, I want to download the Phi-4 mini model from Azure AI Foundry model catalog and register it in Azure ML so that I can use it as the base model for fine-tuning.

**Why this priority**: While critical for the training process, this can be done independently and doesn't block infrastructure setup. The model download is a prerequisite for training but can be prepared in parallel.

**Independent Test**: Can be tested by running the AI Foundry model download script, verifying the model files are downloaded and registered in Azure ML model registry. Delivers a ready-to-use base model.

**Acceptance Scenarios**:

1. **Given** Azure AI Foundry workspace connection and Phi-4 model path, **When** I run the model download script, **Then** the system downloads model weights, tokenizer, and configuration files from AI Foundry catalog
2. **Given** model files are downloaded from AI Foundry, **When** I run the registration script, **Then** the system registers the model in Azure ML with version, tags, and AI Foundry source metadata
3. **Given** download is in progress, **When** I monitor the process, **Then** I see download progress, file sizes, and estimated time remaining
4. **Given** model is registered in Azure ML, **When** I view the model registry, **Then** I see the model with details (size, framework, base architecture, AI Foundry source path)
5. **Given** download fails due to authentication or network issues, **When** the error occurs, **Then** the system provides clear error messages about workspace permissions and retries with exponential backoff

---

### User Story 5 - Create Data Loading, Training, and Validation Code (Priority: P2)

As a data scientist, I want to implement the training pipeline code (data loading, training loop, validation) so that I can customize the fine-tuning process for my specific requirements.

**Why this priority**: This is the core ML engineering work that defines how training happens. While it can be developed independently, it's required before training jobs can run successfully.

**Independent Test**: Can be tested by running the training script locally with a small dataset sample and verifying it completes without errors. Delivers validated training code ready for cloud execution.

**Acceptance Scenarios**:

1. **Given** Azure ML dataset reference and model path, **When** I implement the data loader, **Then** the code efficiently streams data in batches with proper tokenization and padding
2. **Given** training configuration (learning rate, epochs, batch size), **When** I implement the training loop, **Then** the code executes training with gradient accumulation, checkpointing, and logging
3. **Given** validation dataset, **When** I implement validation logic, **Then** the code computes metrics (loss, perplexity, accuracy) at configured intervals
4. **Given** training code is complete, **When** I test locally with small dataset, **Then** training runs successfully for several steps with metrics logged and checkpoints saved
5. **Given** I want to log metrics to Azure ML, **When** I integrate MLflow tracking, **Then** the code logs metrics, parameters, and artifacts to Azure ML experiment runs

---

### User Story 6 - Run Training Job in Azure ML (Priority: P2)

As a data scientist, I want to submit and run a training job on Azure ML compute cluster so that I can fine-tune the model with my custom dataset at scale.

**Why this priority**: This is the execution phase where all previous setup comes together. It's the primary value delivery point but depends on all previous infrastructure and code being ready.

**Independent Test**: Can be tested by submitting a training job, monitoring its execution, and verifying it completes with trained model artifacts. Delivers the fine-tuned model.

**Acceptance Scenarios**:

1. **Given** training script, compute target, and dataset references, **When** I submit the training job via Azure ML SDK/CLI, **Then** the job is queued and starts running on the compute cluster
2. **Given** training job is running, **When** I monitor the job in Azure ML Studio, **Then** I see real-time logs, metrics (loss, learning rate), and resource utilization (GPU %)
3. **Given** training progresses through epochs, **When** I check the metrics, **Then** I see training and validation curves updating in real-time with checkpoints saved periodically
4. **Given** training completes successfully, **When** I check the job outputs, **Then** I find trained model checkpoints, final metrics, and training logs in the job workspace
5. **Given** training job fails due to OOM or code error, **When** the failure occurs, **Then** the system captures error logs, saves last checkpoint, and provides debugging information

---

### User Story 7 - Download Trained Model Weights and Artifacts (Priority: P2)

As a ML engineer, I want to download the trained model weights and associated artifacts from Azure ML so that I can prepare them for containerization and deployment.

**Why this priority**: Bridge between training and deployment phases. Required to move from Azure ML environment to containerized deployment. Critical for operationalization.

**Independent Test**: Can be tested by running download script with job ID and verifying model files are downloaded locally with correct structure. Delivers deployment-ready model artifacts.

**Acceptance Scenarios**:

1. **Given** completed training job ID, **When** I run the model download script, **Then** the system downloads model weights, tokenizer config, and training metadata to local directory
2. **Given** multiple checkpoints exist, **When** I specify which checkpoint to download (best, latest, specific epoch), **Then** the system downloads the requested checkpoint
3. **Given** download is in progress, **When** I monitor the process, **Then** I see download progress for each file and overall completion percentage
4. **Given** model is downloaded, **When** I verify the artifacts, **Then** I find all required files (model weights, tokenizer, config.json, training_args.json)
5. **Given** I want to register the final model, **When** I run the registration script, **Then** the model is registered in Azure ML model registry with version and metadata for traceability

---

### User Story 8 - Create Optimized Scoring Script for Embedded Deployment (Priority: P2)

As a ML engineer, I want to create a highly optimized scoring script that loads the quantized model efficiently and handles inference requests with minimal latency and memory footprint so that the model can run on resource-constrained embedded hardware.

**Why this priority**: Required for deployment but can be developed in parallel with training. The inference interface must be optimized for embedded constraints (CPU-only, limited memory, fast startup).

**Independent Test**: Can be tested locally by loading the optimized model (ONNX/quantized) with the scoring script and sending test inference requests on CPU. Delivers a validated, embedded-ready inference interface.

**Acceptance Scenarios**:

1. **Given** optimized model artifacts (ONNX/quantized), **When** I implement the scoring script init() function, **Then** the code loads the model and tokenizer into memory in <5 seconds with <4GB RAM usage
2. **Given** model is loaded, **When** I implement the run() function, **Then** the code accepts JSON input, performs CPU-based inference, and returns predictions in <50ms P95
3. **Given** scoring script is complete, **When** I test locally on CPU with sample inputs, **Then** the script returns predictions within embedded latency requirements (<50ms) without GPU
4. **Given** various input formats, **When** I send test requests, **Then** the script handles edge cases (empty input, very long text, special characters) gracefully with minimal memory allocation
5. **Given** the embedded hardware has limited resources, **When** I profile memory usage during inference, **Then** the peak memory stays under 4GB and releases unused memory promptly

---

### User Story 9 - Create Highly Optimized Container for Embedded Hardware (Priority: P2)

As a DevOps engineer, I want to build an aggressively optimized Docker container (<500MB) that includes the quantized model and scoring script so that I can deploy the model to resource-constrained embedded devices.

**Why this priority**: Containerization is the final step to create a deployable artifact for embedded hardware. The container must be extremely small and fast-starting due to embedded constraints. This is the critical packaging phase.

**Independent Test**: Can be tested by building the container image, verifying size <500MB, and running it locally on CPU to validate embedded-like conditions. Delivers an embedded-ready container image.

**Acceptance Scenarios**:

1. **Given** optimized model artifacts (ONNX/quantized) and scoring script, **When** I run the container build script with multi-stage Dockerfile, **Then** the system builds a Docker image <500MB with minimal dependencies
2. **Given** container image is built, **When** I run the container on CPU (simulating embedded hardware), **Then** it starts within 10 seconds, loads the optimized model in <5 seconds, and exposes the inference API
3. **Given** container is running, **When** I send health check request to /health endpoint, **Then** I receive 200 OK response indicating model is loaded with minimal overhead
4. **Given** container is running on CPU, **When** I send inference request to /score endpoint with JSON payload, **Then** I receive predictions with latency under 50ms on CPU-only hardware
5. **Given** I want multi-architecture support, **When** I build for both ARM64 and x86_64, **Then** the system creates platform-specific images both under 500MB that run on embedded devices
6. **Given** embedded hardware has limited resources, **When** container runs, **Then** memory usage stays under 4GB during inference and container can be stopped/started quickly

---

### User Story 10 - Upload Container to Azure Container Registry (Priority: P2)

As a DevOps engineer, I want to push the built container image to Azure Container Registry so that it's available for deployment to Azure services and can be pulled for local/embedded deployments.

**Why this priority**: Makes the container available in a central registry for deployment. While important, it's not the final step as we need to actually deploy the container.

**Independent Test**: Can be tested by tagging and pushing the image to ACR, then verifying it appears in the registry. Delivers a centrally hosted, deployment-ready container.

**Acceptance Scenarios**:

1. **Given** built container image and ACR credentials, **When** I run the ACR login and push commands, **Then** the image is authenticated and uploaded to Azure Container Registry
2. **Given** image is being pushed, **When** I monitor the upload, **Then** I see layer-by-layer upload progress with sizes and completion status
3. **Given** image is pushed successfully, **When** I check ACR in Azure Portal, **Then** I see the image with proper tags (version, latest) and manifest details
4. **Given** I want to scan for vulnerabilities, **When** I enable Defender for Containers, **Then** the image is automatically scanned and security findings are reported
5. **Given** I need to deploy from ACR, **When** I configure pull permissions, **Then** I can pull the image using managed identity or credentials for local/embedded deployment

---

### User Story 11 - Deploy Container Locally and to Embedded Hardware (Priority: P2)

As a DevOps engineer, I want to deploy the containerized model both locally for testing and to embedded hardware for production so that I can validate functionality before edge deployment and run inference on resource-constrained devices.

**Why this priority**: Final step that delivers the working inference service. Without deployment, the container remains unused. This completes the end-to-end workflow and enables actual inference.

**Independent Test**: Can be tested by deploying container locally with Docker, running inference requests, then deploying to embedded device (or VM simulating embedded) and validating same functionality. Delivers a production-ready deployed service.

**Acceptance Scenarios**:

1. **Given** container image in ACR (or local), **When** I run the local deployment script with Docker, **Then** the container starts successfully, exposes ports, and responds to health checks
2. **Given** container is running locally, **When** I send test inference requests, **Then** I receive predictions with latency <50ms and can validate model behavior before edge deployment
3. **Given** I want to test with Docker Compose, **When** I run docker-compose up with the provided configuration, **Then** the service starts with proper networking, volumes, and environment variables configured
4. **Given** embedded Linux device with Docker/containerd installed, **When** I run the embedded deployment script, **Then** the script pulls the image from ACR, deploys the container, and configures it to start on boot
5. **Given** container is deployed on embedded hardware, **When** I configure it as a systemd service, **Then** the service starts automatically on boot, restarts on failure, and logs to system journal
6. **Given** container is running on embedded device, **When** I send inference requests, **Then** I receive predictions with latency <50ms and memory usage <4GB
7. **Given** I need to update the model, **When** I run the update script, **Then** the system pulls the new image, gracefully stops the old container, and starts the new version with minimal downtime
8. **Given** deployment fails on embedded device, **When** the error occurs, **Then** the system provides clear logs, rolls back to previous version if configured, and alerts the operator

---

### Edge Cases

- **What happens when training data is corrupted or improperly formatted?**

  - System validates data format before training starts
  - Logs specific errors (line numbers, format issues) with examples
  - Exits gracefully without consuming resources

- **How does system handle GPU out-of-memory errors during training?**

  - Catches OOM exceptions and logs memory usage statistics
  - Suggests reducing batch size or model size in error message
  - Saves checkpoint before terminating to allow configuration adjustment

- **What happens when container initialization fails due to missing model files?**

  - Health check fails and container doesn't become ready
  - Logs clear error about missing files with expected path
  - Container exits with specific error code for orchestration systems

- **How does system handle inference requests when model is still loading?**

  - Returns 503 Service Unavailable with Retry-After header
  - Provides loading progress in response body
  - Queues requests if within buffer limit, rejects if queue full

- **What happens when validation dataset has different schema than training data?**

  - Validates schema compatibility before evaluation
  - Logs schema mismatches with specific field differences
  - Exits with clear error and schema requirements

- **How does system handle extremely large training datasets that don't fit in memory?**

  - Implements streaming data loader with configurable buffer size
  - Loads and processes data in chunks
  - Monitors memory usage and adjusts buffer if approaching limits

- **What happens when container receives malformed inference requests?**

  - Validates request schema and returns 400 Bad Request
  - Provides detailed error message indicating required fields
  - Logs malformed requests for security monitoring

- **How does system handle concurrent training jobs on same GPU?**
  - Checks GPU availability before starting training
  - Returns clear error if GPU already in use
  - Optionally queues job if queue system is enabled

## Requirements _(mandatory)_

### Functional Requirements

#### Training Requirements

- **FR-001**: System MUST accept training data in JSONL format with fields: "prompt" (input text) and "completion" (target output)
- **FR-002**: System MUST support fine-tuning of pre-trained language models (e.g., GPT-2, LLaMA-based models, Phi models)
- **FR-003**: System MUST implement checkpointing every N steps (configurable, default 500 steps) to enable training resumption
- **FR-004**: System MUST log training metrics including loss, learning rate, gradient norm, and throughput to structured log files
- **FR-005**: System MUST validate training data before starting training and report errors for invalid format, missing fields, or corrupted data
- **FR-006**: System MUST support GPU acceleration for training with automatic fallback to CPU if GPU unavailable
- **FR-007**: System MUST allow configuration of hyperparameters: learning rate, batch size, number of epochs, warmup steps, weight decay
- **FR-008**: System MUST implement early stopping based on validation loss with configurable patience parameter
- **FR-009**: System MUST save final model in standardized format (e.g., Hugging Face format) with tokenizer configuration
- **FR-010**: System MUST compute and log training/validation split statistics (sample counts, token distributions)

#### Deployment Requirements

- **FR-011**: System MUST build Docker containers with multi-stage builds to minimize final image size (<500MB target for embedded)
- **FR-012**: System MUST support multi-architecture builds (ARM64 and x86_64) for diverse embedded hardware
- **FR-013**: System MUST use minimal base images (Alpine or distroless) to reduce attack surface and size
- **FR-014**: System MUST include health check endpoint (/health) that returns 200 when model is loaded and ready with minimal overhead
- **FR-015**: System MUST expose REST API endpoint (/predict) accepting POST requests with JSON payload containing "text" field
- **FR-016**: System MUST return inference results as JSON with fields: "prediction" (model output), "confidence" (optional), "latency_ms"
- **FR-017**: System MUST load optimized model (ONNX/quantized) during container initialization in <5 seconds
- **FR-018**: System MUST cache model in memory with peak usage <4GB for embedded hardware compatibility
- **FR-019**: System MUST optimize for single-request efficiency (not batching) as embedded use cases are typically low concurrency
- **FR-020**: System MUST implement request timeout (configurable, default 10 seconds) appropriate for embedded latency expectations
- **FR-021**: System MUST log inference requests efficiently without impacting latency
- **FR-022**: Container MUST accept environment variables for configuration: MODEL_PATH, MODEL_FORMAT, INFERENCE_TIMEOUT, PORT, QUANTIZATION_LEVEL
- **FR-023**: System MUST validate inference requests and return 400 Bad Request with descriptive error for invalid inputs
- **FR-024**: Container startup time MUST be <10 seconds from cold start on embedded hardware

#### Model Optimization Requirements

- **FR-025**: System MUST support post-training quantization with int8 and int4 precision options
- **FR-026**: System MUST convert trained models to ONNX format with optimization level 3
- **FR-027**: System MUST validate that quantized models retain >95% of original accuracy on validation set
- **FR-028**: System MUST support TorchScript compilation as alternative to ONNX for edge cases
- **FR-029**: System MUST generate optimization reports showing: original size, optimized size, compression ratio, speedup factor, accuracy delta
- **FR-030**: System MUST profile optimized model memory usage and ensure <4GB peak during inference
- **FR-031**: System MUST use static quantization with calibration dataset for better accuracy retention than dynamic quantization
- **FR-032**: System MUST optimize ONNX models for CPU inference with target device hints
- **FR-033**: System MUST validate ONNX model outputs match PyTorch outputs within tolerance threshold

#### Evaluation Requirements

- **FR-034**: System MUST compute standard metrics: perplexity, accuracy, precision, recall, F1 score for classification tasks
- **FR-035**: System MUST generate evaluation report in JSON format with all computed metrics and sample predictions
- **FR-036**: System MUST support configurable quality thresholds that fail evaluation if not met (exit code non-zero)
- **FR-037**: System MUST produce confusion matrix for multi-class classification evaluation tasks
- **FR-038**: System MUST log evaluation progress and estimated time remaining during metric computation
- **FR-039**: System MUST evaluate both original and optimized models to measure optimization impact

#### Monitoring Requirements

- **FR-040**: System MUST emit structured logs in JSON format with fields: timestamp, level, message, component, correlation_id
- **FR-041**: System MUST log resource utilization metrics: GPU memory usage, GPU utilization %, CPU usage, RAM usage
- **FR-042**: System MUST provide Prometheus-compatible metrics endpoint (/metrics) exposing training and inference metrics
- **FR-043**: System MUST include distributed tracing support with OpenTelemetry for request tracing across components
- **FR-044**: System MUST monitor and log container resource usage on embedded hardware for optimization feedback

#### Data Processing Requirements

- **FR-045**: System MUST support preprocessing of raw text files into training format with tokenization
- **FR-046**: System MUST implement automatic train/validation split with configurable ratio (default 80/20)
- **FR-047**: System MUST detect and report data quality issues: duplicates, outliers, missing values, encoding errors
- **FR-048**: System MUST support multiple input formats: CSV, JSON, JSONL, plain text with configurable parsing rules

#### Infrastructure Requirements

- **FR-049**: System MUST provide Infrastructure as Code (Terraform) for Azure ML workspace and supporting resources that operates within an existing Azure resource group. Resource naming MUST be managed through the terraform-azurerm-naming module to ensure consistent naming conventions across all Azure resources. The naming module MUST be invoked within each Terraform module (storage, container-registry, azureml-workspace, compute-cluster) with a suffix parameter (typically the environment name). Identity management MUST use a user-assigned managed identity shared across Azure ML workspace and compute cluster resources. The user-assigned managed identity MUST be granted the following RBAC permissions: Storage Blob Data Contributor on the storage account, AcrPull on the container registry. All infrastructure configuration values MUST be explicitly provided either in environment-specific tfvars files (for direct Terraform usage) or via TF*VAR* prefixed environment variables (for azd usage) with no defaults. Terraform modules MUST be organized with separate files for resource definitions (main.tf), variable declarations (variables.tf), outputs (outputs.tf), and provider version constraints (versions.tf). Infrastructure deployment MUST be managed through Azure Developer CLI (azd) with azure.yaml configuration for environment management, with azd passing variables to Terraform via TF*VAR* environment variables.
- **FR-050**: System MUST support embedded hardware deployment with resource constraints documentation
- **FR-051**: System MUST include CI/CD pipeline configuration for automated testing, optimization, and deployment
- **FR-052**: System MUST provide Jupyter notebook interface for all operations: train, evaluate, optimize, deploy, containerize

### Key Entities

- **TrainingDataset**: Represents collection of training examples with fields: samples (list), source_file (path), split_ratio (float), total_tokens (int), created_at (timestamp)

- **ModelCheckpoint**: Represents saved model state with fields: checkpoint_path (path), epoch (int), step (int), training_loss (float), validation_loss (float), timestamp (datetime)

- **TrainingConfiguration**: Encapsulates training hyperparameters with fields: learning_rate (float), batch_size (int), num_epochs (int), warmup_steps (int), gradient_accumulation_steps (int), max_sequence_length (int)

- **TrainedModel**: Represents the final trained model artifact with fields: model_path (path), base_model_name (string), base_model_source (string - AI Foundry path), training_dataset_id (string), final_metrics (dict), created_at (timestamp)

- **OptimizedModel**: Represents optimized model for embedded deployment with fields: original_model_path (path), optimized_model_path (path), format (string - "onnx", "torchscript", "pytorch"), quantization_level (string - "int8", "int4", "fp16"), original_size_mb (float), optimized_size_mb (float), compression_ratio (float), accuracy_retention (float), inference_speedup (float), target_architecture (string - "arm64", "amd64")

- **InferenceRequest**: Represents incoming prediction request with fields: request_id (uuid), text (string), parameters (dict with temperature, max_tokens, etc.), timestamp (datetime)

- **InferenceResponse**: Represents model prediction output with fields: request_id (uuid), prediction (string), confidence_score (float), latency_ms (int), model_version (string)

- **EvaluationMetrics**: Represents model evaluation results with fields: perplexity (float), accuracy (float), f1_score (float), precision (float), recall (float), confusion_matrix (array), sample_predictions (list)

- **ContainerImage**: Represents built Docker image with fields: image_tag (string), image_size_mb (float), model_version (string), build_timestamp (datetime), base_image (string)

- **DeploymentConfiguration**: Represents container deployment settings with fields: resource_group (string), container_name (string), cpu_cores (int), memory_gb (int), replica_count (int), environment_variables (dict)

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Users can successfully train a small language model (up to 1B parameters) on a dataset of 10,000 samples in under 2 hours on a single GPU
- **SC-002**: Training checkpoints are saved every 500 steps and users can resume training from any checkpoint with less than 1% metric divergence
- **SC-003**: Model optimization reduces size by >50% with <2% accuracy degradation from original model
- **SC-004**: Optimized model (ONNX int8) achieves P95 latency under 50ms for single inference requests on CPU-only hardware (simulating embedded)
- **SC-005**: Container images are built successfully in under 10 minutes and are smaller than 500MB for embedded deployment
- **SC-006**: Container images support both ARM64 and x86_64 architectures with size <500MB for each
- **SC-007**: Container startup time from cold start to ready state is under 10 seconds including optimized model loading
- **SC-008**: Peak memory usage during inference stays under 4GB to fit embedded hardware constraints
- **SC-009**: AI Foundry model download completes successfully and model is registered in Azure ML with proper metadata
- **SC-010**: System achieves 100% success rate for valid training data formats and provides actionable error messages for 100% of format errors
- **SC-011**: GPU utilization during training exceeds 80% for batch-based training workloads indicating efficient resource usage
- **SC-012**: Evaluation metrics are computed accurately with less than 0.1% error compared to reference implementations (scikit-learn)
- **SC-013**: 90% of users successfully complete end-to-end workflow (data prep → training → optimization → deployment) on first attempt without external support
- **SC-014**: All Jupyter notebooks provide clear documentation and execute successfully in sequence
- **SC-015**: System logs sufficient information that 95% of common errors can be diagnosed without additional debugging tools
- **SC-016**: Azure deployment automation successfully provisions and deploys containers with 100% success rate when credentials are valid
- **SC-017**: Optimized container runs successfully on both ARM64 and x86_64 embedded hardware without architecture-specific issues
- **SC-018**: System memory usage stays within allocated container limits (<4GB) during inference with no OOM errors for requests within specification
