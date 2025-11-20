# Infrastructure as Code

This directory contains Terraform configuration for provisioning Azure infrastructure required for SLM training and deployment.

## Architecture

The infrastructure is organized into modular Terraform components:

```
infra/
├── terraform/
│   ├── main.tf              # Root module orchestrating all resources
│   ├── variables.tf         # Input variable definitions
│   ├── outputs.tf           # Output value definitions
│   ├── environments/
│   │   ├── dev.tfvars       # Development environment configuration
│   │   └── prod.tfvars      # Production environment configuration
│   └── modules/
│       ├── storage/         # Azure Storage Account + blob container
│       ├── container-registry/  # Azure Container Registry (ACR)
│       ├── azureml-workspace/   # Azure ML workspace with identity
│       └── compute-cluster/     # GPU compute cluster
└── scripts/
    ├── deploy.sh               # Automated deployment script
    ├── destroy.sh              # Infrastructure teardown script
    └── validate-deployment.sh  # Post-deployment validation
```

## Azure Resources

### Resource Group

- **Purpose**: Logical container for all resources
- **Naming**: `rg-slm-train-{environment}`
- **Location**: Configurable (default: `eastus`)

### Storage Account

- **Purpose**: Store training data, models, logs
- **SKU**: Basic (dev), Standard (prod)
- **Features**: Blob container `training-data` auto-created
- **Access**: Managed identity + RBAC

### Container Registry (ACR)

- **Purpose**: Store Docker images for inference
- **SKU**: Basic (dev), Standard/Premium (prod)
- **Features**: Admin enabled, multi-arch support
- **Authentication**: Azure AD + admin credentials

### Azure ML Workspace

- **Purpose**: Centralized ML experiment tracking, model management
- **Identity**: System-assigned managed identity
- **Integration**: Connected to storage account and ACR
- **Features**: Compute management, dataset registration, job submission

### Compute Cluster (Optional)

- **Purpose**: GPU-accelerated training
- **VM Sizes**:
  - Dev: `Standard_NC6s_v3` (V100)
  - Prod: `NC24ads_A100_v4` (A100)
- **Scaling**: Auto-scale 0-4 nodes
- **Idle Time**: 300 seconds before scale-down
- **Priority**: Low priority (cost-optimized)

## Prerequisites

### Required Tools

- **Azure CLI** (>= 2.50.0): `az --version`
- **Terraform** (>= 1.6.0): `terraform --version`
- **Python** (>= 3.11): For validation scripts
- **Bash**: For deployment scripts

### Azure Setup

```bash
# Login to Azure
az login

# Set active subscription (if you have multiple)
az account set --subscription "Your Subscription Name"

# Verify current subscription
az account show
```

### Permissions Required

- **Contributor** role on subscription or resource group
- Ability to create:
  - Resource groups
  - Storage accounts
  - Container registries
  - Machine learning workspaces
  - Compute resources

## Deployment

### Method 1: Using Deployment Script (Recommended)

```bash
# Deploy to dev environment
cd infra/scripts
./deploy.sh dev

# Deploy to production
./deploy.sh prod
```

The script will:

1. Validate Azure CLI authentication
2. Initialize Terraform
3. Validate configuration
4. Create and display execution plan
5. Prompt for confirmation
6. Apply infrastructure changes
7. Display outputs

**Environment Variables:**

- `SKIP_PLAN=true`: Skip plan creation, apply directly
- `AUTO_APPROVE=true`: Auto-approve apply (no prompt)

### Method 2: Manual Terraform

```bash
cd infra/terraform

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Preview changes
terraform plan -var-file=environments/dev.tfvars

# Apply changes
terraform apply -var-file=environments/dev.tfvars
```

### Method 3: Using Jupyter Notebook

Open and run `notebooks/01-setup-infrastructure.ipynb` for interactive deployment with explanations.

## Validation

After deployment, validate that all resources were created correctly:

```bash
cd infra/scripts
./validate-deployment.sh dev
```

The validation script checks:

- ✅ Resource group exists
- ✅ Storage account exists and accessible
- ✅ Blob container `training-data` exists
- ✅ Container registry exists and accessible
- ✅ Azure ML workspace exists and accessible

Expected output:

```
[✓] resource-group exists
[✓] storage-account exists
[✓] container-registry exists
[✓] ml-workspace exists
[✓] Blob container 'training-data' exists
[✓] ACR login server: <acr-name>.azurecr.io
[✓] Azure ML workspace accessible

All checks passed! (6/6)
Infrastructure is ready for use.
```

## Configuration

### Environment Files

**dev.tfvars** (Development):

```hcl
resource_group_name   = "rg-slm-train-dev"
location              = "eastus"
storage_account_name  = "slmtrainstoragedev"  # Must be globally unique
acr_name              = "slmtrainacrdev"      # Must be globally unique
acr_sku               = "Basic"
workspace_name        = "slm-train-workspace-dev"
compute_vm_size       = "Standard_NC6s_v3"    # V100 GPU
compute_max_instances = 2
```

**prod.tfvars** (Production):

```hcl
resource_group_name   = "rg-slm-train-prod"
location              = "eastus"
storage_account_name  = "slmtrainstorageprod"
acr_name              = "slmtrainacrprod"
acr_sku               = "Standard"
workspace_name        = "slm-train-workspace-prod"
compute_vm_size       = "Standard_NC24ads_A100_v4"  # A100 GPU
compute_max_instances = 4
```

### Customizing Resource Names

**Storage Account Names:**

- Must be globally unique across all Azure
- 3-24 characters, lowercase letters and numbers only
- No hyphens, underscores, or special characters
- If deployment fails with "StorageAccountAlreadyTaken", update name

**Container Registry Names:**

- Must be globally unique
- 5-50 characters, alphanumeric only
- If deployment fails with "RegistryNameNotAvailable", update name

### Cost Optimization

**Development Environment:**

```hcl
# Use basic SKUs
acr_sku               = "Basic"

# Smaller GPU VMs
compute_vm_size       = "Standard_NC6s_v3"

# Lower max instances
compute_max_instances = 2

# Shorter idle time
compute_idle_seconds  = 180  # 3 minutes
```

**Production Environment:**

```hcl
# Standard/Premium SKUs
acr_sku               = "Standard"

# Larger GPU VMs
compute_vm_size       = "Standard_NC24ads_A100_v4"

# Higher max instances
compute_max_instances = 4

# Standard idle time
compute_idle_seconds  = 300  # 5 minutes
```

## Outputs

After successful deployment, Terraform outputs these values:

```hcl
resource_group_name     # Name of the resource group
workspace_name          # Azure ML workspace name
workspace_id            # Azure ML workspace ID
storage_account_name    # Storage account name
storage_container_name  # Blob container name ("training-data")
acr_name                # Container registry name
acr_login_server        # ACR login server URL
compute_cluster_name    # Compute cluster name
```

Access outputs:

```bash
terraform output
terraform output -json > outputs.json
terraform output -raw workspace_name
```

These outputs are automatically used by:

- Python configuration (`src/utils/config.py`)
- Data upload scripts (`src/data/upload_to_blob.py`)
- Training jobs (`src/training/`)
- Inference deployment (`src/inference/`)

## Troubleshooting

### "StorageAccountAlreadyTaken" Error

**Problem**: Storage account name is not globally unique.

**Solution**:

```bash
# Check if name is available
az storage account check-name --name slmtrainstoragedev

# Update tfvars with unique name
storage_account_name = "slmtrain<yourname>dev"
```

### "RegistryNameNotAvailable" Error

**Problem**: Container registry name is taken.

**Solution**:

```bash
# Check if name is available
az acr check-name --name slmtrainacrdev

# Update tfvars with unique name
acr_name = "slmtrain<yourname>dev"
```

### "Insufficient Quota" Error

**Problem**: Not enough quota for GPU VMs.

**Solution**:

1. Request quota increase: [Azure Portal → Quotas](https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade)
2. Or use smaller VM size: `Standard_NC6s_v3` instead of `Standard_NC24ads_A100_v4`

### "InvalidAuthenticationToken" Error

**Problem**: Azure CLI session expired.

**Solution**:

```bash
az logout
az login
az account set --subscription "Your Subscription"
```

### Terraform State Lock Error

**Problem**: Previous operation didn't complete cleanly.

**Solution**:

```bash
# Force unlock (use carefully!)
terraform force-unlock <lock-id>
```

## Cleanup

### Destroy All Resources

⚠️ **WARNING**: This permanently deletes all resources and data!

```bash
cd infra/scripts
./destroy.sh dev
```

The script will:

1. Show resources to be destroyed
2. Require typing `destroy-dev` to confirm
3. Prompt final confirmation
4. Destroy all infrastructure

**Safety Options:**

```bash
# Skip confirmations (dangerous!)
FORCE=true ./destroy.sh dev

# Auto-approve (still shows plan)
AUTO_APPROVE=true ./destroy.sh dev
```

### Manual Destroy

```bash
cd infra/terraform
terraform destroy -var-file=environments/dev.tfvars
```

### Delete Resource Group (Alternative)

```bash
# This deletes everything in the resource group
az group delete --name rg-slm-train-dev --yes --no-wait
```

## Testing

Infrastructure tests are located in `tests/integration/test_terraform.py`:

```bash
# Run all infrastructure tests
pytest tests/integration/test_terraform.py -v

# Run specific test
pytest tests/integration/test_terraform.py::test_terraform_validate -v

# Run tests excluding Azure-dependent ones
pytest tests/integration/test_terraform.py \
  -k "not validate and not plan" -v
```

**Test Coverage:**

- ✅ Terraform syntax validation
- ✅ Configuration formatting
- ✅ Module structure verification
- ✅ Variable and output definitions
- ✅ Environment configuration files
- ✅ Deployment script syntax
- ⚠️ Plan generation (requires Azure auth)
- ⚠️ Resource validation (requires Azure auth)

## CI/CD Integration

### GitHub Actions

The infrastructure can be deployed via GitHub Actions:

```yaml
# .github/workflows/deploy-infra.yml
name: Deploy Infrastructure

on:
  push:
    branches: [main]
    paths:
      - "infra/**"

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.6.0

      - name: Deploy Infrastructure
        run: |
          cd infra/scripts
          SKIP_PLAN=false AUTO_APPROVE=true ./deploy.sh dev
```

### Azure DevOps

Similar pipeline can be created in `azure-pipelines.yml`.

## Security Best Practices

1. **Never commit sensitive values**:

   - Use `.gitignore` for `*.tfstate`, `*.tfvars` (except examples)
   - Store secrets in Azure Key Vault
   - Use managed identities when possible

2. **Use RBAC instead of keys**:

   - Assign roles to managed identities
   - Disable storage account key access when possible
   - Use Azure AD authentication for ACR

3. **Enable auditing**:

   - Use Azure Monitor for resource logs
   - Enable storage account logging
   - Track Terraform state changes

4. **Network security**:
   - Consider private endpoints for production
   - Restrict storage account access
   - Use VNet integration for compute

## Next Steps

After infrastructure is provisioned:

1. **Upload training data**: `notebooks/02-prepare-data.ipynb`
2. **Provision compute**: `notebooks/03-provision-compute.ipynb`
3. **Download model**: `notebooks/04-download-model.ipynb`
4. **Run training**: `notebooks/05-fine-tune-model.ipynb`

## Support

For issues or questions:

- Check [Troubleshooting](#troubleshooting) section
- Review [Azure documentation](https://docs.microsoft.com/azure)
- File an issue in the project repository
