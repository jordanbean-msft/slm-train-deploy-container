#!/bin/bash
set -e

# Azure Developer CLI Setup Script for SLM Training Project
# This script helps configure azd environment for infrastructure deployment

echo "==================================="
echo "Azure Developer CLI Configuration"
echo "==================================="
echo ""

# Check if azd is installed
if ! command -v azd &> /dev/null; then
    echo "ERROR: Azure Developer CLI (azd) is not installed"
    echo "Install with: curl -fsSL https://aka.ms/install-azd.sh | bash"
    exit 1
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "ERROR: Azure CLI is not installed"
    echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo "Not logged in to Azure. Running 'az login'..."
    az login
fi

# Get current subscription
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
echo "Current subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"
echo ""

# Prompt for environment name
read -p "Environment name (e.g., dev, staging, prod) [dev]: " ENV_NAME
ENV_NAME=${ENV_NAME:-dev}

# Check if environment already exists
if azd env list 2>/dev/null | grep -q "^$ENV_NAME\$"; then
    read -p "Environment '$ENV_NAME' already exists. Select it? (y/n) [y]: " SELECT_ENV
    SELECT_ENV=${SELECT_ENV:-y}
    if [[ "$SELECT_ENV" == "y" || "$SELECT_ENV" == "Y" ]]; then
        azd env select $ENV_NAME
        echo "Selected existing environment: $ENV_NAME"
    fi
else
    # Create new environment
    azd env new $ENV_NAME
    echo "Created new environment: $ENV_NAME"
fi

echo ""
echo "==================================="
echo "Required Configuration"
echo "==================================="
echo ""

# Resource Group Name
read -p "Azure Resource Group Name (must exist): " RG_NAME
while [ -z "$RG_NAME" ]; do
    echo "ERROR: Resource group name cannot be empty"
    read -p "Azure Resource Group Name (must exist): " RG_NAME
done

# Verify resource group exists
if ! az group show --name "$RG_NAME" &> /dev/null; then
    echo ""
    echo "WARNING: Resource group '$RG_NAME' does not exist"
    read -p "Would you like to create it? (y/n) [y]: " CREATE_RG
    CREATE_RG=${CREATE_RG:-y}

    if [[ "$CREATE_RG" == "y" || "$CREATE_RG" == "Y" ]]; then
        read -p "Azure Location [eastus]: " LOCATION
        LOCATION=${LOCATION:-eastus}
        echo "Creating resource group..."
        az group create --name "$RG_NAME" --location "$LOCATION"
        echo "Resource group created successfully"
    else
        echo "Please create the resource group manually and run this script again"
        exit 1
    fi
else
    LOCATION=$(az group show --name "$RG_NAME" --query location -o tsv)
    echo "Resource group exists in location: $LOCATION"
fi

echo ""
echo "==================================="
echo "Resource Naming Configuration"
echo "==================================="
echo ""
echo "Azure resources will be named using the terraform-azurerm-naming module."
echo "This ensures consistent naming across all resources."
echo ""
echo "Pattern: {resource_type}-{suffix}"
echo "Example: 'st-dev' for storage, 'acr-dev' for container registry"
echo ""
echo "Naming suffix will use environment: $ENV_NAME"
echo ""
echo "Generated naming pattern:"
echo "  Suffix: $ENV_NAME"
echo "  Example Storage Account: st-${ENV_NAME} (with unique hash)"
echo "  Example ACR: acr-${ENV_NAME} (with unique hash)"
echo "  Example Workspace: mlw-${ENV_NAME}"
echo ""

# ACR SKU
echo "ACR SKU options: Basic, Standard, Premium"
read -p "ACR SKU [Basic]: " ACR_SKU
ACR_SKU=${ACR_SKU:-Basic}

# Compute Configuration
echo ""
echo "==================================="
echo "Optional Compute Configuration"
echo "==================================="

read -p "VM Size [Standard_NC6s_v3]: " VM_SIZE
VM_SIZE=${VM_SIZE:-Standard_NC6s_v3}

read -p "Min Instances [0]: " MIN_INSTANCES
MIN_INSTANCES=${MIN_INSTANCES:-0}

read -p "Max Instances [4]: " MAX_INSTANCES
MAX_INSTANCES=${MAX_INSTANCES:-4}

read -p "Idle Seconds [300]: " IDLE_SECONDS
IDLE_SECONDS=${IDLE_SECONDS:-300}

# Set all environment variables
echo ""
echo "Setting environment variables..."

# Set azd environment variables (these will be mapped to Terraform via main.tfvars.json)
azd env set AZURE_SUBSCRIPTION_ID "$SUBSCRIPTION_ID"
azd env set AZURE_ENV_NAME "$ENV_NAME"
azd env set AZURE_RESOURCE_GROUP_NAME "$RG_NAME"
azd env set AZURE_LOCATION "$LOCATION"
azd env set AZURE_ACR_SKU "$ACR_SKU"
azd env set AZURE_COMPUTE_VM_SIZE "$VM_SIZE"
azd env set AZURE_COMPUTE_MIN_INSTANCES "$MIN_INSTANCES"
azd env set AZURE_COMPUTE_MAX_INSTANCES "$MAX_INSTANCES"
azd env set AZURE_COMPUTE_IDLE_SECONDS "$IDLE_SECONDS"

echo ""
echo "==================================="
echo "Configuration Complete!"
echo "==================================="
echo ""
echo "Environment: $ENV_NAME"
echo "Resource Group: $RG_NAME"
echo "Location: $LOCATION"
echo "Naming Suffix: $ENV_NAME"
echo ""
echo "Resources will be named automatically using terraform-azurerm-naming module:"
echo "  Storage Account: st-${ENV_NAME} (with unique hash)"
echo "  Container Registry: acr-${ENV_NAME} (with unique hash)"
echo "  ML Workspace: mlw-${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Deploy infrastructure: azd up"
echo "  2. Or provision only: azd provision"
echo "  3. After provisioning, export Terraform outputs for notebooks:"
echo "     azd env get-values > .env"
echo "  4. The .env file will contain actual resource names created by Terraform"
echo ""
