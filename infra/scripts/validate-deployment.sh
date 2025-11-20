#!/usr/bin/env bash
#
# Validate Azure infrastructure deployment
#
# Usage:
#   ./validate-deployment.sh [environment]
#
# Arguments:
#   environment: dev (default) or prod
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/terraform"
ENVIRONMENT="${1:-dev}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_fail() {
    echo -e "${RED}[✗]${NC} $1"
}

# Validate environment argument
if [[ ! "$ENVIRONMENT" =~ ^(dev|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'dev' or 'prod'"
    exit 1
fi

log_info "Validating deployment for environment: $ENVIRONMENT"
echo

# Check Azure CLI authentication
log_info "Checking Azure CLI authentication..."
if ! az account show &>/dev/null; then
    log_error "Not logged in to Azure CLI. Run 'az login' first."
    exit 1
fi
log_success "Azure CLI authenticated"

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
echo -e "${BLUE}Subscription:${NC} $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"
echo

# Change to Terraform directory
cd "$TERRAFORM_DIR"

# Check if Terraform outputs exist
if [[ ! -f terraform.tfstate ]]; then
    log_error "No Terraform state found. Run deploy.sh first."
    exit 1
fi

# Extract outputs
log_info "Reading Terraform outputs..."
RESOURCE_GROUP=$(terraform output -raw resource_group_name 2>/dev/null || echo "")
WORKSPACE_NAME=$(terraform output -raw workspace_name 2>/dev/null || echo "")
STORAGE_ACCOUNT=$(terraform output -raw storage_account_name 2>/dev/null || echo "")
ACR_NAME=$(terraform output -raw acr_name 2>/dev/null || echo "")
LOCATION=$(terraform output -raw location 2>/dev/null || echo "")

if [[ -z "$RESOURCE_GROUP" ]]; then
    log_error "Could not read Terraform outputs"
    exit 1
fi

echo -e "${BLUE}Resource Group:${NC} $RESOURCE_GROUP"
echo -e "${BLUE}Location:${NC} $LOCATION"
echo

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0

# Function to check resource existence
check_resource() {
    local resource_type=$1
    local resource_name=$2
    local resource_group=$3

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    echo -n "Checking $resource_type '$resource_name'... "

    case $resource_type in
        "resource-group")
            if az group show --name "$resource_name" &>/dev/null; then
                log_success "$resource_type exists"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                return 0
            fi
            ;;
        "storage-account")
            if az storage account show --name "$resource_name" --resource-group "$resource_group" &>/dev/null; then
                log_success "$resource_type exists"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                return 0
            fi
            ;;
        "container-registry")
            if az acr show --name "$resource_name" --resource-group "$resource_group" &>/dev/null; then
                log_success "$resource_type exists"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                return 0
            fi
            ;;
        "ml-workspace")
            if az ml workspace show --name "$resource_name" --resource-group "$resource_group" &>/dev/null; then
                log_success "$resource_type exists"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                return 0
            fi
            ;;
    esac

    log_fail "$resource_type NOT FOUND"
    return 1
}

# Run validations
log_info "Validating Azure resources..."
echo

check_resource "resource-group" "$RESOURCE_GROUP" ""
check_resource "storage-account" "$STORAGE_ACCOUNT" "$RESOURCE_GROUP"
check_resource "container-registry" "$ACR_NAME" "$RESOURCE_GROUP"
check_resource "ml-workspace" "$WORKSPACE_NAME" "$RESOURCE_GROUP"

echo

# Check blob containers
log_info "Checking storage containers..."
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if az storage container exists \
    --name "training-data" \
    --account-name "$STORAGE_ACCOUNT" \
    --auth-mode login \
    --query exists -o tsv 2>/dev/null | grep -q "true"; then
    log_success "Blob container 'training-data' exists"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    log_fail "Blob container 'training-data' NOT FOUND"
fi

echo

# Check ACR status
log_info "Checking ACR configuration..."
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv 2>/dev/null || echo "")
if [[ -n "$ACR_LOGIN_SERVER" ]]; then
    log_success "ACR login server: $ACR_LOGIN_SERVER"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    log_fail "Could not get ACR login server"
fi

echo

# Check Azure ML workspace connection
log_info "Testing Azure ML workspace connection..."
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if az ml workspace show --name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    log_success "Azure ML workspace accessible"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    log_fail "Could not connect to Azure ML workspace"
fi

echo

# Summary
echo "======================================"
if [[ $PASSED_CHECKS -eq $TOTAL_CHECKS ]]; then
    log_success "All checks passed! ($PASSED_CHECKS/$TOTAL_CHECKS)"
    echo
    log_info "Infrastructure is ready for use."
    exit 0
else
    log_error "Some checks failed! ($PASSED_CHECKS/$TOTAL_CHECKS)"
    echo
    log_warn "Please review the failed checks and redeploy if necessary."
    exit 1
fi
