#!/usr/bin/env bash
#
# Destroy Azure infrastructure created by Terraform
#
# Usage:
#   ./destroy.sh [environment]
#
# Arguments:
#   environment: dev (default) or prod
#
# Environment variables:
#   AUTO_APPROVE: Set to "true" to auto-approve destroy
#   FORCE: Set to "true" to skip all confirmations (dangerous!)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/terraform"
ENVIRONMENT="${1:-dev}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Validate environment argument
if [[ ! "$ENVIRONMENT" =~ ^(dev|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'dev' or 'prod'"
    exit 1
fi

TFVARS_FILE="${TERRAFORM_DIR}/environments/${ENVIRONMENT}.tfvars"

log_warn "⚠️  DESTROYING infrastructure for environment: $ENVIRONMENT"
log_warn "⚠️  This will permanently delete all resources!"

# Change to Terraform directory
cd "$TERRAFORM_DIR"

# Check if Terraform is initialized
if [[ ! -d .terraform ]]; then
    log_error "Terraform not initialized. Run deploy.sh first or 'terraform init'"
    exit 1
fi

# Check if tfvars file exists
if [[ ! -f "$TFVARS_FILE" ]]; then
    log_error "Terraform variables file not found: $TFVARS_FILE"
    exit 1
fi

# Check if Azure CLI is logged in
log_info "Verifying Azure CLI authentication..."
if ! az account show &>/dev/null; then
    log_error "Not logged in to Azure CLI. Run 'az login' first."
    exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
log_info "Using subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"

# Show current resources that will be destroyed
log_info "Resources that will be destroyed:"
echo
terraform show -no-color | head -50
echo

# Safety confirmation (unless FORCE is set)
if [[ "${FORCE:-false}" != "true" ]]; then
    log_warn "⚠️  Are you ABSOLUTELY SURE you want to destroy these resources?"
    read -p "Type 'destroy-${ENVIRONMENT}' to confirm: " -r REPLY
    echo
    if [[ "$REPLY" != "destroy-${ENVIRONMENT}" ]]; then
        log_info "Destruction cancelled by user"
        exit 0
    fi
fi

# Plan destruction
log_info "Creating destruction plan..."
if ! terraform plan -destroy -var-file="$TFVARS_FILE" -out="tfplan-destroy"; then
    log_error "Terraform plan failed"
    exit 1
fi

# Final confirmation before destroy (unless AUTO_APPROVE is set)
if [[ "${AUTO_APPROVE:-false}" != "true" ]] && [[ "${FORCE:-false}" != "true" ]]; then
    log_warn "Last chance to cancel!"
    read -p "Proceed with destruction? (yes/no): " -r REPLY
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Destruction cancelled by user"
        rm -f tfplan-destroy
        exit 0
    fi
fi

# Destroy infrastructure
log_info "Destroying infrastructure..."
DESTROY_ARGS=("tfplan-destroy")

if [[ "${AUTO_APPROVE:-false}" == "true" ]] || [[ "${FORCE:-false}" == "true" ]]; then
    # Note: plan already created, no need for -auto-approve
    :
fi

if terraform apply "${DESTROY_ARGS[@]}"; then
    log_info "Infrastructure destroyed successfully!"
    rm -f tfplan-destroy

    # Clean up output files
    rm -f "${TERRAFORM_DIR}/outputs.json"

    exit 0
else
    log_error "Terraform destroy failed"
    log_warn "Some resources may remain. Check Azure Portal and retry."
    rm -f tfplan-destroy
    exit 1
fi
