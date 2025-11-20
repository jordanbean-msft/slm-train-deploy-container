#!/usr/bin/env bash
#
# Deploy infrastructure to Azure using Terraform
#
# Usage:
#   ./deploy.sh [environment]
#
# Arguments:
#   environment: dev (default) or prod
#
# Environment variables:
#   SKIP_PLAN: Set to "true" to skip plan and directly apply
#   AUTO_APPROVE: Set to "true" to auto-approve apply
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
PLAN_FILE="${TERRAFORM_DIR}/tfplan-${ENVIRONMENT}"

log_info "Deploying infrastructure for environment: $ENVIRONMENT"

# Change to Terraform directory
cd "$TERRAFORM_DIR"

# Check if Azure CLI is logged in
log_info "Verifying Azure CLI authentication..."
if ! az account show &>/dev/null; then
    log_error "Not logged in to Azure CLI. Run 'az login' first."
    exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
log_info "Using subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"

# Check if tfvars file exists
if [[ ! -f "$TFVARS_FILE" ]]; then
    log_error "Terraform variables file not found: $TFVARS_FILE"
    exit 1
fi

# Initialize Terraform if needed
if [[ ! -d .terraform ]]; then
    log_info "Initializing Terraform..."
    terraform init
else
    log_info "Terraform already initialized"
fi

# Validate Terraform configuration
log_info "Validating Terraform configuration..."
if ! terraform validate; then
    log_error "Terraform validation failed"
    exit 1
fi

# Format Terraform files
log_info "Formatting Terraform files..."
terraform fmt -recursive

# Create and review plan
if [[ "${SKIP_PLAN:-false}" != "true" ]]; then
    log_info "Creating Terraform plan..."
    if ! terraform plan -var-file="$TFVARS_FILE" -out="$PLAN_FILE"; then
        log_error "Terraform plan failed"
        exit 1
    fi

    log_warn "Review the plan above carefully before proceeding"
    echo
    read -p "Continue with apply? (yes/no): " -r REPLY
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Deployment cancelled by user"
        rm -f "$PLAN_FILE"
        exit 0
    fi
else
    log_warn "Skipping plan creation (SKIP_PLAN=true)"
    PLAN_FILE=""
fi

# Apply infrastructure changes
log_info "Applying Terraform changes..."
APPLY_ARGS=()
if [[ -n "$PLAN_FILE" && -f "$PLAN_FILE" ]]; then
    APPLY_ARGS+=("$PLAN_FILE")
else
    APPLY_ARGS+=("-var-file=$TFVARS_FILE")
fi

if [[ "${AUTO_APPROVE:-false}" == "true" ]]; then
    APPLY_ARGS+=("-auto-approve")
fi

if terraform apply "${APPLY_ARGS[@]}"; then
    log_info "Infrastructure deployed successfully!"

    # Clean up plan file
    rm -f "$PLAN_FILE"

    # Display outputs
    echo
    log_info "Infrastructure outputs:"
    terraform output

    # Save outputs to file for other scripts
    OUTPUT_FILE="${TERRAFORM_DIR}/outputs.json"
    terraform output -json > "$OUTPUT_FILE"
    log_info "Outputs saved to: $OUTPUT_FILE"

    exit 0
else
    log_error "Terraform apply failed"
    rm -f "$PLAN_FILE"
    exit 1
fi
