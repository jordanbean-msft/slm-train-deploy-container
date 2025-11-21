# Use existing resource group (provided as input)
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

# Generate unique suffix based on input suffix
resource "random_string" "unique_suffix" {
  length  = 4
  special = false
  upper   = false
  keepers = {
    suffix = var.suffix
  }
}

locals {
  # Use only random string as suffix to avoid exceeding Azure ML Workspace 33-char limit
  # Original suffix is preserved in tags
  unique_suffix   = random_string.unique_suffix.result
  original_suffix = var.suffix
}

# Create user-assigned managed identity
resource "azurerm_user_assigned_identity" "main" {
  name                = "id-mlworkspace-${local.unique_suffix}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  tags                = var.tags
}

# Log Analytics Workspace Module
module "log_analytics" {
  source = "./modules/log-analytics"

  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  suffix              = local.unique_suffix
  tags                = var.tags
}

# Application Insights Module
module "application_insights" {
  source = "./modules/application-insights"

  resource_group_name        = data.azurerm_resource_group.main.name
  location                   = data.azurerm_resource_group.main.location
  suffix                     = local.unique_suffix
  log_analytics_workspace_id = module.log_analytics.workspace_id
  tags                       = var.tags

  depends_on = [module.log_analytics]
}

# Key Vault Module
module "key_vault" {
  source = "./modules/key-vault"

  resource_group_name                 = data.azurerm_resource_group.main.name
  location                            = data.azurerm_resource_group.main.location
  suffix                              = local.unique_suffix
  user_assigned_identity_principal_id = azurerm_user_assigned_identity.main.principal_id
  log_analytics_workspace_id          = module.log_analytics.workspace_id
  tags                                = var.tags

  depends_on = [module.log_analytics, azurerm_user_assigned_identity.main]
}

# Storage Account Module
module "storage" {
  source = "./modules/storage"

  resource_group_name        = data.azurerm_resource_group.main.name
  location                   = data.azurerm_resource_group.main.location
  suffix                     = local.unique_suffix
  log_analytics_workspace_id = module.log_analytics.workspace_id
  tags                       = var.tags

  depends_on = [module.log_analytics]
}

# Grant managed identity Storage Blob Data Contributor role on storage account
resource "azurerm_role_assignment" "storage_blob_data_contributor" {
  scope                = module.storage.storage_account_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.main.principal_id
}

# Container Registry Module
module "container_registry" {
  source = "./modules/container-registry"

  resource_group_name        = data.azurerm_resource_group.main.name
  location                   = data.azurerm_resource_group.main.location
  suffix                     = local.unique_suffix
  sku                        = var.acr_sku
  log_analytics_workspace_id = module.log_analytics.workspace_id
  tags                       = var.tags

  depends_on = [module.log_analytics]
}

# Grant managed identity AcrPull role on container registry
resource "azurerm_role_assignment" "acr_pull" {
  scope                = module.container_registry.acr_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.main.principal_id
}

# Grant managed identity Contributor role on key vault (required by Azure ML Workspace for control plane operations under RBAC model)
resource "azurerm_role_assignment" "key_vault_contributor" {
  scope                = module.key_vault.key_vault_id
  role_definition_name = "Contributor"
  principal_id         = azurerm_user_assigned_identity.main.principal_id
}

# Grant managed identity Key Vault Administrator role (required by Azure ML Workspace for data plane access in RBAC model)
# Provides full data plane access to secrets, keys, and certificates
resource "azurerm_role_assignment" "key_vault_administrator" {
  scope                = module.key_vault.key_vault_id
  role_definition_name = "Key Vault Administrator"
  principal_id         = azurerm_user_assigned_identity.main.principal_id
}

# Wait for role assignments to propagate before creating workspace (RBAC propagation delay mitigation)
resource "time_sleep" "wait_for_rbac" {
  depends_on = [
    azurerm_role_assignment.storage_blob_data_contributor,
    azurerm_role_assignment.acr_pull,
    azurerm_role_assignment.key_vault_contributor,
    azurerm_role_assignment.key_vault_administrator
  ]

  create_duration = "180s"
}

# Azure ML Workspace Module
module "azureml_workspace" {
  source = "./modules/azureml-workspace"

  resource_group_name        = data.azurerm_resource_group.main.name
  location                   = data.azurerm_resource_group.main.location
  suffix                     = local.unique_suffix
  storage_account_id         = module.storage.storage_account_id
  container_registry_id      = module.container_registry.acr_id
  application_insights_id    = module.application_insights.app_insights_id
  key_vault_id               = module.key_vault.key_vault_id
  user_assigned_identity_id  = azurerm_user_assigned_identity.main.id
  log_analytics_workspace_id = module.log_analytics.workspace_id
  tags                       = var.tags

  depends_on = [
    time_sleep.wait_for_rbac,
    module.application_insights,
    module.key_vault
  ]
}

# Compute Cluster Module
module "compute_cluster" {
  source = "./modules/compute-cluster"

  resource_group_name       = data.azurerm_resource_group.main.name
  location                  = data.azurerm_resource_group.main.location
  workspace_name            = module.azureml_workspace.workspace_name
  suffix                    = local.unique_suffix
  vm_size                   = var.compute_vm_size
  min_instances             = var.compute_min_instances
  max_instances             = var.compute_max_instances
  idle_seconds              = var.compute_idle_seconds
  user_assigned_identity_id = azurerm_user_assigned_identity.main.id
  tags                      = var.tags

  depends_on = [module.azureml_workspace]
}
