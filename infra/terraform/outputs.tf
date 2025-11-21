output "resource_group_name" {
  description = "Name of the resource group"
  value       = data.azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = module.storage.storage_account_name
}

output "storage_container_name" {
  description = "Name of the blob container for training data"
  value       = module.storage.container_name
}

output "acr_name" {
  description = "Name of the Azure Container Registry"
  value       = module.container_registry.acr_name
}

output "acr_login_server" {
  description = "Login server URL for ACR"
  value       = module.container_registry.acr_login_server
}

output "workspace_name" {
  description = "Name of the Azure ML workspace"
  value       = module.azureml_workspace.workspace_name
}

output "workspace_id" {
  description = "ID of the Azure ML workspace"
  value       = module.azureml_workspace.workspace_id
}

output "compute_cluster_name" {
  description = "Name of the compute cluster"
  value       = module.compute_cluster.compute_name
}
output "user_assigned_identity_id" {
  description = "ID of the user-assigned managed identity"
  value       = azurerm_user_assigned_identity.main.id
}

output "user_assigned_identity_principal_id" {
  description = "Principal ID of the user-assigned managed identity"
  value       = azurerm_user_assigned_identity.main.principal_id
}

output "user_assigned_identity_client_id" {
  description = "Client ID of the user-assigned managed identity"
  value       = azurerm_user_assigned_identity.main.client_id
}

# Additional outputs with naming conventions expected by Python config
output "AZURE_RESOURCE_GROUP" {
  description = "Resource group name (alternate naming)"
  value       = data.azurerm_resource_group.main.name
}

output "AZURE_STORAGE_ACCOUNT_NAME" {
  description = "Storage account name (alternate naming)"
  value       = module.storage.storage_account_name
}

output "AZURE_STORAGE_CONTAINER_NAME" {
  description = "Storage container name (alternate naming)"
  value       = module.storage.container_name
}

output "AZUREML_WORKSPACE_NAME" {
  description = "Azure ML workspace name (alternate naming)"
  value       = module.azureml_workspace.workspace_name
}

output "AZUREML_COMPUTE_NAME" {
  description = "Compute cluster name (alternate naming)"
  value       = module.compute_cluster.compute_name
}
