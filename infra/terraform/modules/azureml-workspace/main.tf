variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "workspace_name" {
  description = "Name of the Azure ML workspace"
  type        = string
}

variable "storage_account_id" {
  description = "ID of the storage account"
  type        = string
}

variable "container_registry_id" {
  description = "ID of the container registry"
  type        = string
}

variable "application_insights_id" {
  description = "ID of the Application Insights instance"
  type        = string
  default     = null
}

variable "key_vault_id" {
  description = "ID of the Key Vault"
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

resource "azurerm_machine_learning_workspace" "main" {
  name                    = var.workspace_name
  location                = var.location
  resource_group_name     = var.resource_group_name
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id
  application_insights_id = var.application_insights_id
  key_vault_id            = var.key_vault_id

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

output "workspace_id" {
  description = "ID of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.main.id
}

output "workspace_name" {
  description = "Name of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.main.name
}

output "discovery_url" {
  description = "Discovery URL for the workspace"
  value       = azurerm_machine_learning_workspace.main.discovery_url
}
