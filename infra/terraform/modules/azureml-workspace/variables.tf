variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "suffix" {
  description = "Suffix for resource names (e.g., environment)"
  type        = string
  default     = ""
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

variable "user_assigned_identity_id" {
  description = "ID of the user-assigned managed identity"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace for diagnostics"
  type        = string
}
