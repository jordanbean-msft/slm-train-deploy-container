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

variable "suffix" {
  description = "Suffix for resource names (e.g., environment)"
  type        = string
  default     = ""
}

variable "vm_size" {
  description = "VM size for compute nodes"
  type        = string
}

variable "min_instances" {
  description = "Minimum number of nodes"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of nodes"
  type        = number
  default     = 4
}

variable "idle_seconds" {
  description = "Idle time before scale down"
  type        = number
  default     = 300
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
