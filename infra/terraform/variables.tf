variable "resource_group_name" {
  description = "Name of the existing Azure resource group"
  type        = string
}

variable "location" {
  description = "Azure region for resources (must match resource group location)"
  type        = string
}

variable "suffix" {
  description = "Suffix for resource names (e.g., environment like dev, prod)"
  type        = string
  default     = ""
}

variable "acr_sku" {
  description = "SKU for Azure Container Registry (Basic, Standard, or Premium)"
  type        = string
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be Basic, Standard, or Premium."
  }
}

variable "compute_vm_size" {
  description = "VM size for the compute cluster (e.g., Standard_NC6s_v3 for V100 GPU)"
  type        = string
}

variable "compute_min_instances" {
  description = "Minimum number of nodes in the compute cluster (0 for auto-scale to zero)"
  type        = number
}

variable "compute_max_instances" {
  description = "Maximum number of nodes in the compute cluster"
  type        = number
}

variable "compute_idle_seconds" {
  description = "Idle time before scale down (in seconds)"
  type        = number
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
}
