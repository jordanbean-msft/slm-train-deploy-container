variable "resource_group_name" {
  description = "Name of the Azure resource group"
  type        = string
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "storage_account_name" {
  description = "Name of the Azure Storage Account (must be globally unique)"
  type        = string
}

variable "acr_name" {
  description = "Name of the Azure Container Registry (must be globally unique)"
  type        = string
}

variable "acr_sku" {
  description = "SKU for Azure Container Registry"
  type        = string
  default     = "Basic"
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be Basic, Standard, or Premium."
  }
}

variable "workspace_name" {
  description = "Name of the Azure ML workspace"
  type        = string
}

variable "compute_name" {
  description = "Name of the Azure ML compute cluster"
  type        = string
  default     = "gpu-cluster"
}

variable "compute_vm_size" {
  description = "VM size for the compute cluster"
  type        = string
  default     = "Standard_NC6s_v3"
}

variable "compute_min_instances" {
  description = "Minimum number of nodes in the compute cluster"
  type        = number
  default     = 0
}

variable "compute_max_instances" {
  description = "Maximum number of nodes in the compute cluster"
  type        = number
  default     = 4
}

variable "compute_idle_seconds" {
  description = "Idle time before scale down (in seconds)"
  type        = number
  default     = 300
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    project     = "slm-train-deploy"
    environment = "dev"
    managed_by  = "terraform"
  }
}
