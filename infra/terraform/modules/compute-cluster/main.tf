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

variable "compute_name" {
  description = "Name of the compute cluster"
  type        = string
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

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

resource "azurerm_machine_learning_compute_cluster" "main" {
  name                          = var.compute_name
  location                      = var.location
  vm_priority                   = "Dedicated"
  vm_size                       = var.vm_size
  machine_learning_workspace_id = data.azurerm_machine_learning_workspace.main.id

  scale_settings {
    min_node_count                       = var.min_instances
    max_node_count                       = var.max_instances
    scale_down_nodes_after_idle_duration = "PT${var.idle_seconds}S"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

data "azurerm_machine_learning_workspace" "main" {
  name                = var.workspace_name
  resource_group_name = var.resource_group_name
}

output "compute_id" {
  description = "ID of the compute cluster"
  value       = azurerm_machine_learning_compute_cluster.main.id
}

output "compute_name" {
  description = "Name of the compute cluster"
  value       = azurerm_machine_learning_compute_cluster.main.name
}
