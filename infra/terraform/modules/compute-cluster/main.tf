module "naming" {
  source = "Azure/naming/azurerm"
  suffix = [var.suffix]
}

resource "azurerm_machine_learning_compute_cluster" "main" {
  name                          = "cpu-${var.suffix}"
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
    type = "UserAssigned"
    identity_ids = [
      var.user_assigned_identity_id
    ]
  }

  tags = var.tags
}

data "azurerm_machine_learning_workspace" "main" {
  name                = var.workspace_name
  resource_group_name = var.resource_group_name
}
