module "naming" {
  source = "Azure/naming/azurerm"
  suffix = [var.suffix]
}

resource "azurerm_machine_learning_workspace" "main" {
  name                    = module.naming.machine_learning_workspace.name
  location                = var.location
  resource_group_name     = var.resource_group_name
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id
  application_insights_id = var.application_insights_id
  key_vault_id            = var.key_vault_id

  identity {
    type = "UserAssigned"
    identity_ids = [
      var.user_assigned_identity_id
    ]
  }

  primary_user_assigned_identity = var.user_assigned_identity_id

  tags = var.tags
}

resource "azurerm_monitor_diagnostic_setting" "workspace" {
  name                       = "diag-${azurerm_machine_learning_workspace.main.name}"
  target_resource_id         = azurerm_machine_learning_workspace.main.id
  log_analytics_workspace_id = var.log_analytics_workspace_id

  enabled_log {
    category = "AmlComputeClusterEvent"
  }

  enabled_log {
    category = "AmlComputeClusterNodeEvent"
  }

  enabled_log {
    category = "AmlComputeJobEvent"
  }

  enabled_log {
    category = "AmlRunStatusChangedEvent"
  }

  metric {
    category = "AllMetrics"
  }
}
