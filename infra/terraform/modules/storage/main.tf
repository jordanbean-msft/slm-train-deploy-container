locals {
  # Remove hyphens for storage account naming
  # Storage accounts only allow lowercase letters and numbers
  sanitized_suffix = replace(lower(var.suffix), "-", "")
}

module "naming" {
  source = "Azure/naming/azurerm"
  suffix = [local.sanitized_suffix]
}

resource "azurerm_storage_account" "main" {
  name                      = module.naming.storage_account.name_unique
  resource_group_name       = var.resource_group_name
  location                  = var.location
  account_tier              = "Standard"
  account_replication_type  = "LRS"
  shared_access_key_enabled = true

  tags = var.tags
}

resource "azurerm_storage_container" "training_data" {
  name                  = "training-data"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_monitor_diagnostic_setting" "storage" {
  name                       = "diag-${azurerm_storage_account.main.name}"
  target_resource_id         = azurerm_storage_account.main.id
  log_analytics_workspace_id = var.log_analytics_workspace_id

  metric {
    category = "Transaction"
  }
}
