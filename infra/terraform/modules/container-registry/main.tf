locals {
  # Remove hyphens for container registry naming
  # ACR only allows alphanumeric characters
  sanitized_suffix = replace(lower(var.suffix), "-", "")
}

module "naming" {
  source = "Azure/naming/azurerm"
  suffix = [local.sanitized_suffix]
}

resource "azurerm_container_registry" "main" {
  name                = module.naming.container_registry.name_unique
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = var.sku
  admin_enabled       = true

  tags = var.tags
}

resource "azurerm_monitor_diagnostic_setting" "acr" {
  name                       = "diag-${azurerm_container_registry.main.name}"
  target_resource_id         = azurerm_container_registry.main.id
  log_analytics_workspace_id = var.log_analytics_workspace_id

  enabled_log {
    category = "ContainerRegistryRepositoryEvents"
  }

  enabled_log {
    category = "ContainerRegistryLoginEvents"
  }

  metric {
    category = "AllMetrics"
  }
}
