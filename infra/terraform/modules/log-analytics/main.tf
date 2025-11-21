module "naming" {
  source = "Azure/naming/azurerm"
  suffix = [var.suffix]
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = module.naming.log_analytics_workspace.name_unique
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = var.sku
  retention_in_days   = var.retention_in_days

  tags = var.tags
}
