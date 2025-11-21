module "naming" {
  source = "Azure/naming/azurerm"
  suffix = [var.suffix]
}

resource "azurerm_application_insights" "main" {
  name                = module.naming.application_insights.name_unique
  resource_group_name = var.resource_group_name
  location            = var.location
  application_type    = var.application_type
  workspace_id        = var.log_analytics_workspace_id

  tags = var.tags
}
