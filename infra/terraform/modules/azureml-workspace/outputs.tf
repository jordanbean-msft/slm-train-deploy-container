output "workspace_id" {
  description = "ID of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.main.id
}

output "workspace_name" {
  description = "Name of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.main.name
}

output "discovery_url" {
  description = "Discovery URL for the workspace"
  value       = azurerm_machine_learning_workspace.main.discovery_url
}
