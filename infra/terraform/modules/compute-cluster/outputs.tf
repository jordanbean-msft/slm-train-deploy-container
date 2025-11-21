output "compute_id" {
  description = "ID of the compute cluster"
  value       = azurerm_machine_learning_compute_cluster.main.id
}

output "compute_name" {
  description = "Name of the compute cluster"
  value       = azurerm_machine_learning_compute_cluster.main.name
}
