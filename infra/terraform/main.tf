terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.0"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# Storage Account Module
module "storage" {
  source = "./modules/storage"

  resource_group_name  = azurerm_resource_group.main.name
  location             = azurerm_resource_group.main.location
  storage_account_name = var.storage_account_name
  tags                 = var.tags
}

# Container Registry Module
module "container_registry" {
  source = "./modules/container-registry"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  acr_name            = var.acr_name
  sku                 = var.acr_sku
  tags                = var.tags
}

# Azure ML Workspace Module
module "azureml_workspace" {
  source = "./modules/azureml-workspace"

  resource_group_name     = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  workspace_name          = var.workspace_name
  storage_account_id      = module.storage.storage_account_id
  container_registry_id   = module.container_registry.acr_id
  application_insights_id = null # Optional - can add later
  key_vault_id            = null # Optional - can add later
  tags                    = var.tags
}

# Compute Cluster Module
module "compute_cluster" {
  source = "./modules/compute-cluster"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_name      = module.azureml_workspace.workspace_name
  compute_name        = var.compute_name
  vm_size             = var.compute_vm_size
  min_instances       = var.compute_min_instances
  max_instances       = var.compute_max_instances
  idle_seconds        = var.compute_idle_seconds
  tags                = var.tags

  depends_on = [module.azureml_workspace]
}
