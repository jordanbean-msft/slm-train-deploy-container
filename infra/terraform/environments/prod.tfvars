resource_group_name   = "rg-slm-train-prod"
location              = "eastus"
storage_account_name  = "stslmtrainprod"
acr_name              = "acrslmtrainprod"
acr_sku               = "Standard"
workspace_name        = "mlw-slm-train-prod"
compute_name          = "gpu-cluster-prod"
compute_vm_size       = "Standard_NC24ads_A100_v4"
compute_min_instances = 0
compute_max_instances = 4
compute_idle_seconds  = 600

tags = {
  project     = "slm-train-deploy"
  environment = "prod"
  managed_by  = "terraform"
}
