resource_group_name   = "rg-slm-train-dev"
location              = "eastus"
storage_account_name  = "stslmtraindev"
acr_name              = "acrslmtraindev"
acr_sku               = "Basic"
workspace_name        = "mlw-slm-train-dev"
compute_name          = "gpu-cluster-dev"
compute_vm_size       = "Standard_NC6s_v3"
compute_min_instances = 0
compute_max_instances = 2
compute_idle_seconds  = 300

tags = {
  project     = "slm-train-deploy"
  environment = "dev"
  managed_by  = "terraform"
}
