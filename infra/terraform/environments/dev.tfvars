resource_group_name   = "rg-slm-train-dev"
location              = "eastus"
suffix                = ["dev"]
acr_sku               = "Basic"
compute_vm_size       = "Standard_D4s_v5"
compute_min_instances = 0
compute_max_instances = 2
compute_idle_seconds  = 300

tags = {
  project     = "slm-train-deploy"
  environment = "dev"
  managed_by  = "terraform"
}
