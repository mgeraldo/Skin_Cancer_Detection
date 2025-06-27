variable "location" {
  description = "The Azure region where resources will be created"
  type        = string
  default     = "East US"
}

variable "resource_group_name" {
  description = "The name of the resource group for image dataset"
  type        = string
  default     = "rg-w281-skinvision"
}

variable "container_names" {
  description = "List of container names to create"
  type        = list(string)
  default     = ["isic2019-images", "ham-10_000-images"]
}