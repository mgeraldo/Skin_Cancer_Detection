variable "location" {
  description = "The Azure region where resources will be created"
  type        = string
  default     = "East US"
}

variable "resource_group_name" {
  description = "The name of the resource group for ISIC 2019 dataset"
  type        = string
  default     = "isic2019-resource-group"
}

variable "container_name" {
  description = "The name of the blob container for ISIC 2019 dataset"
  type        = string
  default     = "isic2019-images"
}