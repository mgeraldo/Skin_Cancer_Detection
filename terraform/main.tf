resource "random_string" "suffix" {
  length  = 8
  upper   = false
  special = false
  lower  = true
  numeric = false
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_storage_account" "storage" {
  name                     = "isic2019sa${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier            = "Standard"
  account_replication_type = "LRS"
  account_kind            = "StorageV2"
  min_tls_version = "TLS1_2"
  allow_nested_items_to_be_public = true
}

resource "azurerm_storage_container" "container" {
  name                  = var.container_name
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "blob"
}