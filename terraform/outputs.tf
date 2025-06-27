output "storage_account_name" {
  value = azurerm_storage_account.storage.name
}

output "isic_blob_container_url" {
  value = "https://${azurerm_storage_account.storage.name}.blob.core.windows.net/${var.container_names[0]}"
}

output "ham_blob_container_url" {
  value = "https://${azurerm_storage_account.storage.name}.blob.core.windows.net/${var.container_names[1]}"
}

output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}