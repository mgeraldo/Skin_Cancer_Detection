output "storage_account_name" {
  value = azurerm_storage_account.storage.name
}

output "blob_container_url" {
  value = "https://${azurerm_storage_account.storage.name}.blob.core.windows.net/${azurerm_storage_container.container.name}"
}