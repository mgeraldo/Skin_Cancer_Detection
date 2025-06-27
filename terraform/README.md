# ISIC 2019 Terraform Project

This project provisions Azure resources for the ISIC 2019 dataset using Terraform. It creates a resource group, a storage account (with a unique, compliant name), and a public blob container for storing image data.

## Project Structure

The project consists of the following files in the `terraform/` directory:

- **main.tf**: Main configuration defining the resource group, storage account, and blob container.
- **variables.tf**: Input variables for customization, including resource group name, storage account name (auto-generated and <24 chars), and container name.
- **outputs.tf**: Outputs for the storage account name and blob container URL.
- **provider.tf**: Azure provider configuration and authentication details.
- **README.md**: This documentation.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) installed on your machine.
- An Azure account with sufficient permissions to create resources.

## Setup Instructions

1. Change to the Terraform directory:
   ```
   cd terraform
   ```

2. (Optional) Update the `variables.tf` file if you want to override default values for the resource group name, storage account name, or container name. The storage account name is auto-generated and meets Azure's naming requirements.

3. Initialize Terraform:
   ```
   terraform init
   ```

4. Plan the deployment:
   ```
   terraform plan
   ```

5. Apply the configuration to create the resources:
   ```
   terraform apply
   ```

6. After deployment, check the outputs for the storage account name and blob container URL.

## Cleanup

To remove the resources created by this project, run:
```
terraform destroy
```

This will delete all resources defined in the Terraform configuration.