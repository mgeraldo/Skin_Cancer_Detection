# ISIC 2019 w281 Project SkinVision

## Changelog

### Version 1.0.0
- Initial release of the project.
- Provisioned Azure resources for the ISIC 2019 dataset using Terraform.
- Created a resource group, a storage account with a unique name, and a public blob container for storing image data.

### Version 1.1.0
- Updated the `variables.tf` file to include additional customization options for resource names.
- Enhanced documentation in the `terraform/README.md` to provide clearer setup instructions and cleanup steps.

### Version 1.2.0
- Added output variables in `outputs.tf` for better visibility of the storage account name and blob container URL after deployment.
- Improved the Azure provider configuration in `provider.tf` to streamline authentication processes.

### Version 1.3.0
- Added comments to clarify the purpose of the Terraform configuration for the Azure Storage Account.
- Enhanced the documentation in the Terraform files for better understanding.

### Future Updates
- Plans to integrate automated testing for Terraform configurations.
- Potential enhancements to support additional Azure resources based on user feedback.

## Project Overview

This project provisions Azure resources for the ISIC 2019 dataset using Terraform. It includes detailed documentation and structured files to facilitate easy customization and deployment.