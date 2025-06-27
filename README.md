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

### Version 1.4.0
- Added dataset files `ham_with_diagnosis.csv` and `isic_with_ground_truth.csv` to the project.
- Updated `.gitignore` to exclude these dataset files from version control.

### Future Updates
- Plans to integrate automated testing for Terraform configurations.
- Potential enhancements to support additional Azure resources based on user feedback.

## Project Overview

This project provisions Azure resources for the ISIC 2019 dataset using Terraform. It includes detailed documentation and structured files to facilitate easy customization and deployment. 

### Datasets
The project now includes two datasets:
- **ISIC 2019 Dataset**: Contains images and their corresponding ground truth diagnoses.
- **HAM10000 Dataset**: Contains images with associated diagnoses for skin lesions.
```

### Summary of Changes
1. **Changelog**: Added a new version (1.4.0) with details about the new dataset files and updates to `.gitignore`.
2. **Project Overview**: Added a new section to describe the datasets included in the project.

## Project Overview

:TODO