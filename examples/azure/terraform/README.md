# Azure Confidential Computing Terraform Example

This directory contains Terraform configurations for deploying a confidential computing environment on Microsoft Azure.

## Prerequisites

- Azure subscription with appropriate permissions
- Azure CLI installed and configured
- Terraform installed (version 0.12 or later)

## Configuration

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Update the variables in `terraform.tfvars` with your Azure-specific values:
   - `resource_group_name`: Name for your resource group
   - `location`: Azure region
   - `admin_username`: VM admin username
   - `ssh_public_key`: Your SSH public key
   - `tenant_id`: Your Azure tenant ID
   - `object_id`: Object ID for Key Vault access
   - `key_vault_name`: Name for your Key Vault
   - `attestation_provider_name`: Name for Attestation Provider
   - `storage_account_name`: Name for Storage Account

## Deployment

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Review the planned changes:
   ```bash
   terraform plan
   ```

3. Apply the configuration:
   ```bash
   terraform apply
   ```

## Resources Created

- Resource Group
- Virtual Network and Subnet
- Network Security Group
- Network Interface
- Public IP
- Confidential VM (DCsv2-series)
- Key Vault
- Attestation Provider
- Storage Account and Container

## Security Features

- Uses Azure DCsv2-series VMs with Intel SGX support
- Secure networking with NSG rules
- Key Vault for secure key management
- Attestation Provider for secure enclave verification
- Private storage container for data
- SSH access only through port 22

## Cleanup

To destroy all created resources:
```bash
terraform destroy
```

## References

- [Azure Confidential Computing Documentation](https://docs.microsoft.com/en-us/azure/confidential-computing/)
- [Azure Terraform Provider Documentation](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Azure Security Best Practices](https://docs.microsoft.com/en-us/azure/security/fundamentals/best-practices-and-patterns) 