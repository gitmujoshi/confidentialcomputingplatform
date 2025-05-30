# OCI Confidential Computing Terraform Example

This directory contains Terraform configurations for deploying a confidential computing environment on Oracle Cloud Infrastructure (OCI).

## Prerequisites

- OCI account with appropriate permissions
- OCI CLI configured with API key
- Terraform installed (version 0.12 or later)

## Configuration

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Update the variables in `terraform.tfvars` with your OCI-specific values:
   - `tenancy_ocid`: Your OCI tenancy OCID
   - `user_ocid`: Your OCI user OCID
   - `fingerprint`: Your API key fingerprint
   - `private_key_path`: Path to your OCI API private key
   - `compartment_id`: OCID of your compartment
   - `image_id`: OCID of the image to use
   - `ssh_public_key`: Your SSH public key for instance access

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

- Virtual Cloud Network (VCN)
- Subnet
- Security List
- Route Table
- Internet Gateway
- Confidential Computing Instance
- KMS Vault

## Outputs

After successful deployment, Terraform will output:
- VCN ID
- Subnet ID
- Instance ID
- Instance Public IP
- Vault ID

## Cleanup

To destroy all created resources:
```bash
terraform destroy
```

## Security Notes

- The configuration uses OCI's confidential computing capabilities with the `VM.Standard.E4.Flex` shape
- SSH access is restricted to port 22
- KMS vault is created for secure key management
- All resources are created in a dedicated VCN with proper security rules

## References

- [OCI Confidential Computing Documentation](https://docs.oracle.com/en-us/iaas/Content/confidential-computing/home.htm)
- [OCI Terraform Provider Documentation](https://registry.terraform.io/providers/oracle/oci/latest/docs)
- [OCI Security Best Practices](https://docs.oracle.com/en-us/iaas/Content/Security/Reference/security_best_practices.htm) 