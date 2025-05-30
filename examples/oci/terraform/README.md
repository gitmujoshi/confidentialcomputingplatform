# OCI Confidential Computing Terraform Example

This directory contains Terraform configurations for deploying a confidential computing environment on Oracle Cloud Infrastructure (OCI) using AMD SEV.

## Prerequisites

- OCI account with appropriate permissions
- OCI CLI installed and configured
- Terraform installed (version 0.12 or later)

## Configuration

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Update the variables in `terraform.tfvars` with your OCI-specific values:
   - `tenancy_ocid`: OCID of your tenancy
   - `user_ocid`: OCID of the user calling the API
   - `fingerprint`: Fingerprint for the key pair
   - `private_key_path`: Path to your private key file
   - `region`: OCI region to deploy resources
   - `compartment_id`: OCID of your compartment
   - `image_id`: OCID of the image to use
   - `ssh_public_key`: Your SSH public key
   - `bucket_name`: Name for your Object Storage bucket

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
- Confidential VM Instance (AMD SEV enabled)
- Vault and Key
- Object Storage Bucket

## Security Features

- Uses OCI Confidential Computing with AMD SEV
- Encrypted boot volumes
- Secure networking with security lists
- Vault for key management
- Object Storage with KMS encryption
- No public access to storage bucket
- Versioning enabled for storage

## Cleanup

To destroy all created resources:
```bash
terraform destroy
```

## References

- [OCI Confidential Computing Documentation](https://docs.oracle.com/en-us/iaas/Content/security/confidential-computing.htm)
- [OCI Terraform Provider Documentation](https://registry.terraform.io/providers/oracle/oci/latest/docs)
- [OCI Security Best Practices](https://docs.oracle.com/en-us/iaas/Content/Security/Reference/security_best_practices.htm) 