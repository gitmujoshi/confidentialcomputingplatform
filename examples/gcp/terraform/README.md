# GCP Confidential Computing Terraform Example

This directory contains Terraform configurations for deploying a confidential computing environment on Google Cloud Platform (GCP) using AMD SEV.

## Prerequisites

- GCP project with billing enabled
- GCP CLI installed and configured
- Terraform installed (version 0.12 or later)

## Configuration

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Update the variables in `terraform.tfvars` with your GCP-specific values:
   - `project_id`: Your GCP project ID
   - `region`: GCP region to deploy resources
   - `admin_username`: VM admin username
   - `ssh_public_key`: Your SSH public key
   - `bucket_name`: Name for your Cloud Storage bucket

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

- VPC Network
- Subnet
- Firewall Rules
- Confidential VM Instance (AMD SEV enabled)
- KMS Key Ring and Keys
- Cloud Storage Bucket
- IAM Service Account and Policies

## Security Features

- Uses GCP Confidential Computing with AMD SEV
- Encrypted boot disks using Cloud KMS
- Secure networking with firewall rules
- Cloud KMS encryption for storage
- IAM service account with least privilege
- Cloud Storage bucket with uniform access control

## Cleanup

To destroy all created resources:
```bash
terraform destroy
```

## References

- [GCP Confidential Computing Documentation](https://cloud.google.com/confidential-computing)
- [GCP Terraform Provider Documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices) 