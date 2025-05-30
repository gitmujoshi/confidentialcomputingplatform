# AWS Confidential Computing Terraform Example

This directory contains Terraform configurations for deploying a confidential computing environment on Amazon Web Services (AWS) using Nitro Enclaves.

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured
- Terraform installed (version 0.12 or later)

## Configuration

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Update the variables in `terraform.tfvars` with your AWS-specific values:
   - `aws_region`: AWS region to deploy resources
   - `ami_id`: AMI ID for the EC2 instance (must support Nitro Enclaves)
   - `key_name`: Name of your SSH key pair
   - `bucket_name`: Name for your S3 bucket

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

- VPC with Internet Gateway
- Subnet
- Security Group
- EC2 Instance (Nitro Enclaves enabled)
- KMS Key
- S3 Bucket with encryption
- IAM Role and Policy for Enclaves

## Security Features

- Uses AWS Nitro Enclaves for confidential computing
- Encrypted EBS volumes
- KMS encryption for S3 bucket
- Secure networking with security groups
- IAM roles and policies for least privilege access
- S3 bucket versioning enabled

## Cleanup

To destroy all created resources:
```bash
terraform destroy
```

## References

- [AWS Nitro Enclaves Documentation](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave.html)
- [AWS Terraform Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS Security Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html) 