# AWS Secure Training Infrastructure

This directory contains Terraform configurations for deploying secure training infrastructure on AWS.

## Prerequisites

1. AWS CLI installed and configured
2. Terraform installed (version >= 1.0.0)
3. AWS credentials configured with appropriate permissions

## Infrastructure Components

The Terraform configuration creates the following resources:

1. VPC with public and private subnets
2. KMS key for data encryption
3. S3 bucket for training data with encryption
4. IAM role and policy for SageMaker
5. Security group for SageMaker training jobs

## Usage

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

4. To destroy the infrastructure:
   ```bash
   terraform destroy
   ```

## Configuration

The infrastructure can be configured by modifying the variables in `variables.tf` or by providing a `terraform.tfvars` file with the following variables:

- `aws_region`: AWS region to deploy resources
- `vpc_cidr`: CIDR block for VPC
- `availability_zones`: List of availability zones
- `private_subnet_cidrs`: CIDR blocks for private subnets
- `public_subnet_cidrs`: CIDR blocks for public subnets
- `bucket_name`: Name of the S3 bucket for training data
- `tags`: Tags to apply to all resources

## Security Features

- VPC with public and private subnets
- KMS encryption for data at rest
- S3 bucket encryption
- IAM role-based access control
- Security group for network isolation

## Outputs

The configuration outputs the following values:

- VPC ID
- Subnet IDs
- KMS key ARN
- S3 bucket name
- SageMaker execution role ARN
- Security group ID

## Cleanup

To clean up all resources:
```bash
terraform destroy
```

Note: This will delete all resources created by this Terraform configuration. 