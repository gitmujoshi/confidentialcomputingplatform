# GCP Secure Training Infrastructure

This directory contains Terraform configurations for deploying secure training infrastructure on Google Cloud Platform.

## Prerequisites

1. Google Cloud SDK installed and configured
2. Terraform installed (version >= 1.0.0)
3. GCP project with billing enabled
4. Appropriate IAM permissions

## Infrastructure Components

The Terraform configuration creates the following resources:

1. VPC network with public and private subnets
2. Cloud KMS key ring and key for encryption
3. Cloud Storage bucket for training data
4. Service account for Vertex AI
5. VPC Service Controls
6. Cloud Armor security policy

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

- `project_id`: GCP project ID
- `region`: GCP region to deploy resources
- `private_subnet_cidr`: CIDR block for private subnet
- `public_subnet_cidr`: CIDR block for public subnet
- `bucket_name`: Name of the Cloud Storage bucket
- `access_policy_id`: Access Policy ID for VPC Service Controls
- `blacklisted_ips`: List of IP addresses to block
- `labels`: Labels to apply to all resources

## Security Features

- VPC network with public and private subnets
- Cloud KMS encryption for data at rest
- Cloud Storage bucket encryption
- Service account-based access control
- VPC Service Controls for API access
- Cloud Armor security policy
- Confidential computing support

## Outputs

The configuration outputs the following values:

- VPC network ID
- Subnet IDs
- KMS key name
- Cloud Storage bucket name
- Service account email
- VPC Service Controls perimeter name
- Security policy name

## Cleanup

To clean up all resources:
```bash
terraform destroy
```

Note: This will delete all resources created by this Terraform configuration.

## Additional Security Considerations

1. Enable Cloud Audit Logs for all services
2. Configure Cloud Monitoring alerts
3. Set up Cloud Security Command Center
4. Implement regular key rotation
5. Use Cloud IAM conditions for fine-grained access control 