terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.0.0"
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "3.14.0"

  name = "secure-training-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = var.tags
}

# KMS Key for encryption
resource "aws_kms_key" "training_key" {
  description             = "KMS key for secure training data encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = var.tags
}

resource "aws_kms_alias" "training_key_alias" {
  name          = "alias/secure-training-key"
  target_key_id = aws_kms_key.training_key.key_id
}

# S3 Bucket for training data
resource "aws_s3_bucket" "training_data" {
  bucket = var.bucket_name

  tags = var.tags
}

resource "aws_s3_bucket_versioning" "training_data" {
  bucket = aws_s3_bucket.training_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training_data" {
  bucket = aws_s3_bucket.training_data.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.training_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "secure-training-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Policy for SageMaker
resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "secure-training-sagemaker-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.training_data.arn,
          "${aws_s3_bucket.training_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.training_key.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Security Group for SageMaker
resource "aws_security_group" "sagemaker_sg" {
  name        = "secure-training-sagemaker-sg"
  description = "Security group for SageMaker training jobs"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

# VPC
resource "aws_vpc" "confidential_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "confidential-vpc"
  }
}

# Subnet
resource "aws_subnet" "confidential_subnet" {
  vpc_id            = aws_vpc.confidential_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"

  tags = {
    Name = "confidential-subnet"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "confidential_igw" {
  vpc_id = aws_vpc.confidential_vpc.id

  tags = {
    Name = "confidential-igw"
  }
}

# Route Table
resource "aws_route_table" "confidential_rt" {
  vpc_id = aws_vpc.confidential_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.confidential_igw.id
  }

  tags = {
    Name = "confidential-rt"
  }
}

# Security Group
resource "aws_security_group" "confidential_sg" {
  name        = "confidential-sg"
  description = "Security group for confidential computing"
  vpc_id      = aws_vpc.confidential_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "confidential-sg"
  }
}

# EC2 Instance (Nitro Enclaves enabled)
resource "aws_instance" "confidential_instance" {
  ami           = var.ami_id
  instance_type = "c5.2xlarge"  # Supports Nitro Enclaves
  subnet_id     = aws_subnet.confidential_subnet.id

  vpc_security_group_ids = [aws_security_group.confidential_sg.id]
  key_name              = var.key_name

  root_block_device {
    volume_size = 50
    encrypted   = true
  }

  tags = {
    Name = "confidential-instance"
  }
}

# KMS Key
resource "aws_kms_key" "confidential_key" {
  description             = "KMS key for confidential computing"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "confidential-key"
  }
}

# S3 Bucket
resource "aws_s3_bucket" "confidential_bucket" {
  bucket = var.bucket_name

  tags = {
    Name = "confidential-bucket"
  }
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "confidential_bucket_encryption" {
  bucket = aws_s3_bucket.confidential_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.confidential_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "confidential_bucket_versioning" {
  bucket = aws_s3_bucket.confidential_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# IAM Role for Nitro Enclaves
resource "aws_iam_role" "enclave_role" {
  name = "enclave-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for Enclave
resource "aws_iam_policy" "enclave_policy" {
  name        = "enclave-policy"
  description = "Policy for Nitro Enclaves"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# Attach Policy to Role
resource "aws_iam_role_policy_attachment" "enclave_policy_attachment" {
  role       = aws_iam_role.enclave_role.name
  policy_arn = aws_iam_policy.enclave_policy.arn
} 