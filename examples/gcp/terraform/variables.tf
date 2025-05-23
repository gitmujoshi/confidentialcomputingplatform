variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region to deploy resources"
  type        = string
  default     = "us-central1"
}

variable "private_subnet_cidr" {
  description = "CIDR block for private subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "bucket_name" {
  description = "Name of the Cloud Storage bucket for training data"
  type        = string
  default     = "secure-training-data"
}

variable "access_policy_id" {
  description = "Access Policy ID for VPC Service Controls"
  type        = string
}

variable "blacklisted_ips" {
  description = "List of IP addresses to block"
  type        = list(string)
  default     = []
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    environment = "production"
    project     = "secure-training"
    managed-by  = "terraform"
  }
} 