variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "admin_username" {
  description = "Username for the VM"
  type        = string
  default     = "gcpuser"
}

variable "ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
}

variable "bucket_name" {
  description = "Name of the Cloud Storage bucket"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {
    Environment = "Confidential Computing"
    Project     = "Secure Multi-Party Data Collaboration"
  }
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