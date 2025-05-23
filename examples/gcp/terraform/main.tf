terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.0.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "training_network" {
  name                    = "secure-training-network"
  auto_create_subnetworks = false
}

# Subnets
resource "google_compute_subnetwork" "private_subnet" {
  name          = "secure-training-private-subnet"
  ip_cidr_range = var.private_subnet_cidr
  network       = google_compute_network.training_network.id
  region        = var.region

  private_ip_google_access = true
}

resource "google_compute_subnetwork" "public_subnet" {
  name          = "secure-training-public-subnet"
  ip_cidr_range = var.public_subnet_cidr
  network       = google_compute_network.training_network.id
  region        = var.region
}

# Cloud KMS Key Ring and Key
resource "google_kms_key_ring" "training_keyring" {
  name     = "secure-training-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "training_key" {
  name     = "secure-training-key"
  key_ring = google_kms_key_ring.training_keyring.id

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }

  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }
}

# Cloud Storage Bucket
resource "google_storage_bucket" "training_data" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.training_key.id
  }
}

# Service Account for Vertex AI
resource "google_service_account" "vertex_ai_sa" {
  account_id   = "secure-training-vertex-ai"
  display_name = "Service Account for Secure Training"
}

# IAM Policy for Service Account
resource "google_project_iam_member" "vertex_ai_sa_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

resource "google_project_iam_member" "vertex_ai_sa_kms" {
  project = var.project_id
  role    = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

resource "google_project_iam_member" "vertex_ai_sa_aiplatform" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

# VPC Service Controls
resource "google_access_context_manager_service_perimeter" "training_perimeter" {
  parent = "accessPolicies/${var.access_policy_id}"
  name   = "accessPolicies/${var.access_policy_id}/servicePerimeters/secure-training-perimeter"
  title  = "Secure Training Perimeter"

  status {
    restricted_services = [
      "storage.googleapis.com",
      "aiplatform.googleapis.com",
      "cloudkms.googleapis.com"
    ]

    resources = [
      "projects/${var.project_id}"
    ]

    access_levels = [
      "accessPolicies/${var.access_policy_id}/accessLevels/secure-training-level"
    ]
  }
}

# Cloud Armor Security Policy
resource "google_compute_security_policy" "training_policy" {
  name = "secure-training-policy"

  rule {
    action      = "deny(403)"
    priority    = "1000"
    description = "Deny access to IPs in blacklist"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = var.blacklisted_ips
      }
    }
  }

  rule {
    action      = "allow"
    priority    = "2000"
    description = "Default rule, higher priority overrides it"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }
} 