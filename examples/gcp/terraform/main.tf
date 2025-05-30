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
resource "google_compute_network" "confidential_vpc" {
  name                    = "confidential-vpc"
  auto_create_subnetworks = false
}

# Subnet
resource "google_compute_subnetwork" "confidential_subnet" {
  name          = "confidential-subnet"
  ip_cidr_range = "10.0.0.0/24"
  network       = google_compute_network.confidential_vpc.id
  region        = var.region
}

# Firewall Rules
resource "google_compute_firewall" "confidential_fw" {
  name    = "confidential-fw"
  network = google_compute_network.confidential_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["confidential-vm"]
}

# Confidential VM Instance
resource "google_compute_instance" "confidential_vm" {
  name         = "confidential-vm"
  machine_type = "n2d-standard-2"  # Supports AMD SEV
  zone         = "${var.region}-a"

  tags = ["confidential-vm"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2004-lts"
      size  = 50
    }
    kms_key_self_link = google_kms_crypto_key.vm_key.id
  }

  network_interface {
    subnetwork = google_compute_subnetwork.confidential_subnet.name
    access_config {
      // Ephemeral public IP
    }
  }

  confidential_instance_config {
    enable_confidential_compute = true
  }

  metadata = {
    ssh-keys = "${var.admin_username}:${var.ssh_public_key}"
  }
}

# KMS Key Ring
resource "google_kms_key_ring" "confidential_keyring" {
  name     = "confidential-keyring"
  location = var.region
}

# KMS Crypto Key
resource "google_kms_crypto_key" "vm_key" {
  name     = "vm-key"
  key_ring = google_kms_key_ring.confidential_keyring.id

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }

  rotation_period = "7776000s"  # 90 days
}

# Cloud Storage Bucket
resource "google_storage_bucket" "confidential_bucket" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.bucket_key.id
  }
}

# KMS Crypto Key for Bucket
resource "google_kms_crypto_key" "bucket_key" {
  name     = "bucket-key"
  key_ring = google_kms_key_ring.confidential_keyring.id

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }

  rotation_period = "7776000s"  # 90 days
}

# IAM Service Account
resource "google_service_account" "confidential_sa" {
  account_id   = "confidential-sa"
  display_name = "Confidential Computing Service Account"
}

# IAM Policy Binding
resource "google_project_iam_member" "confidential_sa_binding" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.confidential_sa.email}"
}

# Cloud KMS IAM Binding
resource "google_kms_crypto_key_iam_member" "crypto_key_binding" {
  crypto_key_id = google_kms_crypto_key.vm_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.confidential_sa.email}"
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
} 