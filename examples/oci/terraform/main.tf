terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 4.0"
    }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# VCN
resource "oci_core_vcn" "confidential_vcn" {
  cidr_block     = "10.0.0.0/16"
  compartment_id = var.compartment_id
  display_name   = "confidential-vcn"
  dns_label      = "confidential"
}

# Subnet
resource "oci_core_subnet" "confidential_subnet" {
  cidr_block        = "10.0.1.0/24"
  compartment_id    = var.compartment_id
  vcn_id            = oci_core_vcn.confidential_vcn.id
  display_name      = "confidential-subnet"
  dns_label         = "confidential"
  security_list_ids = [oci_core_security_list.confidential_security_list.id]
  route_table_id    = oci_core_route_table.confidential_route_table.id
}

# Security List
resource "oci_core_security_list" "confidential_security_list" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.confidential_vcn.id
  display_name   = "confidential-security-list"

  egress_security_rules {
    destination      = "0.0.0.0/0"
    protocol         = "all"
    description      = "Allow all outbound traffic"
  }

  ingress_security_rules {
    protocol    = "6"  # TCP
    source      = "0.0.0.0/0"
    description = "Allow SSH access"

    tcp_options {
      min = 22
      max = 22
    }
  }
}

# Route Table
resource "oci_core_route_table" "confidential_route_table" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.confidential_vcn.id
  display_name   = "confidential-route-table"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.confidential_ig.id
  }
}

# Internet Gateway
resource "oci_core_internet_gateway" "confidential_ig" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.confidential_vcn.id
  display_name   = "confidential-ig"
  enabled        = true
}

# Confidential VM Instance
resource "oci_core_instance" "confidential_instance" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_id
  display_name        = "confidential-vm"
  shape               = "VM.Standard.E4.Flex"  # Supports AMD SEV

  shape_config {
    ocpus         = 2
    memory_in_gbs = 32
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.confidential_subnet.id
    display_name     = "confidential-vnic"
    assign_public_ip = true
  }

  source_details {
    source_type = "image"
    source_id   = var.image_id
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
  }

  is_pv_encryption_in_transit_enabled = true
}

# Vault
resource "oci_kms_vault" "confidential_vault" {
  compartment_id = var.compartment_id
  display_name   = "confidential-vault"
  vault_type     = "DEFAULT"
}

# Key
resource "oci_kms_key" "confidential_key" {
  compartment_id = var.compartment_id
  display_name   = "confidential-key"
  key_shape {
    algorithm = "AES"
    length    = 32
  }
  management_endpoint = oci_kms_vault.confidential_vault.management_endpoint
  protection_mode     = "SOFTWARE"
}

# Object Storage Bucket
resource "oci_objectstorage_bucket" "confidential_bucket" {
  compartment_id = var.compartment_id
  name           = var.bucket_name
  namespace      = data.oci_objectstorage_namespace.ns.namespace
  access_type    = "NoPublicAccess"
  versioning     = "Enabled"
  kms_key_id     = oci_kms_key.confidential_key.id
}

# Data Sources
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_id
} 