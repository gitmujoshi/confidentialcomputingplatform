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

# VCN Configuration
resource "oci_core_vcn" "confidential_vcn" {
  compartment_id = var.compartment_id
  display_name   = "confidential-vcn"
  cidr_block     = "10.0.0.0/16"
  dns_label      = "confidentialvcn"
}

# Subnet Configuration
resource "oci_core_subnet" "confidential_subnet" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.confidential_vcn.id
  display_name   = "confidential-subnet"
  cidr_block     = "10.0.1.0/24"
  dns_label      = "confidentialsubnet"
  security_list_ids = [oci_core_security_list.confidential_security_list.id]
  route_table_id    = oci_core_route_table.confidential_route_table.id
}

# Security List Configuration
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

# Route Table Configuration
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

# Internet Gateway Configuration
resource "oci_core_internet_gateway" "confidential_ig" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.confidential_vcn.id
  display_name   = "confidential-ig"
  enabled        = true
}

# Confidential Computing Instance Configuration
resource "oci_core_instance" "confidential_instance" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_id
  display_name        = "confidential-instance"
  shape              = "VM.Standard.E4.Flex"  # Shape that supports confidential computing

  shape_config {
    ocpus         = 1
    memory_in_gbs = 16
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
}

# Vault Configuration for Key Management
resource "oci_kms_vault" "confidential_vault" {
  compartment_id = var.compartment_id
  display_name   = "confidential-vault"
  vault_type     = "DEFAULT"
}

# Data source for availability domains
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
} 