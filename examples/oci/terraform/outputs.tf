output "vcn_id" {
  description = "OCID of the VCN"
  value       = oci_core_vcn.confidential_vcn.id
}

output "subnet_id" {
  description = "OCID of the subnet"
  value       = oci_core_subnet.confidential_subnet.id
}

output "instance_id" {
  description = "OCID of the instance"
  value       = oci_core_instance.confidential_instance.id
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = oci_core_instance.confidential_instance.public_ip
}

output "vault_id" {
  description = "OCID of the KMS vault"
  value       = oci_kms_vault.confidential_vault.id
} 