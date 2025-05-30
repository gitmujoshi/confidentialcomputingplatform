#!/usr/bin/env python3

import os
import sys
import json
import logging
import oci
from oci.core import ComputeClient
from oci.object_storage import ObjectStorageClient
from oci.key_management import KmsCryptoClient
from oci.identity import IdentityClient
from oci.vault import VaultsClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCISecurityVerifier:
    def __init__(self, config_path: str = None):
        """Initialize OCI security verification."""
        self.config = oci.config.from_file(config_path) if config_path else oci.config.from_file()
        self.compute_client = ComputeClient(self.config)
        self.object_storage_client = ObjectStorageClient(self.config)
        self.kms_client = KmsCryptoClient(self.config)
        self.identity_client = IdentityClient(self.config)
        self.vault_client = VaultsClient(self.config)

    def verify_instance_security(self, instance_id: str) -> dict:
        """Verify security configuration of a compute instance."""
        try:
            # Get instance details
            instance = self.compute_client.get_instance(instance_id).data

            # Check confidential computing features
            confidential_computing = {
                'enabled': instance.is_confidential_compute,
                'shape': instance.shape,
                'trusted_platform_module': instance.is_trusted_platform_module_enabled,
                'secure_boot': instance.is_secure_boot_enabled,
                'measured_boot': instance.is_measured_boot_enabled
            }

            # Check network security
            network_security = {
                'vcn_id': instance.vnic_id,
                'subnet_id': instance.subnet_id,
                'private_ip': instance.private_ip,
                'public_ip': instance.public_ip
            }

            return {
                'instance_id': instance_id,
                'confidential_computing': confidential_computing,
                'network_security': network_security
            }
        except Exception as e:
            logger.error(f"Failed to verify instance security: {e}")
            raise

    def verify_storage_security(self, namespace: str, bucket_name: str) -> dict:
        """Verify security configuration of an Object Storage bucket."""
        try:
            # Get bucket details
            bucket = self.object_storage_client.get_bucket(
                namespace_name=namespace,
                bucket_name=bucket_name
            ).data

            # Check encryption
            encryption = {
                'kms_key_id': bucket.kms_key_id,
                'object_events_enabled': bucket.object_events_enabled,
                'versioning': bucket.versioning
            }

            # Check access control
            access_control = {
                'public_access_type': bucket.public_access_type,
                'object_events_enabled': bucket.object_events_enabled,
                'metadata': bucket.metadata
            }

            return {
                'namespace': namespace,
                'bucket_name': bucket_name,
                'encryption': encryption,
                'access_control': access_control
            }
        except Exception as e:
            logger.error(f"Failed to verify storage security: {e}")
            raise

    def verify_kms_security(self, vault_id: str) -> dict:
        """Verify security configuration of Vault keys."""
        try:
            # Get vault details
            vault = self.vault_client.get_vault(vault_id).data

            # List keys in the vault
            keys = []
            for key in self.vault_client.list_keys(vault_id).data:
                key_info = {
                    'id': key.id,
                    'algorithm': key.algorithm,
                    'protection_mode': key.protection_mode,
                    'lifecycle_state': key.lifecycle_state
                }
                keys.append(key_info)

            return {
                'vault_id': vault_id,
                'vault_type': vault.vault_type,
                'keys': keys
            }
        except Exception as e:
            logger.error(f"Failed to verify KMS security: {e}")
            raise

    def verify_iam_security(self, compartment_id: str) -> dict:
        """Verify IAM security settings."""
        try:
            # Get compartment details
            compartment = self.identity_client.get_compartment(compartment_id).data

            # List users
            users = []
            for user in self.identity_client.list_users(compartment_id).data:
                user_info = {
                    'id': user.id,
                    'name': user.name,
                    'email': user.email,
                    'is_mfa_activated': user.is_mfa_activated
                }
                users.append(user_info)

            # List groups
            groups = []
            for group in self.identity_client.list_groups(compartment_id).data:
                group_info = {
                    'id': group.id,
                    'name': group.name,
                    'description': group.description
                }
                groups.append(group_info)

            return {
                'compartment_id': compartment_id,
                'compartment_name': compartment.name,
                'users': users,
                'groups': groups
            }
        except Exception as e:
            logger.error(f"Failed to verify IAM security: {e}")
            raise

    def verify_security_center(self, compartment_id: str) -> dict:
        """Verify Security Center findings."""
        try:
            # Get security assessments
            security_client = oci.cloud_guard.CloudGuardClient(self.config)
            assessments = security_client.list_security_assessments(
                compartment_id=compartment_id
            ).data

            # Process findings
            findings = []
            for assessment in assessments:
                finding_info = {
                    'id': assessment.id,
                    'display_name': assessment.display_name,
                    'severity': assessment.severity,
                    'status': assessment.status
                }
                findings.append(finding_info)

            return {
                'compartment_id': compartment_id,
                'findings': findings
            }
        except Exception as e:
            logger.error(f"Failed to verify Security Center: {e}")
            raise

def main():
    """Main function to verify OCI security."""
    try:
        # Initialize verifier
        verifier = OCISecurityVerifier()

        # Verify instance security
        instance_result = verifier.verify_instance_security(
            'your-instance-id'
        )
        logger.info(f"Instance security verification result: {json.dumps(instance_result, indent=2)}")

        # Verify storage security
        storage_result = verifier.verify_storage_security(
            'your-namespace',
            'your-bucket-name'
        )
        logger.info(f"Storage security verification result: {json.dumps(storage_result, indent=2)}")

        # Verify KMS security
        kms_result = verifier.verify_kms_security(
            'your-vault-id'
        )
        logger.info(f"KMS security verification result: {json.dumps(kms_result, indent=2)}")

        # Verify IAM security
        iam_result = verifier.verify_iam_security(
            'your-compartment-id'
        )
        logger.info(f"IAM security verification result: {json.dumps(iam_result, indent=2)}")

        # Verify Security Center
        security_center_result = verifier.verify_security_center(
            'your-compartment-id'
        )
        logger.info(f"Security Center verification result: {json.dumps(security_center_result, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to verify OCI security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 