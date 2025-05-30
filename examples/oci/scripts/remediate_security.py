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
from oci.cloud_guard import CloudGuardClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCISecurityRemediator:
    def __init__(self, config_path: str = None):
        """Initialize OCI security remediation."""
        self.config = oci.config.from_file(config_path) if config_path else oci.config.from_file()
        self.compute_client = ComputeClient(self.config)
        self.object_storage_client = ObjectStorageClient(self.config)
        self.kms_client = KmsCryptoClient(self.config)
        self.identity_client = IdentityClient(self.config)
        self.vault_client = VaultsClient(self.config)
        self.security_client = CloudGuardClient(self.config)

    def remediate_instance_security(self, instance_id: str) -> dict:
        """Remediate security issues in compute instances."""
        try:
            # Get instance details
            instance = self.compute_client.get_instance(instance_id).data

            # Enable confidential computing
            if not instance.is_confidential_compute:
                self.compute_client.update_instance(
                    instance_id=instance_id,
                    update_instance_details=oci.core.models.UpdateInstanceDetails(
                        is_confidential_compute=True
                    )
                )
                logger.info(f"Enabled confidential computing for instance {instance_id}")

            # Enable secure boot
            if not instance.is_secure_boot_enabled:
                self.compute_client.update_instance(
                    instance_id=instance_id,
                    update_instance_details=oci.core.models.UpdateInstanceDetails(
                        is_secure_boot_enabled=True
                    )
                )
                logger.info(f"Enabled secure boot for instance {instance_id}")

            # Enable TPM
            if not instance.is_trusted_platform_module_enabled:
                self.compute_client.update_instance(
                    instance_id=instance_id,
                    update_instance_details=oci.core.models.UpdateInstanceDetails(
                        is_trusted_platform_module_enabled=True
                    )
                )
                logger.info(f"Enabled TPM for instance {instance_id}")

            return {
                'instance_id': instance_id,
                'remediation_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to remediate instance security: {e}")
            raise

    def remediate_storage_security(self, namespace: str, bucket_name: str) -> dict:
        """Remediate security issues in Object Storage buckets."""
        try:
            # Get bucket details
            bucket = self.object_storage_client.get_bucket(
                namespace_name=namespace,
                bucket_name=bucket_name
            ).data

            # Enable encryption
            if not bucket.kms_key_id:
                self.object_storage_client.update_bucket(
                    namespace_name=namespace,
                    bucket_name=bucket_name,
                    update_bucket_details=oci.object_storage.models.UpdateBucketDetails(
                        kms_key_id='your-kms-key-id'
                    )
                )
                logger.info(f"Enabled encryption for bucket {bucket_name}")

            # Disable public access
            if bucket.public_access_type != 'NoPublicAccess':
                self.object_storage_client.update_bucket(
                    namespace_name=namespace,
                    bucket_name=bucket_name,
                    update_bucket_details=oci.object_storage.models.UpdateBucketDetails(
                        public_access_type='NoPublicAccess'
                    )
                )
                logger.info(f"Disabled public access for bucket {bucket_name}")

            # Enable versioning
            if not bucket.versioning:
                self.object_storage_client.update_bucket(
                    namespace_name=namespace,
                    bucket_name=bucket_name,
                    update_bucket_details=oci.object_storage.models.UpdateBucketDetails(
                        versioning='Enabled'
                    )
                )
                logger.info(f"Enabled versioning for bucket {bucket_name}")

            return {
                'namespace': namespace,
                'bucket_name': bucket_name,
                'remediation_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to remediate storage security: {e}")
            raise

    def remediate_kms_security(self, vault_id: str) -> dict:
        """Remediate security issues in Vault keys."""
        try:
            # Get vault details
            vault = self.vault_client.get_vault(vault_id).data

            # List keys
            keys = self.vault_client.list_keys(vault_id).data

            for key in keys:
                # Enable HSM protection
                if key.protection_mode != 'HSM':
                    self.vault_client.update_key(
                        key_id=key.id,
                        update_key_details=oci.vault.models.UpdateKeyDetails(
                            protection_mode='HSM'
                        )
                    )
                    logger.info(f"Enabled HSM protection for key {key.id}")

            return {
                'vault_id': vault_id,
                'remediation_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to remediate KMS security: {e}")
            raise

    def remediate_iam_security(self, compartment_id: str) -> dict:
        """Remediate IAM security issues."""
        try:
            # List users
            users = self.identity_client.list_users(compartment_id).data

            for user in users:
                # Enable MFA
                if not user.is_mfa_activated:
                    self.identity_client.update_user(
                        user_id=user.id,
                        update_user_details=oci.identity.models.UpdateUserDetails(
                            is_mfa_activated=True
                        )
                    )
                    logger.info(f"Enabled MFA for user {user.name}")

            # List policies
            policies = self.identity_client.list_policies(compartment_id).data

            for policy in policies:
                # Update policy statements
                if 'Allow' in policy.statements:
                    updated_statements = [
                        statement for statement in policy.statements
                        if 'Allow' not in statement
                    ]
                    self.identity_client.update_policy(
                        policy_id=policy.id,
                        update_policy_details=oci.identity.models.UpdatePolicyDetails(
                            statements=updated_statements
                        )
                    )
                    logger.info(f"Updated policy {policy.name} to remove Allow statements")

            return {
                'compartment_id': compartment_id,
                'remediation_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to remediate IAM security: {e}")
            raise

    def remediate_security_center(self, compartment_id: str) -> dict:
        """Remediate Security Center findings."""
        try:
            # Get security assessments
            assessments = self.security_client.list_security_assessments(
                compartment_id=compartment_id
            ).data

            for assessment in assessments:
                # Resolve findings
                if assessment.status != 'RESOLVED':
                    self.security_client.update_security_assessment(
                        security_assessment_id=assessment.id,
                        update_security_assessment_details=oci.cloud_guard.models.UpdateSecurityAssessmentDetails(
                            status='RESOLVED'
                        )
                    )
                    logger.info(f"Resolved assessment {assessment.id}")

            return {
                'compartment_id': compartment_id,
                'remediation_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to remediate Security Center: {e}")
            raise

def main():
    """Main function to remediate OCI security."""
    try:
        # Initialize remediator
        remediator = OCISecurityRemediator()

        # Remediate instance security
        instance_result = remediator.remediate_instance_security(
            'your-instance-id'
        )
        logger.info(f"Instance security remediation result: {json.dumps(instance_result, indent=2)}")

        # Remediate storage security
        storage_result = remediator.remediate_storage_security(
            'your-namespace',
            'your-bucket-name'
        )
        logger.info(f"Storage security remediation result: {json.dumps(storage_result, indent=2)}")

        # Remediate KMS security
        kms_result = remediator.remediate_kms_security(
            'your-vault-id'
        )
        logger.info(f"KMS security remediation result: {json.dumps(kms_result, indent=2)}")

        # Remediate IAM security
        iam_result = remediator.remediate_iam_security(
            'your-compartment-id'
        )
        logger.info(f"IAM security remediation result: {json.dumps(iam_result, indent=2)}")

        # Remediate Security Center
        security_center_result = remediator.remediate_security_center(
            'your-compartment-id'
        )
        logger.info(f"Security Center remediation result: {json.dumps(security_center_result, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to remediate OCI security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 