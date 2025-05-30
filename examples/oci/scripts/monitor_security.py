#!/usr/bin/env python3

import os
import sys
import json
import logging
import time
import oci
from oci.core import ComputeClient
from oci.object_storage import ObjectStorageClient
from oci.key_management import KmsCryptoClient
from oci.identity import IdentityClient
from oci.vault import VaultsClient
from oci.cloud_guard import CloudGuardClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCISecurityMonitor:
    def __init__(self, config_path: str = None):
        """Initialize OCI security monitoring."""
        self.config = oci.config.from_file(config_path) if config_path else oci.config.from_file()
        self.compute_client = ComputeClient(self.config)
        self.object_storage_client = ObjectStorageClient(self.config)
        self.kms_client = KmsCryptoClient(self.config)
        self.identity_client = IdentityClient(self.config)
        self.vault_client = VaultsClient(self.config)
        self.security_client = CloudGuardClient(self.config)

    def monitor_instance_security(self, compartment_id: str, interval: int = 300) -> None:
        """Monitor security of compute instances."""
        try:
            while True:
                # List instances
                instances = self.compute_client.list_instances(
                    compartment_id=compartment_id
                ).data

                for instance in instances:
                    # Check confidential computing
                    if not instance.is_confidential_compute:
                        logger.warning(f"Instance {instance.id} is not using confidential computing")

                    # Check secure boot
                    if not instance.is_secure_boot_enabled:
                        logger.warning(f"Instance {instance.id} does not have secure boot enabled")

                    # Check TPM
                    if not instance.is_trusted_platform_module_enabled:
                        logger.warning(f"Instance {instance.id} does not have TPM enabled")

                time.sleep(interval)
        except Exception as e:
            logger.error(f"Failed to monitor instance security: {e}")
            raise

    def monitor_storage_security(self, compartment_id: str, interval: int = 300) -> None:
        """Monitor security of Object Storage buckets."""
        try:
            while True:
                # List buckets
                buckets = self.object_storage_client.list_buckets(
                    compartment_id=compartment_id
                ).data

                for bucket in buckets:
                    # Check encryption
                    if not bucket.kms_key_id:
                        logger.warning(f"Bucket {bucket.name} is not encrypted with KMS")

                    # Check public access
                    if bucket.public_access_type != 'NoPublicAccess':
                        logger.warning(f"Bucket {bucket.name} has public access enabled")

                    # Check versioning
                    if not bucket.versioning:
                        logger.warning(f"Bucket {bucket.name} does not have versioning enabled")

                time.sleep(interval)
        except Exception as e:
            logger.error(f"Failed to monitor storage security: {e}")
            raise

    def monitor_kms_security(self, compartment_id: str, interval: int = 300) -> None:
        """Monitor security of Vault keys."""
        try:
            while True:
                # List vaults
                vaults = self.vault_client.list_vaults(
                    compartment_id=compartment_id
                ).data

                for vault in vaults:
                    # List keys
                    keys = self.vault_client.list_keys(
                        vault_id=vault.id
                    ).data

                    for key in keys:
                        # Check key state
                        if key.lifecycle_state != 'ACTIVE':
                            logger.warning(f"Key {key.id} is not in ACTIVE state")

                        # Check protection mode
                        if key.protection_mode != 'HSM':
                            logger.warning(f"Key {key.id} is not using HSM protection")

                time.sleep(interval)
        except Exception as e:
            logger.error(f"Failed to monitor KMS security: {e}")
            raise

    def monitor_iam_security(self, compartment_id: str, interval: int = 300) -> None:
        """Monitor IAM security settings."""
        try:
            while True:
                # List users
                users = self.identity_client.list_users(
                    compartment_id=compartment_id
                ).data

                for user in users:
                    # Check MFA
                    if not user.is_mfa_activated:
                        logger.warning(f"User {user.name} does not have MFA enabled")

                # List groups
                groups = self.identity_client.list_groups(
                    compartment_id=compartment_id
                ).data

                for group in groups:
                    # Check group policies
                    policies = self.identity_client.list_policies(
                        compartment_id=compartment_id
                    ).data

                    for policy in policies:
                        if 'Allow' in policy.statements:
                            logger.warning(f"Policy {policy.name} contains Allow statements")

                time.sleep(interval)
        except Exception as e:
            logger.error(f"Failed to monitor IAM security: {e}")
            raise

    def monitor_security_center(self, compartment_id: str, interval: int = 300) -> None:
        """Monitor Security Center findings."""
        try:
            while True:
                # Get security assessments
                assessments = self.security_client.list_security_assessments(
                    compartment_id=compartment_id
                ).data

                for assessment in assessments:
                    # Check severity
                    if assessment.severity in ['HIGH', 'CRITICAL']:
                        logger.warning(f"High severity finding in assessment {assessment.id}")

                    # Check status
                    if assessment.status != 'RESOLVED':
                        logger.warning(f"Unresolved finding in assessment {assessment.id}")

                time.sleep(interval)
        except Exception as e:
            logger.error(f"Failed to monitor Security Center: {e}")
            raise

def main():
    """Main function to monitor OCI security."""
    try:
        # Initialize monitor
        monitor = OCISecurityMonitor()

        # Start monitoring threads
        import threading

        # Monitor instance security
        instance_thread = threading.Thread(
            target=monitor.monitor_instance_security,
            args=('your-compartment-id',)
        )
        instance_thread.start()

        # Monitor storage security
        storage_thread = threading.Thread(
            target=monitor.monitor_storage_security,
            args=('your-compartment-id',)
        )
        storage_thread.start()

        # Monitor KMS security
        kms_thread = threading.Thread(
            target=monitor.monitor_kms_security,
            args=('your-compartment-id',)
        )
        kms_thread.start()

        # Monitor IAM security
        iam_thread = threading.Thread(
            target=monitor.monitor_iam_security,
            args=('your-compartment-id',)
        )
        iam_thread.start()

        # Monitor Security Center
        security_center_thread = threading.Thread(
            target=monitor.monitor_security_center,
            args=('your-compartment-id',)
        )
        security_center_thread.start()

        # Wait for threads to complete
        instance_thread.join()
        storage_thread.join()
        kms_thread.join()
        iam_thread.join()
        security_center_thread.join()

        return 0
    except Exception as e:
        logger.error(f"Failed to monitor OCI security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 