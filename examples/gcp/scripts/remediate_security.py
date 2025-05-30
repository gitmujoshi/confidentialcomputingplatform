#!/usr/bin/env python3

import os
import sys
import json
import logging
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import kms
from google.cloud import securitycenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPSecurityRemediator:
    def __init__(self, project_id: str):
        """Initialize GCP security remediation."""
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.storage_client = storage.Client(project=project_id)
        self.kms_client = kms.KeyManagementServiceClient()
        self.security_client = securitycenter.SecurityCenterClient()

    def remediate_instance_security(self, instance_name: str, zone: str) -> dict:
        """Remediate security issues for a confidential instance."""
        try:
            instance = self.compute_client.get(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )

            # Enable confidential computing if not enabled
            if not instance.confidential_instance_config.enable_confidential_compute:
                instance.confidential_instance_config.enable_confidential_compute = True
                operation = self.compute_client.update(
                    project=self.project_id,
                    zone=zone,
                    instance=instance_name,
                    instance_resource=instance
                )
                operation.result()
                logger.info(f"Enabled confidential computing for instance {instance_name}")

            # Ensure disk encryption
            for disk in instance.disks:
                if not disk.disk_encryption_key:
                    # Create encrypted snapshot
                    snapshot = compute_v1.Snapshot()
                    snapshot.name = f"{disk.source.split('/')[-1]}-encrypted"
                    snapshot.source_disk = disk.source
                    snapshot.disk_size_gb = disk.disk_size_gb

                    operation = self.compute_client.create_snapshot(
                        project=self.project_id,
                        snapshot_resource=snapshot
                    )
                    operation.result()

                    # Create encrypted disk from snapshot
                    new_disk = compute_v1.Disk()
                    new_disk.name = f"{disk.source.split('/')[-1]}-encrypted"
                    new_disk.source_snapshot = f"projects/{self.project_id}/global/snapshots/{snapshot.name}"
                    new_disk.disk_encryption_key = compute_v1.CustomerEncryptionKey(
                        kms_key_name=f"projects/{self.project_id}/locations/{zone}/keyRings/confidential-keyring/cryptoKeys/confidential-key"
                    )

                    operation = self.compute_client.insert_disk(
                        project=self.project_id,
                        zone=zone,
                        disk_resource=new_disk
                    )
                    operation.result()

                    # Detach old disk and attach new one
                    self.compute_client.detach_disk(
                        project=self.project_id,
                        zone=zone,
                        instance=instance_name,
                        device_name=disk.device_name
                    )

                    self.compute_client.attach_disk(
                        project=self.project_id,
                        zone=zone,
                        instance=instance_name,
                        attached_disk_resource=compute_v1.AttachedDisk(
                            source=new_disk.self_link,
                            device_name=disk.device_name
                        )
                    )
                    logger.info(f"Replaced unencrypted disk with encrypted disk for instance {instance_name}")

            return {
                'instance_name': instance_name,
                'remediation_status': 'completed'
            }
        except Exception as e:
            logger.error(f"Failed to remediate instance security: {str(e)}")
            raise

    def remediate_storage_security(self, bucket_name: str) -> dict:
        """Remediate security issues for a Cloud Storage bucket."""
        try:
            bucket = self.storage_client.get_bucket(bucket_name)

            # Enable versioning
            if not bucket.versioning_enabled:
                bucket.versioning_enabled = True
                bucket.patch()
                logger.info(f"Enabled versioning for bucket {bucket_name}")

            # Enable encryption
            if not bucket.default_kms_key_name:
                bucket.default_kms_key_name = f"projects/{self.project_id}/locations/us-central1/keyRings/confidential-keyring/cryptoKeys/confidential-key"
                bucket.patch()
                logger.info(f"Enabled encryption for bucket {bucket_name}")

            # Enable uniform bucket-level access
            if not bucket.iam_configuration.uniform_bucket_level_access_enabled:
                bucket.iam_configuration.uniform_bucket_level_access_enabled = True
                bucket.patch()
                logger.info(f"Enabled uniform bucket-level access for bucket {bucket_name}")

            return {
                'bucket_name': bucket_name,
                'remediation_status': 'completed'
            }
        except Exception as e:
            logger.error(f"Failed to remediate storage security: {str(e)}")
            raise

    def remediate_kms_security(self, keyring_name: str, location: str) -> dict:
        """Remediate security issues for a KMS keyring."""
        try:
            parent = f"projects/{self.project_id}/locations/{location}/keyRings/{keyring_name}"
            keys = self.kms_client.list_crypto_keys(request={"parent": parent})

            for key in keys:
                # Enable key rotation if not enabled
                if not key.rotation_period:
                    self.kms_client.update_crypto_key(
                        request={
                            "crypto_key": {
                                "name": key.name,
                                "rotation_period": {"seconds": 7776000}  # 90 days
                            },
                            "update_mask": {"paths": ["rotation_period"]}
                        }
                    )
                    logger.info(f"Enabled key rotation for key {key.name}")

            return {
                'keyring_name': keyring_name,
                'remediation_status': 'completed'
            }
        except Exception as e:
            logger.error(f"Failed to remediate KMS security: {str(e)}")
            raise

    def remediate_network_security(self, network_name: str) -> dict:
        """Remediate security issues for a VPC network."""
        try:
            # Update firewall rules
            firewall = compute_v1.Firewall()
            firewall.name = f"{network_name}-deny-all"
            firewall.network = f"projects/{self.project_id}/global/networks/{network_name}"
            firewall.direction = "INGRESS"
            firewall.denied = [{"IPProtocol": "all"}]
            firewall.source_ranges = ["0.0.0.0/0"]

            operation = self.compute_client.insert_firewall(
                project=self.project_id,
                firewall_resource=firewall
            )
            operation.result()
            logger.info(f"Updated firewall rules for network {network_name}")

            return {
                'network_name': network_name,
                'remediation_status': 'completed'
            }
        except Exception as e:
            logger.error(f"Failed to remediate network security: {str(e)}")
            raise

def main():
    """Main function to remediate GCP security issues."""
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        remediator = GCPSecurityRemediator(project_id)

        # Remediate instance security
        instance_remediation = remediator.remediate_instance_security(
            "confidential-instance",
            "us-central1-a"
        )
        logger.info(f"Instance remediation status: {json.dumps(instance_remediation, indent=2)}")

        # Remediate storage security
        storage_remediation = remediator.remediate_storage_security("confidential-bucket")
        logger.info(f"Storage remediation status: {json.dumps(storage_remediation, indent=2)}")

        # Remediate KMS security
        kms_remediation = remediator.remediate_kms_security("confidential-keyring", "us-central1")
        logger.info(f"KMS remediation status: {json.dumps(kms_remediation, indent=2)}")

        # Remediate network security
        network_remediation = remediator.remediate_network_security("confidential-network")
        logger.info(f"Network remediation status: {json.dumps(network_remediation, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to remediate GCP security: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 