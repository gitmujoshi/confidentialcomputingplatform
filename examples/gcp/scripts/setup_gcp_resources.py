#!/usr/bin/env python3

import os
import sys
import json
import logging
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import kms
from google.cloud import secretmanager
from google.cloud import monitoring_v3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPResourceManager:
    def __init__(self, project_id: str):
        """Initialize GCP resource manager."""
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.storage_client = storage.Client(project=project_id)
        self.kms_client = kms.KeyManagementServiceClient()
        self.secret_client = secretmanager.SecretManagerServiceClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()

    def create_vpc_network(self, network_name: str, region: str) -> str:
        """Create a VPC network for confidential computing."""
        try:
            network = compute_v1.Network()
            network.name = network_name
            network.auto_create_subnetworks = False

            operation = self.compute_client.insert_network(
                project=self.project_id,
                network_resource=network
            )
            operation.result()

            # Create subnet
            subnet = compute_v1.Subnetwork()
            subnet.name = f"{network_name}-subnet"
            subnet.network = f"projects/{self.project_id}/global/networks/{network_name}"
            subnet.ip_cidr_range = "10.0.0.0/24"
            subnet.region = region

            operation = self.compute_client.insert_subnetwork(
                project=self.project_id,
                region=region,
                subnetwork_resource=subnet
            )
            operation.result()

            logger.info(f"Created VPC network {network_name} and subnet")
            return network_name
        except Exception as e:
            logger.error(f"Error creating VPC network: {str(e)}")
            raise

    def create_firewall_rules(self, network_name: str) -> None:
        """Create firewall rules for the VPC network."""
        try:
            # Deny all ingress
            deny_all = compute_v1.Firewall()
            deny_all.name = f"{network_name}-deny-all"
            deny_all.network = f"projects/{self.project_id}/global/networks/{network_name}"
            deny_all.direction = "INGRESS"
            deny_all.denied = [{"IPProtocol": "all"}]
            deny_all.source_ranges = ["0.0.0.0/0"]

            operation = self.compute_client.insert_firewall(
                project=self.project_id,
                firewall_resource=deny_all
            )
            operation.result()

            # Allow internal traffic
            allow_internal = compute_v1.Firewall()
            allow_internal.name = f"{network_name}-allow-internal"
            allow_internal.network = f"projects/{self.project_id}/global/networks/{network_name}"
            allow_internal.direction = "INGRESS"
            allow_internal.allowed = [{"IPProtocol": "all"}]
            allow_internal.source_ranges = ["10.0.0.0/8"]

            operation = self.compute_client.insert_firewall(
                project=self.project_id,
                firewall_resource=allow_internal
            )
            operation.result()

            logger.info(f"Created firewall rules for network {network_name}")
        except Exception as e:
            logger.error(f"Error creating firewall rules: {str(e)}")
            raise

    def create_confidential_instance(
        self,
        instance_name: str,
        zone: str,
        network_name: str,
        machine_type: str = "n2d-standard-2"
    ) -> str:
        """Create a confidential computing instance."""
        try:
            instance = compute_v1.Instance()
            instance.name = instance_name
            instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

            # Configure confidential computing
            instance.confidential_instance_config = compute_v1.ConfidentialInstanceConfig(
                enable_confidential_compute=True
            )

            # Configure network
            network_interface = compute_v1.NetworkInterface()
            network_interface.network = f"projects/{self.project_id}/global/networks/{network_name}"
            network_interface.subnetwork = f"projects/{self.project_id}/regions/{zone}/subnetworks/{network_name}-subnet"
            network_interface.access_configs = []
            instance.network_interfaces = [network_interface]

            # Configure boot disk
            boot_disk = compute_v1.AttachedDisk()
            boot_disk.boot = True
            boot_disk.auto_delete = True
            boot_disk.disk_size_gb = 50
            boot_disk.disk_encryption_key = compute_v1.CustomerEncryptionKey(
                kms_key_name=f"projects/{self.project_id}/locations/{zone}/keyRings/confidential-keyring/cryptoKeys/confidential-key"
            )
            instance.disks = [boot_disk]

            operation = self.compute_client.insert(
                project=self.project_id,
                zone=zone,
                instance_resource=instance
            )
            operation.result()

            logger.info(f"Created confidential instance {instance_name}")
            return instance_name
        except Exception as e:
            logger.error(f"Error creating confidential instance: {str(e)}")
            raise

    def create_storage_bucket(self, bucket_name: str) -> str:
        """Create a Cloud Storage bucket with encryption."""
        try:
            bucket = self.storage_client.create_bucket(
                bucket_name,
                location="us-central1",
                storage_class="STANDARD"
            )

            # Enable versioning
            bucket.versioning_enabled = True

            # Configure encryption
            bucket.default_kms_key_name = f"projects/{self.project_id}/locations/us-central1/keyRings/confidential-keyring/cryptoKeys/confidential-key"

            # Update bucket
            bucket.patch()

            logger.info(f"Created storage bucket {bucket_name}")
            return bucket_name
        except Exception as e:
            logger.error(f"Error creating storage bucket: {str(e)}")
            raise

    def create_kms_keyring(self, keyring_name: str, location: str) -> str:
        """Create a KMS keyring for encryption."""
        try:
            parent = f"projects/{self.project_id}/locations/{location}"
            keyring = self.kms_client.create_key_ring(
                request={
                    "parent": parent,
                    "key_ring_id": keyring_name,
                    "key_ring": {"name": f"{parent}/keyRings/{keyring_name}"}
                }
            )

            # Create key
            key = self.kms_client.create_crypto_key(
                request={
                    "parent": keyring.name,
                    "crypto_key_id": "confidential-key",
                    "crypto_key": {
                        "purpose": kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT,
                        "version_template": {
                            "algorithm": kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
                        },
                        "rotation_period": {"seconds": 7776000}  # 90 days
                    }
                }
            )

            logger.info(f"Created KMS keyring {keyring_name} and key")
            return keyring_name
        except Exception as e:
            logger.error(f"Error creating KMS keyring: {str(e)}")
            raise

    def setup_resources(self):
        """Set up all GCP resources."""
        try:
            # Create VPC network
            network_name = self.create_vpc_network("confidential-network", "us-central1")
            self.create_firewall_rules(network_name)

            # Create KMS keyring
            self.create_kms_keyring("confidential-keyring", "us-central1")

            # Create storage bucket
            self.create_storage_bucket("confidential-bucket")

            # Create confidential instance
            self.create_confidential_instance(
                "confidential-instance",
                "us-central1-a",
                network_name
            )

            logger.info("All GCP resources set up successfully")
        except Exception as e:
            logger.error(f"Error setting up resources: {str(e)}")
            raise

def main():
    """Main function to set up GCP resources."""
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        manager = GCPResourceManager(project_id)
        manager.setup_resources()
        return 0
    except Exception as e:
        logger.error(f"Failed to set up GCP resources: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 