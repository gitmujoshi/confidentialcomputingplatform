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

class OCIResourceCleanup:
    def __init__(self, config_path: str = None):
        """Initialize OCI resource cleanup."""
        self.config = oci.config.from_file(config_path) if config_path else oci.config.from_file()
        self.compute_client = ComputeClient(self.config)
        self.object_storage_client = ObjectStorageClient(self.config)
        self.kms_client = KmsCryptoClient(self.config)
        self.identity_client = IdentityClient(self.config)
        self.vault_client = VaultsClient(self.config)
        self.security_client = CloudGuardClient(self.config)

    def cleanup_compute_resources(self, compartment_id: str) -> dict:
        """Clean up compute resources."""
        try:
            # List instances
            instances = self.compute_client.list_instances(compartment_id).data

            for instance in instances:
                # Terminate instance
                self.compute_client.terminate_instance(instance.id)
                logger.info(f"Terminated instance: {instance.id}")

            # List subnets
            subnets = self.compute_client.list_subnets(compartment_id).data

            for subnet in subnets:
                # Delete subnet
                self.compute_client.delete_subnet(subnet.id)
                logger.info(f"Deleted subnet: {subnet.id}")

            # List VCNs
            vcns = self.compute_client.list_vcns(compartment_id).data

            for vcn in vcns:
                # Delete VCN
                self.compute_client.delete_vcn(vcn.id)
                logger.info(f"Deleted VCN: {vcn.id}")

            return {
                'compartment_id': compartment_id,
                'cleanup_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to clean up compute resources: {e}")
            raise

    def cleanup_storage_resources(self, compartment_id: str) -> dict:
        """Clean up storage resources."""
        try:
            # List buckets
            buckets = self.object_storage_client.list_buckets(
                namespace_name=self.object_storage_client.get_namespace().data,
                compartment_id=compartment_id
            ).data

            for bucket in buckets:
                # Delete bucket
                self.object_storage_client.delete_bucket(
                    namespace_name=self.object_storage_client.get_namespace().data,
                    bucket_name=bucket.name
                )
                logger.info(f"Deleted bucket: {bucket.name}")

            return {
                'compartment_id': compartment_id,
                'cleanup_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to clean up storage resources: {e}")
            raise

    def cleanup_kms_resources(self, compartment_id: str) -> dict:
        """Clean up KMS resources."""
        try:
            # List vaults
            vaults = self.vault_client.list_vaults(compartment_id).data

            for vault in vaults:
                # List keys
                keys = self.vault_client.list_keys(vault.id).data

                for key in keys:
                    # Delete key
                    self.vault_client.delete_key(key.id)
                    logger.info(f"Deleted key: {key.id}")

                # Delete vault
                self.vault_client.delete_vault(vault.id)
                logger.info(f"Deleted vault: {vault.id}")

            return {
                'compartment_id': compartment_id,
                'cleanup_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to clean up KMS resources: {e}")
            raise

    def cleanup_iam_resources(self, compartment_id: str) -> dict:
        """Clean up IAM resources."""
        try:
            # List policies
            policies = self.identity_client.list_policies(compartment_id).data

            for policy in policies:
                # Delete policy
                self.identity_client.delete_policy(policy.id)
                logger.info(f"Deleted policy: {policy.id}")

            # List groups
            groups = self.identity_client.list_groups(compartment_id).data

            for group in groups:
                # Delete group
                self.identity_client.delete_group(group.id)
                logger.info(f"Deleted group: {group.id}")

            return {
                'compartment_id': compartment_id,
                'cleanup_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to clean up IAM resources: {e}")
            raise

    def cleanup_security_resources(self, compartment_id: str) -> dict:
        """Clean up security resources."""
        try:
            # List security assessments
            assessments = self.security_client.list_security_assessments(
                compartment_id=compartment_id
            ).data

            for assessment in assessments:
                # Delete assessment
                self.security_client.delete_security_assessment(assessment.id)
                logger.info(f"Deleted security assessment: {assessment.id}")

            return {
                'compartment_id': compartment_id,
                'cleanup_status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to clean up security resources: {e}")
            raise

def main():
    """Main function to clean up OCI resources."""
    try:
        # Initialize cleanup
        cleanup = OCIResourceCleanup()

        # Clean up compute resources
        compute_result = cleanup.cleanup_compute_resources(
            'your-compartment-id'
        )
        logger.info(f"Compute resources cleanup result: {json.dumps(compute_result, indent=2)}")

        # Clean up storage resources
        storage_result = cleanup.cleanup_storage_resources(
            'your-compartment-id'
        )
        logger.info(f"Storage resources cleanup result: {json.dumps(storage_result, indent=2)}")

        # Clean up KMS resources
        kms_result = cleanup.cleanup_kms_resources(
            'your-compartment-id'
        )
        logger.info(f"KMS resources cleanup result: {json.dumps(kms_result, indent=2)}")

        # Clean up IAM resources
        iam_result = cleanup.cleanup_iam_resources(
            'your-compartment-id'
        )
        logger.info(f"IAM resources cleanup result: {json.dumps(iam_result, indent=2)}")

        # Clean up security resources
        security_result = cleanup.cleanup_security_resources(
            'your-compartment-id'
        )
        logger.info(f"Security resources cleanup result: {json.dumps(security_result, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to clean up OCI resources: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 