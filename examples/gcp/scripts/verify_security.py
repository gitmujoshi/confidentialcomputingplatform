#!/usr/bin/env python3

import os
import sys
import json
import logging
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import kms
from google.cloud import securitycenter
from google.cloud import asset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPSecurityVerifier:
    def __init__(self, project_id: str):
        """Initialize GCP security verification."""
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.storage_client = storage.Client(project=project_id)
        self.kms_client = kms.KeyManagementServiceClient()
        self.security_client = securitycenter.SecurityCenterClient()
        self.asset_client = asset.AssetClient()

    def verify_instance_security(self, instance_name: str, zone: str) -> dict:
        """Verify security configuration of a compute instance."""
        try:
            # Get instance details
            instance = self.compute_client.get(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )

            # Check confidential computing features
            confidential_computing = {
                'enabled': instance.confidential_instance_config.enable_confidential_compute,
                'type': instance.machine_type,
                'shielded_vm': instance.shielded_instance_config.enable_secure_boot,
                'vTPM': instance.shielded_instance_config.enable_vtpm,
                'integrity_monitoring': instance.shielded_instance_config.enable_integrity_monitoring
            }

            # Check network security
            network_security = {
                'internal_ip_only': not instance.network_interfaces[0].access_configs,
                'service_account': instance.service_accounts[0].email if instance.service_accounts else None,
                'tags': instance.tags.items if instance.tags else []
            }

            return {
                'instance_name': instance_name,
                'confidential_computing': confidential_computing,
                'network_security': network_security
            }
        except Exception as e:
            logger.error(f"Failed to verify instance security: {e}")
            raise

    def verify_storage_security(self, bucket_name: str) -> dict:
        """Verify security configuration of a Cloud Storage bucket."""
        try:
            # Get bucket details
            bucket = self.storage_client.bucket(bucket_name)
            bucket.reload()

            # Check encryption
            encryption = {
                'default_kms_key': bucket.default_kms_key_name,
                'uniform_bucket_level_access': bucket.iam_configuration.uniform_bucket_level_access.enabled,
                'public_access_prevention': bucket.iam_configuration.public_access_prevention
            }

            # Check IAM policies
            iam_policies = {
                'bindings': bucket.get_iam_policy().bindings
            }

            return {
                'bucket_name': bucket_name,
                'encryption': encryption,
                'iam_policies': iam_policies
            }
        except Exception as e:
            logger.error(f"Failed to verify storage security: {e}")
            raise

    def verify_kms_security(self, key_ring_name: str) -> dict:
        """Verify security configuration of Cloud KMS keys."""
        try:
            # Get key ring details
            parent = self.kms_client.key_ring_path(
                self.project_id, "global", key_ring_name
            )

            # List keys in the key ring
            keys = []
            for key in self.kms_client.list_crypto_keys(request={"parent": parent}):
                key_info = {
                    'name': key.name,
                    'purpose': key.purpose,
                    'protection_level': key.version_template.protection_level,
                    'algorithm': key.version_template.algorithm
                }
                keys.append(key_info)

            return {
                'key_ring_name': key_ring_name,
                'keys': keys
            }
        except Exception as e:
            logger.error(f"Failed to verify KMS security: {e}")
            raise

    def verify_iam_security(self) -> dict:
        """Verify IAM security settings."""
        try:
            # Get IAM policy
            policy = self.asset_client.analyze_iam_policy(
                request={
                    "analysis_query": {
                        "scope": f"projects/{self.project_id}",
                        "resource_selector": {
                            "full_resource_name": f"//cloudresourcemanager.googleapis.com/projects/{self.project_id}"
                        }
                    }
                }
            )

            # Analyze IAM bindings
            iam_analysis = {
                'bindings': policy.analysis_results,
                'service_accounts': [],
                'custom_roles': []
            }

            # Get service accounts
            for result in policy.analysis_results:
                if 'serviceAccount' in result.identity:
                    iam_analysis['service_accounts'].append(result.identity)

            return iam_analysis
        except Exception as e:
            logger.error(f"Failed to verify IAM security: {e}")
            raise

    def verify_security_center(self) -> dict:
        """Verify Security Command Center findings."""
        try:
            # Get findings
            findings = []
            for finding in self.security_client.list_findings(
                request={
                    "parent": f"organizations/{self.project_id}/sources/-"
                }
            ):
                finding_info = {
                    'name': finding.name,
                    'category': finding.category,
                    'severity': finding.severity,
                    'state': finding.state
                }
                findings.append(finding_info)

            return {
                'findings': findings
            }
        except Exception as e:
            logger.error(f"Failed to verify Security Center: {e}")
            raise

def main():
    """Main function to verify GCP security."""
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        verifier = GCPSecurityVerifier(project_id)

        # Verify instance security
        instance_result = verifier.verify_instance_security(
            'your-instance-name',
            'your-zone'
        )
        logger.info(f"Instance security verification result: {json.dumps(instance_result, indent=2)}")

        # Verify storage security
        storage_result = verifier.verify_storage_security(
            'your-bucket-name'
        )
        logger.info(f"Storage security verification result: {json.dumps(storage_result, indent=2)}")

        # Verify KMS security
        kms_result = verifier.verify_kms_security(
            'your-key-ring-name'
        )
        logger.info(f"KMS security verification result: {json.dumps(kms_result, indent=2)}")

        # Verify IAM security
        iam_result = verifier.verify_iam_security()
        logger.info(f"IAM security verification result: {json.dumps(iam_result, indent=2)}")

        # Verify Security Center
        security_center_result = verifier.verify_security_center()
        logger.info(f"Security Center verification result: {json.dumps(security_center_result, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to verify GCP security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 