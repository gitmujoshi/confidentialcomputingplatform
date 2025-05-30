#!/usr/bin/env python3

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import kms
from google.cloud import monitoring_v3
from google.cloud import securitycenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPSecurityMonitor:
    def __init__(self, project_id: str):
        """Initialize GCP security monitoring."""
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.storage_client = storage.Client(project=project_id)
        self.kms_client = kms.KeyManagementServiceClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.security_client = securitycenter.SecurityCenterClient()

    def check_instance_security(self, instance_name: str, zone: str) -> dict:
        """Check security status of a confidential instance."""
        try:
            instance = self.compute_client.get(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )

            # Check confidential computing status
            confidential_enabled = instance.confidential_instance_config.enable_confidential_compute

            # Check disk encryption
            encrypted_disks = all(
                disk.disk_encryption_key is not None
                for disk in instance.disks
            )

            # Check network security
            network_secure = all(
                not interface.access_configs
                for interface in instance.network_interfaces
            )

            return {
                'instance_name': instance_name,
                'confidential_computing': confidential_enabled,
                'encrypted_disks': encrypted_disks,
                'network_secure': network_secure,
                'status': instance.status
            }
        except Exception as e:
            logger.error(f"Failed to check instance security: {str(e)}")
            raise

    def monitor_network_traffic(self, network_name: str) -> dict:
        """Monitor network traffic in the VPC."""
        try:
            # Get VPC flow logs
            project_name = f"projects/{self.project_id}"
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(datetime.now().timestamp())},
                "start_time": {"seconds": int((datetime.now() - timedelta(hours=1)).timestamp())}
            })

            # Query network metrics
            response = self.monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": f'metric.type = "compute.googleapis.com/instance/network/received_bytes_count" AND resource.labels.network_name = "{network_name}"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                }
            )

            return {
                'network_name': network_name,
                'traffic_metrics': [
                    {
                        'metric': series.metric.type,
                        'value': series.points[-1].value.double_value
                    }
                    for series in response
                ]
            }
        except Exception as e:
            logger.error(f"Failed to monitor network traffic: {str(e)}")
            raise

    def check_security_findings(self) -> list:
        """Check Security Command Center findings."""
        try:
            parent = f"organizations/{self.project_id}"
            findings = self.security_client.list_findings(
                request={
                    "parent": parent,
                    "filter": "state = 'ACTIVE' AND severity = 'HIGH' OR severity = 'CRITICAL'"
                }
            )

            return [
                {
                    'name': finding.name,
                    'severity': finding.severity,
                    'category': finding.category,
                    'state': finding.state
                }
                for finding in findings
            ]
        except Exception as e:
            logger.error(f"Failed to check security findings: {str(e)}")
            raise

    def check_storage_security(self, bucket_name: str) -> dict:
        """Check Cloud Storage bucket security."""
        try:
            bucket = self.storage_client.get_bucket(bucket_name)

            return {
                'bucket_name': bucket_name,
                'versioning_enabled': bucket.versioning_enabled,
                'encryption_enabled': bucket.default_kms_key_name is not None,
                'public_access': not bucket.iam_configuration.uniform_bucket_level_access_enabled
            }
        except Exception as e:
            logger.error(f"Failed to check storage security: {str(e)}")
            raise

    def check_kms_security(self, keyring_name: str, location: str) -> dict:
        """Check KMS keyring security."""
        try:
            parent = f"projects/{self.project_id}/locations/{location}/keyRings/{keyring_name}"
            keys = self.kms_client.list_crypto_keys(request={"parent": parent})

            key_security = []
            for key in keys:
                key_security.append({
                    'key_name': key.name,
                    'rotation_enabled': key.rotation_period is not None,
                    'purpose': key.purpose,
                    'state': key.state
                })

            return {
                'keyring_name': keyring_name,
                'keys': key_security
            }
        except Exception as e:
            logger.error(f"Failed to check KMS security: {str(e)}")
            raise

def main():
    """Main function to monitor GCP security."""
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        monitor = GCPSecurityMonitor(project_id)

        # Check instance security
        instance_security = monitor.check_instance_security(
            "confidential-instance",
            "us-central1-a"
        )
        logger.info(f"Instance security status: {json.dumps(instance_security, indent=2)}")

        # Monitor network traffic
        network_status = monitor.monitor_network_traffic("confidential-network")
        logger.info(f"Network traffic status: {json.dumps(network_status, indent=2)}")

        # Check security findings
        security_findings = monitor.check_security_findings()
        logger.info(f"Security findings: {json.dumps(security_findings, indent=2)}")

        # Check storage security
        storage_security = monitor.check_storage_security("confidential-bucket")
        logger.info(f"Storage security status: {json.dumps(storage_security, indent=2)}")

        # Check KMS security
        kms_security = monitor.check_kms_security("confidential-keyring", "us-central1")
        logger.info(f"KMS security status: {json.dumps(kms_security, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to monitor GCP security: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 