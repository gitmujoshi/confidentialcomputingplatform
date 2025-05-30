#!/usr/bin/env python3

import boto3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSSecurityMonitor:
    def __init__(self, region: str = 'us-east-1'):
        """Initialize AWS security monitoring with boto3 clients."""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.securityhub = boto3.client('securityhub', region_name=region)
        self.guardduty = boto3.client('guardduty', region_name=region)

    def check_instance_security(self, instance_id: str) -> Dict:
        """Check security status of a confidential instance."""
        try:
            # Get instance details
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]

            # Check security groups
            security_groups = instance.get('SecurityGroups', [])
            security_group_ids = [sg['GroupId'] for sg in security_groups]

            # Check if instance is using encrypted volumes
            block_devices = instance.get('BlockDeviceMappings', [])
            encrypted_volumes = all(
                device.get('Ebs', {}).get('Encrypted', False)
                for device in block_devices
            )

            return {
                'instance_id': instance_id,
                'security_groups': security_group_ids,
                'encrypted_volumes': encrypted_volumes,
                'state': instance['State']['Name']
            }
        except ClientError as e:
            logger.error(f"Failed to check instance security: {e}")
            raise

    def monitor_network_traffic(self, vpc_id: str) -> List[Dict]:
        """Monitor network traffic in the VPC."""
        try:
            # Get VPC Flow Logs
            response = self.ec2.describe_flow_logs(
                Filters=[{'Name': 'resource-id', 'Values': [vpc_id]}]
            )
            flow_logs = response.get('FlowLogs', [])

            # Get CloudWatch metrics for network traffic
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            metrics = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/VPC',
                MetricName='NetworkPacketsIn',
                Dimensions=[{'Name': 'VpcId', 'Value': vpc_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )

            return {
                'flow_logs': flow_logs,
                'network_metrics': metrics.get('Datapoints', [])
            }
        except ClientError as e:
            logger.error(f"Failed to monitor network traffic: {e}")
            raise

    def check_security_hub_findings(self) -> List[Dict]:
        """Check Security Hub findings."""
        try:
            response = self.securityhub.get_findings(
                Filters={
                    'RecordState': [{'Value': 'ACTIVE', 'Comparison': 'EQUALS'}],
                    'SeverityLabel': [
                        {'Value': 'CRITICAL', 'Comparison': 'EQUALS'},
                        {'Value': 'HIGH', 'Comparison': 'EQUALS'}
                    ]
                }
            )
            return response.get('Findings', [])
        except ClientError as e:
            logger.error(f"Failed to check Security Hub findings: {e}")
            raise

    def check_guardduty_findings(self) -> List[Dict]:
        """Check GuardDuty findings."""
        try:
            # Get detector ID
            detectors = self.guardduty.list_detectors()
            if not detectors['DetectorIds']:
                return []

            detector_id = detectors['DetectorIds'][0]
            response = self.guardduty.list_findings(
                DetectorId=detector_id,
                FindingCriteria={
                    'Criterion': {
                        'severity': {
                            'Gte': 7  # High and Critical findings
                        }
                    }
                }
            )

            if not response['FindingIds']:
                return []

            findings = self.guardduty.get_findings(
                DetectorId=detector_id,
                FindingIds=response['FindingIds']
            )
            return findings.get('Findings', [])
        except ClientError as e:
            logger.error(f"Failed to check GuardDuty findings: {e}")
            raise

    def check_kms_key_usage(self, key_id: str) -> Dict:
        """Check KMS key usage and rotation status."""
        try:
            response = self.kms.describe_key(KeyId=key_id)
            key_metadata = response['KeyMetadata']

            return {
                'key_id': key_id,
                'key_state': key_metadata['KeyState'],
                'key_rotation_enabled': key_metadata.get('KeyRotationEnabled', False),
                'creation_date': key_metadata['CreationDate'].isoformat(),
                'key_usage': key_metadata['KeyUsage']
            }
        except ClientError as e:
            logger.error(f"Failed to check KMS key usage: {e}")
            raise

    def check_s3_bucket_security(self, bucket_name: str) -> Dict:
        """Check S3 bucket security settings."""
        try:
            # Check encryption
            encryption = self.s3.get_bucket_encryption(Bucket=bucket_name)
            
            # Check versioning
            versioning = self.s3.get_bucket_versioning(Bucket=bucket_name)
            
            # Check public access block
            public_access = self.s3.get_public_access_block(Bucket=bucket_name)

            return {
                'bucket_name': bucket_name,
                'encryption': encryption.get('ServerSideEncryptionConfiguration', {}),
                'versioning': versioning.get('Status', 'Disabled'),
                'public_access_block': public_access.get('PublicAccessBlockConfiguration', {})
            }
        except ClientError as e:
            logger.error(f"Failed to check S3 bucket security: {e}")
            raise

def main():
    """Main function to monitor AWS security."""
    try:
        # Initialize security monitor
        monitor = AWSSecurityMonitor()

        # Check instance security
        instance_security = monitor.check_instance_security('your-instance-id')
        logger.info(f"Instance security status: {json.dumps(instance_security, indent=2)}")

        # Monitor network traffic
        network_status = monitor.monitor_network_traffic('your-vpc-id')
        logger.info(f"Network traffic status: {json.dumps(network_status, indent=2)}")

        # Check Security Hub findings
        security_hub_findings = monitor.check_security_hub_findings()
        logger.info(f"Security Hub findings: {json.dumps(security_hub_findings, indent=2)}")

        # Check GuardDuty findings
        guardduty_findings = monitor.check_guardduty_findings()
        logger.info(f"GuardDuty findings: {json.dumps(guardduty_findings, indent=2)}")

        # Check KMS key usage
        kms_status = monitor.check_kms_key_usage('your-key-id')
        logger.info(f"KMS key status: {json.dumps(kms_status, indent=2)}")

        # Check S3 bucket security
        s3_security = monitor.check_s3_bucket_security('your-bucket-name')
        logger.info(f"S3 bucket security status: {json.dumps(s3_security, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to monitor AWS security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 