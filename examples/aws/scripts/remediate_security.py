#!/usr/bin/env python3

import boto3
import logging
import json
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSSecurityRemediator:
    def __init__(self, region: str = 'us-east-1'):
        """Initialize AWS security remediation with boto3 clients."""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.kms = boto3.client('kms', region_name=region)
        self.iam = boto3.client('iam', region_name=region)

    def remediate_instance_security(self, instance_id: str) -> Dict:
        """Remediate security issues for a confidential instance."""
        try:
            # Get instance details
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]

            # Check and remediate security groups
            security_groups = instance.get('SecurityGroups', [])
            if not security_groups:
                # Create and attach security group
                sg_id = self._create_security_group(instance['VpcId'])
                self.ec2.modify_instance_attribute(
                    InstanceId=instance_id,
                    Groups=[sg_id]
                )
                logger.info(f"Created and attached security group {sg_id} to instance {instance_id}")

            # Check and remediate volume encryption
            block_devices = instance.get('BlockDeviceMappings', [])
            for device in block_devices:
                if not device.get('Ebs', {}).get('Encrypted', False):
                    # Create encrypted snapshot and new volume
                    snapshot_id = self._create_encrypted_snapshot(device['Ebs']['VolumeId'])
                    new_volume_id = self._create_encrypted_volume(snapshot_id)
                    self.ec2.detach_volume(VolumeId=device['Ebs']['VolumeId'])
                    self.ec2.attach_volume(
                        VolumeId=new_volume_id,
                        InstanceId=instance_id,
                        Device=device['DeviceName']
                    )
                    logger.info(f"Replaced unencrypted volume with encrypted volume {new_volume_id}")

            return {
                'instance_id': instance_id,
                'remediation_status': 'completed'
            }
        except ClientError as e:
            logger.error(f"Failed to remediate instance security: {e}")
            raise

    def _create_security_group(self, vpc_id: str) -> str:
        """Create a security group with strict rules."""
        try:
            response = self.ec2.create_security_group(
                GroupName='confidential-sg',
                Description='Security group for confidential computing',
                VpcId=vpc_id
            )
            sg_id = response['GroupId']

            # Add strict inbound rules
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            return sg_id
        except ClientError as e:
            logger.error(f"Failed to create security group: {e}")
            raise

    def _create_encrypted_snapshot(self, volume_id: str) -> str:
        """Create an encrypted snapshot of a volume."""
        try:
            response = self.ec2.create_snapshot(
                VolumeId=volume_id,
                Description='Encrypted snapshot for remediation'
            )
            snapshot_id = response['SnapshotId']
            
            # Wait for snapshot to complete
            self.ec2.get_waiter('snapshot_completed').wait(SnapshotIds=[snapshot_id])
            return snapshot_id
        except ClientError as e:
            logger.error(f"Failed to create encrypted snapshot: {e}")
            raise

    def _create_encrypted_volume(self, snapshot_id: str) -> str:
        """Create an encrypted volume from a snapshot."""
        try:
            response = self.ec2.create_volume(
                SnapshotId=snapshot_id,
                Encrypted=True,
                VolumeType='gp3'
            )
            volume_id = response['VolumeId']
            
            # Wait for volume to be available
            self.ec2.get_waiter('volume_available').wait(VolumeIds=[volume_id])
            return volume_id
        except ClientError as e:
            logger.error(f"Failed to create encrypted volume: {e}")
            raise

    def remediate_s3_bucket_security(self, bucket_name: str) -> Dict:
        """Remediate security issues for an S3 bucket."""
        try:
            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            # Enable encryption
            self.s3.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [{
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        }
                    }]
                }
            )

            # Block public access
            self.s3.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True
                }
            )

            return {
                'bucket_name': bucket_name,
                'remediation_status': 'completed'
            }
        except ClientError as e:
            logger.error(f"Failed to remediate S3 bucket security: {e}")
            raise

    def remediate_kms_key_rotation(self, key_id: str) -> Dict:
        """Enable key rotation for a KMS key."""
        try:
            self.kms.enable_key_rotation(KeyId=key_id)
            return {
                'key_id': key_id,
                'rotation_status': 'enabled'
            }
        except ClientError as e:
            logger.error(f"Failed to enable key rotation: {e}")
            raise

def main():
    """Main function to remediate AWS security issues."""
    try:
        # Initialize security remediator
        remediator = AWSSecurityRemediator()

        # Remediate instance security
        instance_remediation = remediator.remediate_instance_security('your-instance-id')
        logger.info(f"Instance remediation status: {json.dumps(instance_remediation, indent=2)}")

        # Remediate S3 bucket security
        s3_remediation = remediator.remediate_s3_bucket_security('your-bucket-name')
        logger.info(f"S3 bucket remediation status: {json.dumps(s3_remediation, indent=2)}")

        # Enable KMS key rotation
        kms_remediation = remediator.remediate_kms_key_rotation('your-key-id')
        logger.info(f"KMS key remediation status: {json.dumps(kms_remediation, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to remediate AWS security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 