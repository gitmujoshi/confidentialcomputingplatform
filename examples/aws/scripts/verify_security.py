#!/usr/bin/env python3

import boto3
import logging
import json
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSSecurityVerifier:
    def __init__(self, region: str = 'us-east-1'):
        """Initialize AWS security verification with boto3 clients."""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.kms = boto3.client('kms', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.securityhub = boto3.client('securityhub', region_name=region)

    def verify_instance_security(self, instance_id: str) -> Dict:
        """Verify security configuration of a confidential instance."""
        try:
            # Get instance details
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]

            # Check security groups
            security_groups = instance.get('SecurityGroups', [])
            security_group_ids = [sg['GroupId'] for sg in security_groups]

            # Check volume encryption
            block_devices = instance.get('BlockDeviceMappings', [])
            encrypted_volumes = all(
                device.get('Ebs', {}).get('Encrypted', False)
                for device in block_devices
            )

            # Check instance type for Nitro support
            instance_type = instance['InstanceType']
            nitro_supported = self._check_nitro_support(instance_type)

            return {
                'instance_id': instance_id,
                'security_groups': security_group_ids,
                'encrypted_volumes': encrypted_volumes,
                'nitro_supported': nitro_supported,
                'state': instance['State']['Name']
            }
        except ClientError as e:
            logger.error(f"Failed to verify instance security: {e}")
            raise

    def _check_nitro_support(self, instance_type: str) -> bool:
        """Check if instance type supports Nitro Enclaves."""
        nitro_types = [
            'c5', 'c5a', 'c5ad', 'c5d', 'c5n', 'c6a', 'c6g', 'c6gd', 'c6gn',
            'm5', 'm5a', 'm5ad', 'm5d', 'm5dn', 'm5n', 'm5zn', 'm6a', 'm6g',
            'r5', 'r5a', 'r5ad', 'r5b', 'r5d', 'r5dn', 'r5n', 'r6a', 'r6g'
        ]
        return any(instance_type.startswith(t) for t in nitro_types)

    def verify_s3_bucket_security(self, bucket_name: str) -> Dict:
        """Verify security configuration of an S3 bucket."""
        try:
            # Check versioning
            versioning = self.s3.get_bucket_versioning(Bucket=bucket_name)
            
            # Check encryption
            encryption = self.s3.get_bucket_encryption(Bucket=bucket_name)
            
            # Check public access block
            public_access = self.s3.get_public_access_block(Bucket=bucket_name)

            return {
                'bucket_name': bucket_name,
                'versioning_enabled': versioning.get('Status') == 'Enabled',
                'encryption_enabled': bool(encryption.get('ServerSideEncryptionConfiguration')),
                'public_access_blocked': all(
                    public_access['PublicAccessBlockConfiguration'].values()
                )
            }
        except ClientError as e:
            logger.error(f"Failed to verify S3 bucket security: {e}")
            raise

    def verify_kms_key_security(self, key_id: str) -> Dict:
        """Verify security configuration of a KMS key."""
        try:
            response = self.kms.describe_key(KeyId=key_id)
            key_metadata = response['KeyMetadata']

            return {
                'key_id': key_id,
                'key_state': key_metadata['KeyState'],
                'key_rotation_enabled': key_metadata.get('KeyRotationEnabled', False),
                'key_usage': key_metadata['KeyUsage'],
                'origin': key_metadata['Origin']
            }
        except ClientError as e:
            logger.error(f"Failed to verify KMS key security: {e}")
            raise

    def verify_iam_security(self) -> Dict:
        """Verify IAM security configuration."""
        try:
            # Check password policy
            password_policy = self.iam.get_account_password_policy()
            
            # Check MFA status
            mfa_devices = self.iam.list_virtual_mfa_devices()
            
            # Check access keys
            users = self.iam.list_users()
            access_keys = []
            for user in users['Users']:
                keys = self.iam.list_access_keys(UserName=user['UserName'])
                access_keys.extend(keys['AccessKeyMetadata'])

            return {
                'password_policy': password_policy['PasswordPolicy'],
                'mfa_devices_count': len(mfa_devices['VirtualMFADevices']),
                'access_keys_count': len(access_keys)
            }
        except ClientError as e:
            logger.error(f"Failed to verify IAM security: {e}")
            raise

def main():
    """Main function to verify AWS security configuration."""
    try:
        # Initialize security verifier
        verifier = AWSSecurityVerifier()

        # Verify instance security
        instance_security = verifier.verify_instance_security('your-instance-id')
        logger.info(f"Instance security status: {json.dumps(instance_security, indent=2)}")

        # Verify S3 bucket security
        s3_security = verifier.verify_s3_bucket_security('your-bucket-name')
        logger.info(f"S3 bucket security status: {json.dumps(s3_security, indent=2)}")

        # Verify KMS key security
        kms_security = verifier.verify_kms_key_security('your-key-id')
        logger.info(f"KMS key security status: {json.dumps(kms_security, indent=2)}")

        # Verify IAM security
        iam_security = verifier.verify_iam_security()
        logger.info(f"IAM security status: {json.dumps(iam_security, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to verify AWS security: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 