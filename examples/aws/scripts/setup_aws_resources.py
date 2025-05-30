#!/usr/bin/env python3

import boto3
import logging
import os
import sys
from botocore.exceptions import ClientError
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSResourceManager:
    def __init__(self, region: str = 'us-east-1'):
        """Initialize AWS resource manager with boto3 clients."""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.kms = boto3.client('kms', region_name=region)
        self.iam = boto3.client('iam', region_name=region)

    def create_vpc(self, cidr_block: str = '10.0.0.0/16') -> str:
        """Create a VPC for confidential computing."""
        try:
            response = self.ec2.create_vpc(
                CidrBlock=cidr_block,
                EnableDnsSupport=True,
                EnableDnsHostnames=True,
                TagSpecifications=[{
                    'ResourceType': 'vpc',
                    'Tags': [{'Key': 'Name', 'Value': 'confidential-vpc'}]
                }]
            )
            vpc_id = response['Vpc']['VpcId']
            logger.info(f"Created VPC: {vpc_id}")
            return vpc_id
        except ClientError as e:
            logger.error(f"Failed to create VPC: {e}")
            raise

    def create_subnet(self, vpc_id: str, cidr_block: str = '10.0.1.0/24') -> str:
        """Create a subnet in the VPC."""
        try:
            response = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=cidr_block,
                TagSpecifications=[{
                    'ResourceType': 'subnet',
                    'Tags': [{'Key': 'Name', 'Value': 'confidential-subnet'}]
                }]
            )
            subnet_id = response['Subnet']['SubnetId']
            logger.info(f"Created subnet: {subnet_id}")
            return subnet_id
        except ClientError as e:
            logger.error(f"Failed to create subnet: {e}")
            raise

    def create_security_group(self, vpc_id: str) -> str:
        """Create a security group for the confidential instance."""
        try:
            response = self.ec2.create_security_group(
                GroupName='confidential-sg',
                Description='Security group for confidential computing',
                VpcId=vpc_id
            )
            sg_id = response['GroupId']

            # Add SSH rule
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            logger.info(f"Created security group: {sg_id}")
            return sg_id
        except ClientError as e:
            logger.error(f"Failed to create security group: {e}")
            raise

    def create_kms_key(self) -> str:
        """Create a KMS key for encryption."""
        try:
            response = self.kms.create_key(
                Description='KMS key for confidential computing',
                KeyUsage='ENCRYPT_DECRYPT',
                Origin='AWS_KMS',
                Tags=[{
                    'TagKey': 'Name',
                    'TagValue': 'confidential-key'
                }]
            )
            key_id = response['KeyMetadata']['KeyId']
            logger.info(f"Created KMS key: {key_id}")
            return key_id
        except ClientError as e:
            logger.error(f"Failed to create KMS key: {e}")
            raise

    def create_s3_bucket(self, bucket_name: str, kms_key_id: str) -> str:
        """Create an encrypted S3 bucket."""
        try:
            self.s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )

            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            # Configure encryption
            self.s3.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [{
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'aws:kms',
                            'KMSMasterKeyID': kms_key_id
                        }
                    }]
                }
            )
            logger.info(f"Created S3 bucket: {bucket_name}")
            return bucket_name
        except ClientError as e:
            logger.error(f"Failed to create S3 bucket: {e}")
            raise

    def create_iam_role(self) -> str:
        """Create an IAM role for Nitro Enclaves."""
        try:
            # Create role
            response = self.iam.create_role(
                RoleName='enclave-role',
                AssumeRolePolicyDocument=json.dumps({
                    'Version': '2012-10-17',
                    'Statement': [{
                        'Effect': 'Allow',
                        'Principal': {'Service': 'ec2.amazonaws.com'},
                        'Action': 'sts:AssumeRole'
                    }]
                })
            )
            role_arn = response['Role']['Arn']

            # Create and attach policy
            policy_document = {
                'Version': '2012-10-17',
                'Statement': [{
                    'Effect': 'Allow',
                    'Action': [
                        'kms:Decrypt',
                        'kms:GenerateDataKey',
                        's3:GetObject',
                        's3:PutObject'
                    ],
                    'Resource': '*'
                }]
            }

            self.iam.put_role_policy(
                RoleName='enclave-role',
                PolicyName='enclave-policy',
                PolicyDocument=json.dumps(policy_document)
            )
            logger.info(f"Created IAM role: {role_arn}")
            return role_arn
        except ClientError as e:
            logger.error(f"Failed to create IAM role: {e}")
            raise

    def launch_confidential_instance(
        self,
        subnet_id: str,
        security_group_id: str,
        key_name: str,
        ami_id: str
    ) -> str:
        """Launch a confidential computing instance."""
        try:
            response = self.ec2.run_instances(
                ImageId=ami_id,
                InstanceType='c5.2xlarge',  # Supports Nitro Enclaves
                MinCount=1,
                MaxCount=1,
                SubnetId=subnet_id,
                SecurityGroupIds=[security_group_id],
                KeyName=key_name,
                BlockDeviceMappings=[{
                    'DeviceName': '/dev/xvda',
                    'Ebs': {
                        'VolumeSize': 50,
                        'Encrypted': True
                    }
                }],
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': 'Name', 'Value': 'confidential-instance'}]
                }]
            )
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"Launched confidential instance: {instance_id}")
            return instance_id
        except ClientError as e:
            logger.error(f"Failed to launch instance: {e}")
            raise

def main():
    """Main function to set up AWS resources."""
    try:
        # Initialize resource manager
        manager = AWSResourceManager()

        # Create VPC and networking
        vpc_id = manager.create_vpc()
        subnet_id = manager.create_subnet(vpc_id)
        security_group_id = manager.create_security_group(vpc_id)

        # Create security resources
        kms_key_id = manager.create_kms_key()
        bucket_name = manager.create_s3_bucket('your-confidential-bucket', kms_key_id)
        role_arn = manager.create_iam_role()

        # Launch instance
        instance_id = manager.launch_confidential_instance(
            subnet_id=subnet_id,
            security_group_id=security_group_id,
            key_name='your-key-pair',
            ami_id='ami-0c55b159cbfafe1f0'  # Ubuntu 20.04 LTS
        )

        logger.info("Successfully set up AWS confidential computing environment")
        return 0
    except Exception as e:
        logger.error(f"Failed to set up AWS resources: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 