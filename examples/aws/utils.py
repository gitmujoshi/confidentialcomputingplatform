#!/usr/bin/env python3

import boto3
import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import subprocess
import shutil
import tempfile

class SecureEnvironmentManager:
    """Utility class for managing secure training environment on AWS."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the secure environment manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.session = boto3.Session()
        
        # Initialize AWS clients
        self.s3_client = self.session.client('s3')
        self.kms_client = self.session.client('kms')
        self.sagemaker_client = self.session.client('sagemaker')
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def setup_secure_storage(self) -> None:
        """Set up secure storage for training data and models."""
        try:
            # Create S3 bucket if it doesn't exist
            bucket_name = self.config["storage"]["bucket_name"]
            
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except Exception:
                # Create bucket with encryption
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.config["aws"]["region"]
                    }
                )
                
                # Enable default encryption
                self.s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration={
                        'Rules': [
                            {
                                'ApplyServerSideEncryptionByDefault': {
                                    'SSEAlgorithm': 'aws:kms',
                                    'KMSMasterKeyID': self.config["aws"]["kms_key_id"]
                                }
                            }
                        ]
                    }
                )
                
            # Create folders for data and models
            for folder in ['data/train', 'data/val', 'models']:
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"{folder}/",
                    Body=''
                )
                
            self.logger.info(f"Secure storage is ready")
            
        except Exception as e:
            self.logger.error(f"Error setting up secure storage: {str(e)}")
            raise
            
    def setup_encryption_keys(self) -> None:
        """Set up encryption keys for data and model encryption."""
        try:
            # Create or get KMS key
            key_id = self.config["aws"]["kms_key_id"]
            
            try:
                self.kms_client.describe_key(KeyId=key_id)
            except Exception:
                # Create KMS key
                response = self.kms_client.create_key(
                    Description="Key for secure training",
                    KeyUsage='ENCRYPT_DECRYPT',
                    Origin='AWS_KMS',
                    BypassPolicyLockoutSafetyCheck=False
                )
                key_id = response['KeyMetadata']['KeyId']
                
                # Enable key rotation
                self.kms_client.enable_key_rotation(KeyId=key_id)
                
            self.logger.info("Encryption keys are ready")
            
        except Exception as e:
            self.logger.error(f"Error setting up encryption keys: {str(e)}")
            raise
            
    def prepare_training_data(self, local_data_path: str) -> None:
        """
        Prepare and upload training data securely.
        
        Args:
            local_data_path: Path to local training data
        """
        try:
            # Create temporary directory for encrypted data
            with tempfile.TemporaryDirectory() as temp_dir:
                # Encrypt data
                encrypted_data_path = self._encrypt_data(local_data_path, temp_dir)
                
                # Upload to secure storage
                self._upload_to_storage(encrypted_data_path)
                
            self.logger.info("Training data prepared and uploaded securely")
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
            
    def _encrypt_data(self, data_path: str, output_dir: str) -> str:
        """Encrypt training data using AWS KMS key."""
        try:
            # Read data
            with open(data_path, 'rb') as f:
                data = f.read()
                
            # Encrypt data
            response = self.kms_client.encrypt(
                KeyId=self.config["aws"]["kms_key_id"],
                Plaintext=data
            )
            
            # Save encrypted data
            encrypted_path = os.path.join(output_dir, 'encrypted_data.bin')
            with open(encrypted_path, 'wb') as f:
                f.write(response['CiphertextBlob'])
                
            return encrypted_path
            
        except Exception as e:
            self.logger.error(f"Error encrypting data: {str(e)}")
            raise
            
    def _upload_to_storage(self, data_path: str) -> None:
        """Upload encrypted data to secure storage."""
        try:
            # Upload to S3
            self.s3_client.upload_file(
                data_path,
                self.config["storage"]["bucket_name"],
                f"data/train/{os.path.basename(data_path)}",
                ExtraArgs={
                    'ServerSideEncryption': 'aws:kms',
                    'SSEKMSKeyId': self.config["aws"]["kms_key_id"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error uploading data: {str(e)}")
            raise
            
    def cleanup_resources(self) -> None:
        """Clean up temporary resources and files."""
        try:
            # Implementation to clean up resources
            pass
            
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {str(e)}")
            raise
            
    def verify_environment(self) -> bool:
        """
        Verify the secure training environment.
        
        Returns:
            bool: True if environment is ready, False otherwise
        """
        try:
            # Check storage access
            self._verify_storage_access()
            
            # Check encryption keys
            self._verify_encryption_keys()
            
            # Check SageMaker resources
            self._verify_sagemaker_resources()
            
            self.logger.info("Secure environment verification completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment verification failed: {str(e)}")
            return False
            
    def _verify_storage_access(self) -> None:
        """Verify access to secure storage."""
        try:
            self.s3_client.list_objects_v2(
                Bucket=self.config["storage"]["bucket_name"],
                MaxKeys=1
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying storage access: {str(e)}")
            raise
            
    def _verify_encryption_keys(self) -> None:
        """Verify encryption key access and validity."""
        try:
            self.kms_client.describe_key(
                KeyId=self.config["aws"]["kms_key_id"]
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying encryption keys: {str(e)}")
            raise
            
    def _verify_sagemaker_resources(self) -> None:
        """Verify SageMaker resources availability."""
        try:
            self.sagemaker_client.list_training_jobs(
                MaxResults=1
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying SageMaker resources: {str(e)}")
            raise

def main():
    # Example usage
    manager = SecureEnvironmentManager()
    
    # Setup secure environment
    manager.setup_secure_storage()
    manager.setup_encryption_keys()
    
    # Prepare training data
    manager.prepare_training_data("/path/to/training/data")
    
    # Verify environment
    if manager.verify_environment():
        print("Secure environment is ready for training")
    else:
        print("Secure environment setup failed")
        
    # Cleanup
    manager.cleanup_resources()

if __name__ == "__main__":
    main() 