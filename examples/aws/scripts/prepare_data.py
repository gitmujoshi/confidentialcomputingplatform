#!/usr/bin/env python3

import boto3
import logging
import json
import os
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSDataPreparator:
    def __init__(self, region: str = 'us-east-1'):
        """Initialize AWS data preparation with boto3 clients."""
        self.region = region
        self.s3 = boto3.client('s3', region_name=region)
        self.kms = boto3.client('kms', region_name=region)

    def prepare_training_data(self, local_path: str, bucket_name: str, key_id: str) -> Dict:
        """Prepare and upload training data to S3 with encryption."""
        try:
            # Create a data key for encryption
            data_key = self.kms.generate_data_key(
                KeyId=key_id,
                KeySpec='AES_256'
            )

            # Encrypt the data
            encrypted_data = self._encrypt_data(local_path, data_key['Plaintext'])

            # Upload encrypted data to S3
            s3_key = f"training-data/{os.path.basename(local_path)}.enc"
            self.s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=encrypted_data,
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=key_id,
                Metadata={
                    'encrypted-key': data_key['CiphertextBlob'].decode('utf-8')
                }
            )

            return {
                'bucket': bucket_name,
                'key': s3_key,
                'encryption_key': data_key['CiphertextBlob'].decode('utf-8')
            }
        except ClientError as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise

    def _encrypt_data(self, file_path: str, key: bytes) -> bytes:
        """Encrypt data using AES-256."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64

            # Generate a salt
            salt = os.urandom(16)
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key))

            # Create Fernet instance
            f = Fernet(key)

            # Read and encrypt file
            with open(file_path, 'rb') as file:
                data = file.read()
                encrypted_data = f.encrypt(data)

            return encrypted_data
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise

    def prepare_model_data(self, model_path: str, bucket_name: str, key_id: str) -> Dict:
        """Prepare and upload model data to S3 with encryption."""
        try:
            # Create a data key for encryption
            data_key = self.kms.generate_data_key(
                KeyId=key_id,
                KeySpec='AES_256'
            )

            # Encrypt the model
            encrypted_model = self._encrypt_data(model_path, data_key['Plaintext'])

            # Upload encrypted model to S3
            s3_key = f"models/{os.path.basename(model_path)}.enc"
            self.s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=encrypted_model,
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=key_id,
                Metadata={
                    'encrypted-key': data_key['CiphertextBlob'].decode('utf-8')
                }
            )

            return {
                'bucket': bucket_name,
                'key': s3_key,
                'encryption_key': data_key['CiphertextBlob'].decode('utf-8')
            }
        except ClientError as e:
            logger.error(f"Failed to prepare model data: {e}")
            raise

    def prepare_config_data(self, config: Dict, bucket_name: str, key_id: str) -> Dict:
        """Prepare and upload configuration data to S3 with encryption."""
        try:
            # Create a data key for encryption
            data_key = self.kms.generate_data_key(
                KeyId=key_id,
                KeySpec='AES_256'
            )

            # Encrypt the configuration
            encrypted_config = self._encrypt_data(
                json.dumps(config).encode('utf-8'),
                data_key['Plaintext']
            )

            # Upload encrypted configuration to S3
            s3_key = "config/confidential-config.enc"
            self.s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=encrypted_config,
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=key_id,
                Metadata={
                    'encrypted-key': data_key['CiphertextBlob'].decode('utf-8')
                }
            )

            return {
                'bucket': bucket_name,
                'key': s3_key,
                'encryption_key': data_key['CiphertextBlob'].decode('utf-8')
            }
        except ClientError as e:
            logger.error(f"Failed to prepare configuration data: {e}")
            raise

def main():
    """Main function to prepare AWS data."""
    try:
        # Initialize data preparator
        preparator = AWSDataPreparator()

        # Prepare training data
        training_data = preparator.prepare_training_data(
            'path/to/training/data',
            'your-bucket-name',
            'your-key-id'
        )
        logger.info(f"Training data prepared: {json.dumps(training_data, indent=2)}")

        # Prepare model data
        model_data = preparator.prepare_model_data(
            'path/to/model',
            'your-bucket-name',
            'your-key-id'
        )
        logger.info(f"Model data prepared: {json.dumps(model_data, indent=2)}")

        # Prepare configuration data
        config_data = preparator.prepare_config_data(
            {'key': 'value'},
            'your-bucket-name',
            'your-key-id'
        )
        logger.info(f"Configuration data prepared: {json.dumps(config_data, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to prepare AWS data: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 