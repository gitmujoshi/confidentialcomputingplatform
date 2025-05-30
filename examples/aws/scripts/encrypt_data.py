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

class AWSDataEncryptor:
    def __init__(self, region: str = 'us-east-1'):
        """Initialize AWS data encryption with boto3 clients."""
        self.region = region
        self.kms = boto3.client('kms', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)

    def encrypt_file(self, file_path: str, key_id: str) -> Dict:
        """Encrypt a file using KMS."""
        try:
            # Generate a data key
            data_key = self.kms.generate_data_key(
                KeyId=key_id,
                KeySpec='AES_256'
            )

            # Read file content
            with open(file_path, 'rb') as file:
                content = file.read()

            # Encrypt the content
            encrypted_content = self._encrypt_content(content, data_key['Plaintext'])

            # Save encrypted file
            encrypted_path = f"{file_path}.enc"
            with open(encrypted_path, 'wb') as file:
                file.write(encrypted_content)

            return {
                'original_file': file_path,
                'encrypted_file': encrypted_path,
                'encryption_key': data_key['CiphertextBlob'].decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Failed to encrypt file: {e}")
            raise

    def _encrypt_content(self, content: bytes, key: bytes) -> bytes:
        """Encrypt content using AES-256."""
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

            # Encrypt content
            encrypted_content = f.encrypt(content)

            return encrypted_content
        except Exception as e:
            logger.error(f"Failed to encrypt content: {e}")
            raise

    def encrypt_s3_object(self, bucket: str, key: str, key_id: str) -> Dict:
        """Encrypt an S3 object using KMS."""
        try:
            # Generate a data key
            data_key = self.kms.generate_data_key(
                KeyId=key_id,
                KeySpec='AES_256'
            )

            # Get object content
            response = self.s3.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()

            # Encrypt the content
            encrypted_content = self._encrypt_content(content, data_key['Plaintext'])

            # Upload encrypted content
            encrypted_key = f"{key}.enc"
            self.s3.put_object(
                Bucket=bucket,
                Key=encrypted_key,
                Body=encrypted_content,
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=key_id,
                Metadata={
                    'encrypted-key': data_key['CiphertextBlob'].decode('utf-8')
                }
            )

            return {
                'bucket': bucket,
                'original_key': key,
                'encrypted_key': encrypted_key,
                'encryption_key': data_key['CiphertextBlob'].decode('utf-8')
            }
        except ClientError as e:
            logger.error(f"Failed to encrypt S3 object: {e}")
            raise

    def encrypt_directory(self, directory_path: str, key_id: str) -> List[Dict]:
        """Encrypt all files in a directory."""
        try:
            results = []
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    result = self.encrypt_file(file_path, key_id)
                    results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to encrypt directory: {e}")
            raise

    def encrypt_bucket(self, bucket: str, key_id: str) -> List[Dict]:
        """Encrypt all objects in an S3 bucket."""
        try:
            results = []
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket):
                for obj in page.get('Contents', []):
                    result = self.encrypt_s3_object(bucket, obj['Key'], key_id)
                    results.append(result)
            return results
        except ClientError as e:
            logger.error(f"Failed to encrypt bucket: {e}")
            raise

def main():
    """Main function to encrypt AWS data."""
    try:
        # Initialize data encryptor
        encryptor = AWSDataEncryptor()

        # Encrypt a file
        file_result = encryptor.encrypt_file(
            'path/to/file',
            'your-key-id'
        )
        logger.info(f"File encryption result: {json.dumps(file_result, indent=2)}")

        # Encrypt an S3 object
        s3_result = encryptor.encrypt_s3_object(
            'your-bucket',
            'your-object-key',
            'your-key-id'
        )
        logger.info(f"S3 object encryption result: {json.dumps(s3_result, indent=2)}")

        # Encrypt a directory
        directory_results = encryptor.encrypt_directory(
            'path/to/directory',
            'your-key-id'
        )
        logger.info(f"Directory encryption results: {json.dumps(directory_results, indent=2)}")

        # Encrypt a bucket
        bucket_results = encryptor.encrypt_bucket(
            'your-bucket',
            'your-key-id'
        )
        logger.info(f"Bucket encryption results: {json.dumps(bucket_results, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to encrypt AWS data: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 