#!/usr/bin/env python3

import os
import sys
import json
import logging
from google.cloud import storage
from google.cloud import kms
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPDataEncryptor:
    def __init__(self, project_id: str):
        """Initialize GCP data encryption."""
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.kms_client = kms.KeyManagementServiceClient()

    def encrypt_file(self, file_path: str, key_id: str) -> dict:
        """Encrypt a file using Cloud KMS."""
        try:
            # Generate a data key
            parent = self.kms_client.crypto_key_path(
                self.project_id, "global", "confidential-keyring", "confidential-key"
            )
            response = self.kms_client.encrypt(
                request={
                    "name": parent,
                    "plaintext": os.urandom(32)
                }
            )

            # Read file content
            with open(file_path, 'rb') as file:
                content = file.read()

            # Encrypt the content
            encrypted_content = self._encrypt_content(content, response.plaintext)

            # Save encrypted file
            encrypted_path = f"{file_path}.enc"
            with open(encrypted_path, 'wb') as file:
                file.write(encrypted_content)

            return {
                'original_file': file_path,
                'encrypted_file': encrypted_path,
                'encryption_key': base64.b64encode(response.ciphertext).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Failed to encrypt file: {e}")
            raise

    def _encrypt_content(self, content: bytes, key: bytes) -> bytes:
        """Encrypt content using AES-256."""
        try:
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

    def encrypt_storage_object(self, bucket_name: str, object_name: str, key_id: str) -> dict:
        """Encrypt a Cloud Storage object."""
        try:
            # Generate a data key
            parent = self.kms_client.crypto_key_path(
                self.project_id, "global", "confidential-keyring", "confidential-key"
            )
            response = self.kms_client.encrypt(
                request={
                    "name": parent,
                    "plaintext": os.urandom(32)
                }
            )

            # Get object content
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            content = blob.download_as_bytes()

            # Encrypt the content
            encrypted_content = self._encrypt_content(content, response.plaintext)

            # Upload encrypted content
            encrypted_name = f"{object_name}.enc"
            encrypted_blob = bucket.blob(encrypted_name)
            encrypted_blob.upload_from_string(
                encrypted_content,
                content_type='application/octet-stream',
                metadata={
                    'encryption-key': base64.b64encode(response.ciphertext).decode('utf-8')
                }
            )

            return {
                'bucket': bucket_name,
                'original_object': object_name,
                'encrypted_object': encrypted_name,
                'encryption_key': base64.b64encode(response.ciphertext).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Failed to encrypt storage object: {e}")
            raise

    def encrypt_directory(self, directory_path: str, key_id: str) -> list:
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

    def encrypt_bucket(self, bucket_name: str, key_id: str) -> list:
        """Encrypt all objects in a Cloud Storage bucket."""
        try:
            results = []
            bucket = self.storage_client.bucket(bucket_name)
            for blob in bucket.list_blobs():
                result = self.encrypt_storage_object(bucket_name, blob.name, key_id)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to encrypt bucket: {e}")
            raise

def main():
    """Main function to encrypt GCP data."""
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        encryptor = GCPDataEncryptor(project_id)

        # Encrypt a file
        file_result = encryptor.encrypt_file(
            'path/to/file',
            'your-key-id'
        )
        logger.info(f"File encryption result: {json.dumps(file_result, indent=2)}")

        # Encrypt a storage object
        storage_result = encryptor.encrypt_storage_object(
            'your-bucket',
            'your-object-name',
            'your-key-id'
        )
        logger.info(f"Storage object encryption result: {json.dumps(storage_result, indent=2)}")

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
        logger.error(f"Failed to encrypt GCP data: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 