#!/usr/bin/env python3

import os
import sys
import json
import logging
import oci
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCIDataEncryptor:
    def __init__(self, config_path: str = None):
        """Initialize OCI data encryption."""
        self.config = oci.config.from_file(config_path) if config_path else oci.config.from_file()
        self.kms_client = oci.key_management.KmsCryptoClient(self.config)
        self.object_storage_client = oci.object_storage.ObjectStorageClient(self.config)

    def encrypt_file(self, file_path: str, key_id: str) -> dict:
        """Encrypt a file using OCI KMS."""
        try:
            # Generate a data key
            response = self.kms_client.generate_data_encryption_key(
                generate_data_encryption_key_details=oci.key_management.models.GenerateDataEncryptionKeyDetails(
                    include_plaintext_key=True,
                    key_id=key_id
                )
            )

            # Read file content
            with open(file_path, 'rb') as file:
                content = file.read()

            # Encrypt the content
            encrypted_content = self._encrypt_content(content, response.data.plaintext)

            # Save encrypted file
            encrypted_path = f"{file_path}.enc"
            with open(encrypted_path, 'wb') as file:
                file.write(encrypted_content)

            return {
                'original_file': file_path,
                'encrypted_file': encrypted_path,
                'encryption_key': base64.b64encode(response.data.ciphertext).decode('utf-8')
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

    def encrypt_storage_object(self, namespace: str, bucket_name: str, object_name: str, key_id: str) -> dict:
        """Encrypt an Object Storage object."""
        try:
            # Generate a data key
            response = self.kms_client.generate_data_encryption_key(
                generate_data_encryption_key_details=oci.key_management.models.GenerateDataEncryptionKeyDetails(
                    include_plaintext_key=True,
                    key_id=key_id
                )
            )

            # Get object content
            get_object_response = self.object_storage_client.get_object(
                namespace_name=namespace,
                bucket_name=bucket_name,
                object_name=object_name
            )
            content = get_object_response.data.content

            # Encrypt the content
            encrypted_content = self._encrypt_content(content, response.data.plaintext)

            # Upload encrypted content
            encrypted_name = f"{object_name}.enc"
            self.object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=bucket_name,
                object_name=encrypted_name,
                put_object_body=encrypted_content,
                metadata={
                    'encryption-key': base64.b64encode(response.data.ciphertext).decode('utf-8')
                }
            )

            return {
                'namespace': namespace,
                'bucket': bucket_name,
                'original_object': object_name,
                'encrypted_object': encrypted_name,
                'encryption_key': base64.b64encode(response.data.ciphertext).decode('utf-8')
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

    def encrypt_bucket(self, namespace: str, bucket_name: str, key_id: str) -> list:
        """Encrypt all objects in an Object Storage bucket."""
        try:
            results = []
            list_objects_response = self.object_storage_client.list_objects(
                namespace_name=namespace,
                bucket_name=bucket_name
            )
            for obj in list_objects_response.data.objects:
                result = self.encrypt_storage_object(namespace, bucket_name, obj.name, key_id)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to encrypt bucket: {e}")
            raise

def main():
    """Main function to encrypt OCI data."""
    try:
        # Initialize encryptor
        encryptor = OCIDataEncryptor()

        # Encrypt a file
        file_result = encryptor.encrypt_file(
            'path/to/file',
            'your-key-id'
        )
        logger.info(f"File encryption result: {json.dumps(file_result, indent=2)}")

        # Encrypt a storage object
        storage_result = encryptor.encrypt_storage_object(
            'your-namespace',
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
            'your-namespace',
            'your-bucket',
            'your-key-id'
        )
        logger.info(f"Bucket encryption results: {json.dumps(bucket_results, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to encrypt OCI data: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 