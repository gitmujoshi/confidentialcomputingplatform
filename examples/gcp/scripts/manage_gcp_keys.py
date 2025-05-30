#!/usr/bin/env python3

from google.cloud import kms
import logging
from typing import Dict, Optional
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCPKeyManager:
    """Utility class for managing GCP KMS keys."""
    
    def __init__(self, project_id: str, location: str = "global"):
        """Initialize GCP KMS client."""
        self.client = kms.KeyManagementServiceClient()
        self.project_id = project_id
        self.location = location
        self.key_ring_path = self.client.key_ring_path(
            project_id, location, "secure-key-ring"
        )
        
    def create_key_ring(self) -> str:
        """Create a new key ring."""
        try:
            key_ring = self.client.create_key_ring(
                request={
                    "parent": f"projects/{self.project_id}/locations/{self.location}",
                    "key_ring_id": "secure-key-ring",
                    "key_ring": {}
                }
            )
            logger.info(f"Created key ring: {key_ring.name}")
            return key_ring.name
            
        except Exception as e:
            logger.error(f"Error creating key ring: {str(e)}")
            raise
            
    def create_key(self, key_id: str, purpose: str = "ENCRYPT_DECRYPT") -> str:
        """Create a new KMS key."""
        try:
            key = self.client.create_crypto_key(
                request={
                    "parent": self.key_ring_path,
                    "crypto_key_id": key_id,
                    "crypto_key": {
                        "purpose": getattr(kms.CryptoKey.CryptoKeyPurpose, purpose),
                        "version_template": {
                            "algorithm": kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
                        }
                    }
                }
            )
            logger.info(f"Created KMS key: {key.name}")
            return key.name
            
        except Exception as e:
            logger.error(f"Error creating KMS key: {str(e)}")
            raise
            
    def encrypt_data(self, key_name: str, data: bytes) -> bytes:
        """Encrypt data using KMS key."""
        try:
            response = self.client.encrypt(
                request={
                    "name": key_name,
                    "plaintext": data
                }
            )
            return response.ciphertext
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise
            
    def decrypt_data(self, key_name: str, encrypted_data: bytes) -> bytes:
        """Decrypt data using KMS key."""
        try:
            response = self.client.decrypt(
                request={
                    "name": key_name,
                    "ciphertext": encrypted_data
                }
            )
            return response.plaintext
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise
            
    def rotate_key(self, key_name: str) -> None:
        """Rotate KMS key."""
        try:
            self.client.create_crypto_key_version(
                request={
                    "parent": key_name,
                    "crypto_key_version": {
                        "state": kms.CryptoKeyVersion.CryptoKeyVersionState.ENABLED
                    }
                }
            )
            logger.info(f"Created new key version for: {key_name}")
            
        except Exception as e:
            logger.error(f"Error rotating key: {str(e)}")
            raise
            
    def destroy_key_version(self, key_version_name: str) -> None:
        """Destroy a key version."""
        try:
            self.client.destroy_crypto_key_version(
                request={"name": key_version_name}
            )
            logger.info(f"Destroyed key version: {key_version_name}")
            
        except Exception as e:
            logger.error(f"Error destroying key version: {str(e)}")
            raise

def main():
    # Example usage
    key_manager = GCPKeyManager(project_id="your-project-id")
    
    try:
        # Create key ring
        key_ring_name = key_manager.create_key_ring()
        
        # Create a new key
        key_name = key_manager.create_key(
            key_id="example-key",
            purpose="ENCRYPT_DECRYPT"
        )
        
        # Encrypt some data
        data = b"Hello, GCP KMS!"
        encrypted_data = key_manager.encrypt_data(key_name, data)
        
        # Decrypt the data
        decrypted_data = key_manager.decrypt_data(key_name, encrypted_data)
        print(f"Decrypted data: {decrypted_data.decode()}")
        
        # Rotate the key
        key_manager.rotate_key(key_name)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 