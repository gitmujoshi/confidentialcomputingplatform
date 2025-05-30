#!/usr/bin/env python3

import boto3
import logging
from typing import Dict, Optional
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AWSKeyManager:
    """Utility class for managing AWS KMS keys."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize AWS KMS client."""
        self.kms_client = boto3.client('kms', region_name=region_name)
        
    def create_key(self, description: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create a new KMS key."""
        try:
            response = self.kms_client.create_key(
                Description=description,
                KeyUsage='ENCRYPT_DECRYPT',
                Origin='AWS_KMS',
                Tags=[{'TagKey': k, 'TagValue': v} for k, v in (tags or {}).items()]
            )
            key_id = response['KeyMetadata']['KeyId']
            logger.info(f"Created KMS key: {key_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Error creating KMS key: {str(e)}")
            raise
            
    def encrypt_data(self, key_id: str, data: bytes) -> bytes:
        """Encrypt data using KMS key."""
        try:
            response = self.kms_client.encrypt(
                KeyId=key_id,
                Plaintext=data
            )
            return response['CiphertextBlob']
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise
            
    def decrypt_data(self, key_id: str, encrypted_data: bytes) -> bytes:
        """Decrypt data using KMS key."""
        try:
            response = self.kms_client.decrypt(
                KeyId=key_id,
                CiphertextBlob=encrypted_data
            )
            return response['Plaintext']
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise
            
    def rotate_key(self, key_id: str) -> None:
        """Rotate KMS key."""
        try:
            self.kms_client.enable_key_rotation(KeyId=key_id)
            logger.info(f"Enabled key rotation for: {key_id}")
            
        except Exception as e:
            logger.error(f"Error rotating key: {str(e)}")
            raise
            
    def delete_key(self, key_id: str, pending_window_days: int = 7) -> None:
        """Schedule KMS key deletion."""
        try:
            self.kms_client.schedule_key_deletion(
                KeyId=key_id,
                PendingWindowInDays=pending_window_days
            )
            logger.info(f"Scheduled key deletion for: {key_id}")
            
        except Exception as e:
            logger.error(f"Error deleting key: {str(e)}")
            raise

def main():
    # Example usage
    key_manager = AWSKeyManager()
    
    try:
        # Create a new key
        key_id = key_manager.create_key(
            description="Example KMS key",
            tags={"Purpose": "Testing", "Environment": "Development"}
        )
        
        # Encrypt some data
        data = b"Hello, AWS KMS!"
        encrypted_data = key_manager.encrypt_data(key_id, data)
        
        # Decrypt the data
        decrypted_data = key_manager.decrypt_data(key_id, encrypted_data)
        print(f"Decrypted data: {decrypted_data.decode()}")
        
        # Enable key rotation
        key_manager.rotate_key(key_id)
        
        # Schedule key deletion (uncomment to use)
        # key_manager.delete_key(key_id)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 