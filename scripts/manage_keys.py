#!/usr/bin/env python3

import os
import sys
import json
import logging
import base64
from typing import Dict, List, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import azure.keyvault.keys
import oci.key_management
import boto3
from google.cloud import kms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeyManager:
    """Utility class for managing encryption keys."""
    
    def __init__(self, cloud_provider: str = None):
        self.cloud_provider = cloud_provider
        self.key_store = {}
        
    def generate_local_key(self) -> bytes:
        """Generate a local encryption key."""
        try:
            key = Fernet.generate_key()
            logger.info("Generated local encryption key")
            return key
            
        except Exception as e:
            logger.error(f"Error generating local key: {str(e)}")
            raise
            
    def store_key_azure(self, key: bytes, key_name: str) -> bool:
        """Store key in Azure Key Vault."""
        try:
            credential = azure.identity.DefaultAzureCredential()
            client = azure.keyvault.keys.KeyClient(
                vault_url="https://your-key-vault.vault.azure.net/",
                credential=credential
            )
            
            # Convert key to base64
            key_b64 = base64.b64encode(key).decode('utf-8')
            
            # Create key
            client.create_key(
                name=key_name,
                key_type="oct",
                size=256,
                key_operations=["encrypt", "decrypt"]
            )
            
            logger.info(f"Stored key '{key_name}' in Azure Key Vault")
            return True
            
        except Exception as e:
            logger.error(f"Error storing key in Azure: {str(e)}")
            return False
            
    def store_key_oci(self, key: bytes, key_name: str) -> bool:
        """Store key in OCI Vault."""
        try:
            config = oci.config.from_file()
            kms_client = oci.key_management.KmsVaultClient(config)
            
            # Create key
            key_details = oci.key_management.models.CreateKeyDetails(
                compartment_id=config["tenancy"],
                display_name=key_name,
                key_shape=oci.key_management.models.KeyShape(
                    algorithm="AES",
                    length=256
                )
            )
            
            kms_client.create_key(key_details)
            
            logger.info(f"Stored key '{key_name}' in OCI Vault")
            return True
            
        except Exception as e:
            logger.error(f"Error storing key in OCI: {str(e)}")
            return False
            
    def store_key_aws(self, key: bytes, key_name: str) -> bool:
        """Store key in AWS KMS."""
        try:
            kms_client = boto3.client('kms')
            
            # Create key
            response = kms_client.create_key(
                Description=f"Key for {key_name}",
                KeyUsage='ENCRYPT_DECRYPT',
                Origin='AWS_KMS',
                BypassPolicyLockoutSafetyCheck=False
            )
            
            logger.info(f"Stored key '{key_name}' in AWS KMS")
            return True
            
        except Exception as e:
            logger.error(f"Error storing key in AWS: {str(e)}")
            return False
            
    def store_key_gcp(self, key: bytes, key_name: str) -> bool:
        """Store key in GCP KMS."""
        try:
            client = kms.KeyManagementServiceClient()
            
            # Create key ring
            parent = client.key_ring_path(
                "your-project",
                "global",
                "your-key-ring"
            )
            
            # Create key
            key = client.create_crypto_key(
                request={
                    "parent": parent,
                    "crypto_key_id": key_name,
                    "crypto_key": {
                        "purpose": kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
                    }
                }
            )
            
            logger.info(f"Stored key '{key_name}' in GCP KMS")
            return True
            
        except Exception as e:
            logger.error(f"Error storing key in GCP: {str(e)}")
            return False
            
    def rotate_key(self, key_name: str) -> bool:
        """Rotate an existing key."""
        try:
            if self.cloud_provider == 'azure':
                return self._rotate_key_azure(key_name)
            elif self.cloud_provider == 'oci':
                return self._rotate_key_oci(key_name)
            elif self.cloud_provider == 'aws':
                return self._rotate_key_aws(key_name)
            elif self.cloud_provider == 'gcp':
                return self._rotate_key_gcp(key_name)
            else:
                logger.error("No cloud provider specified for key rotation")
                return False
                
        except Exception as e:
            logger.error(f"Error rotating key: {str(e)}")
            return False
            
    def _rotate_key_azure(self, key_name: str) -> bool:
        """Rotate key in Azure Key Vault."""
        try:
            credential = azure.identity.DefaultAzureCredential()
            client = azure.keyvault.keys.KeyClient(
                vault_url="https://your-key-vault.vault.azure.net/",
                credential=credential
            )
            
            # Create new key version
            client.create_key(
                name=key_name,
                key_type="oct",
                size=256,
                key_operations=["encrypt", "decrypt"]
            )
            
            logger.info(f"Rotated key '{key_name}' in Azure Key Vault")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating key in Azure: {str(e)}")
            return False
            
    def _rotate_key_oci(self, key_name: str) -> bool:
        """Rotate key in OCI Vault."""
        try:
            config = oci.config.from_file()
            kms_client = oci.key_management.KmsVaultClient(config)
            
            # Create new key version
            kms_client.create_key_version(
                key_id=key_name,
                key_version_details=oci.key_management.models.CreateKeyVersionDetails()
            )
            
            logger.info(f"Rotated key '{key_name}' in OCI Vault")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating key in OCI: {str(e)}")
            return False
            
    def _rotate_key_aws(self, key_name: str) -> bool:
        """Rotate key in AWS KMS."""
        try:
            kms_client = boto3.client('kms')
            
            # Create new key version
            kms_client.create_alias(
                AliasName=f"alias/{key_name}-new",
                TargetKeyId=key_name
            )
            
            logger.info(f"Rotated key '{key_name}' in AWS KMS")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating key in AWS: {str(e)}")
            return False
            
    def _rotate_key_gcp(self, key_name: str) -> bool:
        """Rotate key in GCP KMS."""
        try:
            client = kms.KeyManagementServiceClient()
            
            # Create new key version
            parent = client.crypto_key_path(
                "your-project",
                "global",
                "your-key-ring",
                key_name
            )
            
            client.create_crypto_key_version(
                request={
                    "parent": parent,
                    "crypto_key_version": {
                        "state": kms.CryptoKeyVersion.CryptoKeyVersionState.ENABLED
                    }
                }
            )
            
            logger.info(f"Rotated key '{key_name}' in GCP KMS")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating key in GCP: {str(e)}")
            return False

def main():
    # Example usage
    key_manager = KeyManager(cloud_provider='azure')
    
    # Generate and store a new key
    key = key_manager.generate_local_key()
    key_manager.store_key_azure(key, "test-key")
    
    # Rotate the key
    key_manager.rotate_key("test-key")

if __name__ == "__main__":
    main() 