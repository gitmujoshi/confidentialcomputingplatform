#!/usr/bin/env python3

from google.cloud import storage
from google.cloud import kms
from google.cloud import aiplatform
import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import subprocess
import shutil
import tempfile

class SecureEnvironmentManager:
    """Utility class for managing secure training environment on GCP."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the secure environment manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize GCP clients
        self.storage_client = storage.Client()
        self.kms_client = kms.KeyManagementServiceClient()
        
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
            # Create GCS bucket if it doesn't exist
            bucket_name = self.config["storage"]["bucket_name"]
            
            try:
                bucket = self.storage_client.get_bucket(bucket_name)
            except Exception:
                # Create bucket with encryption
                bucket = self.storage_client.create_bucket(
                    bucket_name,
                    location=self.config["gcp"]["location"]
                )
                
                # Set default encryption
                bucket.default_kms_key_name = self.config["gcp"]["kms_key_name"]
                bucket.patch()
                
            # Create folders for data and models
            for folder in ['data/train', 'data/val', 'models']:
                blob = bucket.blob(f"{folder}/")
                blob.upload_from_string('')
                
            self.logger.info(f"Secure storage is ready")
            
        except Exception as e:
            self.logger.error(f"Error setting up secure storage: {str(e)}")
            raise
            
    def setup_encryption_keys(self) -> None:
        """Set up encryption keys for data and model encryption."""
        try:
            # Create or get key ring
            key_ring_name = self.config["gcp"]["key_ring_name"]
            key_ring_path = self.kms_client.key_ring_path(
                self.config["gcp"]["project_id"],
                self.config["gcp"]["location"],
                key_ring_name
            )
            
            try:
                self.kms_client.get_key_ring(name=key_ring_path)
            except Exception:
                # Create key ring
                self.kms_client.create_key_ring(
                    request={
                        "parent": f"projects/{self.config['gcp']['project_id']}/locations/{self.config['gcp']['location']}",
                        "key_ring_id": key_ring_name,
                        "key_ring": {}
                    }
                )
                
            # Create or get key
            key_name = self.config["gcp"]["kms_key_name"]
            key_path = self.kms_client.crypto_key_path(
                self.config["gcp"]["project_id"],
                self.config["gcp"]["location"],
                key_ring_name,
                key_name
            )
            
            try:
                self.kms_client.get_crypto_key(name=key_path)
            except Exception:
                # Create key
                self.kms_client.create_crypto_key(
                    request={
                        "parent": key_ring_path,
                        "crypto_key_id": key_name,
                        "crypto_key": {
                            "purpose": kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT,
                            "version_template": {
                                "algorithm": kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
                            }
                        }
                    }
                )
                
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
        """Encrypt training data using GCP KMS key."""
        try:
            # Read data
            with open(data_path, 'rb') as f:
                data = f.read()
                
            # Encrypt data
            response = self.kms_client.encrypt(
                request={
                    "name": self.config["gcp"]["kms_key_name"],
                    "plaintext": data
                }
            )
            
            # Save encrypted data
            encrypted_path = os.path.join(output_dir, 'encrypted_data.bin')
            with open(encrypted_path, 'wb') as f:
                f.write(response.ciphertext)
                
            return encrypted_path
            
        except Exception as e:
            self.logger.error(f"Error encrypting data: {str(e)}")
            raise
            
    def _upload_to_storage(self, data_path: str) -> None:
        """Upload encrypted data to secure storage."""
        try:
            # Upload to GCS
            bucket = self.storage_client.bucket(self.config["storage"]["bucket_name"])
            blob = bucket.blob(f"data/train/{os.path.basename(data_path)}")
            blob.upload_from_filename(data_path)
            
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
            
            # Check Vertex AI resources
            self._verify_vertex_ai_resources()
            
            self.logger.info("Secure environment verification completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment verification failed: {str(e)}")
            return False
            
    def _verify_storage_access(self) -> None:
        """Verify access to secure storage."""
        try:
            bucket = self.storage_client.bucket(self.config["storage"]["bucket_name"])
            list(bucket.list_blobs(max_results=1))
            
        except Exception as e:
            self.logger.error(f"Error verifying storage access: {str(e)}")
            raise
            
    def _verify_encryption_keys(self) -> None:
        """Verify encryption key access and validity."""
        try:
            self.kms_client.get_crypto_key(
                name=self.config["gcp"]["kms_key_name"]
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying encryption keys: {str(e)}")
            raise
            
    def _verify_vertex_ai_resources(self) -> None:
        """Verify Vertex AI resources availability."""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=self.config["gcp"]["project_id"],
                location=self.config["gcp"]["location"]
            )
            
            # List custom jobs
            aiplatform.CustomJob.list()
            
        except Exception as e:
            self.logger.error(f"Error verifying Vertex AI resources: {str(e)}")
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