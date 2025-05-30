#!/usr/bin/env python3

import os
import sys
import json
import logging
import oci
from typing import Dict, List, Optional
from datetime import datetime
import base64
from cryptography.fernet import Fernet
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCIManager:
    """Utility class for managing OCI resources and secure enclaves."""
    
    def __init__(self, config_path: str = "~/.oci/config"):
        """
        Initialize OCI manager.
        
        Args:
            config_path: Path to OCI configuration file
        """
        self.config = oci.config.from_file(config_path)
        self.identity_client = oci.identity.IdentityClient(self.config)
        self.compute_client = oci.core.ComputeClient(self.config)
        self.vault_client = oci.vault.VaultsClient(self.config)
        self.kms_client = oci.key_management.KmsVaultClient(self.config)
        self.storage_client = oci.object_storage.ObjectStorageClient(self.config)
        
    def setup_secure_enclave(self, compartment_id: str, display_name: str) -> str:
        """
        Set up a secure enclave instance.
        
        Args:
            compartment_id: OCI compartment ID
            display_name: Display name for the instance
            
        Returns:
            str: Instance ID
        """
        try:
            # Create instance with confidential computing enabled
            instance_details = oci.core.models.LaunchInstanceDetails(
                compartment_id=compartment_id,
                display_name=display_name,
                shape="VM.Standard.E4.Flex",  # Confidential computing shape
                shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
                    ocpus=4,
                    memory_in_gbs=64
                ),
                source_details=oci.core.models.InstanceSourceViaImageDetails(
                    image_id="ocid1.image.oc1..example",  # Replace with actual image ID
                    source_type="image"
                ),
                metadata={
                    "confidential-computing": "true"
                }
            )
            
            response = self.compute_client.launch_instance(instance_details)
            instance_id = response.data.id
            
            logger.info(f"Created secure enclave instance: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating secure enclave: {str(e)}")
            raise
            
    def setup_secure_storage(self, compartment_id: str, bucket_name: str) -> str:
        """
        Set up secure object storage bucket.
        
        Args:
            compartment_id: OCI compartment ID
            bucket_name: Name for the bucket
            
        Returns:
            str: Bucket name
        """
        try:
            # Create namespace
            namespace = self.storage_client.get_namespace().data
            
            # Create bucket with encryption
            bucket_details = oci.object_storage.models.CreateBucketDetails(
                name=bucket_name,
                compartment_id=compartment_id,
                metadata={
                    "confidential-computing": "true"
                },
                public_access_type="NoPublicAccess",
                versioning="Enabled"
            )
            
            response = self.storage_client.create_bucket(
                namespace_name=namespace,
                create_bucket_details=bucket_details
            )
            
            logger.info(f"Created secure storage bucket: {bucket_name}")
            return bucket_name
            
        except Exception as e:
            logger.error(f"Error creating secure storage: {str(e)}")
            raise
            
    def setup_encryption_keys(self, compartment_id: str, vault_name: str) -> str:
        """
        Set up encryption keys in OCI Vault.
        
        Args:
            compartment_id: OCI compartment ID
            vault_name: Name for the vault
            
        Returns:
            str: Vault ID
        """
        try:
            # Create vault
            vault_details = oci.vault.models.CreateVaultDetails(
                compartment_id=compartment_id,
                display_name=vault_name,
                vault_type="VIRTUAL_PRIVATE"
            )
            
            vault = self.vault_client.create_vault(vault_details)
            vault_id = vault.data.id
            
            # Create master encryption key
            key_details = oci.key_management.models.CreateKeyDetails(
                compartment_id=compartment_id,
                display_name=f"{vault_name}-master-key",
                key_shape=oci.key_management.models.KeyShape(
                    algorithm="AES",
                    length=256
                )
            )
            
            self.kms_client.create_key(
                vault_id=vault_id,
                create_key_details=key_details
            )
            
            logger.info(f"Created vault and encryption keys: {vault_id}")
            return vault_id
            
        except Exception as e:
            logger.error(f"Error setting up encryption keys: {str(e)}")
            raise
            
    def verify_enclave_setup(self, instance_id: str) -> bool:
        """
        Verify secure enclave setup.
        
        Args:
            instance_id: OCI instance ID
            
        Returns:
            bool: True if setup is valid
        """
        try:
            # Get instance details
            instance = self.compute_client.get_instance(instance_id)
            
            # Verify confidential computing features
            if not instance.data.metadata.get("confidential-computing"):
                logger.error("Instance is not configured for confidential computing")
                return False
                
            # Verify shape
            if not instance.data.shape.startswith("VM.Standard.E4.Flex"):
                logger.error("Instance is not using a confidential computing shape")
                return False
                
            logger.info("Secure enclave setup is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying enclave setup: {str(e)}")
            return False
            
    def cleanup_resources(self, instance_id: str, bucket_name: str, vault_id: str) -> None:
        """
        Clean up OCI resources.
        
        Args:
            instance_id: OCI instance ID
            bucket_name: Bucket name
            vault_id: Vault ID
        """
        try:
            # Terminate instance
            self.compute_client.terminate_instance(instance_id)
            
            # Delete bucket
            namespace = self.storage_client.get_namespace().data
            self.storage_client.delete_bucket(namespace, bucket_name)
            
            # Delete vault
            self.vault_client.delete_vault(vault_id)
            
            logger.info("Cleaned up OCI resources")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
            raise

def main():
    # Example usage
    oci_manager = OCIManager()
    
    try:
        # Setup secure environment
        instance_id = oci_manager.setup_secure_enclave(
            compartment_id="ocid1.compartment.oc1..example",
            display_name="secure-enclave-1"
        )
        
        bucket_name = oci_manager.setup_secure_storage(
            compartment_id="ocid1.compartment.oc1..example",
            bucket_name="secure-storage-1"
        )
        
        vault_id = oci_manager.setup_encryption_keys(
            compartment_id="ocid1.compartment.oc1..example",
            vault_name="secure-vault-1"
        )
        
        # Verify setup
        if oci_manager.verify_enclave_setup(instance_id):
            print("Secure environment setup completed successfully")
        else:
            print("Secure environment setup failed")
            
        # Cleanup (uncomment to use)
        # oci_manager.cleanup_resources(instance_id, bucket_name, vault_id)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 