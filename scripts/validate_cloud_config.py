#!/usr/bin/env python3

import os
import json
import logging
from typing import Dict, List, Optional
import boto3
import oci
import azure.identity
from google.cloud import storage
from google.cloud import compute
from google.cloud import kms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudConfigValidator:
    """Utility class for validating cloud provider configurations."""
    
    def __init__(self):
        self.config_paths = {
            'aws': '~/.aws/credentials',
            'azure': '~/.azure/credentials.json',
            'gcp': '~/.config/gcloud/application_default_credentials.json',
            'oci': '~/.oci/config'
        }
        
    def validate_aws_config(self) -> bool:
        """Validate AWS configuration."""
        try:
            # Check credentials file
            if not os.path.exists(os.path.expanduser(self.config_paths['aws'])):
                logger.error("AWS credentials file not found")
                return False
                
            # Test AWS connection
            session = boto3.Session()
            sts = session.client('sts')
            sts.get_caller_identity()
            
            logger.info("AWS configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error validating AWS configuration: {str(e)}")
            return False
            
    def validate_azure_config(self) -> bool:
        """Validate Azure configuration."""
        try:
            # Check credentials
            credential = azure.identity.DefaultAzureCredential()
            token = credential.get_token("https://management.azure.com/.default")
            
            logger.info("Azure configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error validating Azure configuration: {str(e)}")
            return False
            
    def validate_gcp_config(self) -> bool:
        """Validate GCP configuration."""
        try:
            # Check credentials file
            if not os.path.exists(os.path.expanduser(self.config_paths['gcp'])):
                logger.error("GCP credentials file not found")
                return False
                
            # Test GCP connection
            storage_client = storage.Client()
            storage_client.list_buckets(max_results=1)
            
            logger.info("GCP configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error validating GCP configuration: {str(e)}")
            return False
            
    def validate_oci_config(self) -> bool:
        """Validate OCI configuration."""
        try:
            # Check config file
            if not os.path.exists(os.path.expanduser(self.config_paths['oci'])):
                logger.error("OCI config file not found")
                return False
                
            # Test OCI connection
            config = oci.config.from_file()
            identity = oci.identity.IdentityClient(config)
            identity.get_user(config["user"])
            
            logger.info("OCI configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error validating OCI configuration: {str(e)}")
            return False
            
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all cloud provider configurations."""
        results = {
            'aws': self.validate_aws_config(),
            'azure': self.validate_azure_config(),
            'gcp': self.validate_gcp_config(),
            'oci': self.validate_oci_config()
        }
        
        return results

def main():
    validator = CloudConfigValidator()
    results = validator.validate_all_configs()
    
    # Print results
    print("\nCloud Configuration Validation Results:")
    print("-" * 40)
    for provider, is_valid in results.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{provider.upper()}: {status}")
    print("-" * 40)
    
    # Exit with error if any configuration is invalid
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main() 