#!/usr/bin/env python3

import os
import sys
import json
import logging
import azure.identity
import azure.mgmt.resource
import azure.mgmt.keyvault
import azure.mgmt.storage
import azure.mgmt.attestation
import azure.mgmt.compute
from azure.mgmt.keyvault.models import (
    VaultProperties,
    Sku,
    AccessPolicyEntry,
    Permissions,
    KeyPermissions,
    SecretPermissions,
    CertificatePermissions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureResourceManager:
    def __init__(self, config_path: str = "config.json"):
        """Initialize Azure resource manager."""
        self.config = self._load_config(config_path)
        self.credential = azure.identity.DefaultAzureCredential()
        self.subscription_id = self.config["subscription_id"]
        
        # Initialize Azure clients
        self.resource_client = azure.mgmt.resource.ResourceManagementClient(
            self.credential, self.subscription_id
        )
        self.keyvault_client = azure.mgmt.keyvault.KeyVaultManagementClient(
            self.credential, self.subscription_id
        )
        self.storage_client = azure.mgmt.storage.StorageManagementClient(
            self.credential, self.subscription_id
        )
        self.attestation_client = azure.mgmt.attestation.AttestationManagementClient(
            self.credential, self.subscription_id
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def create_resource_group(self):
        """Create Azure resource group."""
        try:
            self.resource_client.resource_groups.create_or_update(
                self.config["resource_group_name"],
                {"location": self.config["location"]}
            )
            logger.info(f"Resource group {self.config['resource_group_name']} created/updated")
        except Exception as e:
            logger.error(f"Error creating resource group: {str(e)}")
            raise
            
    def create_key_vault(self):
        """Create Azure Key Vault with enhanced security."""
        try:
            # Get current user/service principal
            current_user = self.credential.get_token("https://vault.azure.net/.default")
            
            # Create Key Vault with enhanced security
            vault_properties = VaultProperties(
                tenant_id=current_user.tenant_id,
                sku=Sku(family="A", name="standard"),
                enable_rbac_authorization=True,
                enable_soft_delete=True,
                soft_delete_retention_in_days=90,
                enable_purge_protection=True,
                network_acls={
                    "default_action": "Deny",
                    "bypass": "AzureServices",
                    "ip_rules": [],
                    "virtual_network_rules": []
                }
            )
            
            self.keyvault_client.vaults.begin_create_or_update(
                self.config["resource_group_name"],
                self.config["key_vault_name"],
                {
                    "location": self.config["location"],
                    "properties": vault_properties
                }
            ).wait()
            
            logger.info(f"Key Vault {self.config['key_vault_name']} created/updated")
            
            # Create encryption keys
            self._create_encryption_keys()
            
        except Exception as e:
            logger.error(f"Error creating Key Vault: {str(e)}")
            raise
            
    def _create_encryption_keys(self):
        """Create encryption keys in Key Vault."""
        try:
            # Create training data key
            self.keyvault_client.keys.begin_create_or_update(
                self.config["resource_group_name"],
                self.config["key_vault_name"],
                self.config["data"]["key_name"],
                {
                    "kty": "RSA",
                    "key_size": 2048,
                    "key_ops": ["encrypt", "decrypt", "sign", "verify", "wrapKey", "unwrapKey"],
                    "attributes": {
                        "enabled": True
                    }
                }
            ).wait()
            
            # Create model encryption key
            self.keyvault_client.keys.begin_create_or_update(
                self.config["resource_group_name"],
                self.config["key_vault_name"],
                self.config["data"]["model_key_name"],
                {
                    "kty": "RSA",
                    "key_size": 2048,
                    "key_ops": ["encrypt", "decrypt", "sign", "verify", "wrapKey", "unwrapKey"],
                    "attributes": {
                        "enabled": True
                    }
                }
            ).wait()
            
            logger.info("Encryption keys created successfully")
            
        except Exception as e:
            logger.error(f"Error creating encryption keys: {str(e)}")
            raise
            
    def create_storage_account(self):
        """Create Azure Storage account with enhanced security."""
        try:
            self.storage_client.storage_accounts.begin_create(
                self.config["resource_group_name"],
                self.config["storage_account_name"],
                {
                    "sku": {
                        "name": "Standard_LRS"
                    },
                    "kind": "StorageV2",
                    "location": self.config["location"],
                    "enable_https_traffic_only": True,
                    "minimum_tls_version": "TLS1_2",
                    "network_rule_set": {
                        "default_action": "Deny",
                        "bypass": ["AzureServices"],
                        "ip_rules": [],
                        "virtual_network_rules": []
                    },
                    "encryption": {
                        "services": {
                            "blob": {
                                "enabled": True
                            }
                        },
                        "key_source": "Microsoft.Keyvault",
                        "key_vault_properties": {
                            "key_name": self.config["data"]["key_name"],
                            "key_vault_uri": f"https://{self.config['key_vault_name']}.vault.azure.net"
                        }
                    }
                }
            ).wait()
            
            # Create container
            storage_client = azure.storage.blob.BlobServiceClient(
                account_url=f"https://{self.config['storage_account_name']}.blob.core.windows.net",
                credential=self.credential
            )
            
            storage_client.create_container(
                self.config["container_name"],
                metadata={
                    "confidential-computing": "true"
                }
            )
            
            logger.info(f"Storage account {self.config['storage_account_name']} created/updated")
            
        except Exception as e:
            logger.error(f"Error creating storage account: {str(e)}")
            raise
            
    def create_attestation_provider(self):
        """Create Azure Attestation provider."""
        try:
            self.attestation_client.attestation_providers.create(
                self.config["resource_group_name"],
                self.config["attestation_endpoint"].split(".")[0],
                {
                    "location": self.config["location"],
                    "properties": {
                        "status": "Ready",
                        "attestUri": f"https://{self.config['attestation_endpoint']}",
                        "trustModel": "AAD"
                    }
                }
            )
            
            logger.info(f"Attestation provider created at {self.config['attestation_endpoint']}")
            
        except Exception as e:
            logger.error(f"Error creating attestation provider: {str(e)}")
            raise

    def create_network_security_group(self):
        """Create network security group with strict rules."""
        try:
            nsg_name = f"{self.config['resource_group_name']}-nsg"
            
            # Create NSG
            nsg = self.network_client.network_security_groups.begin_create_or_update(
                self.config["resource_group_name"],
                nsg_name,
                {
                    "location": self.config["location"],
                    "security_rules": [
                        {
                            "name": "DenyAllInbound",
                            "protocol": "*",
                            "source_address_prefix": "*",
                            "destination_address_prefix": "*",
                            "access": "Deny",
                            "priority": 4096,
                            "direction": "Inbound",
                            "source_port_range": "*",
                            "destination_port_range": "*"
                        },
                        {
                            "name": "AllowVnetInbound",
                            "protocol": "*",
                            "source_address_prefix": "VirtualNetwork",
                            "destination_address_prefix": "VirtualNetwork",
                            "access": "Allow",
                            "priority": 1000,
                            "direction": "Inbound",
                            "source_port_range": "*",
                            "destination_port_range": "*"
                        },
                        {
                            "name": "AllowAzureLoadBalancerInbound",
                            "protocol": "*",
                            "source_address_prefix": "AzureLoadBalancer",
                            "destination_address_prefix": "*",
                            "access": "Allow",
                            "priority": 2000,
                            "direction": "Inbound",
                            "source_port_range": "*",
                            "destination_port_range": "*"
                        },
                        {
                            "name": "AllowKeyVaultOutbound",
                            "protocol": "Tcp",
                            "source_address_prefix": "*",
                            "destination_address_prefix": "AzureKeyVault",
                            "access": "Allow",
                            "priority": 1000,
                            "direction": "Outbound",
                            "source_port_range": "*",
                            "destination_port_range": "443"
                        },
                        {
                            "name": "AllowStorageOutbound",
                            "protocol": "Tcp",
                            "source_address_prefix": "*",
                            "destination_address_prefix": "Storage",
                            "access": "Allow",
                            "priority": 1100,
                            "direction": "Outbound",
                            "source_port_range": "*",
                            "destination_port_range": "443"
                        },
                        {
                            "name": "DenyAllOutbound",
                            "protocol": "*",
                            "source_address_prefix": "*",
                            "destination_address_prefix": "*",
                            "access": "Deny",
                            "priority": 4096,
                            "direction": "Outbound",
                            "source_port_range": "*",
                            "destination_port_range": "*"
                        }
                    ]
                }
            ).wait()
            
            logger.info(f"Network security group {nsg_name} created/updated")
            
        except Exception as e:
            logger.error(f"Error creating network security group: {str(e)}")
            raise

    def setup_resources(self):
        """Set up all Azure resources."""
        try:
            self.create_resource_group()
            self.create_network_security_group()
            self.create_key_vault()
            self.create_storage_account()
            self.create_attestation_provider()
            logger.info("All Azure resources set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up resources: {str(e)}")
            raise

def main():
    manager = AzureResourceManager()
    manager.setup_resources()

if __name__ == "__main__":
    main() 