#!/usr/bin/env python3

import os
import sys
import json
import logging
import azure.identity
import azure.mgmt.resource
import azure.mgmt.keyvault
import azure.mgmt.storage
import azure.mgmt.network
import azure.mgmt.attestation
from azure.core.exceptions import ResourceNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityVerifier:
    def __init__(self, config_path: str = "config.json"):
        """Initialize security verifier."""
        self.config = self._load_config(config_path)
        self.credential = azure.identity.DefaultAzureCredential()
        self.subscription_id = self.config["subscription_id"]
        
        # Initialize Azure clients
        self.keyvault_client = azure.mgmt.keyvault.KeyVaultManagementClient(
            self.credential, self.subscription_id
        )
        self.storage_client = azure.mgmt.storage.StorageManagementClient(
            self.credential, self.subscription_id
        )
        self.network_client = azure.mgmt.network.NetworkManagementClient(
            self.credential, self.subscription_id
        )
        self.attestation_client = azure.mgmt.attestation.AttestationManagementClient(
            self.credential, self.subscription_id
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def verify_key_vault_security(self) -> bool:
        """Verify Key Vault security settings."""
        try:
            vault = self.keyvault_client.vaults.get(
                self.config["resource_group_name"],
                self.config["key_vault_name"]
            )
            
            # Check RBAC authorization
            if not vault.properties.enable_rbac_authorization:
                logger.error("Key Vault RBAC authorization is not enabled")
                return False
                
            # Check soft delete
            if not vault.properties.enable_soft_delete:
                logger.error("Key Vault soft delete is not enabled")
                return False
                
            # Check purge protection
            if not vault.properties.enable_purge_protection:
                logger.error("Key Vault purge protection is not enabled")
                return False
                
            # Check network rules
            if vault.properties.network_acls.default_action != "Deny":
                logger.error("Key Vault network rules are not set to deny by default")
                return False
                
            logger.info("Key Vault security settings are properly configured")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying Key Vault security: {str(e)}")
            return False
            
    def verify_storage_security(self) -> bool:
        """Verify Storage account security settings."""
        try:
            storage = self.storage_client.storage_accounts.get_properties(
                self.config["resource_group_name"],
                self.config["storage_account_name"]
            )
            
            # Check HTTPS only
            if not storage.enable_https_traffic_only:
                logger.error("Storage account does not require HTTPS")
                return False
                
            # Check TLS version
            if storage.minimum_tls_version != "TLS1_2":
                logger.error("Storage account minimum TLS version is not 1.2")
                return False
                
            # Check network rules
            if storage.network_rule_set.default_action != "Deny":
                logger.error("Storage account network rules are not set to deny by default")
                return False
                
            # Check encryption
            if not storage.encryption.services.blob.enabled:
                logger.error("Storage account blob encryption is not enabled")
                return False
                
            logger.info("Storage account security settings are properly configured")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying storage security: {str(e)}")
            return False
            
    def verify_network_security(self) -> bool:
        """Verify network security group settings."""
        try:
            nsg_name = f"{self.config['resource_group_name']}-nsg"
            nsg = self.network_client.network_security_groups.get(
                self.config["resource_group_name"],
                nsg_name
            )
            
            # Verify required security rules
            required_rules = {
                "DenyAllInbound": {
                    "direction": "Inbound",
                    "access": "Deny",
                    "priority": 4096
                },
                "AllowVnetInbound": {
                    "direction": "Inbound",
                    "access": "Allow",
                    "priority": 1000
                },
                "AllowAzureLoadBalancerInbound": {
                    "direction": "Inbound",
                    "access": "Allow",
                    "priority": 2000
                }
            }
            
            for rule_name, expected in required_rules.items():
                rule = next(
                    (r for r in nsg.security_rules if r.name == rule_name),
                    None
                )
                if not rule:
                    logger.error(f"Required security rule {rule_name} is missing")
                    return False
                    
                if (rule.direction != expected["direction"] or
                    rule.access != expected["access"] or
                    rule.priority != expected["priority"]):
                    logger.error(f"Security rule {rule_name} is not properly configured")
                    return False
                    
            logger.info("Network security group settings are properly configured")
            return True
            
        except ResourceNotFoundError:
            logger.error("Network security group not found")
            return False
        except Exception as e:
            logger.error(f"Error verifying network security: {str(e)}")
            return False
            
    def verify_attestation_security(self) -> bool:
        """Verify attestation provider security settings."""
        try:
            attestation = self.attestation_client.attestation_providers.get(
                self.config["resource_group_name"],
                self.config["attestation_endpoint"].split(".")[0]
            )
            
            # Check trust model
            if attestation.properties.trust_model != "AAD":
                logger.error("Attestation provider is not using AAD trust model")
                return False
                
            # Check status
            if attestation.properties.status != "Ready":
                logger.error("Attestation provider is not in Ready state")
                return False
                
            logger.info("Attestation provider security settings are properly configured")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying attestation security: {str(e)}")
            return False
            
    def verify_all_security(self) -> bool:
        """Verify security settings for all resources."""
        try:
            key_vault_secure = self.verify_key_vault_security()
            storage_secure = self.verify_storage_security()
            network_secure = self.verify_network_security()
            attestation_secure = self.verify_attestation_security()
            
            all_secure = all([
                key_vault_secure,
                storage_secure,
                network_secure,
                attestation_secure
            ])
            
            if all_secure:
                logger.info("All security settings are properly configured")
            else:
                logger.warning("Some security settings need attention")
                
            return all_secure
            
        except Exception as e:
            logger.error(f"Error verifying security settings: {str(e)}")
            return False

def main():
    verifier = SecurityVerifier()
    verifier.verify_all_security()

if __name__ == "__main__":
    main() 