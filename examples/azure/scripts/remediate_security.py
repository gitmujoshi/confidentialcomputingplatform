#!/usr/bin/env python3

import os
import sys
import json
import logging
import azure.identity
import azure.mgmt.keyvault
import azure.mgmt.storage
import azure.mgmt.network
import azure.mgmt.security
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityRemediator:
    def __init__(self, config_path: str = "config.json"):
        """Initialize security remediator."""
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
        self.security_client = azure.mgmt.security.SecurityCenter(
            self.credential, self.subscription_id
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def remediate_key_vault(self) -> Tuple[bool, List[str]]:
        """Remediate Key Vault security issues."""
        issues_fixed = []
        try:
            vault = self.keyvault_client.vaults.get(
                self.config["resource_group_name"],
                self.config["key_vault_name"]
            )
            
            # Fix RBAC authorization
            if not vault.properties.enable_rbac_authorization:
                self.keyvault_client.vaults.begin_create_or_update(
                    self.config["resource_group_name"],
                    self.config["key_vault_name"],
                    {
                        "location": vault.location,
                        "properties": {
                            **vault.properties.as_dict(),
                            "enable_rbac_authorization": True
                        }
                    }
                ).wait()
                issues_fixed.append("Enabled RBAC authorization")
                
            # Fix soft delete
            if not vault.properties.enable_soft_delete:
                self.keyvault_client.vaults.begin_create_or_update(
                    self.config["resource_group_name"],
                    self.config["key_vault_name"],
                    {
                        "location": vault.location,
                        "properties": {
                            **vault.properties.as_dict(),
                            "enable_soft_delete": True,
                            "soft_delete_retention_in_days": 90
                        }
                    }
                ).wait()
                issues_fixed.append("Enabled soft delete")
                
            # Fix network rules
            if vault.properties.network_acls.default_action != "Deny":
                self.keyvault_client.vaults.begin_create_or_update(
                    self.config["resource_group_name"],
                    self.config["key_vault_name"],
                    {
                        "location": vault.location,
                        "properties": {
                            **vault.properties.as_dict(),
                            "network_acls": {
                                "default_action": "Deny",
                                "bypass": "AzureServices",
                                "ip_rules": [],
                                "virtual_network_rules": []
                            }
                        }
                    }
                ).wait()
                issues_fixed.append("Updated network rules to deny by default")
                
            return True, issues_fixed
            
        except Exception as e:
            logger.error(f"Error remediating Key Vault: {str(e)}")
            return False, issues_fixed
            
    def remediate_storage(self) -> Tuple[bool, List[str]]:
        """Remediate Storage account security issues."""
        issues_fixed = []
        try:
            storage = self.storage_client.storage_accounts.get_properties(
                self.config["resource_group_name"],
                self.config["storage_account_name"]
            )
            
            # Fix HTTPS only
            if not storage.enable_https_traffic_only:
                self.storage_client.storage_accounts.begin_update(
                    self.config["resource_group_name"],
                    self.config["storage_account_name"],
                    {
                        "enable_https_traffic_only": True
                    }
                ).wait()
                issues_fixed.append("Enabled HTTPS-only traffic")
                
            # Fix TLS version
            if storage.minimum_tls_version != "TLS1_2":
                self.storage_client.storage_accounts.begin_update(
                    self.config["resource_group_name"],
                    self.config["storage_account_name"],
                    {
                        "minimum_tls_version": "TLS1_2"
                    }
                ).wait()
                issues_fixed.append("Updated minimum TLS version to 1.2")
                
            # Fix network rules
            if storage.network_rule_set.default_action != "Deny":
                self.storage_client.storage_accounts.begin_update(
                    self.config["resource_group_name"],
                    self.config["storage_account_name"],
                    {
                        "network_rule_set": {
                            "default_action": "Deny",
                            "bypass": ["AzureServices"],
                            "ip_rules": [],
                            "virtual_network_rules": []
                        }
                    }
                ).wait()
                issues_fixed.append("Updated network rules to deny by default")
                
            return True, issues_fixed
            
        except Exception as e:
            logger.error(f"Error remediating Storage: {str(e)}")
            return False, issues_fixed
            
    def remediate_network_security(self) -> Tuple[bool, List[str]]:
        """Remediate network security group issues."""
        issues_fixed = []
        try:
            nsg_name = f"{self.config['resource_group_name']}-nsg"
            nsg = self.network_client.network_security_groups.get(
                self.config["resource_group_name"],
                nsg_name
            )
            
            # Check and fix required rules
            required_rules = {
                "DenyAllInbound": {
                    "direction": "Inbound",
                    "access": "Deny",
                    "priority": 4096,
                    "protocol": "*",
                    "source_address_prefix": "*",
                    "destination_address_prefix": "*",
                    "source_port_range": "*",
                    "destination_port_range": "*"
                },
                "AllowVnetInbound": {
                    "direction": "Inbound",
                    "access": "Allow",
                    "priority": 1000,
                    "protocol": "*",
                    "source_address_prefix": "VirtualNetwork",
                    "destination_address_prefix": "VirtualNetwork",
                    "source_port_range": "*",
                    "destination_port_range": "*"
                }
            }
            
            for rule_name, rule_config in required_rules.items():
                existing_rule = next(
                    (r for r in nsg.security_rules if r.name == rule_name),
                    None
                )
                
                if not existing_rule or any(
                    getattr(existing_rule, k) != v
                    for k, v in rule_config.items()
                ):
                    self.network_client.security_rules.begin_create_or_update(
                        self.config["resource_group_name"],
                        nsg_name,
                        rule_name,
                        rule_config
                    ).wait()
                    issues_fixed.append(f"Updated security rule: {rule_name}")
                    
            return True, issues_fixed
            
        except Exception as e:
            logger.error(f"Error remediating network security: {str(e)}")
            return False, issues_fixed
            
    def remediate_all(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Remediate all security issues."""
        results = {
            "key_vault": self.remediate_key_vault(),
            "storage": self.remediate_storage(),
            "network": self.remediate_network_security()
        }
        
        # Log results
        for resource, (success, issues) in results.items():
            if success:
                if issues:
                    logger.info(f"Fixed {len(issues)} issues in {resource}:")
                    for issue in issues:
                        logger.info(f"  - {issue}")
                else:
                    logger.info(f"No issues found in {resource}")
            else:
                logger.error(f"Failed to remediate {resource}")
                
        return results

def main():
    remediator = SecurityRemediator()
    results = remediator.remediate_all()
    
    # Print summary
    total_issues = sum(len(issues) for _, (_, issues) in results.items())
    if total_issues > 0:
        logger.info(f"Successfully fixed {total_issues} security issues")
    else:
        logger.info("No security issues found")

if __name__ == "__main__":
    main() 