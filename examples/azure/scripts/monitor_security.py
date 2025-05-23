#!/usr/bin/env python3

import os
import sys
import json
import logging
import azure.identity
import azure.mgmt.monitor
import azure.mgmt.security
import azure.mgmt.resource
from datetime import datetime, timedelta
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityMonitor:
    def __init__(self, config_path: str = "config.json"):
        """Initialize security monitor."""
        self.config = self._load_config(config_path)
        self.credential = azure.identity.DefaultAzureCredential()
        self.subscription_id = self.config["subscription_id"]
        
        # Initialize Azure clients
        self.monitor_client = azure.mgmt.monitor.MonitorManagementClient(
            self.credential, self.subscription_id
        )
        self.security_client = azure.mgmt.security.SecurityCenter(
            self.credential, self.subscription_id
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def setup_security_alerts(self):
        """Set up security alerts for critical events."""
        try:
            # Create action group for alerts
            action_group_name = f"{self.config['resource_group_name']}-security-alerts"
            self.monitor_client.action_groups.create_or_update(
                self.config["resource_group_name"],
                action_group_name,
                {
                    "location": "global",
                    "group_short_name": "SecAlert",
                    "enabled": True,
                    "email_receivers": [
                        {
                            "name": "Security Team",
                            "email_address": self.config.get("security", {}).get("alert_email", ""),
                            "use_common_alert_schema": True
                        }
                    ]
                }
            )
            
            # Set up alerts for Key Vault
            self._setup_key_vault_alerts(action_group_name)
            
            # Set up alerts for Storage
            self._setup_storage_alerts(action_group_name)
            
            # Set up alerts for Network Security
            self._setup_network_alerts(action_group_name)
            
            logger.info("Security alerts configured successfully")
            
        except Exception as e:
            logger.error(f"Error setting up security alerts: {str(e)}")
            raise
            
    def _setup_key_vault_alerts(self, action_group_name: str):
        """Set up alerts for Key Vault security events."""
        try:
            # Alert for failed authentication attempts
            self.monitor_client.metric_alerts.create_or_update(
                self.config["resource_group_name"],
                "keyvault-auth-failures",
                {
                    "location": self.config["location"],
                    "scopes": [
                        f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group_name']}/providers/Microsoft.KeyVault/vaults/{self.config['key_vault_name']}"
                    ],
                    "severity": 1,
                    "enabled": True,
                    "evaluation_frequency": "PT5M",
                    "window_size": "PT5M",
                    "criteria": {
                        "odata.type": "Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria",
                        "all_of": [
                            {
                                "name": "Failed Authentication Attempts",
                                "metric_name": "FailedRequests",
                                "operator": "GreaterThan",
                                "threshold": 5,
                                "time_aggregation": "Total"
                            }
                        ]
                    },
                    "actions": [
                        {
                            "action_group_id": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group_name']}/providers/microsoft.insights/actionGroups/{action_group_name}"
                        }
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Error setting up Key Vault alerts: {str(e)}")
            raise
            
    def _setup_storage_alerts(self, action_group_name: str):
        """Set up alerts for Storage security events."""
        try:
            # Alert for anonymous access attempts
            self.monitor_client.metric_alerts.create_or_update(
                self.config["resource_group_name"],
                "storage-anonymous-access",
                {
                    "location": self.config["location"],
                    "scopes": [
                        f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group_name']}/providers/Microsoft.Storage/storageAccounts/{self.config['storage_account_name']}"
                    ],
                    "severity": 1,
                    "enabled": True,
                    "evaluation_frequency": "PT5M",
                    "window_size": "PT5M",
                    "criteria": {
                        "odata.type": "Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria",
                        "all_of": [
                            {
                                "name": "Anonymous Access Attempts",
                                "metric_name": "AnonymousAccess",
                                "operator": "GreaterThan",
                                "threshold": 0,
                                "time_aggregation": "Total"
                            }
                        ]
                    },
                    "actions": [
                        {
                            "action_group_id": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group_name']}/providers/microsoft.insights/actionGroups/{action_group_name}"
                        }
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Error setting up Storage alerts: {str(e)}")
            raise
            
    def _setup_network_alerts(self, action_group_name: str):
        """Set up alerts for Network security events."""
        try:
            # Alert for network security rule changes
            self.monitor_client.activity_log_alerts.create_or_update(
                self.config["resource_group_name"],
                "network-rule-changes",
                {
                    "location": "global",
                    "scopes": [
                        f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group_name']}"
                    ],
                    "enabled": True,
                    "condition": {
                        "all_of": [
                            {
                                "field": "category",
                                "equals": "Security"
                            },
                            {
                                "field": "resourceType",
                                "equals": "Microsoft.Network/networkSecurityGroups"
                            }
                        ]
                    },
                    "actions": {
                        "action_groups": [
                            {
                                "action_group_id": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group_name']}/providers/microsoft.insights/actionGroups/{action_group_name}"
                            }
                        ]
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error setting up Network alerts: {str(e)}")
            raise
            
    def get_security_events(self, hours: int = 24) -> List[Dict]:
        """Get recent security events."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            events = self.monitor_client.activity_logs.list(
                filter=f"eventTimestamp ge '{start_time.isoformat()}' and eventTimestamp le '{end_time.isoformat()}' and category eq 'Security'"
            )
            
            return [event.as_dict() for event in events]
            
        except Exception as e:
            logger.error(f"Error getting security events: {str(e)}")
            return []
            
    def get_security_recommendations(self) -> List[Dict]:
        """Get security recommendations from Azure Security Center."""
        try:
            recommendations = self.security_client.recommendations.list()
            return [rec.as_dict() for rec in recommendations]
            
        except Exception as e:
            logger.error(f"Error getting security recommendations: {str(e)}")
            return []

def main():
    monitor = SecurityMonitor()
    
    # Set up security alerts
    monitor.setup_security_alerts()
    
    # Get recent security events
    events = monitor.get_security_events()
    logger.info(f"Found {len(events)} security events in the last 24 hours")
    
    # Get security recommendations
    recommendations = monitor.get_security_recommendations()
    logger.info(f"Found {len(recommendations)} security recommendations")

if __name__ == "__main__":
    main() 