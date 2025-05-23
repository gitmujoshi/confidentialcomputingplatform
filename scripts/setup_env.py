#!/usr/bin/env python3

import subprocess
import sys
import os
import platform
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Utility class for setting up the development environment."""
    
    def __init__(self):
        self.required_packages = {
            'azure': [
                'azure-mgmt-compute',
                'azure-mgmt-storage',
                'azure-mgmt-keyvault',
                'azure-storage-blob',
                'azure-identity'
            ],
            'oci': [
                'oci',
                'oci-cli'
            ],
            'aws': [
                'boto3',
                'awscli'
            ],
            'gcp': [
                'google-cloud-storage',
                'google-cloud-compute',
                'google-cloud-kms'
            ],
            'common': [
                'torch',
                'numpy',
                'pandas',
                'scikit-learn',
                'cryptography',
                'python-dotenv'
            ]
        }
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < required_version:
            logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
            return False
        return True
        
    def create_virtual_env(self) -> bool:
        """Create and activate virtual environment."""
        try:
            if not os.path.exists('.venv'):
                logger.info("Creating virtual environment...")
                subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating virtual environment: {str(e)}")
            return False
            
    def install_packages(self, cloud_provider: str = None) -> bool:
        """Install required packages."""
        try:
            packages = self.required_packages['common']
            if cloud_provider and cloud_provider in self.required_packages:
                packages.extend(self.required_packages[cloud_provider])
                
            logger.info(f"Installing packages: {', '.join(packages)}")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                '--upgrade', 'pip'
            ], check=True)
            
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                *packages
            ], check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing packages: {str(e)}")
            return False
            
    def setup_config_files(self) -> bool:
        """Create necessary configuration files."""
        try:
            # Create .env file if it doesn't exist
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write("# Environment variables\n")
                    f.write("# Add your configuration here\n")
                    
            # Create config directory if it doesn't exist
            if not os.path.exists('config'):
                os.makedirs('config')
                
            return True
            
        except Exception as e:
            logger.error(f"Error setting up config files: {str(e)}")
            return False
            
    def verify_setup(self) -> bool:
        """Verify the setup is complete."""
        try:
            # Check virtual environment
            if not os.path.exists('.venv'):
                logger.error("Virtual environment not found")
                return False
                
            # Check required files
            if not os.path.exists('.env'):
                logger.error(".env file not found")
                return False
                
            # Check Python packages
            import pkg_resources
            installed_packages = {pkg.key for pkg in pkg_resources.working_set}
            required_packages = set(sum(self.required_packages.values(), []))
            
            missing_packages = required_packages - installed_packages
            if missing_packages:
                logger.error(f"Missing packages: {', '.join(missing_packages)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying setup: {str(e)}")
            return False

def main():
    setup = EnvironmentSetup()
    
    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)
        
    # Create virtual environment
    if not setup.create_virtual_env():
        sys.exit(1)
        
    # Install packages
    if not setup.install_packages():
        sys.exit(1)
        
    # Setup config files
    if not setup.setup_config_files():
        sys.exit(1)
        
    # Verify setup
    if not setup.verify_setup():
        sys.exit(1)
        
    logger.info("Environment setup completed successfully!")

if __name__ == "__main__":
    main() 