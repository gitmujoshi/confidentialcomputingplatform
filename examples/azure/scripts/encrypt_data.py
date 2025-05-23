#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import base64
from typing import Dict, Optional
import azure.identity
import azure.keyvault.keys
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataEncryptor:
    """Utility class for encrypting training data."""
    
    def __init__(self, key_vault_name: str):
        """
        Initialize data encryptor.
        
        Args:
            key_vault_name: Name of the Azure Key Vault
        """
        self.key_vault_name = key_vault_name
        self.credential = azure.identity.DefaultAzureCredential()
        self.keyvault_client = azure.keyvault.keys.KeyClient(
            vault_url=f"https://{key_vault_name}.vault.azure.net",
            credential=self.credential
        )
        
    def get_encryption_key(self, key_name: str) -> bytes:
        """Get encryption key from Key Vault."""
        try:
            key = self.keyvault_client.get_key(key_name)
            return base64.b64decode(key.key)
        except Exception as e:
            logger.error(f"Error getting encryption key: {str(e)}")
            raise
            
    def encrypt_data(self, data: bytes, key_name: str) -> bytes:
        """
        Encrypt data using Key Vault key.
        
        Args:
            data: Data to encrypt
            key_name: Name of the key in Key Vault
            
        Returns:
            bytes: Encrypted data
        """
        try:
            # Get key from Key Vault
            key = self.get_encryption_key(key_name)
            
            # Generate a Fernet key from the Key Vault key
            salt = b'secure_training_salt'  # In production, use a secure random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            fernet_key = base64.urlsafe_b64encode(kdf.derive(key))
            
            # Create Fernet instance
            f = Fernet(fernet_key)
            
            # Encrypt data
            encrypted_data = f.encrypt(data)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise
            
    def encrypt_file(self, input_path: str, output_path: str, key_name: str) -> None:
        """
        Encrypt a file and save the encrypted data.
        
        Args:
            input_path: Path to input file
            output_path: Path to save encrypted data
            key_name: Name of the key in Key Vault
        """
        try:
            # Read input file
            with open(input_path, 'rb') as f:
                data = f.read()
                
            # Encrypt data
            encrypted_data = self.encrypt_data(data, key_name)
            
            # Save encrypted data
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
                
            logger.info(f"Successfully encrypted {input_path} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error encrypting file: {str(e)}")
            raise
            
    def encrypt_numpy_data(self, input_path: str, output_path: str, key_name: str) -> None:
        """
        Encrypt NumPy data and save the encrypted data.
        
        Args:
            input_path: Path to input NumPy file
            output_path: Path to save encrypted data
            key_name: Name of the key in Key Vault
        """
        try:
            # Load NumPy data
            data = np.load(input_path)
            
            # Convert to bytes
            data_bytes = data.tobytes()
            
            # Encrypt data
            encrypted_data = self.encrypt_data(data_bytes, key_name)
            
            # Save encrypted data
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
                
            logger.info(f"Successfully encrypted NumPy data from {input_path} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error encrypting NumPy data: {str(e)}")
            raise
            
    def encrypt_csv_data(self, input_path: str, output_path: str, key_name: str) -> None:
        """
        Encrypt CSV data and save the encrypted data.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save encrypted data
            key_name: Name of the key in Key Vault
        """
        try:
            # Load CSV data
            data = pd.read_csv(input_path)
            
            # Convert to bytes
            data_bytes = data.to_csv(index=False).encode()
            
            # Encrypt data
            encrypted_data = self.encrypt_data(data_bytes, key_name)
            
            # Save encrypted data
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
                
            logger.info(f"Successfully encrypted CSV data from {input_path} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error encrypting CSV data: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Encrypt training data')
    parser.add_argument('--input-path', required=True, help='Path to input data file')
    parser.add_argument('--output-path', required=True, help='Path to save encrypted data')
    parser.add_argument('--key-name', required=True, help='Name of the key in Key Vault')
    parser.add_argument('--key-vault-name', required=True, help='Name of the Azure Key Vault')
    parser.add_argument('--data-type', choices=['file', 'numpy', 'csv'], default='file',
                      help='Type of data to encrypt')
    
    args = parser.parse_args()
    
    try:
        # Initialize encryptor
        encryptor = DataEncryptor(args.key_vault_name)
        
        # Encrypt data based on type
        if args.data_type == 'numpy':
            encryptor.encrypt_numpy_data(args.input_path, args.output_path, args.key_name)
        elif args.data_type == 'csv':
            encryptor.encrypt_csv_data(args.input_path, args.output_path, args.key_name)
        else:
            encryptor.encrypt_file(args.input_path, args.output_path, args.key_name)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 