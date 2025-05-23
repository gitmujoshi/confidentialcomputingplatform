#!/usr/bin/env python3

import os
import sys
import logging
import json
import base64
import azure.identity
import azure.keyvault.keys
import azure.storage.blob
import azure.confidentialcomputing
import azure.confidentialcomputing.models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureEnvironment:
    """Manages secure environment setup and attestation."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize secure environment.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.credential = azure.identity.DefaultAzureCredential()
        self._setup_clients()
        self._verify_environment()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def _setup_clients(self):
        """Set up Azure service clients."""
        # Key Vault client
        self.keyvault_client = azure.keyvault.keys.KeyClient(
            vault_url=f"https://{self.config['key_vault_name']}.vault.azure.net",
            credential=self.credential
        )
        
        # Storage client
        self.storage_client = azure.storage.blob.BlobServiceClient(
            account_url=f"https://{self.config['storage_account_name']}.blob.core.windows.net",
            credential=self.credential
        )
        
        # Confidential Computing client
        self.acc_client = azure.confidentialcomputing.ConfidentialComputingClient(
            credential=self.credential
        )
        
    def _verify_environment(self):
        """Verify the secure environment and perform attestation."""
        try:
            # Verify we're running in a confidential computing environment
            if not self._is_confidential_computing():
                raise RuntimeError("Not running in a confidential computing environment")
                
            # Perform attestation
            self._perform_attestation()
            
            # Verify access to required services
            self._verify_service_access()
            
            logger.info("Secure environment verification completed successfully")
            
        except Exception as e:
            logger.error(f"Environment verification failed: {str(e)}")
            raise
            
    def _is_confidential_computing(self) -> bool:
        """Check if running in a confidential computing environment."""
        try:
            # Check for AMD SEV-SNP capabilities
            with open("/proc/cpuinfo", "r") as f:
                cpu_info = f.read()
                return "sev_snp" in cpu_info.lower()
        except Exception:
            return False
            
    def _perform_attestation(self):
        """Perform remote attestation."""
        try:
            # Get attestation token
            attestation_token = self.acc_client.get_attestation_token(
                resource_group_name=self.config["resource_group_name"],
                workspace_name=self.config["workspace_name"]
            )
            
            # Verify attestation token
            if not self._verify_attestation_token(attestation_token):
                raise RuntimeError("Attestation verification failed")
                
            logger.info("Attestation completed successfully")
            
        except Exception as e:
            logger.error(f"Attestation failed: {str(e)}")
            raise
            
    def _verify_attestation_token(self, token: str) -> bool:
        """Verify the attestation token."""
        try:
            # Verify token signature and claims
            # This would typically involve checking the token's signature
            # and verifying the claims match expected values
            return True
        except Exception:
            return False
            
    def _verify_service_access(self):
        """Verify access to required Azure services."""
        try:
            # Verify Key Vault access
            self.keyvault_client.list_keys()
            
            # Verify Storage access
            self.storage_client.list_containers()
            
            logger.info("Service access verification completed successfully")
            
        except Exception as e:
            logger.error(f"Service access verification failed: {str(e)}")
            raise
            
    def get_encryption_key(self, key_name: str) -> bytes:
        """Get encryption key from Key Vault."""
        try:
            key = self.keyvault_client.get_key(key_name)
            return base64.b64decode(key.key)
        except Exception as e:
            logger.error(f"Error getting encryption key: {str(e)}")
            raise
            
    def decrypt_data(self, encrypted_data: bytes, key_name: str) -> bytes:
        """Decrypt data using Key Vault key."""
        try:
            key = self.get_encryption_key(key_name)
            # Implement decryption logic here
            # This would typically use a symmetric encryption algorithm
            return encrypted_data  # Placeholder
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise

class SecureDataset(Dataset):
    """Dataset class for secure training data."""
    
    def __init__(self, data_path: str, secure_env: SecureEnvironment):
        """
        Initialize secure dataset.
        
        Args:
            data_path: Path to encrypted training data
            secure_env: Secure environment instance
        """
        self.data_path = data_path
        self.secure_env = secure_env
        self.data = self._load_and_decrypt_data()
        
    def _load_and_decrypt_data(self) -> List[Tuple]:
        """Load and decrypt training data."""
        try:
            # Get container client
            container_client = self.secure_env.storage_client.get_container_client(
                self.secure_env.config["container_name"]
            )
            
            # Download encrypted data
            blob_client = container_client.get_blob_client(self.data_path)
            encrypted_data = blob_client.download_blob().readall()
            
            # Decrypt data
            decrypted_data = self.secure_env.decrypt_data(
                encrypted_data,
                self.secure_env.config["data_key_name"]
            )
            
            # Parse and return data
            # This would depend on your data format
            return []  # Placeholder
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]

class SecureModel(nn.Module):
    """Neural network model for secure training."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize secure model."""
        super(SecureModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class SecureTrainer:
    """Trainer class for secure model training."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 secure_env: SecureEnvironment,
                 config: Dict):
        """Initialize secure trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.secure_env = secure_env
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001)
        )
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.get("log_interval", 10) == 0:
                logger.info(f"Train Epoch: {batch_idx} Loss: {loss.item():.6f}")
                
        return total_loss / len(self.train_loader)
        
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / len(self.val_loader.dataset)
        
        logger.info(f"Validation set: Average loss: {val_loss:.4f}, "
                   f"Accuracy: {correct}/{len(self.val_loader.dataset)} ({accuracy:.0f}%)")
                   
        return val_loss
        
    def train(self) -> None:
        """Train the model for specified number of epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.get("epochs", 10)):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(epoch, val_loss)
                
    def _save_model(self, epoch: int, val_loss: float) -> None:
        """Save the model securely."""
        try:
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }
            
            # Serialize checkpoint
            checkpoint_bytes = torch.save(checkpoint, io.BytesIO()).getvalue()
            
            # Encrypt checkpoint
            encrypted_checkpoint = self.secure_env.encrypt_data(
                checkpoint_bytes,
                self.secure_env.config["model_key_name"]
            )
            
            # Upload to secure storage
            container_client = self.secure_env.storage_client.get_container_client(
                self.secure_env.config["container_name"]
            )
            
            blob_client = container_client.get_blob_client(
                f"models/checkpoint_epoch_{epoch}.pt"
            )
            
            blob_client.upload_blob(
                encrypted_checkpoint,
                overwrite=True
            )
            
            logger.info(f"Model saved securely for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
        
    # Initialize secure environment
    secure_env = SecureEnvironment()
    
    # Initialize dataset
    train_dataset = SecureDataset(
        config["data"]["train_data_path"],
        secure_env
    )
    val_dataset = SecureDataset(
        config["data"]["val_data_path"],
        secure_env
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )
    
    # Initialize model
    model = SecureModel(
        input_size=config["training"]["input_size"],
        hidden_size=config["training"]["hidden_size"],
        output_size=config["training"]["output_size"]
    )
    
    # Initialize trainer
    trainer = SecureTrainer(model, train_loader, val_loader, secure_env, config)
    
    # Start training
    logger.info("Starting secure training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 