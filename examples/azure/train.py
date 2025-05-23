#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import os
import json
from typing import Dict, List, Tuple
import time
from datetime import datetime
import azure.storage.blob
import azure.keyvault.secrets
import azure.identity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureDataset(Dataset):
    """Dataset class for secure training data."""
    
    def __init__(self, data_path: str, storage_account: str, container_name: str):
        """
        Initialize the secure dataset.
        
        Args:
            data_path: Path to the encrypted training data
            storage_account: Azure Storage account name
            container_name: Blob container name
        """
        self.data_path = data_path
        self.storage_account = storage_account
        self.container_name = container_name
        self.data = self._load_and_decrypt_data()
        
    def _load_and_decrypt_data(self) -> List[Tuple]:
        """Load and decrypt the training data from Azure Blob Storage."""
        try:
            # Get credentials
            credential = azure.identity.DefaultAzureCredential()
            
            # Create blob service client
            blob_service_client = azure.storage.blob.BlobServiceClient(
                account_url=f"https://{self.storage_account}.blob.core.windows.net",
                credential=credential
            )
            
            # Get container client
            container_client = blob_service_client.get_container_client(self.container_name)
            
            # Download and decrypt data
            # Implementation to download and decrypt data
            # This would use Azure Key Vault for key management
            pass
            
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
        """
        Initialize the secure model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output layer
        """
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
                 config: Dict):
        """
        Initialize the secure trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        """Save the model securely to Azure Blob Storage."""
        try:
            # Get credentials
            credential = azure.identity.DefaultAzureCredential()
            
            # Create blob service client
            blob_service_client = azure.storage.blob.BlobServiceClient(
                account_url=f"https://{self.config['storage_account']}.blob.core.windows.net",
                credential=credential
            )
            
            # Get container client
            container_client = blob_service_client.get_container_client(
                self.config['container_name']
            )
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }
            
            # Save checkpoint securely
            # Implementation to save checkpoint to Azure Blob Storage
            # This would use Azure Key Vault for encryption
            pass
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
        
    # Initialize dataset
    train_dataset = SecureDataset(
        config["data"]["train_data_path"],
        config["storage"]["account_name"],
        config["storage"]["container_name"]
    )
    val_dataset = SecureDataset(
        config["data"]["val_data_path"],
        config["storage"]["account_name"],
        config["storage"]["container_name"]
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
    trainer = SecureTrainer(model, train_loader, val_loader, config)
    
    # Start training
    logger.info("Starting secure training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 