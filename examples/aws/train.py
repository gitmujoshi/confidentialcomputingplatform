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
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import kms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureDataset(Dataset):
    """Dataset class for secure training data."""
    
    def __init__(self, data_path: str, bucket_name: str):
        """
        Initialize the secure dataset.
        
        Args:
            data_path: Path to the encrypted training data
            bucket_name: S3 bucket name
        """
        self.data_path = data_path
        self.bucket_name = bucket_name
        self.data = self._load_and_decrypt_data()
        
    def _load_and_decrypt_data(self) -> List[Tuple]:
        """Load and decrypt the training data from S3."""
        try:
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Initialize KMS client
            kms_client = boto3.client('kms')
            
            # Download encrypted data
            response = s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.data_path
            )
            encrypted_data = response['Body'].read()
            
            # Decrypt data using KMS
            decrypted_data = kms_client.decrypt(
                CiphertextBlob=encrypted_data,
                KeyId=os.environ['KMS_KEY_ID']
            )['Plaintext']
            
            # Process decrypted data
            # Implementation to process data
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
    """Trainer class for secure model training on AWS SageMaker."""
    
    def __init__(self, config: Dict):
        """
        Initialize the secure trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.sagemaker_session = sagemaker.Session()
        
        # Initialize SageMaker estimator
        self.estimator = PyTorch(
            entry_point='train.py',
            role=self.config['aws']['role'],
            instance_type=self.config['aws']['instance_type'],
            instance_count=1,
            framework_version='1.8.1',
            py_version='py3',
            hyperparameters={
                'epochs': self.config['training']['epochs'],
                'batch-size': self.config['training']['batch_size'],
                'learning-rate': self.config['training']['learning_rate']
            },
            output_path=f"s3://{self.config['aws']['bucket']}/models",
            code_location=f"s3://{self.config['aws']['bucket']}/code",
            encrypt_inter_container_traffic=True,
            enable_network_isolation=True
        )
        
    def train(self) -> None:
        """Train the model using SageMaker."""
        try:
            # Set up training data
            train_data = sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{self.config['aws']['bucket']}/data/train",
                distribution='FullyReplicated',
                s3_data_type='S3Prefix'
            )
            
            val_data = sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{self.config['aws']['bucket']}/data/val",
                distribution='FullyReplicated',
                s3_data_type='S3Prefix'
            )
            
            # Start training
            self.estimator.fit(
                inputs={
                    'train': train_data,
                    'val': val_data
                },
                wait=True
            )
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def save_model(self) -> None:
        """Save the trained model securely to S3."""
        try:
            # Get model artifacts
            model_path = self.estimator.model_data
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Initialize KMS client
            kms_client = boto3.client('kms')
            
            # Download model artifacts
            response = s3_client.get_object(
                Bucket=self.config['aws']['bucket'],
                Key=model_path
            )
            model_data = response['Body'].read()
            
            # Encrypt model data
            encrypted_data = kms_client.encrypt(
                KeyId=os.environ['KMS_KEY_ID'],
                Plaintext=model_data
            )['CiphertextBlob']
            
            # Upload encrypted model
            s3_client.put_object(
                Bucket=self.config['aws']['bucket'],
                Key=f"models/encrypted/{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
                Body=encrypted_data,
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=os.environ['KMS_KEY_ID']
            )
            
            logger.info("Model saved securely to S3")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
        
    # Initialize trainer
    trainer = SecureTrainer(config)
    
    # Start training
    logger.info("Starting secure training on AWS SageMaker...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 