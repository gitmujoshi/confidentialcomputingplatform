#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from cryptography.fernet import Fernet
import json
import os

logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated learning client."""
    
    def __init__(self,
                 model: nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: str = 'cuda'):
        """
        Initialize federated client.
        
        Args:
            model: Neural network model
            data_loader: Data loader for client's data
            device: Training device
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train model on client's data.
        
        Args:
            global_weights: Global model weights
            
        Returns:
            Updated model weights
        """
        try:
            # Load global weights
            self.model.load_state_dict(global_weights)
            self.model.to(self.device)
            
            # Train model
            optimizer = torch.optim.Adam(self.model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            self.model.train()
            for data, labels in self.data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            # Get updated weights
            updated_weights = self.model.state_dict()
            
            # Encrypt weights
            encrypted_weights = self._encrypt_weights(updated_weights)
            
            return encrypted_weights
            
        except Exception as e:
            logger.error(f"Error in client training: {str(e)}")
            raise
            
    def _encrypt_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, bytes]:
        """Encrypt model weights."""
        try:
            encrypted_weights = {}
            for key, value in weights.items():
                # Convert tensor to bytes
                weight_bytes = value.cpu().numpy().tobytes()
                
                # Encrypt weights
                encrypted_weights[key] = self.cipher_suite.encrypt(weight_bytes)
                
            return encrypted_weights
            
        except Exception as e:
            logger.error(f"Error encrypting weights: {str(e)}")
            raise
            
class FederatedServer:
    """Federated learning server."""
    
    def __init__(self,
                 model: nn.Module,
                 clients: List[FederatedClient],
                 num_rounds: int = 100):
        """
        Initialize federated server.
        
        Args:
            model: Global model
            clients: List of federated clients
            num_rounds: Number of training rounds
        """
        self.model = model
        self.clients = clients
        self.num_rounds = num_rounds
        
    def train(self) -> Dict[str, List[float]]:
        """
        Perform federated training.
        
        Returns:
            Dictionary containing training metrics
        """
        try:
            metrics = {
                'loss': [],
                'accuracy': []
            }
            
            for round_idx in range(self.num_rounds):
                logger.info(f"Starting round {round_idx + 1}/{self.num_rounds}")
                
                # Train on each client
                client_weights = []
                for client in self.clients:
                    weights = client.train(self.model.state_dict())
                    client_weights.append(weights)
                    
                # Aggregate weights
                aggregated_weights = self._aggregate_weights(client_weights)
                
                # Update global model
                self.model.load_state_dict(aggregated_weights)
                
                # Evaluate model
                round_metrics = self._evaluate_model()
                metrics['loss'].append(round_metrics['loss'])
                metrics['accuracy'].append(round_metrics['accuracy'])
                
                logger.info(f"Round {round_idx + 1} metrics: {round_metrics}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in federated training: {str(e)}")
            raise
            
    def _aggregate_weights(self, 
                          client_weights: List[Dict[str, bytes]]) -> Dict[str, torch.Tensor]:
        """Aggregate encrypted weights from clients."""
        try:
            aggregated_weights = {}
            
            # Get all weight keys
            weight_keys = client_weights[0].keys()
            
            for key in weight_keys:
                # Decrypt and aggregate weights
                decrypted_weights = []
                for client_weights_dict in client_weights:
                    # Decrypt weights
                    decrypted_bytes = self.clients[0].cipher_suite.decrypt(
                        client_weights_dict[key]
                    )
                    
                    # Convert bytes to tensor
                    weight_tensor = torch.from_numpy(
                        np.frombuffer(decrypted_bytes, dtype=np.float32)
                    )
                    decrypted_weights.append(weight_tensor)
                    
                # Average weights
                aggregated_weights[key] = torch.mean(
                    torch.stack(decrypted_weights),
                    dim=0
                )
                
            return aggregated_weights
            
        except Exception as e:
            logger.error(f"Error aggregating weights: {str(e)}")
            raise
            
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate global model."""
        try:
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            criterion = nn.CrossEntropyLoss()
            
            # Evaluate on each client's validation data
            for client in self.clients:
                for data, labels in client.data_loader:
                    data, labels = data.to(self.model.device), labels.to(self.model.device)
                    
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
            return {
                'loss': total_loss / len(self.clients),
                'accuracy': 100. * correct / total
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
class SecureFederatedLearning:
    """Secure federated learning implementation."""
    
    def __init__(self,
                 model: nn.Module,
                 num_clients: int,
                 data_loaders: List[torch.utils.data.DataLoader],
                 num_rounds: int = 100,
                 device: str = 'cuda'):
        """
        Initialize secure federated learning.
        
        Args:
            model: Global model
            num_clients: Number of clients
            data_loaders: List of data loaders for each client
            num_rounds: Number of training rounds
            device: Training device
        """
        self.model = model
        self.num_clients = num_clients
        self.data_loaders = data_loaders
        self.num_rounds = num_rounds
        self.device = device
        
        # Initialize clients
        self.clients = [
            FederatedClient(
                model=model,
                data_loader=data_loader,
                device=device
            )
            for data_loader in data_loaders
        ]
        
        # Initialize server
        self.server = FederatedServer(
            model=model,
            clients=self.clients,
            num_rounds=num_rounds
        )
        
    def train(self) -> Dict[str, List[float]]:
        """
        Perform secure federated training.
        
        Returns:
            Dictionary containing training metrics
        """
        try:
            return self.server.train()
            
        except Exception as e:
            logger.error(f"Error in secure federated training: {str(e)}")
            raise 