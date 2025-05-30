#!/usr/bin/env python3

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from phe import paillier
import json
import os

logger = logging.getLogger(__name__)

class HomomorphicEncryption:
    """Homomorphic encryption implementation."""
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize homomorphic encryption.
        
        Args:
            key_size: Key size in bits
        """
        self.key_size = key_size
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_size
        )
        
    def encrypt(self, data: np.ndarray) -> np.ndarray:
        """
        Encrypt data.
        
        Args:
            data: Input data
            
        Returns:
            Encrypted data
        """
        try:
            # Convert to float
            data_float = data.astype(np.float64)
            
            # Encrypt each element
            encrypted_data = np.vectorize(self.public_key.encrypt)(data_float)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise
            
    def decrypt(self, encrypted_data: np.ndarray) -> np.ndarray:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            # Decrypt each element
            decrypted_data = np.vectorize(self.private_key.decrypt)(encrypted_data)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise
            
    def add(self, 
            encrypted_a: np.ndarray,
            encrypted_b: np.ndarray) -> np.ndarray:
        """
        Add encrypted values.
        
        Args:
            encrypted_a: First encrypted array
            encrypted_b: Second encrypted array
            
        Returns:
            Encrypted sum
        """
        try:
            return encrypted_a + encrypted_b
            
        except Exception as e:
            logger.error(f"Error adding encrypted values: {str(e)}")
            raise
            
    def multiply(self,
                encrypted_a: np.ndarray,
                scalar: float) -> np.ndarray:
        """
        Multiply encrypted value by scalar.
        
        Args:
            encrypted_a: Encrypted array
            scalar: Scalar value
            
        Returns:
            Encrypted product
        """
        try:
            return encrypted_a * scalar
            
        except Exception as e:
            logger.error(f"Error multiplying encrypted value: {str(e)}")
            raise
            
class HomomorphicModel:
    """Homomorphic encryption for model parameters."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 he: HomomorphicEncryption):
        """
        Initialize homomorphic model.
        
        Args:
            model: Neural network model
            he: Homomorphic encryption instance
        """
        self.model = model
        self.he = he
        
    def encrypt_parameters(self) -> Dict[str, np.ndarray]:
        """
        Encrypt model parameters.
        
        Returns:
            Dictionary of encrypted parameters
        """
        try:
            encrypted_params = {}
            for name, param in self.model.named_parameters():
                # Convert to numpy
                param_np = param.detach().cpu().numpy()
                
                # Encrypt parameters
                encrypted_params[name] = self.he.encrypt(param_np)
                
            return encrypted_params
            
        except Exception as e:
            logger.error(f"Error encrypting parameters: {str(e)}")
            raise
            
    def decrypt_parameters(self,
                          encrypted_params: Dict[str, np.ndarray]) -> None:
        """
        Decrypt and update model parameters.
        
        Args:
            encrypted_params: Dictionary of encrypted parameters
        """
        try:
            for name, param in self.model.named_parameters():
                # Decrypt parameters
                decrypted_param = self.he.decrypt(encrypted_params[name])
                
                # Update parameter
                param.data = torch.from_numpy(decrypted_param).to(param.device)
                
        except Exception as e:
            logger.error(f"Error decrypting parameters: {str(e)}")
            raise
            
    def secure_update(self,
                      gradients: Dict[str, torch.Tensor],
                      learning_rate: float) -> None:
        """
        Perform secure parameter update.
        
        Args:
            gradients: Dictionary of gradients
            learning_rate: Learning rate
        """
        try:
            # Encrypt parameters
            encrypted_params = self.encrypt_parameters()
            
            # Encrypt gradients
            encrypted_gradients = {}
            for name, grad in gradients.items():
                grad_np = grad.detach().cpu().numpy()
                encrypted_gradients[name] = self.he.encrypt(grad_np)
                
            # Update parameters
            for name in encrypted_params.keys():
                # Multiply gradient by learning rate
                scaled_grad = self.he.multiply(
                    encrypted_gradients[name],
                    -learning_rate
                )
                
                # Add to parameters
                encrypted_params[name] = self.he.add(
                    encrypted_params[name],
                    scaled_grad
                )
                
            # Decrypt and update parameters
            self.decrypt_parameters(encrypted_params)
            
        except Exception as e:
            logger.error(f"Error in secure update: {str(e)}")
            raise
            
class HomomorphicTraining:
    """Homomorphic encryption for training."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 key_size: int = 2048):
        """
        Initialize homomorphic training.
        
        Args:
            model: Neural network model
            key_size: Key size in bits
        """
        self.he = HomomorphicEncryption(key_size=key_size)
        self.homomorphic_model = HomomorphicModel(model, self.he)
        
    def train_step(self,
                  data: torch.Tensor,
                  labels: torch.Tensor,
                  learning_rate: float) -> Dict[str, float]:
        """
        Perform one training step with homomorphic encryption.
        
        Args:
            data: Input data
            labels: Target labels
            learning_rate: Learning rate
            
        Returns:
            Dictionary containing loss
        """
        try:
            # Forward pass
            outputs = self.homomorphic_model.model(data)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass
            self.homomorphic_model.model.zero_grad()
            loss.backward()
            
            # Get gradients
            gradients = {
                name: param.grad
                for name, param in self.homomorphic_model.model.named_parameters()
            }
            
            # Secure update
            self.homomorphic_model.secure_update(gradients, learning_rate)
            
            return {
                'loss': loss.item()
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise 