#!/usr/bin/env python3

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from diffprivlib.mechanisms import Laplace, Gaussian
from diffprivlib.accountant import BudgetAccountant

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential privacy implementation for secure training."""
    
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 max_norm: float = 1.0,
                 noise_type: str = 'laplace'):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            max_norm: Maximum gradient norm
            noise_type: Type of noise ('laplace' or 'gaussian')
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_norm = max_norm
        self.noise_type = noise_type
        self.accountant = BudgetAccountant(epsilon=epsilon, delta=delta)
        
        # Initialize noise mechanism
        if noise_type == 'laplace':
            self.mechanism = Laplace(epsilon=epsilon)
        else:
            self.mechanism = Gaussian(epsilon=epsilon, delta=delta)
            
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add noise to data."""
        try:
            return self.mechanism.randomise(data)
        except Exception as e:
            logger.error(f"Error adding noise: {str(e)}")
            raise
            
    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Clip gradients to bound sensitivity."""
        try:
            norm = torch.norm(gradients)
            if norm > self.max_norm:
                gradients = gradients * self.max_norm / norm
            return gradients
        except Exception as e:
            logger.error(f"Error clipping gradients: {str(e)}")
            raise
            
    def add_noise_to_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add noise to gradients."""
        try:
            # Convert to numpy for noise addition
            grad_np = gradients.detach().cpu().numpy()
            
            # Add noise
            noisy_grad = self.add_noise(grad_np)
            
            # Convert back to tensor
            return torch.from_numpy(noisy_grad).to(gradients.device)
        except Exception as e:
            logger.error(f"Error adding noise to gradients: {str(e)}")
            raise
            
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get privacy budget spent."""
        return {
            'epsilon': self.accountant.epsilon,
            'delta': self.accountant.delta
        }
        
    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is available."""
        return self.accountant.epsilon > 0
        
class PrivacyPreservingOptimizer:
    """Privacy-preserving optimizer wrapper."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 dp: DifferentialPrivacy):
        """
        Initialize privacy-preserving optimizer.
        
        Args:
            optimizer: Base optimizer
            dp: Differential privacy instance
        """
        self.optimizer = optimizer
        self.dp = dp
        
    def step(self) -> None:
        """Perform privacy-preserving optimization step."""
        try:
            # Get gradients
            gradients = []
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    gradients.append(param.grad)
                    
            # Clip and add noise to gradients
            for grad in gradients:
                clipped_grad = self.dp.clip_gradients(grad)
                noisy_grad = self.dp.add_noise_to_gradients(clipped_grad)
                grad.copy_(noisy_grad)
                
            # Perform optimization step
            self.optimizer.step()
            
        except Exception as e:
            logger.error(f"Error in privacy-preserving optimization: {str(e)}")
            raise
            
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
        
class PrivacyPreservingTrainer:
    """Privacy-preserving model trainer."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dp: DifferentialPrivacy,
                 device: str = 'cuda'):
        """
        Initialize privacy-preserving trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            dp: Differential privacy instance
            device: Training device
        """
        self.model = model
        self.optimizer = PrivacyPreservingOptimizer(optimizer, dp)
        self.dp = dp
        self.device = device
        
    def train_step(self, 
                  data: torch.Tensor,
                  labels: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step with privacy guarantees.
        
        Args:
            data: Input data
            labels: Target labels
            
        Returns:
            Dictionary containing loss and privacy metrics
        """
        try:
            # Move data to device
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Privacy-preserving optimization
            self.optimizer.step()
            
            # Get privacy metrics
            privacy_metrics = self.dp.get_privacy_spent()
            
            return {
                'loss': loss.item(),
                'epsilon': privacy_metrics['epsilon'],
                'delta': privacy_metrics['delta']
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise 