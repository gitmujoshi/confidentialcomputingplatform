#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    """Utility class for preparing training data."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize data preparator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def prepare_csv_data(self, input_path: str, output_dir: str) -> None:
        """
        Prepare CSV data for training.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save processed data
        """
        try:
            # Load data
            data = pd.read_csv(input_path)
            
            # Split into train and validation sets
            train_data = data.sample(frac=0.8, random_state=42)
            val_data = data.drop(train_data.index)
            
            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
            val_data.to_csv(os.path.join(output_dir, "val.csv"), index=False)
            
            logger.info(f"Successfully prepared CSV data from {input_path}")
            
        except Exception as e:
            logger.error(f"Error preparing CSV data: {str(e)}")
            raise
            
    def prepare_numpy_data(self, input_path: str, output_dir: str) -> None:
        """
        Prepare NumPy data for training.
        
        Args:
            input_path: Path to input NumPy file
            output_dir: Directory to save processed data
        """
        try:
            # Load data
            data = np.load(input_path)
            
            # Split into train and validation sets
            n_samples = len(data)
            indices = np.random.permutation(n_samples)
            train_size = int(0.8 * n_samples)
            
            train_data = data[indices[:train_size]]
            val_data = data[indices[train_size:]]
            
            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "train.npy"), train_data)
            np.save(os.path.join(output_dir, "val.npy"), val_data)
            
            logger.info(f"Successfully prepared NumPy data from {input_path}")
            
        except Exception as e:
            logger.error(f"Error preparing NumPy data: {str(e)}")
            raise
            
    def prepare_custom_data(self, input_path: str, output_dir: str, 
                          data_loader: callable) -> None:
        """
        Prepare custom data for training.
        
        Args:
            input_path: Path to input data file
            output_dir: Directory to save processed data
            data_loader: Custom function to load and process data
        """
        try:
            # Load and process data using custom loader
            train_data, val_data = data_loader(input_path)
            
            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            torch.save(train_data, os.path.join(output_dir, "train.pt"))
            torch.save(val_data, os.path.join(output_dir, "val.pt"))
            
            logger.info(f"Successfully prepared custom data from {input_path}")
            
        except Exception as e:
            logger.error(f"Error preparing custom data: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--input-path', required=True, help='Path to input data file')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed data')
    parser.add_argument('--data-type', choices=['csv', 'numpy', 'custom'], required=True,
                      help='Type of data to prepare')
    parser.add_argument('--config-path', default='config.json',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize preparator
        preparator = DataPreparator(args.config_path)
        
        # Prepare data based on type
        if args.data_type == 'csv':
            preparator.prepare_csv_data(args.input_path, args.output_dir)
        elif args.data_type == 'numpy':
            preparator.prepare_numpy_data(args.input_path, args.output_dir)
        else:
            # For custom data, you need to provide a data loader function
            logger.error("Custom data preparation requires a data loader function")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 