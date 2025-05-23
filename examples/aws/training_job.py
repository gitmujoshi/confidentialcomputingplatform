#!/usr/bin/env python3

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import json
import logging
import os
from typing import Dict, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureTrainingJob:
    """Class for managing secure training jobs on AWS SageMaker."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the secure training job manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.session = sagemaker.Session()
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker')
        self.kms_client = boto3.client('kms')
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        with open(config_path, "r") as f:
            return json.load(f)
            
    def create_training_job(self,
                          entry_point: str = "train.py",
                          source_dir: str = ".",
                          hyperparameters: Optional[Dict] = None) -> str:
        """
        Create and start a secure training job.
        
        Args:
            entry_point: Training script entry point
            source_dir: Directory containing training code
            hyperparameters: Optional hyperparameters for training
            
        Returns:
            str: Training job name
        """
        try:
            # Create estimator
            estimator = PyTorch(
                entry_point=entry_point,
                source_dir=source_dir,
                role=self.config["aws"]["sagemaker_role"],
                instance_count=self.config["training"]["instance_count"],
                instance_type=self.config["training"]["instance_type"],
                volume_size=self.config["training"]["volume_size"],
                max_run=self.config["training"]["max_runtime_in_seconds"],
                framework_version="1.12.1",
                py_version="py38",
                hyperparameters=hyperparameters or {},
                output_path=f"s3://{self.config['storage']['bucket_name']}/models",
                code_location=f"s3://{self.config['storage']['bucket_name']}/code",
                sagemaker_session=self.session,
                subnets=self.config["security"]["vpc_config"]["subnets"],
                security_group_ids=self.config["security"]["vpc_config"]["security_groups"],
                encrypt_inter_container_traffic=self.config["security"]["encryption"]["in_transit"],
                enable_network_isolation=True
            )
            
            # Start training
            estimator.fit(
                inputs={
                    "train": f"s3://{self.config['storage']['bucket_name']}/data/train",
                    "validation": f"s3://{self.config['storage']['bucket_name']}/data/val"
                }
            )
            
            logger.info(f"Training job started: {estimator.latest_training_job.name}")
            return estimator.latest_training_job.name
            
        except Exception as e:
            logger.error(f"Error creating training job: {str(e)}")
            raise
            
    def monitor_training_job(self, job_name: str) -> None:
        """
        Monitor the progress of a training job.
        
        Args:
            job_name: Name of the training job
        """
        try:
            while True:
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                
                status = response["TrainingJobStatus"]
                logger.info(f"Training job status: {status}")
                
                if status in ["Completed", "Failed", "Stopped"]:
                    break
                    
                # Log metrics if available
                if "FinalMetricDataList" in response:
                    for metric in response["FinalMetricDataList"]:
                        logger.info(f"Metric: {metric['MetricName']} = {metric['Value']}")
                        
        except Exception as e:
            logger.error(f"Error monitoring training job: {str(e)}")
            raise
            
    def stop_training_job(self, job_name: str) -> None:
        """
        Stop a running training job.
        
        Args:
            job_name: Name of the training job
        """
        try:
            self.sagemaker_client.stop_training_job(
                TrainingJobName=job_name
            )
            logger.info(f"Stopped training job: {job_name}")
            
        except Exception as e:
            logger.error(f"Error stopping training job: {str(e)}")
            raise
            
    def get_training_job_metrics(self, job_name: str) -> Dict:
        """
        Get metrics for a completed training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            Dict: Training job metrics
        """
        try:
            response = self.sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            
            metrics = {}
            if "FinalMetricDataList" in response:
                for metric in response["FinalMetricDataList"]:
                    metrics[metric["MetricName"]] = metric["Value"]
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting training job metrics: {str(e)}")
            raise

def main():
    # Example usage
    training_job = SecureTrainingJob()
    
    # Define hyperparameters
    hyperparameters = {
        "epochs": 10,
        "batch-size": 32,
        "learning-rate": 0.001
    }
    
    # Create and start training job
    job_name = training_job.create_training_job(
        entry_point="train.py",
        source_dir=".",
        hyperparameters=hyperparameters
    )
    
    # Monitor training progress
    training_job.monitor_training_job(job_name)
    
    # Get final metrics
    metrics = training_job.get_training_job_metrics(job_name)
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main() 