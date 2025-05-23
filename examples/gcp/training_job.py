#!/usr/bin/env python3

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import kms
import json
import logging
import os
from typing import Dict, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureTrainingJob:
    """Class for managing secure training jobs on GCP Vertex AI."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the secure training job manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize GCP clients
        self.storage_client = storage.Client()
        self.kms_client = kms.KeyManagementServiceClient()
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.config["gcp"]["project_id"],
            location=self.config["gcp"]["location"]
        )
        
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
            # Create custom job
            job = aiplatform.CustomJob(
                display_name=f"secure-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                worker_pool_specs=[{
                    "machine_spec": {
                        "machine_type": self.config["training"]["machine_type"],
                        "accelerator_type": self.config["training"]["accelerator_type"],
                        "accelerator_count": self.config["training"]["accelerator_count"]
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-12:latest",
                        "args": [
                            "--epochs", str(hyperparameters.get("epochs", 10)),
                            "--batch-size", str(hyperparameters.get("batch-size", 32)),
                            "--learning-rate", str(hyperparameters.get("learning-rate", 0.001))
                        ],
                        "command": ["python", entry_point]
                    },
                    "disk_spec": {
                        "boot_disk_size_gb": self.config["training"]["disk_size_gb"]
                    }
                }],
                base_output_dir=f"gs://{self.config['storage']['bucket_name']}/models",
                network=self.config["security"]["vpc_config"]["network"],
                service_account=self.config["security"]["service_account"],
                encryption_spec_key_name=self.config["gcp"]["kms_key_name"]
            )
            
            # Start training
            job.run(
                sync=False,
                timeout=self.config["training"]["max_runtime_seconds"]
            )
            
            logger.info(f"Training job started: {job.name}")
            return job.name
            
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
            job = aiplatform.CustomJob.get(job_name)
            
            while True:
                job.refresh()
                status = job.state
                logger.info(f"Training job status: {status}")
                
                if status in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                    break
                    
                # Log metrics if available
                if hasattr(job, "metrics"):
                    for metric in job.metrics:
                        logger.info(f"Metric: {metric.name} = {metric.value}")
                        
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
            job = aiplatform.CustomJob.get(job_name)
            job.cancel()
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
            job = aiplatform.CustomJob.get(job_name)
            metrics = {}
            
            if hasattr(job, "metrics"):
                for metric in job.metrics:
                    metrics[metric.name] = metric.value
                    
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