#!/usr/bin/env python3

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPDataPreparator:
    def __init__(self, project_id: str):
        """Initialize GCP data preparation."""
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bigquery_client = bigquery.Client(project=project_id)

    def prepare_training_data(self, input_path: str, output_path: str) -> dict:
        """Prepare training data for confidential computing."""
        try:
            # Read input data
            if input_path.startswith('gs://'):
                # Read from Cloud Storage
                bucket_name, blob_name = input_path[5:].split('/', 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                data = pd.read_csv(blob.download_as_string())
            else:
                # Read from local file
                data = pd.read_csv(input_path)

            # Preprocess data
            data = self._preprocess_data(data)

            # Split data
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            # Save processed data
            if output_path.startswith('gs://'):
                # Save to Cloud Storage
                bucket_name, blob_name = output_path[5:].split('/', 1)
                bucket = self.storage_client.bucket(bucket_name)
                
                # Save training data
                train_blob = bucket.blob(f"{blob_name}/train.csv")
                train_blob.upload_from_string(train_data.to_csv(index=False))
                
                # Save test data
                test_blob = bucket.blob(f"{blob_name}/test.csv")
                test_blob.upload_from_string(test_data.to_csv(index=False))
            else:
                # Save to local files
                os.makedirs(output_path, exist_ok=True)
                train_data.to_csv(f"{output_path}/train.csv", index=False)
                test_data.to_csv(f"{output_path}/test.csv", index=False)

            return {
                'input_path': input_path,
                'output_path': output_path,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'features': list(data.columns)
            }
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise

    def prepare_model_data(self, model_path: str, output_path: str) -> dict:
        """Prepare model data for confidential computing."""
        try:
            # Load model
            if model_path.startswith('gs://'):
                # Load from Cloud Storage
                bucket_name, blob_name = model_path[5:].split('/', 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                model_data = blob.download_as_string()
            else:
                # Load from local file
                with open(model_path, 'rb') as f:
                    model_data = f.read()

            # Process model data
            processed_data = self._process_model_data(model_data)

            # Save processed model data
            if output_path.startswith('gs://'):
                # Save to Cloud Storage
                bucket_name, blob_name = output_path[5:].split('/', 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.upload_from_string(processed_data)
            else:
                # Save to local file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(processed_data)

            return {
                'model_path': model_path,
                'output_path': output_path,
                'model_size': len(processed_data)
            }
        except Exception as e:
            logger.error(f"Failed to prepare model data: {e}")
            raise

    def prepare_config_data(self, config_path: str, output_path: str) -> dict:
        """Prepare configuration data for confidential computing."""
        try:
            # Load configuration
            if config_path.startswith('gs://'):
                # Load from Cloud Storage
                bucket_name, blob_name = config_path[5:].split('/', 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                config_data = json.loads(blob.download_as_string())
            else:
                # Load from local file
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

            # Process configuration
            processed_config = self._process_config_data(config_data)

            # Save processed configuration
            if output_path.startswith('gs://'):
                # Save to Cloud Storage
                bucket_name, blob_name = output_path[5:].split('/', 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.upload_from_string(json.dumps(processed_config))
            else:
                # Save to local file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(processed_config, f, indent=2)

            return {
                'config_path': config_path,
                'output_path': output_path,
                'config_size': len(json.dumps(processed_config))
            }
        except Exception as e:
            logger.error(f"Failed to prepare configuration data: {e}")
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        try:
            # Handle missing values
            data = data.fillna(data.mean())

            # Scale numerical features
            scaler = StandardScaler()
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

            # Encode categorical features
            categorical_cols = data.select_dtypes(include=['object']).columns
            data = pd.get_dummies(data, columns=categorical_cols)

            return data
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise

    def _process_model_data(self, model_data: bytes) -> bytes:
        """Process model data for confidential computing."""
        try:
            # Add model metadata
            metadata = {
                'version': '1.0',
                'timestamp': pd.Timestamp.now().isoformat(),
                'size': len(model_data)
            }

            # Combine metadata and model data
            processed_data = json.dumps(metadata).encode() + b'\n' + model_data

            return processed_data
        except Exception as e:
            logger.error(f"Failed to process model data: {e}")
            raise

    def _process_config_data(self, config_data: dict) -> dict:
        """Process configuration data for confidential computing."""
        try:
            # Add configuration metadata
            config_data['metadata'] = {
                'version': '1.0',
                'timestamp': pd.Timestamp.now().isoformat(),
                'environment': 'confidential-computing'
            }

            # Validate configuration
            required_fields = ['model_type', 'parameters', 'security_settings']
            for field in required_fields:
                if field not in config_data:
                    raise ValueError(f"Missing required field: {field}")

            return config_data
        except Exception as e:
            logger.error(f"Failed to process configuration data: {e}")
            raise

def main():
    """Main function to prepare GCP data."""
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        preparator = GCPDataPreparator(project_id)

        # Prepare training data
        training_result = preparator.prepare_training_data(
            'path/to/input.csv',
            'path/to/output'
        )
        logger.info(f"Training data preparation result: {json.dumps(training_result, indent=2)}")

        # Prepare model data
        model_result = preparator.prepare_model_data(
            'path/to/model.pkl',
            'path/to/processed_model.pkl'
        )
        logger.info(f"Model data preparation result: {json.dumps(model_result, indent=2)}")

        # Prepare configuration data
        config_result = preparator.prepare_config_data(
            'path/to/config.json',
            'path/to/processed_config.json'
        )
        logger.info(f"Configuration data preparation result: {json.dumps(config_result, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to prepare GCP data: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 