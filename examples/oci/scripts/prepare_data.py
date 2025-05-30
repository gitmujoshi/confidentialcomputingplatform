#!/usr/bin/env python3

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import oci
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCIDataPreparator:
    def __init__(self, config_path: str = None):
        """Initialize OCI data preparation."""
        self.config = oci.config.from_file(config_path) if config_path else oci.config.from_file()
        self.object_storage_client = oci.object_storage.ObjectStorageClient(self.config)

    def prepare_training_data(self, namespace: str, input_bucket: str, input_object: str, output_bucket: str) -> dict:
        """Prepare training data for confidential computing."""
        try:
            # Read input data
            get_object_response = self.object_storage_client.get_object(
                namespace_name=namespace,
                bucket_name=input_bucket,
                object_name=input_object
            )
            data = pd.read_csv(get_object_response.data.content)

            # Preprocess data
            data = self._preprocess_data(data)

            # Split data
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            # Save processed data
            # Save training data
            train_object_name = f"{input_object.rsplit('.', 1)[0]}_train.csv"
            self.object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=output_bucket,
                object_name=train_object_name,
                put_object_body=train_data.to_csv(index=False)
            )

            # Save test data
            test_object_name = f"{input_object.rsplit('.', 1)[0]}_test.csv"
            self.object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=output_bucket,
                object_name=test_object_name,
                put_object_body=test_data.to_csv(index=False)
            )

            return {
                'input_bucket': input_bucket,
                'input_object': input_object,
                'output_bucket': output_bucket,
                'train_object': train_object_name,
                'test_object': test_object_name,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'features': list(data.columns)
            }
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise

    def prepare_model_data(self, namespace: str, input_bucket: str, input_object: str, output_bucket: str) -> dict:
        """Prepare model data for confidential computing."""
        try:
            # Get model data
            get_object_response = self.object_storage_client.get_object(
                namespace_name=namespace,
                bucket_name=input_bucket,
                object_name=input_object
            )
            model_data = get_object_response.data.content

            # Process model data
            processed_data = self._process_model_data(model_data)

            # Save processed model data
            output_object_name = f"{input_object.rsplit('.', 1)[0]}_processed{input_object[input_object.rfind('.'):]}"
            self.object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=output_bucket,
                object_name=output_object_name,
                put_object_body=processed_data
            )

            return {
                'input_bucket': input_bucket,
                'input_object': input_object,
                'output_bucket': output_bucket,
                'output_object': output_object_name,
                'model_size': len(processed_data)
            }
        except Exception as e:
            logger.error(f"Failed to prepare model data: {e}")
            raise

    def prepare_config_data(self, namespace: str, input_bucket: str, input_object: str, output_bucket: str) -> dict:
        """Prepare configuration data for confidential computing."""
        try:
            # Get configuration data
            get_object_response = self.object_storage_client.get_object(
                namespace_name=namespace,
                bucket_name=input_bucket,
                object_name=input_object
            )
            config_data = json.loads(get_object_response.data.content)

            # Process configuration
            processed_config = self._process_config_data(config_data)

            # Save processed configuration
            output_object_name = f"{input_object.rsplit('.', 1)[0]}_processed.json"
            self.object_storage_client.put_object(
                namespace_name=namespace,
                bucket_name=output_bucket,
                object_name=output_object_name,
                put_object_body=json.dumps(processed_config)
            )

            return {
                'input_bucket': input_bucket,
                'input_object': input_object,
                'output_bucket': output_bucket,
                'output_object': output_object_name,
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
    """Main function to prepare OCI data."""
    try:
        # Initialize preparator
        preparator = OCIDataPreparator()

        # Prepare training data
        training_result = preparator.prepare_training_data(
            'your-namespace',
            'input-bucket',
            'input-data.csv',
            'output-bucket'
        )
        logger.info(f"Training data preparation result: {json.dumps(training_result, indent=2)}")

        # Prepare model data
        model_result = preparator.prepare_model_data(
            'your-namespace',
            'input-bucket',
            'model.pkl',
            'output-bucket'
        )
        logger.info(f"Model data preparation result: {json.dumps(model_result, indent=2)}")

        # Prepare configuration data
        config_result = preparator.prepare_config_data(
            'your-namespace',
            'input-bucket',
            'config.json',
            'output-bucket'
        )
        logger.info(f"Configuration data preparation result: {json.dumps(config_result, indent=2)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to prepare OCI data: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 