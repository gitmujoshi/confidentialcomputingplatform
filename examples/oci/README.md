# OCI Confidential Computing Examples

This directory contains example implementations for running secure machine learning training jobs on Oracle Cloud Infrastructure (OCI) using confidential computing features.

## Prerequisites

1. OCI CLI installed and configured
2. Python 3.8+ installed
3. Required Python packages:
   - oci-sdk
   - torch
   - numpy
   - logging

## Setup

1. Install the required packages:
```bash
pip install oci-sdk torch numpy
```

2. Configure OCI credentials:
```bash
oci setup config
```

3. Update the `config.json` file with your OCI-specific values:
   - `compartment_id`: Your OCI compartment ID
   - `subnet_id`: Your VCN subnet ID
   - `encryption_key_id`: Your OCI Vault key ID
   - Update other configuration values as needed

## Files

- `training_job.py`: Main script for creating and managing secure training jobs
- `train.py`: Training script that runs inside the secure enclave
- `config.json`: Configuration file for the training job

## Usage

1. Start a secure training job:
```bash
python training_job.py
```

This will:
- Create a secure enclave instance
- Set up the training environment
- Start the training process
- Monitor and log the training progress

## Security Features

The implementation includes several security features:

1. **Secure Enclave**: Training runs in a secure enclave with memory encryption
2. **Data Encryption**: Training data is encrypted at rest and in transit
3. **Secure Storage**: Model checkpoints are stored securely in OCI Object Storage
4. **Key Management**: Uses OCI Vault for key management
5. **Secure Communication**: All communication is encrypted

## Monitoring

The training progress can be monitored through:
- OCI Console
- Cloud Watch logs
- Application logs

## Best Practices

1. Always use the latest OCI SDK version
2. Regularly rotate encryption keys
3. Monitor resource usage and costs
4. Implement proper error handling and logging
5. Use appropriate instance shapes for your workload
6. Implement proper backup and recovery procedures

## Troubleshooting

Common issues and solutions:

1. **Instance Creation Fails**
   - Check compartment permissions
   - Verify subnet configuration
   - Ensure sufficient quota

2. **Training Fails**
   - Check data access permissions
   - Verify encryption key configuration
   - Check instance resources

3. **Storage Issues**
   - Verify bucket permissions
   - Check storage quota
   - Verify encryption configuration

## Support

For issues and support:
1. Check OCI documentation
2. Contact OCI support
3. Open an issue in this repository

## License

This example code is provided under the MIT License. See the LICENSE file for details. 