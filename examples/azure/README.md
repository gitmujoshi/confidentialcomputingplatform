# Azure Confidential Computing Examples

This directory contains example implementations for running secure machine learning training jobs on Microsoft Azure using confidential computing features.

## Prerequisites

1. Azure CLI installed and configured
2. Python 3.8+ installed
3. Required Python packages:
   - azure-mgmt-compute
   - azure-mgmt-storage
   - azure-mgmt-keyvault
   - azure-storage-blob
   - azure-identity
   - torch
   - numpy
   - logging

## Setup

1. Install the required packages:
```bash
pip install azure-mgmt-compute azure-mgmt-storage azure-mgmt-keyvault azure-storage-blob azure-identity torch numpy
```

2. Configure Azure credentials:
```bash
az login
```

3. Update the `config.json` file with your Azure-specific values:
   - `subscription_id`: Your Azure subscription ID
   - `resource_group`: Your Azure resource group name
   - `key_vault_name`: Your Azure Key Vault name
   - `storage_account`: Your Azure Storage account name
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
- Create a confidential VM instance
- Set up the training environment
- Start the training process
- Monitor and log the training progress

## Security Features

The implementation includes several security features:

1. **Confidential VM**: Training runs in a confidential VM with memory encryption
2. **Data Encryption**: Training data is encrypted at rest and in transit
3. **Secure Storage**: Model checkpoints are stored securely in Azure Blob Storage
4. **Key Management**: Uses Azure Key Vault for key management
5. **Secure Communication**: All communication is encrypted

## Monitoring

The training progress can be monitored through:
- Azure Portal
- Azure Monitor
- Application logs

## Best Practices

1. Always use the latest Azure SDK version
2. Regularly rotate encryption keys
3. Monitor resource usage and costs
4. Implement proper error handling and logging
5. Use appropriate VM sizes for your workload
6. Implement proper backup and recovery procedures

## Troubleshooting

Common issues and solutions:

1. **VM Creation Fails**
   - Check resource group permissions
   - Verify network configuration
   - Ensure sufficient quota

2. **Training Fails**
   - Check data access permissions
   - Verify Key Vault configuration
   - Check VM resources

3. **Storage Issues**
   - Verify storage account permissions
   - Check storage quota
   - Verify encryption configuration

## Support

For issues and support:
1. Check Azure documentation
2. Contact Azure support
3. Open an issue in this repository

## License

This example code is provided under the MIT License. See the LICENSE file for details. 