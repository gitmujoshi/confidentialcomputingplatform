# Azure Secure Training with Confidential Computing

This guide provides step-by-step instructions for setting up and running secure machine learning training in Azure using confidential computing capabilities.

## Prerequisites

### 1. Azure Account and Tools
- Azure subscription with owner/contributor access
- Azure CLI installed
- Terraform installed
- Python 3.8+ installed
- Git installed

### 2. Install Required Tools
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install Terraform
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

# Install Python dependencies
pip install -r requirements.txt
```

## Setup Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ConfidentialComputing/examples/azure
```

### 2. Configure Azure Environment

#### 2.1 Login to Azure
```bash
az login
az account set --subscription <your-subscription-id>
```

#### 2.2 Deploy Infrastructure
```bash
# Navigate to Terraform directory
cd terraform/azure

# Initialize Terraform
terraform init

# Create terraform.tfvars from example
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
# Required values:
# - resource_group_name
# - storage_account_name
# - key_vault_name
# - log_analytics_workspace_name

# Deploy infrastructure
terraform plan -out=tfplan
terraform apply tfplan
```

### 3. Configure Training Environment

#### 3.1 Create and Configure Key Vault Keys
```bash
# Create data encryption key
az keyvault key create \
  --vault-name <your-key-vault-name> \
  --name training-data-key \
  --kty RSA \
  --size 2048

# Create model encryption key
az keyvault key create \
  --vault-name <your-key-vault-name> \
  --name model-key \
  --kty RSA \
  --size 2048
```

#### 3.2 Prepare Training Data
```bash
# Create data directory
mkdir -p data

# Encrypt your training data
python scripts/encrypt_data.py \
  --input-path /path/to/your/training/data \
  --output-path data/train.enc \
  --key-name training-data-key \
  --key-vault-name <your-key-vault-name>

# Encrypt your validation data
python scripts/encrypt_data.py \
  --input-path /path/to/your/validation/data \
  --output-path data/val.enc \
  --key-name training-data-key \
  --key-vault-name <your-key-vault-name>

# Upload encrypted data to Azure Storage
az storage blob upload-batch \
  --account-name <your-storage-account> \
  --container-name training-data \
  --source data
```

#### 3.3 Configure Training Settings
```bash
# Copy and edit configuration file
cp config.json.example config.json

# Update config.json with your values:
# - resource_group_name
# - workspace_name
# - key_vault_name
# - storage_account_name
# - container_name
# - data paths
# - training parameters
```

### 4. Run Secure Training

#### 4.1 Connect to Confidential Computing VM
```bash
# Get VM IP address
VM_IP=$(az vm list-ip-addresses \
  --resource-group <your-resource-group> \
  --name secure-training-vm \
  --query "[0].virtualMachine.network.publicIpAddresses[0].ipAddress" \
  --output tsv)

# Connect to VM
ssh azureuser@$VM_IP
```

#### 4.2 Install Dependencies on VM
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3-pip
pip3 install -r requirements.txt
```

#### 4.3 Start Training
```bash
# Run training script
python secure_train.py
```

## Monitoring Training

### 1. View Training Logs
```bash
# View real-time logs
tail -f training.log

# View Azure Monitor logs
az monitor log-analytics query \
  --workspace <your-workspace-name> \
  --analytics-query "AzureActivity | where Category == 'Administrative'"
```

### 2. Monitor Resources
```bash
# View VM metrics
az monitor metrics list \
  --resource /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.Compute/virtualMachines/secure-training-vm \
  --metric "CPU Credits Consumed,CPU Credits Remaining"

# View storage metrics
az monitor metrics list \
  --resource /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.Storage/storageAccounts/<storage-account> \
  --metric "Transactions,SuccessE2ELatency"
```

## Security Features

### 1. Confidential Computing
- Training runs in AMD SEV-SNP secure enclaves
- Memory encryption at rest
- Secure boot enabled
- vTPM for secure key storage

### 2. Data Protection
- Data encrypted at rest in Azure Storage
- Data decrypted only within secure enclave
- Model checkpoints encrypted before storage
- Key rotation support

### 3. Access Control
- Managed identities for service authentication
- Role-based access control (RBAC)
- Private endpoints for Azure services
- Network security groups

## Troubleshooting

### 1. Common Issues

#### Attestation Failures
```bash
# Check attestation status
python scripts/verify_attestation.py

# Verify VM configuration
az vm show \
  --resource-group <your-resource-group> \
  --name secure-training-vm \
  --query "securityProfile"
```

#### Data Access Issues
```bash
# Verify storage access
az storage blob list \
  --account-name <your-storage-account> \
  --container-name training-data

# Check Key Vault access
az keyvault key list \
  --vault-name <your-key-vault-name>
```

### 2. Performance Issues
```bash
# Check VM performance
az vm get-instance-view \
  --resource-group <your-resource-group> \
  --name secure-training-vm \
  --query "instanceView.statuses"

# Monitor network performance
az network watcher connection-monitor show \
  --resource-group <your-resource-group> \
  --name secure-training-monitor
```

## Cleanup

### 1. Stop Training
```bash
# Stop the training process
pkill -f secure_train.py

# Clean up temporary files
rm -rf data/*.enc
```

### 2. Destroy Infrastructure
```bash
# Navigate to Terraform directory
cd terraform/azure

# Destroy resources
terraform destroy
```

## Additional Resources

### 1. Documentation
- [Azure Confidential Computing](https://docs.microsoft.com/azure/confidential-computing)
- [Azure Key Vault](https://docs.microsoft.com/azure/key-vault)
- [Azure Storage](https://docs.microsoft.com/azure/storage)
- [Azure Monitor](https://docs.microsoft.com/azure/azure-monitor)

### 2. Support
- Azure Support
- GitHub Issues
- Stack Overflow
- Azure Community

## Security Best Practices

1. **Key Management**
   - Rotate encryption keys regularly
   - Use separate keys for data and models
   - Enable soft-delete and purge protection

2. **Network Security**
   - Use private endpoints
   - Restrict access with NSGs
   - Enable network isolation

3. **Monitoring**
   - Enable diagnostic settings
   - Set up alerts for security events
   - Monitor resource usage

4. **Access Control**
   - Use managed identities
   - Implement least privilege
   - Regular access reviews

## Using Your Own Data and Models

### 1. Prepare Your Data

#### 1.1 Data Preparation
```bash
# For CSV data
python scripts/prepare_data.py \
  --input-path /path/to/your/data.csv \
  --output-dir data/processed \
  --data-type csv

# For NumPy data
python scripts/prepare_data.py \
  --input-path /path/to/your/data.npy \
  --output-dir data/processed \
  --data-type numpy

# For custom data formats
# Create a custom data loader function and use it with prepare_data.py
```

#### 1.2 Data Encryption
```bash
# Encrypt training data
python scripts/encrypt_data.py \
  --input-path data/processed/train.csv \
  --output-path data/encrypted/train.enc \
  --key-name training-data-key \
  --key-vault-name <your-key-vault-name> \
  --data-type csv

# Encrypt validation data
python scripts/encrypt_data.py \
  --input-path data/processed/val.csv \
  --output-path data/encrypted/val.enc \
  --key-name training-data-key \
  --key-vault-name <your-key-vault-name> \
  --data-type csv
```

### 2. Using Custom Models

#### 2.1 Create Custom Model
Create a new file `custom_model.py`:
```python
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(YourCustomModel, self).__init__()
        # Define your model architecture here
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Define your forward pass here
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```

#### 2.2 Custom Dataset
For custom data formats, create a custom dataset class:
```python
from torch.utils.data import Dataset

class YourCustomDataset(Dataset):
    def __init__(self, data_path, storage_account, container_name):
        self.data = self._load_and_decrypt_data(data_path, storage_account, container_name)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def _load_and_decrypt_data(self, data_path, storage_account, container_name):
        # Implement your custom data loading and decryption logic here
        pass
```

#### 2.3 Update Configuration
Modify `config.json` to match your data and model requirements:
```json
{
  "training": {
    "input_size": 784,  // Update based on your data
    "hidden_size": 512, // Update based on your model
    "output_size": 10,  // Update based on your task
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10
  },
  "data": {
    "train_data_path": "data/encrypted/train.enc",
    "val_data_path": "data/encrypted/val.enc",
    "key_name": "training-data-key",
    "key_vault_name": "<your-key-vault-name>"
  }
}
```

#### 2.4 Modify Training Script
Update `secure_train.py` to use your custom model:
```python
from custom_model import YourCustomModel

# In the main function, replace the model initialization:
model = YourCustomModel(
    input_size=config["training"]["input_size"],
    hidden_size=config["training"]["hidden_size"],
    output_size=config["training"]["output_size"]
)
```

### 3. Data Format Support

The setup supports various data formats:

1. **CSV Data**:
   - Tabular data in CSV format
   - Automatically splits into train/validation sets
   - Handles missing values and data types

2. **NumPy Arrays**:
   - Numerical data in .npy format
   - Supports multi-dimensional arrays
   - Preserves data types and shapes

3. **Custom Formats**:
   - Implement custom data loader
   - Define data preprocessing steps
   - Handle specific data structures

### 4. Model Customization

You can customize various aspects of the training:

1. **Model Architecture**:
   - Define custom layers
   - Implement custom loss functions
   - Add regularization techniques

2. **Training Process**:
   - Custom optimizers
   - Learning rate schedules
   - Early stopping criteria

3. **Evaluation Metrics**:
   - Custom metrics
   - Validation strategies
   - Model checkpointing

### 5. Security Considerations

When using custom data and models:

1. **Data Security**:
   - All data is encrypted at rest
   - Secure data transfer
   - Access control through managed identities

2. **Model Security**:
   - Encrypted model checkpoints
   - Secure model deployment
   - Access logging and monitoring

3. **Environment Security**:
   - Secure enclave execution
   - Memory encryption
   - Network isolation

// ... rest of existing README content ... 