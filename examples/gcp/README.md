# GCP Secure Training Setup

This directory contains scripts and configurations for running secure machine learning training jobs on Google Cloud Platform using Vertex AI with encryption and security features.

## Prerequisites

1. Google Cloud SDK installed and configured
2. Python 3.8 or later
3. Required Python packages:
   ```bash
   pip install google-cloud-aiplatform google-cloud-storage google-cloud-kms torch torchvision
   ```

## Configuration

1. Update `config.json` with your GCP settings:
   - Replace `your-project-id` with your GCP project ID
   - Update KMS key names and paths
   - Configure VPC network and subnet settings
   - Set appropriate machine types and accelerator configurations

2. Ensure service account permissions:
   - Vertex AI service account with necessary permissions
   - Cloud KMS key access permissions
   - Cloud Storage bucket access permissions

## Directory Structure

```
gcp/
├── config.json           # GCP configuration
├── train.py             # Training script
├── training_job.py      # Training job management
└── utils.py            # Utility functions
```

## Usage

1. Set up secure environment:
   ```bash
   python utils.py
   ```

2. Start training job:
   ```bash
   python training_job.py
   ```

3. Monitor training progress:
   ```bash
   # The training job script includes monitoring
   # Or use Google Cloud Console to monitor jobs
   ```

## Security Features

- Data encryption at rest using Cloud KMS
- Network isolation using VPC
- Service account-based access control
- Secure storage with Cloud Storage encryption
- Confidential computing support

## Monitoring and Logging

- Training metrics available through Vertex AI
- Cloud Logging for detailed monitoring
- Cloud Storage access logs for storage monitoring

## Cleanup

To clean up resources:
1. Stop any running training jobs
2. Delete model artifacts from Cloud Storage
3. Clean up temporary files
4. Delete KMS keys if no longer needed

## Troubleshooting

Common issues and solutions:
1. Service account permission errors: Check IAM permissions
2. VPC configuration issues: Verify network and subnet settings
3. Storage access errors: Verify Cloud Storage permissions
4. KMS key errors: Check key policy and access

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai)
- [Cloud KMS Documentation](https://cloud.google.com/kms)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices) 