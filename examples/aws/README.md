# AWS Confidential Computing Security Scripts

This directory contains Python scripts for monitoring and securing AWS confidential computing environments.

## Prerequisites

- Python 3.7 or higher
- AWS account with appropriate permissions
- AWS credentials configured
- Required AWS services enabled:
  - EC2 with Nitro Enclaves
  - S3
  - KMS
  - IAM
  - Security Hub

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="your-region"
```

## Scripts

### Verify Security (`verify_security.py`)

This script verifies the security configuration of your AWS environment, including:
- EC2 instance security (Nitro Enclaves support)
- S3 bucket security
- KMS key security
- IAM security settings

Usage:
```bash
python scripts/verify_security.py
```

### Prepare Data (`prepare_data.py`)

This script prepares and encrypts data for confidential computing:
- Training data preparation
- Model data preparation
- Configuration data preparation

Usage:
```bash
python scripts/prepare_data.py
```

### Encrypt Data (`encrypt_data.py`)

This script provides encryption utilities for AWS data:
- File encryption
- S3 object encryption
- Directory encryption
- Bucket encryption

Usage:
```bash
python scripts/encrypt_data.py
```

## Security Features

The scripts implement the following security features:

1. **Confidential Computing**
   - Nitro Enclaves support verification
   - Secure instance configuration

2. **Encryption**
   - KMS-based encryption
   - S3 server-side encryption
   - Client-side encryption

3. **Access Control**
   - IAM security verification
   - S3 bucket policies
   - KMS key policies

4. **Data Protection**
   - Secure data preparation
   - Encrypted data storage
   - Secure data transfer

5. **Monitoring and Compliance**
   - Security configuration verification
   - Compliance checks
   - Security best practices enforcement

## Best Practices

1. Run security verification regularly
2. Use encryption for all sensitive data
3. Follow the principle of least privilege
4. Enable AWS CloudTrail for audit logging
5. Use AWS Config for compliance monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 