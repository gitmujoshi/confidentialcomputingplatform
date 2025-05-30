# GCP Confidential Computing Security Scripts

This directory contains Python scripts for monitoring and remediating security issues in GCP confidential computing environments.

## Prerequisites

- Python 3.7 or higher
- GCP project with billing enabled
- GCP credentials configured
- Required GCP APIs enabled:
  - Compute Engine API
  - Cloud Storage API
  - Cloud KMS API
  - Cloud Monitoring API
  - Security Command Center API

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

3. Set up GCP credentials:
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Scripts

### Monitor Security (`monitor_security.py`)

This script monitors the security status of your GCP confidential computing environment, including:
- Confidential instance status
- Network traffic monitoring
- Security Command Center findings
- Storage bucket security
- KMS keyring security

Usage:
```bash
python scripts/monitor_security.py
```

### Remediate Security (`remediate_security.py`)

This script automatically remediates common security issues in your GCP environment, including:
- Enabling confidential computing on instances
- Encrypting unencrypted disks
- Enabling versioning and encryption on storage buckets
- Enabling key rotation for KMS keys
- Updating network firewall rules

Usage:
```bash
python scripts/remediate_security.py
```

## Security Features

The scripts implement the following security features:

1. **Confidential Computing**
   - Enables confidential computing on instances
   - Monitors confidential computing status

2. **Encryption**
   - Ensures disk encryption using customer-managed keys
   - Enables storage bucket encryption
   - Manages KMS key rotation

3. **Network Security**
   - Monitors network traffic
   - Updates firewall rules
   - Restricts public access

4. **Storage Security**
   - Enables versioning
   - Enforces uniform bucket-level access
   - Monitors bucket security settings

5. **Monitoring and Alerting**
   - Tracks security findings
   - Monitors resource configurations
   - Logs security events

## Best Practices

1. Run the monitoring script regularly to check security status
2. Review remediation actions before applying them
3. Keep dependencies up to date
4. Use service accounts with minimal required permissions
5. Enable audit logging for all actions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 