# Common Confidential Computing Scripts

This directory contains shared scripts used across different cloud providers for confidential computing operations.

## Scripts

### Validate Cloud Configuration (`validate_cloud_config.py`)

Validates cloud provider configurations for confidential computing:
- Checks required services and APIs
- Verifies security settings
- Validates network configurations
- Ensures compliance with best practices

### Manage Keys (`manage_keys.py`)

Common key management utilities:
- Key generation
- Key rotation
- Key backup and recovery
- Key policy management

### Test Enclave (`test_enclave.py`)

Enclave testing utilities:
- Enclave attestation testing
- Enclave communication testing
- Enclave performance testing
- Security validation

### Setup Environment (`setup_env.py`)

Environment setup utilities:
- Common dependencies installation
- Environment variable configuration
- Security baseline setup
- Cross-cloud provider setup

## Usage

These scripts are used by the cloud provider-specific scripts to perform common operations. They should not be used directly unless you understand their dependencies and requirements.

## Dependencies

The common scripts require:
- Python 3.7 or higher
- Cloud provider SDKs (AWS, GCP, OCI)
- Common security libraries
- Testing frameworks

## Best Practices

1. Keep common code DRY (Don't Repeat Yourself)
2. Maintain backward compatibility
3. Document all shared functionality
4. Test across all supported cloud providers
5. Follow security best practices

## Contributing

When adding new common functionality:
1. Ensure it's truly common across providers
2. Add appropriate documentation
3. Include tests
4. Update this README
5. Follow the project's coding standards 