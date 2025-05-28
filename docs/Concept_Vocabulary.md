# Confidential Computing Platform - Concept Vocabulary

## Core Concepts

### Confidential Computing
A cloud computing technology that protects data in use by performing computation in a hardware-based Trusted Execution Environment (TEE). It ensures that data remains encrypted even during processing.

### Trusted Execution Environment (TEE)
A secure area within a processor that ensures the confidentiality and integrity of code and data. It provides an isolated execution environment that protects against unauthorized access or modification.

### Secure Enclave
A protected region of memory that provides confidentiality and integrity for code and data. It ensures that even privileged system software cannot access the contents of the enclave.

## Cloud Provider Specific Terms

### Azure
- **DCsv2-series VMs**: Azure virtual machines with Intel SGX support for confidential computing
- **Azure Attestation Service**: Service that verifies the trustworthiness of a platform and the integrity of the binaries running inside it
- **Intel SGX**: Intel Software Guard Extensions, a set of CPU instructions that enable secure enclaves

### AWS
- **Nitro Enclaves**: Isolated compute environments that provide additional security for processing sensitive data
- **Nitro System**: The underlying platform that provides the security and isolation capabilities for AWS Nitro Enclaves
- **CloudHSM**: AWS Cloud Hardware Security Module service for secure key storage and cryptographic operations

### GCP
- **Confidential VMs**: Virtual machines with AMD SEV technology for memory encryption
- **AMD SEV**: AMD Secure Encrypted Virtualization, a technology that encrypts VM memory
- **VPC Service Controls**: Security feature that helps mitigate data exfiltration risks

## Security Terms

### Encryption
- **End-to-End Encryption**: Encryption of data throughout its entire lifecycle, from creation to storage and processing
- **Encryption at Rest**: Protection of data when it is stored
- **Memory Encryption**: Protection of data while it is being processed in memory

### Access Control
- **Zero-Trust Architecture**: Security model that requires strict identity verification for every person and device trying to access resources
- **Role-Based Access Control (RBAC)**: Method of regulating access to resources based on the roles of individual users
- **Least Privilege Access**: Security principle that users should have only the minimum permissions necessary to perform their tasks

### Authentication & Authorization
- **Remote Attestation**: Process of verifying the integrity and security of a remote system
- **Hardware Security Module (HSM)**: Physical computing device that safeguards and manages digital keys
- **Service Principal**: Identity used by a service or application to access specific resources

## Infrastructure Terms

### Compute
- **Virtual Machine (VM)**: Software emulation of a physical computer
- **Container**: Lightweight, standalone package that includes everything needed to run a piece of software
- **Instance**: A virtual server in the cloud

### Storage
- **Object Storage**: Storage architecture that manages data as objects
- **Block Storage**: Storage that manages data as blocks within sectors and tracks
- **Data Lifecycle Management**: Process of managing data throughout its lifecycle from creation to deletion

### Networking
- **VPC (Virtual Private Cloud)**: Isolated network environment in the cloud
- **Private Endpoint**: Network interface that connects privately and securely to a service
- **Security Groups**: Virtual firewall that controls inbound and outbound traffic

## Monitoring & Operations

### Monitoring
- **CloudWatch**: AWS service for monitoring and observability
- **Azure Monitor**: Azure service for collecting and analyzing telemetry
- **Cloud Monitoring**: GCP service for monitoring and observability

### Logging
- **Audit Trail**: Chronological record of system activities
- **Security Logs**: Records of security-related events
- **System Logs**: Records of system events and operations

## Compliance & Governance

### Compliance
- **FIPS 140-2**: Federal Information Processing Standards for cryptographic modules
- **Data Residency**: Physical or geographical location of an organization's data
- **Privacy Regulations**: Laws and regulations governing data privacy

### Governance
- **Security Policy**: Set of rules and procedures that define how an organization manages and protects its information
- **Compliance Monitoring**: Process of monitoring and ensuring adherence to compliance requirements
- **Security Analytics**: Analysis of security data to identify patterns and potential threats

## AI/ML Terms

### Machine Learning
- **Model Training**: Process of teaching a machine learning model to make predictions
- **Checkpoint**: Saved state of a model during training
- **Inference**: Process of using a trained model to make predictions

### Data Processing
- **Data Validation**: Process of ensuring data is correct and useful
- **Batch Processing**: Processing of data in groups
- **Data Pipeline**: Set of data processing elements connected in series

## References

- [NIST Glossary](https://csrc.nist.gov/glossary)
- [Cloud Security Alliance](https://cloudsecurityalliance.org/)
- [Confidential Computing Consortium](https://confidentialcomputing.io/) 