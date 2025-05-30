terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "confidential_rg" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# Virtual Network
resource "azurerm_virtual_network" "confidential_vnet" {
  name                = "confidential-vnet"
  resource_group_name = azurerm_resource_group.confidential_rg.name
  location            = azurerm_resource_group.confidential_rg.location
  address_space       = ["10.0.0.0/16"]
  tags                = var.tags
}

# Subnet
resource "azurerm_subnet" "confidential_subnet" {
  name                 = "confidential-subnet"
  resource_group_name  = azurerm_resource_group.confidential_rg.name
  virtual_network_name = azurerm_virtual_network.confidential_vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "confidential_nsg" {
  name                = "confidential-nsg"
  location            = azurerm_resource_group.confidential_rg.location
  resource_group_name = azurerm_resource_group.confidential_rg.name
  tags                = var.tags

  security_rule {
    name                       = "SSH"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Network Interface
resource "azurerm_network_interface" "confidential_nic" {
  name                = "confidential-nic"
  location            = azurerm_resource_group.confidential_rg.location
  resource_group_name = azurerm_resource_group.confidential_rg.name
  tags                = var.tags

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.confidential_subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.confidential_pip.id
  }
}

# Public IP
resource "azurerm_public_ip" "confidential_pip" {
  name                = "confidential-pip"
  resource_group_name = azurerm_resource_group.confidential_rg.name
  location            = azurerm_resource_group.confidential_rg.location
  allocation_method   = "Dynamic"
  tags                = var.tags
}

# Confidential VM
resource "azurerm_linux_virtual_machine" "confidential_vm" {
  name                = "confidential-vm"
  resource_group_name = azurerm_resource_group.confidential_rg.name
  location            = azurerm_resource_group.confidential_rg.location
  size                = "Standard_DC2s_v2"  # Azure confidential computing VM size
  admin_username      = var.admin_username
  tags                = var.tags

  network_interface_ids = [
    azurerm_network_interface.confidential_nic.id,
  ]

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "UbuntuServer"
    sku       = "18.04-LTS"
    version   = "latest"
  }
}

# Key Vault
resource "azurerm_key_vault" "confidential_kv" {
  name                        = var.key_vault_name
  location                    = azurerm_resource_group.confidential_rg.location
  resource_group_name         = azurerm_resource_group.confidential_rg.name
  enabled_for_disk_encryption = true
  tenant_id                   = var.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
  sku_name                   = "standard"
  tags                       = var.tags

  access_policy {
    tenant_id = var.tenant_id
    object_id = var.object_id

    key_permissions = [
      "Get",
      "List",
      "Create",
      "Delete",
      "Update",
      "Import",
      "Backup",
      "Restore",
      "Recover",
      "Purge"
    ]

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete",
      "Backup",
      "Restore",
      "Recover",
      "Purge"
    ]
  }
}

# Attestation Provider
resource "azurerm_attestation_provider" "confidential_attestation" {
  name                = var.attestation_provider_name
  resource_group_name = azurerm_resource_group.confidential_rg.name
  location            = azurerm_resource_group.confidential_rg.location
  tags                = var.tags
}

# Storage Account
resource "azurerm_storage_account" "confidential_storage" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.confidential_rg.name
  location                 = azurerm_resource_group.confidential_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = var.tags
}

# Storage Container
resource "azurerm_storage_container" "confidential_container" {
  name                  = "confidential-data"
  storage_account_name  = azurerm_storage_account.confidential_storage.name
  container_access_type = "private"
} 