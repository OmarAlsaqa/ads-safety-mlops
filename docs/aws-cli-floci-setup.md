# AWS CLI Installation & Floci Setup Guide

[![Back to Main README](https://img.shields.io/badge/Back_to-Main_README-181717?style=flat-square&logo=github&logoColor=white)](../README.md)

This guide documents the steps to install the official **AWS CLI v2** on Linux/WSL and configure it to connect seamlessly to **Floci** (local AWS emulator running on port 4566).

---

## 1. Install AWS CLI v2

Run the following commands in your WSL terminal to download, unpack, install, and clean up the AWS CLI v2 installer:

```bash
sudo apt update && sudo apt install -y curl unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws
```

Verify installation:
```bash
aws --version
```

---

## 2. Configure AWS CLI for Floci

Configure AWS CLI globally (`~/.aws/`) so terminal commands automatically route to the local Floci container (`http://localhost:4566`) without needing extra flags or temporary `export` environment variables.

### Step 1: Create or update AWS config file

Create the directory and open/edit `~/.aws/config`:

```bash
mkdir -p ~/.aws
nano ~/.aws/config
```

Paste the following configuration:

```ini
[default]
region = us-east-1
endpoint_url = http://localhost:4566
```

### Step 2: Create or update AWS credentials file

Open or edit `~/.aws/credentials`:

```bash
nano ~/.aws/credentials
```

Paste the dummy credentials:

```ini
[default]
aws_access_key_id = test
aws_secret_access_key = test
```

---

## 3. Verify Configuration

Run the following command to verify that the active profile settings, region, and credentials are correctly loaded:

```bash
aws configure list
```
