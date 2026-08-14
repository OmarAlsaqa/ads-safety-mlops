# Installing Docker Engine directly on WSL (Ubuntu / Debian)

[![Back to Main README](https://img.shields.io/badge/Back_to-Main_README-181717?style=flat-square&logo=github&logoColor=white)](../README.md)

This guide documents the steps to install and configure Docker Engine (standalone daemon) natively inside WSL 2 without requiring Docker Desktop for Windows.

---

## 1. Download and Run the Official Docker Install Script

Download the convenient automated installation script provided by Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

*(Optional)* Cleanup the installer script after installation:
```bash
rm get-docker.sh
```

---

## 2. Manage Docker as a Non-Root User

To avoid using `sudo` before every `docker` command, add your current user to the `docker` group:

```bash
sudo usermod -aG docker $USER
```

### Apply Group Changes immediately
Group membership updates require a new shell session. To apply the changes immediately without closing the terminal:

```bash
newgrp docker
```

---

## 3. Start the Docker Daemon

Depending on whether `systemd` is enabled in your WSL configuration:

### Option A: Using `service` (Standard WSL)
```bash
sudo service docker start
```

### Option B: Using `systemctl` (WSL with systemd enabled)
```bash
sudo systemctl enable --now docker.service
```

---

## 4. Verify Installation

Run the official test image to confirm Docker is working properly:

```bash
docker run hello-world
```

If successful, you will see a message beginning with:
> **Hello from Docker!**
> This message shows that your installation appears to be working correctly.

---

## Troubleshooting & Notes

- **Permission Denied Error**: `permission denied while trying to connect to the Docker daemon socket`
  - Ensure your user was added to the `docker` group (`sudo usermod -aG docker $USER`).
  - Run `newgrp docker` or restart your WSL session (`wsl --shutdown` from PowerShell/CMD).
- **Daemon Not Running**: If commands fail with `Is the docker daemon running?`, verify the service status with `sudo service docker status` or `sudo service docker start`.
