# Python MLOps Environment Setup Guide

[![Back to Main README](https://img.shields.io/badge/Back_to-Main_README-181717?style=flat-square&logo=github&logoColor=white)](../README.md)

This guide documents how to create the Python 3.11 Conda environment (`ops`) and install the core MLOps dependencies from `requirements.txt`.

---

## 🚀 Quick Setup Commands

### Step 1: Create & Activate Conda Environment

```bash
# 1. Create Conda environment with Python 3.11
conda create -n ops python=3.11 -y

# 2. Activate the environment
conda activate ops
```


### Step 2: Install Dependencies from `requirements.txt`

Run the following command from the project root to install the exact dependencies:

```bash
pip install -r requirements.txt
```


## 📦 Installed MLOps Package Stack (`requirements.txt`)

| Category | Package | Version | Purpose |
| --- | --- | --- | --- |
| **Data Processing & EDA** | `pandas`, `seaborn` | `0.13.2` | Data manipulation, statistical profiling, and EDA visualizations. |
| **Data Validation** | `pandera` | `0.32.1` | Schema enforcement, data quality checks, and contract validation. |
| **Experiment Tracking** | `mlflow` | `3.15.1` | Model logging, parameter tracking, metrics visualization, and model registry. |
| **Cloud & S3 SDK** | `boto3` | `1.43.56` | AWS S3 SDK for interacting with local Floci S3 storage (`:4566`). |
| **Data Versioning** | `dvc[s3]` | `3.67.1` | Data version control for raw datasets, processed data, and ML models with AWS S3 remote backend. |
| **Model & Data Monitoring** | `evidently` | `0.7.21` | Data drift detection, model performance monitoring, and HTML reports. |

## ✅ Verify Environment Installation

Test that all packages load properly inside your `ops` environment:

```bash
python -c "import pandas, seaborn, matplotlib, pandera, mlflow, boto3, dvc, evidently; print('✅ All MLOps & EDA packages imported successfully!')"
```
