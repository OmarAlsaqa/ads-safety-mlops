# DVC Data Versioning & Local S3 Storage Guide

[![Back to Main README](https://img.shields.io/badge/Back_to-Main_README-181717?style=flat-square&logo=github&logoColor=white)](../README.md)

This guide documents how to set up **Data Version Control (DVC)** with a **Local S3 Remote (Floci)** to track datasets, models, and artifacts without committing large binary files directly to Git.

---

## 🛠️ Step-by-Step DVC Setup & Workflow

### Step 1: Install DVC with S3 Support
Ensure `dvc[s3]` (or `dvc-s3`) is installed in your Python environment:

```bash
pip install "dvc[s3]"
# Or install from requirements.txt:
pip install -r requirements.txt
```

---

### Step 2: Ensure Local S3 Stack is Running & Buckets Created
Start the local Floci container stack and create your S3 storage bucket:

```bash
# 1. Start Docker Compose stack
docker compose up -d

# 2. Create local S3 bucket for DVC data storage
aws --endpoint-url=http://localhost:4566 s3 mb s3://mlops-data
```

---

### Step 3: Initialize DVC in the Repository
Initialize DVC inside your project root:

```bash
dvc init
```

---

### Step 4: Configure Local S3 Remote (`floci`)
Add `floci` as your default DVC remote pointing to your local S3 endpoint (`http://localhost:4566`):

```bash
# 1. Set default DVC remote pointing to s3://mlops-data
dvc remote add -d floci s3://mlops-data

# 2. Configure local Floci endpoint URL & credentials
dvc remote modify floci endpointurl http://localhost:4566
dvc remote modify floci access_key_id test
dvc remote modify floci secret_access_key test
```

---

### Step 5: Track Raw Dataset with DVC
Track the raw dataset using DVC instead of Git:

```bash
dvc add data/raw/iaq_classification.csv
```

This creates `data/raw/iaq_classification.csv.dvc` (a lightweight pointer file) and adds `iaq_classification.csv` to `data/raw/.gitignore`.

---

### Step 6: Commit DVC Metadata to Git
Commit the `.dvc` tracking files and configuration to Git:

```bash
git add .dvc .gitignore data/raw/iaq_classification.csv.dvc data/raw/.gitignore
git commit -m "init dvc and track raw IAQ classification dataset"
```

---

### Step 7: Push Dataset to Local S3 Remote
Push the actual raw data files to your local S3 bucket (`floci`):

```bash
dvc push
```

---

### Step 8: Define & Reproduce DVC Pipeline (`dvc repro`)
The `dvc.yaml` file defines reproducibility stages:

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - src/schema.py
      - data/raw/iaq_classification.csv
    outs:
      - data/processed
      - data/simulated_production/production_drift_stream.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.parquet
      - data/processed/val.parquet
    outs:
      - models/model.pkl

  evaluate_drift:
    cmd: python src/evaluate_drift.py
    deps:
      - src/evaluate_drift.py
      - data/processed/train.parquet
      - data/simulated_production/production_drift_stream.csv
    outs:
      - docs/reports/data_drift_report.html
      - docs/reports/data_drift_summary.json
```

#### Execute Pipeline & Push Artifacts:
```bash
# 1. Reproduce pipeline stage (runs preprocess.py and generates dvc.lock)
dvc repro

# 2. Push processed artifacts to local S3
dvc push

# 3. Commit pipeline configuration and lock file to Git
git add dvc.yaml dvc.lock src/schema.py src/preprocess.py
git commit -m "add DVC preprocessing pipeline stage and lock file"
```

---

## 🔍 Verifying Data in Local S3 & Floci UI

- **Verify via AWS CLI:**
  ```bash
  aws --endpoint-url=http://localhost:4566 s3 ls s3://mlops-data/files/md5/ --recursive
  ```
- **Verify via Floci Web UI:**
  Open **`http://localhost:8080`** in your browser and open the `mlops-data` bucket.
