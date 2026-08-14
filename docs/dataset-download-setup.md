# Longitudinal Indoor Air Quality Dataset Setup Guide

[![Back to Main README](https://img.shields.io/badge/Back_to-Main_README-181717?style=flat-square&logo=github&logoColor=white)](../README.md)

This guide documents how to manually download, extract, clean up, and set up the **Longitudinal Indoor Air Quality Dataset (Classification Subset)** for model training and local S3 object storage upload.

---

## 📌 Dataset Overview

- **Dataset Source:** [Mendeley Data - Longitudinal Indoor Air Quality Dataset (v2)](https://data.mendeley.com/datasets/b5jvs7kykn/2)
- **Selected File:** `IAQ_Classification_Subset.csv`
- **Description:** Contains 139,074 high-frequency instances of indoor air quality sensor readings (Temperature, Humidity, Pressure, Gas Resistance, PM2.5, TVOC, eCO2, VOC Index, MQ135, Voltage, PPM) cleaned for indoor air quality risk classification and alerting.
- **Download Requirement:** Requires **manual downloading** from Mendeley Data due to portal download restrictions.
- **Target Local File:** `data/raw/iaq_classification.csv`

---

## 🚀 Manual Download & Setup Commands

### Step 1: Download Archive Manually
1. Open [Mendeley Data - Longitudinal Indoor Air Quality Dataset](https://data.mendeley.com/datasets/b5jvs7kykn/2).
2. Download the ZIP file: **`Longitudinal Indoor Air Quality Dataset Collected.zip`**.
3. Place the downloaded `.zip` file into your project's `data/raw/` directory.

---

### Step 2: Extract & Clean Up Commands

Run the following commands in your terminal from `data/raw/`:

```bash
# 0. Make the data/raw/ directory
mkdir -p data/raw data/processed data/simulated_production

# 1. Navigate to the raw data directory
cd data/raw

# 2. Extract the downloaded Mendeley dataset archive
unzip "Longitudinal Indoor Air Quality Dataset Collected.zip"

# 3. Move and rename the Classification Subset CSV to data/raw/iaq_classification.csv
mv "Longitudinal Indoor Air Quality Dataset Collected/Research_Subsets/IAQ_Classification_Subset.csv" ./iaq_classification.csv

# 4. Remove unneeded extracted folders, zip archive, and old UCI dataset files
rm -rf "Longitudinal Indoor Air Quality Dataset Collected" \
       "Longitudinal Indoor Air Quality Dataset Collected.zip"
```

---

## 📂 Expected Directory Structure

After running the cleanup commands, your `data/raw/` directory will cleanly contain only:

```text
data/raw/
└── iaq_classification.csv     # Primary dataset (139,074 rows x 12 attributes)
```
