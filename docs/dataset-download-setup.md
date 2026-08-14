# Kaggle Dataset Download Guide

This guide details how to download and extract the full [TalkingData AdTracking Fraud Detection](https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection) dataset (~1.94 GB compressed) into `data/raw/`.

---

## ⚠️ Prerequisite: Accept Competition Rules (Mandatory)

Before Kaggle allows API downloads, you must accept the competition terms on their site:
1. Open **[talkingdata-adtracking-fraud-detection/rules](https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/rules)**.
2. Click **"I Understand and Accept"**.

---

## 🔑 How to Get Your Kaggle API Credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings).
2. Scroll to the **API** section.
3. Click **"Create New Token"** (downloads a `kaggle.json` file).
4. Open `kaggle.json` to find:
   - `"username"`: your Kaggle username
   - `"key"`: your 32-character API key

---

## 🚀 Step-by-Step Download & Extraction Commands

Follow these clear, step-by-step commands in your terminal:

### Step 1: Install the Kaggle CLI
```bash
pip install kaggle
```

### Step 2: Set Your Kaggle Credentials in the Terminal Session
Set your environment variables (replace with your actual username and key/token):
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key_or_token"
```

### Step 3: Create the Raw Data Directory
```bash
cd /mnt/d/Projects/fcma/ads-safety-mlops
mkdir -p data/raw
```

### Step 4: Download the Full Competition Dataset
This downloads the 1.94 GB archive (`talkingdata-adtracking-fraud-detection.zip`) directly into `data/raw/`:
```bash
kaggle competitions download -c talkingdata-adtracking-fraud-detection -p data/raw/
```

### Step 5: Unzip the Dataset Archive
Unzipping the archive directly inflates all competition files (`train_sample.csv`, `train.csv`, `test.csv`):
```bash
cd data/raw
unzip talkingdata-adtracking-fraud-detection.zip
rm talkingdata-adtracking-fraud-detection.zip
cd ../..
```

---

## 🔍 Step 6: Verify the Extracted Dataset

Check the first few rows:
```bash
head -n 5 data/raw/train_sample.csv
```

Expected schema:
```csv
ip,app,device,os,channel,click_time,attributed_time,is_attributed
```
