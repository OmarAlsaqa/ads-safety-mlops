import pandas as pd
import numpy as np
import os
from schema import RawIAQSchema, ProcessedIAQSchema

def derive_iaq_class(row):
    """
    Derives standard EPA/WHO indoor air quality risk class (0: Good, 1: Moderate, 2: High Risk).
    """
    pm25 = row["PM2.5"]
    tvoc = row["TVOC (ppb)"]

    if pm25 > 35.0 or tvoc > 1000.0:
        return 2  # High Risk / Unhealthy
    elif pm25 > 12.0 or tvoc > 300.0:
        return 1  # Moderate Risk
    else:
        return 0  # Good / Low Risk

def preprocess():
    raw_path = "data/raw/iaq_classification.csv"
    output_dir = "data/processed"
    prod_dir = "data/simulated_production"

    print("1. Reading raw IAQ classification dataset...")
    df = pd.read_csv(raw_path)

    print("2. Handling missing values (Interpolating null PM2.5 readings)...")
    clean_df = df.copy()
    clean_df["PM2.5"] = clean_df["PM2.5"].interpolate(method="linear").bfill()

    print("3. Deriving target classification label (IAQ_Class)...")
    clean_df["IAQ_Class"] = clean_df.apply(derive_iaq_class, axis=1)

    print("4. Performing Day-Based Chronological Split...")
    clean_df["dt"] = pd.to_datetime(clean_df["Timestamp"])
    clean_df["date_str"] = clean_df["dt"].dt.strftime("%Y-%m-%d")

    # Day-based splits (9 calendar days total: 2026-06-29 to 2026-07-07)
    # - Train Set:       Days 1–5 (2026-06-29 -> 2026-07-03)
    # - Validation Set:  Day 6    (2026-07-04)
    # - Test Set:        Day 7    (2026-07-05)
    # - Production/Drift Stream: Days 8–9 (2026-07-06 -> 2026-07-07)
    train_mask = clean_df["date_str"].isin(["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"])
    val_mask = clean_df["date_str"] == "2026-07-04"
    test_mask = clean_df["date_str"] == "2026-07-05"
    drift_mask = clean_df["date_str"].isin(["2026-07-06", "2026-07-07"])

    train_df = clean_df[train_mask].drop(columns=["dt", "date_str"])
    val_df = clean_df[val_mask].drop(columns=["dt", "date_str"])
    test_df = clean_df[test_mask].drop(columns=["dt", "date_str"])
    drift_df = clean_df[drift_mask].drop(columns=["dt", "date_str"])

    print("5. Validating preprocessed Training Set against ProcessedIAQSchema...")
    # Validates preprocessed features + derived target label (IAQ_Class) with Pandera
    ProcessedIAQSchema.validate(train_df)

    # Ensure target output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prod_dir, exist_ok=True)

    # Export split datasets
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)
    drift_df.to_csv(f"{prod_dir}/production_drift_stream.csv", index=False)

    print(f"\n✅ Preprocessing complete!\n"
          f" - Train Set (Days 1-5):         {len(train_df):,} rows\n"
          f" - Validation Set (Day 6):       {len(val_df):,} rows\n"
          f" - Test Set (Day 7):             {len(test_df):,} rows\n"
          f" - Production Drift Stream (Days 8-9): {len(drift_df):,} rows")

if __name__ == "__main__":
    preprocess()