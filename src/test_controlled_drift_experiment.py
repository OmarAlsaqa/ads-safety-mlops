import os
import time
import pandas as pd
import requests
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset

def run_experiment():
    raw_path = "data/raw/iaq_classification.csv"
    reports_dir = "docs/reports"
    api_metrics_url = "http://localhost:8000/metrics"

    print("======================================================================")
    print("🧪 CONTROLLED DRIFT EXPERIMENT: Phase 1 (Normal) vs Phase 2 (Drifted)")
    print("======================================================================\n")

    raw_df = pd.read_csv(raw_path)
    raw_df["dt"] = pd.to_datetime(raw_df["Timestamp"])
    raw_df["date_str"] = raw_df["dt"].dt.strftime("%Y-%m-%d")

    # Define exact day groups as requested
    ref_days = ["2026-06-29", "2026-06-30"]          # Days 1 & 2 (Baseline Reference)
    normal_curr_days = ["2026-06-29", "2026-06-30"]  # Days 1 & 2 (Same as Reference)
    drift_curr_days = ["2026-07-06", "2026-07-07"]   # Days 8 & 9 (Drifted Production Stream)

    drop_cols = ["Timestamp", "dt", "date_str"]
    eval_cols = [c for c in raw_df.columns if c not in drop_cols]

    ref_df = raw_df[raw_df["date_str"].isin(ref_days)][eval_cols]
    normal_df = raw_df[raw_df["date_str"].isin(normal_curr_days)][eval_cols]
    drift_df = raw_df[raw_df["date_str"].isin(drift_curr_days)][eval_cols]

    os.makedirs(reports_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # PHASE 1: Reference (Days 1-2) vs Current (Days 1-2) -> MUST BE NORMAL (0%)
    # --------------------------------------------------------------------------
    print("📍 PHASE 1: Testing Baseline (Days 1 & 2) vs Baseline (Days 1 & 2)...")
    print(f"   Reference samples: {len(ref_df):,} | Current samples: {len(normal_df):,}")

    report_p1 = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report_p1.run(reference_data=ref_df, current_data=normal_df)

    json_path = os.path.join(reports_dir, "data_drift_summary.json")
    report_p1.save_json(json_path)

    # Trigger FastAPI Prometheus metrics update
    try:
        requests.get(api_metrics_url, timeout=3)
    except Exception as e:
        print(f"Notice API trigger: {e}")

    dict_p1 = report_p1.as_dict()
    for m in dict_p1.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            d_share = res.get("share_of_drifted_columns", 0.0) * 100
            is_drifted = res.get("dataset_drift", False)
            d_cnt = res.get("number_of_drifted_columns", 0)
            status_p1 = "🔴 DRIFT DETECTED" if is_drifted else "🟢 NORMAL"
            print(f"   => PHASE 1 RESULT: Status={status_p1} | Drift Share={d_share:.1f}% | Drifted Features={d_cnt}/{len(eval_cols)}")

    print("   Pausing 10s for Prometheus scraping and Grafana visualization...")
    time.sleep(10)

    # --------------------------------------------------------------------------
    # PHASE 2: Reference (Days 1-2) vs Drifted Current (Days 8-9) -> HIGH DRIFT
    # --------------------------------------------------------------------------
    print("\n📍 PHASE 2: Testing Baseline (Days 1 & 2) vs Production Stream (Days 8 & 9)...")
    print(f"   Reference samples: {len(ref_df):,} | Current samples: {len(drift_df):,}")

    report_p2 = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report_p2.run(reference_data=ref_df, current_data=drift_df)

    report_p2.save_json(json_path)

    # Trigger FastAPI Prometheus metrics update
    try:
        requests.get(api_metrics_url, timeout=3)
    except Exception as e:
        print(f"Notice API trigger: {e}")

    dict_p2 = report_p2.as_dict()
    for m in dict_p2.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            d_share = res.get("share_of_drifted_columns", 0.0) * 100
            is_drifted = res.get("dataset_drift", False)
            d_cnt = res.get("number_of_drifted_columns", 0)
            status_p2 = "🔴 DRIFT DETECTED" if is_drifted else "🟢 NORMAL"
            print(f"   => PHASE 2 RESULT: Status={status_p2} | Drift Share={d_share:.1f}% | Drifted Features={d_cnt}/{len(eval_cols)}")

    print("\n✅ Experiment Complete! Check Grafana (http://localhost/grafana/) to see Phase 1 (0% Green) transition into Phase 2 (High Drift Red)!")

if __name__ == "__main__":
    run_experiment()
