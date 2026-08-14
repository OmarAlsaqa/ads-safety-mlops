import os
import time
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


def run_controlled_drift_experiment():
    raw_path = "data/raw/train_sample.csv"
    ref_path = "data/processed/train.parquet"
    reports_dir = "docs/reports"
    api_drift_url = "http://localhost:8000/metrics/drift"

    print("======================================================================")
    print("🧪 CONTROLLED DRIFT EXPERIMENT: Phase 1 (0% Normal) vs Phase 2 (High Drift)")
    print("======================================================================\n")

    if not os.path.exists(raw_path) or not os.path.exists(ref_path):
        raise FileNotFoundError("Required data files not found. Run 'dvc repro' first.")

    ref_df = pd.read_parquet(ref_path)
    eval_cols = ["hour", "day", "ip_click_count", "ip_unique_apps", "app_freq", "channel_freq", "device_freq"]
    if "is_attributed" in ref_df.columns:
        eval_cols.append("is_attributed")

    # Baseline slice (Nov 7 & Nov 8)
    raw_df = pd.read_csv(raw_path)
    raw_df["click_time"] = pd.to_datetime(raw_df["click_time"])
    raw_df["date_str"] = raw_df["click_time"].dt.strftime("%Y-%m-%d")
    raw_df["hour"] = raw_df["click_time"].dt.hour
    raw_df["day"] = raw_df["click_time"].dt.day

    ip_counts = raw_df["ip"].value_counts()
    app_counts = raw_df["app"].value_counts()
    chan_counts = raw_df["channel"].value_counts()
    dev_counts = raw_df["device"].value_counts()
    ip_app_counts = raw_df.groupby("ip")["app"].nunique()

    raw_df["ip_click_count"] = raw_df["ip"].map(ip_counts).fillna(1)
    raw_df["ip_unique_apps"] = raw_df["ip"].map(ip_app_counts).fillna(1)
    raw_df["app_freq"] = raw_df["app"].map(app_counts).fillna(1)
    raw_df["channel_freq"] = raw_df["channel"].map(chan_counts).fillna(1)
    raw_df["device_freq"] = raw_df["device"].map(dev_counts).fillna(1)

    baseline_df = raw_df[raw_df["date_str"].isin(["2017-11-07", "2017-11-08"])][eval_cols]
    drifted_df = raw_df[raw_df["date_str"] == "2017-11-06"][eval_cols]

    ref_data = baseline_df.sample(n=min(20000, len(baseline_df)), random_state=42)
    normal_data = baseline_df.sample(n=min(20000, len(baseline_df)), random_state=123)
    drift_data = drifted_df

    os.makedirs(reports_dir, exist_ok=True)
    html_path = os.path.join(reports_dir, "data_drift_report.html")
    json_path = os.path.join(reports_dir, "data_drift_summary.json")

    # --------------------------------------------------------------------------
    # PHASE 1: Baseline vs Baseline -> 0% DRIFT (GREEN)
    # --------------------------------------------------------------------------
    print("📍 PHASE 1: Testing Baseline (Nov 7-8) vs Baseline (Nov 7-8)...")
    print(f"   Reference: {len(ref_data):,} rows | Current: {len(normal_data):,} rows")

    report_p1 = Report(metrics=[DataDriftPreset()])
    report_p1.run(reference_data=ref_data, current_data=normal_data)
    report_p1.save_html(html_path)
    report_p1.save_json(json_path)

    dict_p1 = report_p1.as_dict()
    metrics_p1 = dict_p1.get("metrics", [])
    d_share_p1 = 0.0
    d_cnt_p1 = 0
    is_drifted_p1 = False
    scores_p1 = {}

    for m in metrics_p1:
        res = m.get("result", {})
        if m.get("metric") == "DatasetDriftMetric":
            is_drifted_p1 = bool(res.get("dataset_drift", False))
            d_share_p1 = float(res.get("share_of_drifted_columns", 0.0))
            d_cnt_p1 = int(res.get("number_of_drifted_columns", 0))

        drift_by_cols = res.get("drift_by_columns") or res.get("drift_by_column") or res.get("columns") or {}
        if drift_by_cols:
            for col_name, col_data in drift_by_cols.items():
                if isinstance(col_data, dict):
                    scores_p1[col_name] = float(col_data.get("drift_score", col_data.get("p_value", 0.0)))

    payload_p1 = {
        "input_dataset_drift": is_drifted_p1,
        "input_drift_share": d_share_p1,
        "input_drifted_features_count": d_cnt_p1,
        "output_prediction_drift": False,
        "output_prediction_drift_score": 0.0,
        "feature_drift_scores": scores_p1,
    }
    requests.post(api_drift_url, json=payload_p1, timeout=5)

    status_p1 = "🔴 DRIFT DETECTED" if is_drifted_p1 else "🟢 NORMAL"
    print(f"   => PHASE 1 RESULT: Status={status_p1} | Drift Share={d_share_p1 * 100:.1f}% | Drifted Features={d_cnt_p1}/{len(eval_cols)}")
    print("   ⏳ Pausing 10s for Prometheus scraping & Grafana visualization (Watch Grafana turn GREEN)...")
    time.sleep(10)

    # --------------------------------------------------------------------------
    # PHASE 2: Baseline vs Drifted Stream -> HIGH DRIFT (RED)
    # --------------------------------------------------------------------------
    print("\n📍 PHASE 2: Testing Baseline (Nov 7-8) vs Drifted Stream (Nov 6)...")
    print(f"   Reference: {len(ref_data):,} rows | Current: {len(drift_data):,} rows")

    report_p2 = Report(metrics=[DataDriftPreset()])
    report_p2.run(reference_data=ref_data, current_data=drift_data)
    report_p2.save_html(html_path)
    report_p2.save_json(json_path)

    dict_p2 = report_p2.as_dict()
    metrics_p2 = dict_p2.get("metrics", [])
    d_share_p2 = 0.0
    d_cnt_p2 = 0
    is_drifted_p2 = False
    scores_p2 = {}

    for m in metrics_p2:
        res = m.get("result", {})
        if m.get("metric") == "DatasetDriftMetric":
            is_drifted_p2 = bool(res.get("dataset_drift", False))
            d_share_p2 = float(res.get("share_of_drifted_columns", 0.0))
            d_cnt_p2 = int(res.get("number_of_drifted_columns", 0))

        drift_by_cols = res.get("drift_by_columns") or res.get("drift_by_column") or res.get("columns") or {}
        if drift_by_cols:
            for col_name, col_data in drift_by_cols.items():
                if isinstance(col_data, dict):
                    scores_p2[col_name] = float(col_data.get("drift_score", col_data.get("p_value", 0.0)))

    payload_p2 = {
        "input_dataset_drift": is_drifted_p2,
        "input_drift_share": d_share_p2,
        "input_drifted_features_count": d_cnt_p2,
        "output_prediction_drift": False,
        "output_prediction_drift_score": 0.0,
        "feature_drift_scores": scores_p2,
    }
    requests.post(api_drift_url, json=payload_p2, timeout=5)

    status_p2 = "🔴 DRIFT DETECTED" if is_drifted_p2 else "🟢 NORMAL"
    print(f"   => PHASE 2 RESULT: Status={status_p2} | Drift Share={d_share_p2 * 100:.1f}% | Drifted Features={d_cnt_p2}/{len(eval_cols)}")

    print("\n✅ Controlled Drift Experiment Complete!")
    print("👉 Check Grafana (http://localhost:3000) to see Phase 1 (0% Green) transition directly into Phase 2 (High Drift Red)!")


if __name__ == "__main__":
    run_controlled_drift_experiment()
