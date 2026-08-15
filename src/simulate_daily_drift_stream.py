import os
import sys
import time
import json
import warnings
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Modern Evidently for Workspace UI
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace

# Legacy Evidently for full standalone reports
from evidently.legacy.report import Report as LegacyReport
from evidently.legacy.metric_preset import DataDriftPreset as LegacyDataDriftPreset


def run_daily_drift_simulation():
    raw_path = "data/raw/train_sample.csv"
    ref_path = "data/processed/train.parquet"
    reports_dir = "docs/reports"
    workspace_dir = "workspace"
    api_drift_url = "http://localhost:8000/metrics/drift"
    api_predict_url = "http://localhost:8000/predict/ad-click"

    print("🚀 Starting Chronological Multi-Day Data Drift Simulation Stream...\n")
    if not os.path.exists(raw_path) or not os.path.exists(ref_path):
        raise FileNotFoundError("Required data files not found. Run 'dvc repro' first.")

    # 1. Load Reference Training Data
    ref_df = pd.read_parquet(ref_path)
    eval_cols = [
        "hour", "day", "ip_click_count", "ip_unique_apps", "app_freq", "channel_freq", "device_freq",
        "next_click_delta", "prev_click_delta", "ip_device_os_cumcount", "ip_app_cumcount",
        "ip_hh_app_count", "ip_hh_device_count", "app_channel_count", "ip_unique_channels"
    ]
    if "is_attributed" in ref_df.columns:
        eval_cols.append("is_attributed")
    ref_data = ref_df[eval_cols]

    # 2. Load and Preprocess Chronological Stream Data
    raw_df = pd.read_csv(raw_path)
    raw_df["click_time"] = pd.to_datetime(raw_df["click_time"])
    raw_df["dt"] = raw_df["click_time"]
    raw_df["date_str"] = raw_df["click_time"].dt.strftime("%Y-%m-%d")
    raw_df["hour"] = raw_df["click_time"].dt.hour
    raw_df["day"] = raw_df["click_time"].dt.day
    raw_df = raw_df.sort_values("dt").reset_index(drop=True)

    # Compute rolling behavioral feature counts
    raw_df["ip_click_count"] = raw_df.groupby("ip")["app"].transform("count").astype("float32")
    raw_df["ip_unique_apps"] = raw_df.groupby("ip")["app"].transform("nunique").astype("float32")
    raw_df["app_freq"] = raw_df.groupby("app")["ip"].transform("count").astype("float32")
    raw_df["channel_freq"] = raw_df.groupby("channel")["ip"].transform("count").astype("float32")
    raw_df["device_freq"] = raw_df.groupby("device")["ip"].transform("count").astype("float32")

    # Advanced Temporal Deltas & Sequences
    next_sec = raw_df.groupby(["ip", "app", "device", "os"])["dt"].shift(-1)
    next_delta = (pd.to_datetime(next_sec) - raw_df["dt"]).dt.total_seconds().fillna(3600.0)
    raw_df["next_click_delta"] = np.log1p(np.clip(next_delta, 0.0, 86400.0)).astype("float32")

    prev_sec = raw_df.groupby(["ip", "channel"])["dt"].shift(1)
    prev_delta = (raw_df["dt"] - pd.to_datetime(prev_sec)).dt.total_seconds().fillna(3600.0)
    raw_df["prev_click_delta"] = np.log1p(np.clip(prev_delta, 0.0, 86400.0)).astype("float32")

    raw_df["ip_device_os_cumcount"] = np.log1p(raw_df.groupby(["ip", "device", "os"]).cumcount()).astype("float32")
    raw_df["ip_app_cumcount"] = np.log1p(raw_df.groupby(["ip", "app"]).cumcount()).astype("float32")

    # High-Order Cross Interactions & Diversity
    raw_df["ip_hh_app_count"] = raw_df.groupby(["ip", "hour", "app"])["channel"].transform("count").astype("float32")
    raw_df["ip_hh_device_count"] = raw_df.groupby(["ip", "hour", "device"])["channel"].transform("count").astype("float32")
    raw_df["app_channel_count"] = raw_df.groupby(["app", "channel"])["ip"].transform("count").astype("float32")
    raw_df["ip_unique_channels"] = raw_df.groupby("ip")["channel"].transform("nunique").astype("float32")

    unique_days = sorted(raw_df["date_str"].unique())
    print(f"Total Chronological Days in Dataset ({len(unique_days)} days): {unique_days}\n")

    # 3. Initialize Local Workspace for Evidently UI Container
    local_ws = None
    project = None
    try:
        local_ws = Workspace.create(workspace_dir)
        project_name = "Ads Safety Monitoring"
        projects = local_ws.search_project(project_name)
        if projects:
            project = projects[0]
        else:
            project = local_ws.create_project(project_name)
            project.description = "Ads Safety Real-time Botnet & Graph Drift Telemetry"

        # Configure visual dashboard panels if not already present
        try:
            from evidently.ui.dashboards import (
                DashboardPanelCounter,
                DashboardPanelPlot,
                PanelValue,
                PlotType,
                ReportFilter,
                CounterAgg,
            )
            if hasattr(project, "dashboard") and len(project.dashboard.panels) == 0:
                project.dashboard.add_panel(
                    DashboardPanelCounter(
                        title="Share of Drifted Features",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        value=PanelValue(
                            metric_id="DatasetDriftMetric",
                            field_path="share_of_drifted_columns",
                            legend="Drift Share",
                        ),
                        text="Drift Share %",
                        agg=CounterAgg.LAST,
                        size=1,
                    )
                )
                project.dashboard.add_panel(
                    DashboardPanelCounter(
                        title="Number of Drifted Features",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        value=PanelValue(
                            metric_id="DatasetDriftMetric",
                            field_path="number_of_drifted_columns",
                            legend="Drifted Columns",
                        ),
                        text="Features",
                        agg=CounterAgg.LAST,
                        size=1,
                    )
                )
                project.dashboard.add_panel(
                    DashboardPanelPlot(
                        title="Data Drift Progression Over Time",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        values=[
                            PanelValue(
                                metric_id="DatasetDriftMetric",
                                field_path="share_of_drifted_columns",
                                legend="Share of Drifted Features",
                            ),
                        ],
                        plot_type=PlotType.LINE,
                        size=2,
                    )
                )
        except Exception as panel_err:
            pass

        project.save()
    except Exception as e:
        print(f"Notice: Workspace connection: {e}")
        print(f"Notice: Workspace connection: {e}")

    os.makedirs(reports_dir, exist_ok=True)

    for step, day in enumerate(unique_days, start=1):
        day_df = raw_df[raw_df["date_str"] == day]
        day_data = day_df[eval_cols]
        sample_count = len(day_data)

        print(f"----------------------------------------------------------------------")
        print(f"📅 [Step {step}/{len(unique_days)}] Processing Date: {day} ({sample_count:,} ad click events)")

        # A. Modern Evidently Snapshot for Web UI
        try:
            drift_report = Report(metrics=[DataDriftPreset()])
            snapshot = drift_report.run(reference_data=ref_data, current_data=day_data)
            if local_ws and project:
                local_ws.add_run(project.id, snapshot)
        except Exception as e:
            print(f"Notice: UI snapshot save: {e}")

        # B. Legacy Report for Metrics Parsing & Standalone HTML
        legacy_report = LegacyReport(metrics=[LegacyDataDriftPreset()])
        legacy_report.run(reference_data=ref_data, current_data=day_data)

        html_path = os.path.join(reports_dir, "data_drift_report.html")
        json_path = os.path.join(reports_dir, "data_drift_summary.json")
        legacy_report.save_html(html_path)
        legacy_report.save_json(json_path)

        # C. Extract Dynamic Metrics & Post Telemetry to FastAPI / Prometheus
        dict_res = legacy_report.as_dict()
        metrics_list = dict_res.get("metrics", [])
        dataset_drift = False
        drift_share = 0.0
        drifted_cnt = 0
        feature_drift_scores = {}

        for m in metrics_list:
            res = m.get("result", {})
            if m.get("metric") == "DatasetDriftMetric":
                dataset_drift = bool(res.get("dataset_drift", False))
                drift_share = float(res.get("share_of_drifted_columns", 0.0))
                drifted_cnt = int(res.get("number_of_drifted_columns", 0))

            drift_by_cols = res.get("drift_by_columns") or res.get("drift_by_column") or res.get("columns") or {}
            if drift_by_cols:
                for col_name, col_data in drift_by_cols.items():
                    if isinstance(col_data, dict):
                        score = col_data.get("drift_score", col_data.get("p_value", 0.0))
                        feature_drift_scores[col_name] = float(score)

        # Send live inference predictions for a sample batch to update latency & prediction gauges
        sample_batch = day_df.sample(n=min(20, len(day_df)), random_state=42)
        for _, row in sample_batch.iterrows():
            payload = {
                "ip": int(row["ip"]),
                "app": int(row["app"]),
                "device": int(row["device"]),
                "os": int(row["os"]),
                "channel": int(row["channel"]),
                "click_time": datetime.utcnow().isoformat(),
            }
            try:
                requests.post(api_predict_url, json=payload, timeout=2)
            except Exception:
                pass

        # Push Drift Telemetry to FastAPI -> Prometheus
        telemetry_payload = {
            "input_dataset_drift": dataset_drift,
            "input_drift_share": drift_share,
            "input_drifted_features_count": drifted_cnt,
            "output_prediction_drift": False,
            "output_prediction_drift_score": 0.0,
            "feature_drift_scores": feature_drift_scores,
        }

        try:
            resp = requests.post(api_drift_url, json=telemetry_payload, timeout=5)
            status_str = "🔴 DRIFT DETECTED" if dataset_drift else "🟢 NORMAL"
            print(f"   Telemetry Metric: Status={status_str} | Drift Share={drift_share * 100:.1f}% | Drifted Features={drifted_cnt}/{len(eval_cols)}")
            print(f"   Ingested {len(feature_drift_scores)} Per-Feature Drift Scores into Prometheus/Grafana.")
        except Exception as err:
            print(f"   Notice: API metrics post: {err}")

        # Sleep 3s so Prometheus scrapes each timestamped day
        print("   ⏳ Pausing 3s for Prometheus scraping...")
        time.sleep(3)

    print("\n✅ Simulation Complete! All Chronological Days Processed & Telemetry Streamed.")
    print("👉 Open Grafana Dashboard (http://localhost:3000) and Evidently UI (http://localhost:8085) to view the dynamic time-series curves!")


if __name__ == "__main__":
    run_daily_drift_simulation()
