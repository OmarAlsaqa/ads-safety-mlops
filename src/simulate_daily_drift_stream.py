import os
import time
import pandas as pd
import requests
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.ui.dashboards.reports import DashboardPanelCounter, DashboardPanelPlot, PanelValue, PlotType
from evidently.ui.workspace import Workspace, RemoteWorkspace

def run_simulation():
    raw_path = "data/raw/iaq_classification.csv"
    ref_path = "data/processed/train.parquet"
    reports_dir = "docs/reports"
    workspace_dir = "workspace"
    evidently_service_url = "http://localhost:8085"
    api_metrics_url = "http://localhost:8000/metrics"

    print("🚀 Starting 9-Day Data Drift Telemetry Simulation Stream...\n")
    if not os.path.exists(raw_path) or not os.path.exists(ref_path):
        raise FileNotFoundError("Data files not found. Run 'dvc repro' first.")

    raw_df = pd.read_csv(raw_path)
    raw_df["dt"] = pd.to_datetime(raw_df["Timestamp"])
    raw_df["date_str"] = raw_df["dt"].dt.strftime("%Y-%m-%d")

    unique_days = sorted(raw_df["date_str"].unique())
    print(f"Total Unique Days in Dataset ({len(unique_days)} days): {unique_days}\n")

    ref_df = pd.read_parquet(ref_path)
    drop_cols = ["Timestamp", "dt", "date_str"]
    eval_cols = [col for col in ref_df.columns if col not in drop_cols and col in raw_df.columns]
    
    ref_data = ref_df[eval_cols]

    # Initialize Remote Evidently UI Workspace
    try:
        remote_ws = RemoteWorkspace(evidently_service_url)
        project_name = "IAQ Safety Monitoring"
        remote_projects = remote_ws.search_project(project_name)
        remote_project = remote_projects[0] if remote_projects else remote_ws.create_project(project_name)
    except Exception as e:
        print(f"Notice: Could not connect to Remote Workspace: {e}")
        remote_ws = None

    os.makedirs(reports_dir, exist_ok=True)

    for step, day in enumerate(unique_days, start=1):
        day_df = raw_df[raw_df["date_str"] == day]
        day_data = day_df[eval_cols]
        sample_count = len(day_data)

        print(f"----------------------------------------------------------------------")
        print(f"📅 [Step {step}/{len(unique_days)}] Processing Date: {day} ({sample_count:,} sensor samples)")

        # Run Evidently Report
        drift_report = Report(metrics=[
            DataDriftPreset(stattest_threshold=0.25),
            TargetDriftPreset()
        ])
        drift_report.run(reference_data=ref_data, current_data=day_data)

        # 1. Save static JSON summary for Prometheus Exporter
        json_path = os.path.join(reports_dir, "data_drift_summary.json")
        drift_report.save_json(json_path)

        # 2. Push to Evidently Interactive UI Container
        if remote_ws and remote_project:
            try:
                remote_ws.add_run(remote_project.id, drift_report._get_snapshot())
            except Exception as ex:
                pass

        # 3. Trigger FastAPI /metrics endpoint to update Prometheus Gauges
        try:
            resp = requests.get(api_metrics_url, timeout=3)
            # Parse drift metric result for clean terminal printing
            dict_res = drift_report.as_dict()
            metrics_list = dict_res.get("metrics", [])
            drift_share = 0.0
            is_drifted = False
            drifted_cnt = 0

            for m in metrics_list:
                if m.get("metric") == "DatasetDriftMetric":
                    res = m.get("result", {})
                    drift_share = res.get("share_of_drifted_columns", 0.0) * 100
                    is_drifted = res.get("dataset_drift", False)
                    drifted_cnt = res.get("number_of_drifted_columns", 0)

            status_str = "🔴 DRIFT DETECTED" if is_drifted else "🟢 NORMAL"
            print(f"   Telemetry Metric: Status={status_str} | Drift Share={drift_share:.1f}% | Drifted Features={drifted_cnt}/{len(eval_cols)}")
        except Exception as err:
            print(f"   Notice: API metrics trigger: {err}")

        # Sleep 3s so Prometheus scrapes each timestamped day
        print("   Pausing 3s for Prometheus scraping...")
        time.sleep(3)

    print("\n✅ Simulation Complete! All 9 Days Processed & Telemetry Streamed.")
    print("👉 Open Grafana Dashboard (http://localhost/grafana/) to view the dynamic time-series curve!")

if __name__ == "__main__":
    run_simulation()
