import os
import json
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Modern Evidently 0.7+ for Workspace UI
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace, RemoteWorkspace

# Legacy Evidently for standalone visual HTML and JSON reports
from evidently.legacy.report import Report as LegacyReport
from evidently.legacy.metric_preset import DataDriftPreset as LegacyDataDriftPreset


def evaluate_drift():
    ref_path = "data/processed/train.parquet"
    prod_path = "data/simulated_production/production_drift_stream.csv"
    reports_dir = "docs/reports"
    workspace_dir = "workspace"

    print("1. Loading reference training dataset and production drift stream...")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference dataset not found at {ref_path}. Run 'dvc repro' first.")
    if not os.path.exists(prod_path):
        raise FileNotFoundError(f"Production stream dataset not found at {prod_path}. Run 'dvc repro' first.")

    ref_df = pd.read_parquet(ref_path)
    prod_df = pd.read_csv(prod_path)

    # Select behavioral, graph-derived and target features (excluding high-cardinality raw IDs like ip)
    eval_cols = ["hour", "day", "ip_click_count", "ip_unique_apps", "app_freq", "channel_freq", "device_freq"]
    if "is_attributed" in ref_df.columns and "is_attributed" in prod_df.columns:
        eval_cols.append("is_attributed")

    ref_data = ref_df[eval_cols]
    prod_data = prod_df[eval_cols]

    print(f"Reference dataset (train.parquet): {len(ref_data):,} rows")
    print(f"Current production dataset (production_drift_stream.csv): {len(prod_data):,} rows")

    # --------------------------------------------------------------------------
    # 1. Modern Evidently 0.7+ Report for Workspace UI Sync
    # --------------------------------------------------------------------------
    print("2. Computing Evidently AI Data Drift metrics for Workspace UI...")
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    snapshot = drift_report.run(reference_data=ref_data, current_data=prod_data)

    try:
        local_ws = Workspace.create(workspace_dir)
        project_name = "Ads Safety Monitoring"
        projects = local_ws.search_project(project_name)
        if projects:
            project = projects[0]
        else:
            project = local_ws.create_project(project_name)
            project.description = "Ads Safety Real-time Botnet & Graph Drift Telemetry"
        project.save()

        # Workspace.add_run expects a Snapshot object in evidently 0.7.x
        local_ws.add_run(project.id, snapshot)
        print("✅ Successfully logged drift report to local workspace (Evidently UI will auto-sync)!")
    except Exception as e:
        print(f"Notice: Workspace save failed: {e}")

    # --------------------------------------------------------------------------
    # 2. Legacy Evidently Report for Standalone HTML and JSON Reports
    # --------------------------------------------------------------------------
    print("3. Generating full standalone interactive HTML & JSON reports for DVC...")
    os.makedirs(reports_dir, exist_ok=True)
    html_path = os.path.join(reports_dir, "data_drift_report.html")
    json_path = os.path.join(reports_dir, "data_drift_summary.json")

    try:
        legacy_report = LegacyReport(metrics=[LegacyDataDriftPreset()])
        legacy_report.run(reference_data=ref_data, current_data=prod_data)
        legacy_report.save_html(html_path)
        legacy_report.save_json(json_path)
        print(f"✅ Generated full visual interactive HTML report at {html_path}")
        print(f"✅ Generated metrics summary JSON at {json_path}")

        # ----------------------------------------------------------------------
        # 3. Push Live Telemetry to FastAPI / Prometheus / Grafana
        # ----------------------------------------------------------------------
        dict_res = legacy_report.as_dict()
        metrics_list = dict_res.get("metrics", [])
        
        dataset_drift = False
        drift_share = 0.0
        drifted_cnt = 0
        feature_drift_scores = {}

        for m in metrics_list:
            res = m.get("result", {})
            metric_type = m.get("metric", "")
            
            if metric_type == "DatasetDriftMetric":
                dataset_drift = bool(res.get("dataset_drift", False))
                drift_share = float(res.get("share_of_drifted_columns", 0.0))
                drifted_cnt = int(res.get("number_of_drifted_columns", 0))

            drift_by_cols = res.get("drift_by_columns") or res.get("drift_by_column") or res.get("columns") or {}
            if drift_by_cols:
                for col_name, col_data in drift_by_cols.items():
                    if isinstance(col_data, dict):
                        score = col_data.get("drift_score", col_data.get("p_value", 0.0))
                        feature_drift_scores[col_name] = float(score)

        telemetry_payload = {
            "input_dataset_drift": dataset_drift,
            "input_drift_share": drift_share,
            "input_drifted_features_count": drifted_cnt,
            "output_prediction_drift": False,
            "output_prediction_drift_score": 0.0,
            "feature_drift_scores": feature_drift_scores,
        }

        api_url = "http://localhost:8000/metrics/drift"
        resp = requests.post(api_url, json=telemetry_payload, timeout=5)
        if resp.status_code == 200:
            print(f"✅ Ingested {len(feature_drift_scores)} per-feature drift scores into Prometheus/Grafana! ({drifted_cnt}/{len(eval_cols)} drifted)")
    except Exception as e:
        print(f"Notice on telemetry ingestion: {e}")

    print(f"\n🎉 Drift Reports Complete:\n - Live Workspace UI: http://localhost:8085\n - HTML Report:       {html_path}\n - JSON Report:       {json_path}")


if __name__ == "__main__":
    evaluate_drift()
