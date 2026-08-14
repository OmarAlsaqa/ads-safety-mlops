import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace, RemoteWorkspace

def evaluate_drift():
    ref_path = "data/processed/train.parquet"
    prod_path = "data/simulated_production/production_drift_stream.csv"
    reports_dir = "docs/reports"
    workspace_dir = "workspace"
    evidently_service_url = "http://localhost:8085"

    print("1. Loading reference training dataset and production drift stream...")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference dataset not found at {ref_path}. Run 'dvc repro' first.")
    if not os.path.exists(prod_path):
        raise FileNotFoundError(f"Production stream dataset not found at {prod_path}. Run 'dvc repro' first.")

    ref_df = pd.read_parquet(ref_path)
    prod_df = pd.read_csv(prod_path)

    # Filter feature and target columns (excluding metadata)
    drop_cols = ["Timestamp", "dt", "date_str"]
    eval_cols = [col for col in ref_df.columns if col not in drop_cols]

    ref_data = ref_df[eval_cols]
    prod_data = prod_df[eval_cols]

    print(f"Reference dataset (train.parquet): {len(ref_data):,} rows")
    print(f"Current production dataset (production_drift_stream.csv): {len(prod_data):,} rows")

    print("2. Computing Evidently AI Data Drift metrics...")
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    snapshot = drift_report.run(reference_data=ref_data, current_data=prod_data)

    # Save static JSON and HTML reports for Prometheus Exporter & Local Artifacts
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "data_drift_summary.json")

    # In Evidently 0.7+, we rely on the Workspace snapshot for UI rather than save_html
    # You can extract JSON directly if needed via snapshot.dict()
    print("Skipping local HTML/JSON generation (using UI Workspace snapshot instead)")

    # B. Local Workspace for Evidently UI Container Service
    try:
        # We use the local Workspace instead of RemoteWorkspace to bypass HTTP timeout/serialization bugs.
        # The Docker container mounts this local directory so it will instantly pick up the new snapshot.
        workspace_dir = "workspace"
        local_ws = Workspace.create(workspace_dir)
        project_name = "IAQ Safety Monitoring"
        projects = local_ws.search_project(project_name)
        if projects:
            project = projects[0]
        else:
            project = local_ws.create_project(project_name)
            project.description = "Indoor Air Quality MLOps Data Drift & Concept Drift Telemetry"
            project.save()

        # Workspace.add_run expects a Snapshot object in evidently 0.7.x
        local_ws.add_run(project.id, snapshot)
        
        print("✅ Successfully logged drift report to local workspace (Evidently UI will auto-sync)!")
    except Exception as e:
        print(f"Notice: Workspace save failed: {e}")

if __name__ == "__main__":
    evaluate_drift()
