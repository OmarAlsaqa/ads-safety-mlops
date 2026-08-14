import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient

# Ensure src and project root are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from src.models.graph_nc import GraphNC
except ImportError:
    from models.graph_nc import GraphNC

# Local S3 and MLflow environment configuration
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:4566")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

MODEL_NAME = "GraphNC-AdFraud-Detector"


def apply_edge_dropping_attack(edge_index: torch.Tensor, drop_ratio: float) -> torch.Tensor:
    """Simulates structural camouflage attack where botnets rotate infrastructure to hide co-occurrence edges."""
    if drop_ratio <= 0.0:
        return edge_index

    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges) >= drop_ratio
    # Ensure at least self-loops or some edges remain
    if keep_mask.sum() == 0:
        return edge_index[:, :10]
    return edge_index[:, keep_mask]


def apply_feature_jitter_attack(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Simulates adversarial timing & device jittering to evade static rule-based filters."""
    if noise_std <= 0.0:
        return x
    noise = torch.randn_like(x) * noise_std
    return x + noise


def evaluate_adversarial_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🛡️ Running GraphNC Adversarial Evasion Benchmarks on: {device}")

    # 1. MLflow Experiment Setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = "ads-safety-graphnc"

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"   -> Connected to MLflow at: {mlflow_tracking_uri}")
    except Exception as e:
        print(f"   -> MLflow Notice ({e}). Logging locally to ./mlruns")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

    # 2. Load Processed Graph and Champion Model Checkpoint
    graph_path = "data/processed/ad_click_graph.pt"
    model_path = "models/graph_nc.pt"

    if not os.path.exists(graph_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Missing graph tensor or model checkpoint. Run 'dvc repro' first.")

    data = torch.load(graph_path, weights_only=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    hyperparams = checkpoint.get("hyperparams", {})
    threshold = checkpoint.get("threshold", 0.6566)

    model = GraphNC(
        in_dim=hyperparams.get("in_dim", 7),
        hidden_dim=hyperparams.get("hidden_dim", 64),
        readout=hyperparams.get("readout", "avg"),
        beta=hyperparams.get("beta", 0.5),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_y = data.y[data.test_mask].cpu().numpy()
    pos_ratio = float((data.y == 1).float().mean().item())

    # 3. Clean Baseline Evaluation (Day 4 Test Set)
    with torch.no_grad():
        clean_probs = model(data.x, data.edge_index)[data.test_mask].cpu().numpy()
        clean_auc = float(roc_auc_score(test_y, clean_probs))
        clean_pr_auc = float(average_precision_score(test_y, clean_probs))
        clean_preds = (clean_probs >= threshold).astype(int)
        clean_f1 = float(f1_score(test_y, clean_preds, zero_division=0))

    print(f"\n📊 Baseline (Clean Traffic - 0% Perturbation):\n"
          f"   - Clean AUC-ROC: {clean_auc:.4f} | Clean PR-AUC: {clean_pr_auc:.4f} | F1: {clean_f1:.4f}")

    benchmark_results = {
        "baseline": {
            "auc_roc": clean_auc,
            "pr_auc": clean_pr_auc,
            "f1_score": clean_f1,
            "threshold": threshold,
        },
        "structural_camouflage_attack": [],
        "feature_jitter_attack": [],
    }

    # ==============================================================================
    # Stress Test 1: Structural Camouflage Attack (Edge Dropping 10% to 80%)
    # ==============================================================================
    print("\n⚔️ Stress Test 1: Structural Camouflage Attack (Dropping Co-occurrence Edges)...")
    edge_drop_levels = [0.10, 0.25, 0.50, 0.75, 0.90]

    for drop_rate in edge_drop_levels:
        perturbed_edges = apply_edge_dropping_attack(data.edge_index, drop_rate)
        with torch.no_grad():
            pert_probs = model(data.x, perturbed_edges)[data.test_mask].cpu().numpy()
            pert_auc = float(roc_auc_score(test_y, pert_probs))
            pert_pr_auc = float(average_precision_score(test_y, pert_probs))
            pert_preds = (pert_probs >= threshold).astype(int)
            pert_f1 = float(f1_score(test_y, pert_preds, zero_division=0))

        degradation = clean_auc - pert_auc
        print(f"   [Drop {int(drop_rate*100):02d}% Edges] AUC-ROC: {pert_auc:.4f} (Δ -{degradation:.4f}) | PR-AUC: {pert_pr_auc:.4f} | F1: {pert_f1:.4f}")

        benchmark_results["structural_camouflage_attack"].append({
            "edge_drop_ratio": drop_rate,
            "auc_roc": pert_auc,
            "pr_auc": pert_pr_auc,
            "f1_score": pert_f1,
            "auc_degradation": degradation,
        })

    # ==============================================================================
    # Stress Test 2: Feature Jittering Attack (Gaussian Noise Injection)
    # ==============================================================================
    print("\n⚔️ Stress Test 2: Feature Jittering Attack (Injecting Adversarial Parameter Noise)...")
    noise_levels = [0.10, 0.25, 0.50, 1.00, 2.00]

    for noise_std in noise_levels:
        perturbed_x = apply_feature_jitter_attack(data.x, noise_std)
        with torch.no_grad():
            pert_probs = model(perturbed_x, data.edge_index)[data.test_mask].cpu().numpy()
            pert_auc = float(roc_auc_score(test_y, pert_probs))
            pert_pr_auc = float(average_precision_score(test_y, pert_probs))
            pert_preds = (pert_probs >= threshold).astype(int)
            pert_f1 = float(f1_score(test_y, pert_preds, zero_division=0))

        degradation = clean_auc - pert_auc
        print(f"   [Noise σ = {noise_std:.2f}] AUC-ROC: {pert_auc:.4f} (Δ -{degradation:.4f}) | PR-AUC: {pert_pr_auc:.4f} | F1: {pert_f1:.4f}")

        benchmark_results["feature_jitter_attack"].append({
            "noise_std": noise_std,
            "auc_roc": pert_auc,
            "pr_auc": pert_pr_auc,
            "f1_score": pert_f1,
            "auc_degradation": degradation,
        })

    # ==============================================================================
    # 4. Generate Publication-Quality Robustness Degradation Charts
    # ==============================================================================
    os.makedirs("docs/reports", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Chart 1: Structural Camouflage Degradation
    drop_pcts = [item["edge_drop_ratio"] * 100 for item in benchmark_results["structural_camouflage_attack"]]
    struct_aucs = [item["auc_roc"] for item in benchmark_results["structural_camouflage_attack"]]

    ax1.plot(drop_pcts, struct_aucs, marker="o", color="#c0392b", lw=2.5, label="GraphNC (ICML 2026)")
    ax1.axhline(clean_auc, color="gray", linestyle="--", alpha=0.7, label=f"Clean Baseline ({clean_auc:.4f})")
    ax1.axhline(0.50, color="black", linestyle=":", alpha=0.5, label="Random Guessing (0.50)")
    ax1.set_title("Structural Camouflage Robustness (Edge Dropping)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("% Co-occurrence Edges Deleted by Botnet")
    ax1.set_ylabel("Test Set AUC-ROC")
    ax1.set_ylim(0.45, 1.0)
    ax1.legend(loc="lower left")

    # Chart 2: Feature Jittering Degradation
    noises = [item["noise_std"] for item in benchmark_results["feature_jitter_attack"]]
    feature_aucs = [item["auc_roc"] for item in benchmark_results["feature_jitter_attack"]]

    ax2.plot(noises, feature_aucs, marker="s", color="#2980b9", lw=2.5, label="GraphNC (ICML 2026)")
    ax2.axhline(clean_auc, color="gray", linestyle="--", alpha=0.7, label=f"Clean Baseline ({clean_auc:.4f})")
    ax2.axhline(0.50, color="black", linestyle=":", alpha=0.5, label="Random Guessing (0.50)")
    ax2.set_title("Feature Jittering Robustness (Adversarial Noise Injection)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Noise Magnitude (σ)")
    ax2.set_ylabel("Test Set AUC-ROC")
    ax2.set_ylim(0.45, 1.0)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    chart_path = "docs/reports/adversarial_robustness_curve.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5. Export JSON Report
    report_path = "docs/reports/adversarial_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # 6. Log Everything to MLflow
    print("\n📝 Logging Adversarial Robustness Benchmarks to MLflow...")
    with mlflow.start_run(run_name="graphnc_adversarial_benchmarks") as run:
        mlflow.log_param("evaluated_model", MODEL_NAME)
        mlflow.log_param("evaluation_dataset", "TalkingData Ad-Click Stream (Day 4)")
        mlflow.log_metric("baseline_clean_auc_roc", clean_auc)
        mlflow.log_metric("baseline_clean_pr_auc", clean_pr_auc)

        # Log stress test metrics
        for item in benchmark_results["structural_camouflage_attack"]:
            k = int(item["edge_drop_ratio"] * 100)
            mlflow.log_metric(f"edge_drop_{k}pct_auc_roc", item["auc_roc"])

        for item in benchmark_results["feature_jitter_attack"]:
            k = str(item["noise_std"]).replace(".", "_")
            mlflow.log_metric(f"noise_sigma_{k}_auc_roc", item["auc_roc"])

        mlflow.log_artifact(chart_path, artifact_path="adversarial_reports")
        mlflow.log_artifact(report_path, artifact_path="adversarial_reports")

    print(f"\n✅ Adversarial Evaluation Complete!\n"
          f" - Chart Saved:   {chart_path}\n"
          f" - Report Saved:  {report_path}\n"
          f" - MLflow Artifacts Uploaded to http://localhost:5000")


if __name__ == "__main__":
    evaluate_adversarial_robustness()
