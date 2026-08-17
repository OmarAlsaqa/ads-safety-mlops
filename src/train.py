import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import sys
from pathlib import Path
import mlflow
import mlflow.pytorch
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:4566")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

try:
    from torch_geometric.loader import NeighborLoader
except ImportError:
    from torch_geometric.data import NeighborLoader


def generate_evaluation_plots(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, output_dir: str = "models"):
    """Generates and saves publication-quality evaluation figures for MLflow."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, color="#2980b9", lw=2, label=f"GraphNC (AUC = {auc_score:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
    ax_roc.set_title("ROC Curve - Ad-Fraud Detection", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    fig_roc.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(rec, prec, color="#27ae60", lw=2, label=f"GraphNC (PR-AUC = {pr_auc:.4f})")
    ax_pr.set_title("Precision-Recall Curve (Severe Imbalance)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")
    pr_path = os.path.join(output_dir, "pr_curve.png")
    fig_pr.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig_pr)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig_cm)

    return roc_path, pr_path, cm_path


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training GraphNC Model on Device: {device} (Mini-Batch Neighbor Sampling & AMP Enabled)")

    # 1. MLflow Experiment Setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = "ads-safety-graphnc"

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"   -> Connected to MLflow at: {mlflow_tracking_uri}")
    except Exception as e:
        print(f"   -> Notice: MLflow server offline ({e}). Logging locally to sqlite:///mlflow.db")
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

    # 2. Load Processed Graph Data
    graph_path = "data/processed/ad_click_graph.pt"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Processed graph not found at {graph_path}. Run 'dvc repro' first.")

    print(f"1. Loading Graph Dataset from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    data.x = torch.nan_to_num(data.x.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    if hasattr(data, "x_cat") and data.x_cat is not None:
        data.x_cat = torch.nan_to_num(data.x_cat.contiguous(), nan=0)
    data.edge_index = data.edge_index.contiguous()
    data.y = torch.nan_to_num(data.y.contiguous(), nan=0)
    data.train_mask = data.train_mask.contiguous()
    data.val_mask = data.val_mask.contiguous()
    data.test_mask = data.test_mask.contiguous()

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    in_dim = data.num_features
    pos_ratio = (data.y == 1).float().mean().item()

    print(f"   -> Nodes: {num_nodes:,} | Edges: {num_edges:,} | Features: {in_dim}")
    print(f"   -> Class Imbalance: Positive Fraud Rate = {pos_ratio * 100:.3f}%")

    # 3. Setup Adaptive Execution Strategy (Optimized Mini-Batch for bounded VRAM)
    use_minibatch = num_nodes > 500000
    if use_minibatch:
        batch_size = 8192
        train_loader = NeighborLoader(
            data,
            num_neighbors=[8, 4],
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=0,
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors=[8, 4],
            batch_size=batch_size,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=0,
        )
        test_loader = NeighborLoader(
            data,
            num_neighbors=[8, 4],
            batch_size=batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_loader = None
        val_loader = None
        test_loader = None
        # Fast full-graph GPU execution (<1.5GB VRAM for 1M nodes)
        data = data.to(device)

    # 4. Hyperparameters
    hyperparams = {
        "model_architecture": "GraphNC-GATv2 (ICML 2026)",
        "in_dim": in_dim,
        "hidden_dim": 128,
        "readout": "avg",
        "lr": 0.001,
        "weight_decay": 5e-5,
        "epochs": 35,
        "early_stopping_patience": 10,
        "beta": 0.5,
        "score_da_weight": 0.05,
        "distill_weight": 0.05,
        "norm_reg_weight": 0.01,
        "noise_var": 0.01,
        "focal_gamma": 2.0,
        "focal_alpha": 0.75,
        "use_minibatch": use_minibatch,
        "batch_nodes": num_nodes,
        "total_edges": num_edges,
    }

    # 5. Instantiate Model, Optimizer & Scaler
    model = GraphNC(
        in_dim=hyperparams["in_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        readout=hyperparams["readout"],
        beta=hyperparams["beta"],
        score_da_weight=hyperparams["score_da_weight"],
        distill_weight=hyperparams["distill_weight"],
        norm_reg_weight=hyperparams["norm_reg_weight"],
        noise_var=hyperparams["noise_var"],
        focal_gamma=hyperparams["focal_gamma"],
        focal_alpha=hyperparams["focal_alpha"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams["epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    model_name = "GraphNC-AdFraud-Detector"
    os.makedirs("models", exist_ok=True)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n2. Starting MLflow Run & GraphNC Training (Strategy: {'Mini-Batch' if use_minibatch else 'Fast Full-Graph GPU'})...")
    with mlflow.start_run(run_name="graphnc_production_training") as run:
        mlflow.log_params(hyperparams)

        for epoch in range(1, hyperparams["epochs"] + 1):
            model.train()
            optimizer.zero_grad()

            if use_minibatch:
                total_loss = 0.0
                total_bce = 0.0
                num_batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    target_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
                    target_mask[:batch.batch_size] = True
                    batch_x_cat = getattr(batch, "x_cat", None)
                    
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        loss, loss_dict = model.compute_loss(
                            batch.x,
                            batch.edge_index,
                            batch.y,
                            target_mask,
                            x_cat=batch_x_cat,
                            is_training=True,
                        )

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss_dict["loss_total"]
                    total_bce += loss_dict["loss_bce"]
                    num_batches += 1
                avg_loss = total_loss / max(1, num_batches)
                avg_bce = total_bce / max(1, num_batches)
            else:
                data_x_cat = getattr(data, "x_cat", None)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    loss, loss_dict = model.compute_loss(
                        data.x,
                        data.edge_index,
                        data.y,
                        data.train_mask,
                        x_cat=data_x_cat,
                        is_training=True,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                avg_loss = loss_dict["loss_total"]
                avg_bce = loss_dict["loss_bce"]

            scheduler.step()

            # --- Validation Step ---
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if use_minibatch:
                    val_preds = []
                    val_targets = []
                    for batch in val_loader:
                        batch = batch.to(device)
                        batch_x_cat = getattr(batch, "x_cat", None)
                        probs = model(batch.x, batch.edge_index, x_cat=batch_x_cat)[:batch.batch_size]
                        val_preds.append(probs.cpu().numpy())
                        val_targets.append(batch.y[:batch.batch_size].cpu().numpy())
                    val_probs = np.concatenate(val_preds)
                    val_y = np.concatenate(val_targets)
                else:
                    data_x_cat = getattr(data, "x_cat", None)
                    val_probs = model(data.x, data.edge_index, x_cat=data_x_cat)[data.val_mask].cpu().numpy()
                    val_y = data.y[data.val_mask].cpu().numpy()

                val_probs = np.nan_to_num(val_probs, nan=0.0)
                val_y = np.nan_to_num(val_y, nan=0.0)
                val_auc = float(roc_auc_score(val_y, val_probs))
                val_pr_auc = float(average_precision_score(val_y, val_probs))

            # Composite ranking score: 40% Global AUC-ROC + 60% Sharp PR-AUC
            val_score = 0.40 * val_auc + 0.60 * val_pr_auc

            # Log metrics per epoch
            mlflow.log_metrics({
                "epoch_loss": avg_loss,
                "loss_bce": avg_bce,
                "val_auc_roc": val_auc,
                "val_pr_auc": val_pr_auc,
                "val_composite_score": val_score,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            # Check for best model checkpoint based on Composite Score
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or epoch == best_epoch:
                print(f"   Epoch [{epoch:02d}/{hyperparams['epochs']:02d}] "
                      f"Loss: {avg_loss:.4f} (BCE: {avg_bce:.3f}) | "
                      f"Val AUC-ROC: {val_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Peak Ep: {best_epoch} (Score: {best_val_score:.4f})")

            # Early stopping check
            if patience_counter >= hyperparams["early_stopping_patience"]:
                print(f"   🛑 Early stopping triggered at epoch {epoch} (Best model checkpoint from epoch {best_epoch})")
                break

        # 6. Final Evaluation on Held-Out Test Set (Day 4)
        print(f"\n3. Evaluating Best Champion Model (from Epoch {best_epoch}) on Held-Out Test Set...")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if use_minibatch:
                val_preds = []
                val_targets = []
                for batch in val_loader:
                    batch = batch.to(device)
                    batch_x_cat = getattr(batch, "x_cat", None)
                    probs = model(batch.x, batch.edge_index, x_cat=batch_x_cat)[:batch.batch_size]
                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(batch.y[:batch.batch_size].cpu().numpy())
                val_probs = np.concatenate(val_preds)
                val_y = np.concatenate(val_targets)
            else:
                data_x_cat = getattr(data, "x_cat", None)
                val_probs = model(data.x, data.edge_index, x_cat=data_x_cat)[data.val_mask].cpu().numpy()
                val_y = data.y[data.val_mask].cpu().numpy()

            # Pareto Threshold Tuning: Find threshold maximizing F1 with ≥80% Precision & ≥80% Recall target
            best_thresh = 0.5
            best_val_f1 = 0.0
            pareto_thresh = None  # Threshold meeting both P≥0.80 and R≥0.80
            pareto_f1 = 0.0

            for t in np.linspace(max(val_probs.min() + 1e-4, 0.01), min(val_probs.max() - 1e-4, 0.99), 300):
                preds = (val_probs >= t).astype(int)
                f1 = f1_score(val_y, preds, zero_division=0)
                prec = precision_score(val_y, preds, zero_division=0)
                rec = recall_score(val_y, preds, zero_division=0)

                # Track best overall F1
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_thresh = float(t)

                # Track Pareto-optimal: both P≥0.80 and R≥0.80
                if prec >= 0.80 and rec >= 0.80 and f1 > pareto_f1:
                    pareto_f1 = f1
                    pareto_thresh = float(t)

            # Prefer Pareto threshold if found, otherwise fall back to best F1 threshold
            if pareto_thresh is not None:
                best_thresh = pareto_thresh
                best_val_f1 = pareto_f1
                print(f"   ✅ Pareto threshold found: {best_thresh:.4f} (F1={pareto_f1:.4f}, P≥80%, R≥80%)")
            else:
                print(f"   ⚠️ No Pareto threshold with P≥80% & R≥80% found. Using best F1 threshold: {best_thresh:.4f} (F1={best_val_f1:.4f})")

            # Apply tuned threshold on independent Held-out Test Set
            if use_minibatch:
                test_preds_list = []
                test_targets_list = []
                for batch in test_loader:
                    batch = batch.to(device)
                    batch_x_cat = getattr(batch, "x_cat", None)
                    probs = model(batch.x, batch.edge_index, x_cat=batch_x_cat)[:batch.batch_size]
                    test_preds_list.append(probs.cpu().numpy())
                    test_targets_list.append(batch.y[:batch.batch_size].cpu().numpy())
                test_probs = np.concatenate(test_preds_list)
                test_y = np.concatenate(test_targets_list)
            else:
                data_x_cat = getattr(data, "x_cat", None)
                test_probs = model(data.x, data.edge_index, x_cat=data_x_cat)[data.test_mask].cpu().numpy()
                test_y = data.y[data.test_mask].cpu().numpy()

            test_probs = np.nan_to_num(test_probs, nan=0.0)
            test_y = np.nan_to_num(test_y, nan=0.0)
            test_auc = float(roc_auc_score(test_y, test_probs))
            test_pr_auc = float(average_precision_score(test_y, test_probs))
            test_preds = (test_probs >= best_thresh).astype(int)

            test_f1 = float(f1_score(test_y, test_preds, zero_division=0))
            test_prec = float(precision_score(test_y, test_preds, zero_division=0))
            test_rec = float(recall_score(test_y, test_preds, zero_division=0))

        print(f"   📊 Final Test Performance (Held-out Chronological Day 4):\n"
              f"      - AUC-ROC:         {test_auc:.4f} (Ranking Power)\n"
              f"      - PR-AUC:          {test_pr_auc:.4f} ({test_pr_auc / pos_ratio:.1f}x higher than random baseline)\n"
              f"      - F1-Score:        {test_f1:.4f}\n"
              f"      - Precision:       {test_prec:.4f}\n"
              f"      - Recall:          {test_rec:.4f}\n"
              f"      - Calibrated Thresh: {best_thresh:.4f}")

        # Log final metrics to MLflow
        mlflow.log_metrics({
            "test_auc_roc": test_auc,
            "test_pr_auc": test_pr_auc,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "optimal_threshold": best_thresh,
        })

        # 6. Save Local Model Artifact & Metrics JSON
        local_model_path = "models/graph_nc.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "hyperparams": hyperparams,
            "threshold": best_thresh,
            "metrics": {
                "test_auc_roc": float(test_auc),
                "test_pr_auc": float(test_pr_auc),
                "test_f1": float(test_f1),
                "best_val_auc_roc": float(best_val_score),
            }
        }, local_model_path)
        print(f"\n✅ Local PyG model artifact saved to: {local_model_path}")

        metrics_file = "models/training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "test_auc_roc": float(test_auc),
                "test_pr_auc": float(test_pr_auc),
                "test_f1": float(test_f1),
                "test_precision": float(test_prec),
                "test_recall": float(test_rec),
                "best_val_auc_roc": float(best_val_score),
                "best_epoch": int(best_epoch),
                "threshold": float(best_thresh),
            }, f, indent=2)

        # 7. Generate and Log Evaluation Plots to MLflow
        print("4. Generating & Logging Performance Plots (ROC, PR Curve, Confusion Matrix)...")
        roc_p, pr_p, cm_p = generate_evaluation_plots(test_y, test_probs, test_preds, output_dir="models")
        mlflow.log_artifact(roc_p, artifact_path="evaluation_charts")
        mlflow.log_artifact(pr_p, artifact_path="evaluation_charts")
        mlflow.log_artifact(cm_p, artifact_path="evaluation_charts")
        mlflow.log_artifact(metrics_file, artifact_path="metrics")
        mlflow.log_artifact(local_model_path, artifact_path="model_checkpoints")

        # 8. Register Model in MLflow Registry
        try:
            print(f"5. Registering Model in MLflow Model Registry as '{model_name}'...")
            client = MlflowClient()
            try:
                client.create_registered_model(model_name)
            except Exception:
                pass

            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model_checkpoints"
            mv = client.create_model_version(
                name=model_name,
                source=f"{run.info.artifact_uri}/model_checkpoints/graph_nc.pt",
                run_id=run_id,
            )

            if test_auc >= 0.70:
                print(f"   -> Setting alias '@champion' for {model_name} version {mv.version}...")
                client.set_registered_model_alias(name=model_name, alias="champion", version=mv.version)
        except Exception as e:
            print(f"   -> Notice on Model Registry: {e}")

    print("\n🎉 Training, Artifact Logging & MLflow Tracking Complete!")


if __name__ == "__main__":
    train()
