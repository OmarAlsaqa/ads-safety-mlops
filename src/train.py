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
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from models.graph_nc import GraphNC

# Local S3 and MLflow environment configuration
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:4566")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


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
    print(f"🚀 Training GraphNC Model on Device: {device}")

    # 1. MLflow Experiment Setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = "ads-safety-graphnc"

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"   -> Connected to MLflow at: {mlflow_tracking_uri}")
    except Exception as e:
        print(f"   -> Notice: MLflow server offline ({e}). Logging locally to ./mlruns")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

    # 2. Load Processed Graph Data
    graph_path = "data/processed/ad_click_graph.pt"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Processed graph not found at {graph_path}. Run 'dvc repro' first.")

    print(f"1. Loading Graph Dataset from {graph_path}...")
    data = torch.load(graph_path, weights_only=False).to(device)

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    in_dim = data.num_features
    pos_ratio = (data.y == 1).float().mean().item()

    print(f"   -> Nodes: {num_nodes:,} | Edges: {num_edges:,} | Features: {in_dim}")
    print(f"   -> Class Imbalance: Positive Fraud Rate = {pos_ratio * 100:.3f}%")

    # 3. Hyperparameters
    hyperparams = {
        "model_architecture": "GraphNC (ICML 2026)",
        "in_dim": in_dim,
        "hidden_dim": 64,
        "readout": "avg",
        "lr": 0.002,
        "weight_decay": 1e-4,
        "epochs": 40,
        "early_stopping_patience": 12,
        "beta": 0.5,
        "score_da_weight": 1.0,
        "distill_weight": 1.0,
        "norm_reg_weight": 0.5,
        "noise_var": 0.01,
        "batch_nodes": num_nodes,
        "total_edges": num_edges,
    }

    # 4. Instantiate Model, Optimizer & Scheduler
    model = GraphNC(
        in_dim=hyperparams["in_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        readout=hyperparams["readout"],
        beta=hyperparams["beta"],
        score_da_weight=hyperparams["score_da_weight"],
        distill_weight=hyperparams["distill_weight"],
        norm_reg_weight=hyperparams["norm_reg_weight"],
        noise_var=hyperparams["noise_var"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams["epochs"], eta_min=1e-5)

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    model_name = "GraphNC-AdFraud-Detector"
    os.makedirs("models", exist_ok=True)

    print("\n2. Starting MLflow Run & GraphNC Training...")
    with mlflow.start_run(run_name="graphnc_production_training") as run:
        mlflow.log_params(hyperparams)

        for epoch in range(1, hyperparams["epochs"] + 1):
            # --- Training Step ---
            model.train()
            optimizer.zero_grad()

            loss, loss_dict = model.compute_loss(
                data.x,
                data.edge_index,
                data.y,
                data.train_mask,
                is_training=True,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Validation Step ---
            model.eval()
            with torch.no_grad():
                val_probs = model(data.x, data.edge_index)[data.val_mask].cpu().numpy()
                val_y = data.y[data.val_mask].cpu().numpy()

                val_auc = roc_auc_score(val_y, val_probs)
                val_pr_auc = average_precision_score(val_y, val_probs)

            # Log metrics per epoch
            mlflow.log_metrics({
                "epoch_loss": loss_dict["loss_total"],
                "loss_bce": loss_dict["loss_bce"],
                "loss_norm_reg": loss_dict["loss_norm_reg"],
                "loss_distill": loss_dict["loss_distill"],
                "loss_score_da": loss_dict["loss_score_da"],
                "val_auc_roc": val_auc,
                "val_pr_auc": val_pr_auc,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            # Check for best model checkpoint based on Validation AUC-ROC & PR-AUC
            val_composite_score = 0.7 * val_auc + 0.3 * (val_pr_auc * 10)
            if val_composite_score > best_val_auc:
                best_val_auc = val_composite_score
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"   Epoch [{epoch:02d}/{hyperparams['epochs']:02d}] "
                      f"Loss: {loss_dict['loss_total']:.4f} (BCE: {loss_dict['loss_bce']:.3f}) | "
                      f"Val AUC-ROC: {val_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Peak Ep: {best_epoch}")

            # Early stopping check
            if patience_counter >= hyperparams["early_stopping_patience"]:
                print(f"   🛑 Early stopping triggered at epoch {epoch} (Best model checkpoint from epoch {best_epoch})")
                break

        # 5. Final Evaluation on Held-Out Test Set (Day 4)
        print(f"\n3. Evaluating Best Champion Model (from Epoch {best_epoch}) on Held-Out Test Set...")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()

        with torch.no_grad():
            # Get validation predictions to tune optimal operational decision threshold
            val_probs = model(data.x, data.edge_index)[data.val_mask].cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()

            # Find optimal threshold that maximizes Validation F1 Score
            best_thresh = 0.5
            best_val_f1 = 0.0
            for t in np.linspace(val_probs.min() + 1e-4, val_probs.max() - 1e-4, 150):
                preds = (val_probs >= t).astype(int)
                f1 = f1_score(val_y, preds, zero_division=0)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_thresh = float(t)

            # Apply tuned threshold on independent Held-out Test Set
            test_probs = model(data.x, data.edge_index)[data.test_mask].cpu().numpy()
            test_y = data.y[data.test_mask].cpu().numpy()

            test_auc = roc_auc_score(test_y, test_probs)
            test_pr_auc = average_precision_score(test_y, test_probs)
            test_preds = (test_probs >= best_thresh).astype(int)

            test_f1 = f1_score(test_y, test_preds, zero_division=0)
            test_prec = precision_score(test_y, test_preds, zero_division=0)
            test_rec = recall_score(test_y, test_preds, zero_division=0)

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
                "best_val_auc_roc": float(best_val_auc),
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
                "best_val_auc_roc": float(best_val_auc),
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
