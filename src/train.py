import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Set AWS S3 local endpoint for MLflow artifact logging to Floci (:4566)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:4566"
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

def train():
    # 1. MLflow Setup
    mlflow_tracking_uri = "http://localhost:5000"
    experiment_name = "air-quality-safety-classification"
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Enable MLflow Autologging for Scikit-Learn
    # Automatically logs model parameters, metrics, feature importances, and model signatures
    mlflow.sklearn.autolog(log_models=False)

    print("Reading dataset splits (train.parquet, val.parquet)...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    # Separate features and target
    drop_cols = ["Timestamp", "dt", "date_str", "IAQ_Class"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["IAQ_Class"]

    X_val = val_df[feature_cols]
    y_val = val_df["IAQ_Class"]

    # Define hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": 42,
        "n_jobs": -1
    }

    model_name = "AirQualityClassifier"

    print("Starting MLflow training run...")
    with mlflow.start_run() as run:
        # Fit Model (Autolog captures fit parameters & training score automatically)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions & Validation Metrics
        val_preds = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        val_f1_macro = f1_score(y_val, val_preds, average="macro")

        print(f"Validation Metrics -> Accuracy: {val_accuracy:.4f} | F1-Score (Macro): {val_f1_macro:.4f}")

        # Explicitly log validation split metrics
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_macro", val_f1_macro)

        # Log Model & Register in MLflow Model Registry
        signature = infer_signature(X_val, val_preds)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name=model_name
        )

        # MLflow Model Evaluation (Auto-generates Confusion Matrix, ROC & PR curves)
        eval_data = X_val.copy()
        eval_data["IAQ_Class"] = y_val
        
        print("Running MLflow Model Evaluation...")
        mlflow.models.evaluate(
            model_info.model_uri,
            data=eval_data,
            targets="IAQ_Class",
            model_type="classifier"
        )

        # Deployment Aliasing (@champion)
        client = MlflowClient()
        if val_accuracy >= 0.85:
            model_versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max([int(v.version) for v in model_versions]) if model_versions else 1
            
            print(f"Setting alias '@champion' for {model_name} version {latest_version}...")
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=latest_version
            )
        else:
            print(f"Validation accuracy ({val_accuracy:.4f}) below baseline threshold (0.85). Skipping @champion alias.")

        # Save local model artifact for DVC pipeline tracking
        os.makedirs("models", exist_ok=True)
        local_model_path = "models/model.pkl"
        joblib.dump(model, local_model_path)
        print(f"Local model saved to {local_model_path}")

if __name__ == "__main__":
    train()
