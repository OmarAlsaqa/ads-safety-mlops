import os
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field, ConfigDict
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

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

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "GraphNC-AdFraud-Detector"
MODEL_ALIAS = "champion"

# ==============================================================================
# Prometheus Telemetry Metrics
# ==============================================================================
PREDICTION_COUNT = Counter(
    "ad_click_predictions_total",
    "Total number of ad click predictions served",
    ["risk_tier"]
)

FRAUD_DETECTED_COUNT = Counter(
    "ad_click_fraud_detected_total",
    "Total number of ad click fraud incidents flagged"
)

PREDICTION_LATENCY = Histogram(
    "ad_click_prediction_latency_seconds",
    "Time spent processing ad click graph inference in seconds",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FRAUD_SCORE_GAUGE = Gauge(
    "ad_click_latest_fraud_score",
    "Latest ad click fraud risk probability score"
)

# Evidently AI Monitoring Gauges
EVIDENTLY_INPUT_DATASET_DRIFT = Gauge(
    "evidently_input_dataset_drift",
    "Input feature dataset drift status (1 if feature drift detected, 0 otherwise)"
)
EVIDENTLY_INPUT_DRIFT_SHARE = Gauge(
    "evidently_input_drift_share",
    "Share of drifted input features across monitored dataset (0.0 to 1.0)"
)
EVIDENTLY_INPUT_DRIFTED_FEATURES_COUNT = Gauge(
    "evidently_input_number_of_drifted_columns",
    "Number of drifted input features in current production window"
)
EVIDENTLY_FEATURE_DRIFT_SCORE = Gauge(
    "evidently_column_drift_score",
    "Drift p-value / distance score per feature",
    ["column_name"]
)
EVIDENTLY_OUTPUT_PREDICTION_DRIFT = Gauge(
    "evidently_output_prediction_drift",
    "Output target / prediction drift status (1 if prediction drift detected, 0 otherwise)"
)
EVIDENTLY_OUTPUT_PREDICTION_DRIFT_SCORE = Gauge(
    "evidently_output_prediction_drift_score",
    "Output target / prediction drift score"
)

# In-memory Global Model Store
model_store: Dict[str, Any] = {}


# ==============================================================================
# Pydantic Schemas for Request & Response
# ==============================================================================
class AdClickRequest(BaseModel):
    """Payload representing a single real-time ad click event."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ip": 5348,
                "app": 3,
                "device": 1,
                "os": 19,
                "channel": 379,
                "click_time": "2026-08-14 12:00:00"
            }
        }
    )
    ip: int = Field(..., ge=0, description="IP address ID")
    app: int = Field(..., ge=0, description="Target app ID")
    device: int = Field(..., ge=0, description="Device type ID")
    os: int = Field(..., ge=0, description="Operating system ID")
    channel: int = Field(..., ge=0, description="Publisher / channel ID")
    click_time: Optional[str] = Field(None, description="ISO timestamp of the click event")


class BatchAdClickRequest(BaseModel):
    """Batch payload for high-throughput stream inference."""
    clicks: List[AdClickRequest]


class AdClickResponse(BaseModel):
    """Response payload with calibrated risk score and operational tier."""
    ip: int
    app: int
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    decision_threshold: float
    inference_latency_ms: float
    model_version: str
    timestamp: str


class BatchAdClickResponse(BaseModel):
    total_clicks: int
    fraud_count: int
    fraud_rate: float
    results: List[AdClickResponse]


# ==============================================================================
# Model Loading & Warm-Up Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads GraphNC model weights and initializes in-memory graph context buffer."""
    print("🔄 Initializing GraphNC Real-Time Inference Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = "models/graph_nc.pt"

    loaded = False
    model_version = "v1-local"
    threshold = 0.6566

    # Attempt 1: Load from MLflow Model Registry (@champion)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        model_version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_version = f"v{model_version_info.version}"
        print(f"   -> Found MLflow Champion Model version: {model_version}")
    except Exception as e:
        print(f"   -> MLflow Registry lookup notice ({e}). Using local checkpoint.")

    # Attempt 2: Load local PyG model checkpoint
    if os.path.exists(local_path):
        checkpoint = torch.load(local_path, map_location=device, weights_only=False)
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
        loaded = True
        print(f"   ✅ GraphNC model loaded successfully from {local_path} on {device} (Threshold: {threshold:.4f})")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {local_path}. Run training first.")

    # Initialize Feast Feature Store
    try:
        from feast import FeatureStore
        fs_path = os.getenv("FEAST_REPO_PATH", "feature_store")
        if os.path.exists(fs_path):
            model_store["feature_store"] = FeatureStore(repo_path=fs_path)
            print(f"   ✅ Feast Feature Store online retrieval ready from {fs_path}")
    except Exception as fs_err:
        print(f"   Notice on Feast Feature Store init: {fs_err}")

    # Initialize rolling graph context state (fallback frequency tables)
    model_store["model"] = model
    model_store["device"] = device
    model_store["threshold"] = threshold
    model_store["model_version"] = model_version
    model_store["ip_counts"] = {}
    model_store["app_counts"] = {}
    model_store["channel_counts"] = {}
    model_store["device_counts"] = {}
    model_store["load_time"] = datetime.utcnow().isoformat()

    yield
    print("🛑 Shutting down GraphNC Inference Engine...")
    model_store.clear()


app = FastAPI(
    title="Ads Safety - GraphNC Ad-Fraud Detection Service",
    description="Sub-millisecond Graph Neural Network (ICML 2026) API for real-time botnet and ad-click fraud detection.",
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# Helper Inference Logic
# ==============================================================================
def process_single_click(click: AdClickRequest) -> Dict[str, Any]:
    """Extracts features, builds local graph tensor, and executes GraphNC forward pass."""
    model: GraphNC = model_store["model"]
    device = model_store["device"]
    threshold = model_store["threshold"]
    model_version = model_store["model_version"]

    start_time = time.perf_counter()

    # 1. Fetch enriched online entity features from Feast Feature Store
    fs = model_store.get("feature_store")
    ip_cnt = None
    ip_uniq_apps = 1.0
    ip_uniq_chan = 1.0
    app_cum = 0.0
    next_delta = 8.19
    prev_delta = 8.19
    app_cnt = None
    app_chan_cnt = 10.0
    chan_cnt = None
    dev_cnt = None
    dev_os_cum = 0.0

    if fs is not None:
        try:
            response = fs.get_online_features(
                features=[
                    "ip_features:ip_click_count",
                    "ip_features:ip_unique_apps",
                    "ip_features:ip_unique_channels",
                    "ip_features:ip_app_cumcount",
                    "ip_features:next_click_delta",
                    "ip_features:prev_click_delta",
                    "app_features:app_freq",
                    "app_features:app_channel_count",
                    "channel_features:channel_freq",
                    "device_features:device_freq",
                    "device_features:ip_device_os_cumcount",
                ],
                entity_rows=[{
                    "ip": click.ip,
                    "app": click.app,
                    "channel": click.channel,
                    "device": click.device,
                }],
            ).to_dict()
            if response.get("ip_click_count") and response["ip_click_count"][0] is not None:
                ip_cnt = float(response["ip_click_count"][0])
            if response.get("ip_unique_apps") and response["ip_unique_apps"][0] is not None:
                ip_uniq_apps = float(response["ip_unique_apps"][0])
            if response.get("ip_unique_channels") and response["ip_unique_channels"][0] is not None:
                ip_uniq_chan = float(response["ip_unique_channels"][0])
            if response.get("ip_app_cumcount") and response["ip_app_cumcount"][0] is not None:
                app_cum = float(response["ip_app_cumcount"][0])
            if response.get("next_click_delta") and response["next_click_delta"][0] is not None:
                next_delta = float(response["next_click_delta"][0])
            if response.get("prev_click_delta") and response["prev_click_delta"][0] is not None:
                prev_delta = float(response["prev_click_delta"][0])
            if response.get("app_freq") and response["app_freq"][0] is not None:
                app_cnt = float(response["app_freq"][0])
            if response.get("app_channel_count") and response["app_channel_count"][0] is not None:
                app_chan_cnt = float(response["app_channel_count"][0])
            if response.get("channel_freq") and response["channel_freq"][0] is not None:
                chan_cnt = float(response["channel_freq"][0])
            if response.get("device_freq") and response["device_freq"][0] is not None:
                dev_cnt = float(response["device_freq"][0])
            if response.get("ip_device_os_cumcount") and response["ip_device_os_cumcount"][0] is not None:
                dev_os_cum = float(response["ip_device_os_cumcount"][0])
        except Exception:
            pass

    # Fallback to local rolling counter if Feast store is unpopulated
    if ip_cnt is None:
        ip_cnt = float(model_store["ip_counts"].get(click.ip, 1) + 1)
        model_store["ip_counts"][click.ip] = int(ip_cnt)
    if app_cnt is None:
        app_cnt = float(model_store["app_counts"].get(click.app, 1) + 1)
        model_store["app_counts"][click.app] = int(app_cnt)
    if chan_cnt is None:
        chan_cnt = float(model_store["channel_counts"].get(click.channel, 1) + 1)
        model_store["channel_counts"][click.channel] = int(chan_cnt)
    if dev_cnt is None:
        dev_cnt = float(model_store["device_counts"].get(click.device, 1) + 1)
        model_store["device_counts"][click.device] = int(dev_cnt)

    # 2. Extract temporal signals
    try:
        dt = datetime.fromisoformat(click.click_time) if click.click_time else datetime.utcnow()
    except Exception:
        dt = datetime.utcnow()

    hour = float(dt.hour)
    day = float(dt.day)

    ip_hh_app = min(ip_cnt, 10.0)
    ip_hh_dev = min(ip_cnt, 20.0)

    # 3. Assemble normalized feature vector [15 features]
    raw_feats = np.array([
        hour, day, ip_cnt, ip_uniq_apps, app_cnt, chan_cnt, dev_cnt,
        next_delta, prev_delta, dev_os_cum, app_cum,
        ip_hh_app, ip_hh_dev, app_chan_cnt, ip_uniq_chan
    ], dtype=np.float32)

    # Standardize with approximate dataset statistics
    means = np.array([9.0, 7.0, 50.0, 2.0, 100.0, 50.0, 500.0, 7.0, 7.0, 0.5, 0.5, 3.0, 5.0, 15.0, 2.0], dtype=np.float32)
    stds = np.array([6.0, 1.0, 150.0, 3.0, 250.0, 100.0, 1000.0, 3.0, 3.0, 1.5, 1.5, 10.0, 20.0, 40.0, 3.0], dtype=np.float32)
    x_norm = (raw_feats - means) / (stds + 1e-6)

    x_tensor = torch.tensor(x_norm, dtype=torch.float).unsqueeze(0).to(device)
    x_cat_tensor = torch.tensor([[click.app, click.channel, click.device, click.os, int(hour)]], dtype=torch.long).to(device)
    # Self-loop edge for single-node graph inference
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    # 4. GraphNC Model Forward Pass
    with torch.no_grad():
        raw_score = model(x_tensor, edge_index, x_cat=x_cat_tensor).item()

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Calculate dynamic fraud risk score factoring GraphNC embeddings + IP velocity surge
    # High frequency bursts (click-flooding) exponentially increase fraud risk
    velocity_penalty = max(0.0, (ip_cnt - 10) / 100.0)
    fraud_prob = min(1.0, max(0.0, raw_score + velocity_penalty))

    is_fraud = bool(fraud_prob >= threshold or ip_cnt >= 40)

    # Determine operational risk tier
    if fraud_prob >= 0.70 or ip_cnt >= 60:
        risk_tier = "HIGH_BLOCK"
    elif fraud_prob >= threshold or ip_cnt >= 15:
        risk_tier = "MEDIUM_CHALLENGE"
    else:
        risk_tier = "LOW_ALLOW"

    # Prometheus telemetry tracking
    PREDICTION_COUNT.labels(risk_tier=risk_tier).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    FRAUD_SCORE_GAUGE.set(fraud_prob)
    if is_fraud:
        FRAUD_DETECTED_COUNT.inc()

    return {
        "ip": click.ip,
        "app": click.app,
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "risk_tier": risk_tier,
        "decision_threshold": round(threshold, 4),
        "inference_latency_ms": round(latency_ms, 2),
        "model_version": model_version,
        "timestamp": dt.isoformat(),
    }


# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Docker Compose & Kubernetes probes."""
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="GraphNC model is not initialized")
    return {
        "status": "HEALTHY",
        "service": "Ads Safety - GraphNC Detection Engine",
        "model_name": MODEL_NAME,
        "model_version": model_store.get("model_version", "unknown"),
        "decision_threshold": model_store.get("threshold", 0.6566),
        "device": str(model_store.get("device", "cpu")),
        "loaded_at": model_store.get("load_time"),
    }


@app.post("/predict/ad-click", response_model=AdClickResponse, status_code=status.HTTP_200_OK)
def predict_ad_click(click: AdClickRequest):
    """
    Real-Time Ad Click Fraud Inference:
    Scores an incoming ad click event against the GraphNC model in sub-milliseconds.
    """
    try:
        return process_single_click(click)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict", response_model=AdClickResponse, status_code=status.HTTP_200_OK)
def predict_alias(click: AdClickRequest):
    """Alias for /predict/ad-click."""
    return process_single_click(click)


@app.post("/batch-predict", response_model=BatchAdClickResponse, status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchAdClickRequest):
    """High-Throughput Batch Stream Inference."""
    results = [process_single_click(click) for click in payload.clicks]
    fraud_count = sum(1 for r in results if r["is_fraud"])
    fraud_rate = fraud_count / len(results) if results else 0.0

    return {
        "total_clicks": len(results),
        "fraud_count": fraud_count,
        "fraud_rate": round(fraud_rate, 4),
        "results": results,
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/metrics/drift", status_code=status.HTTP_200_OK)
def update_drift_metrics(payload: Dict[str, Any]):
    """Receives Evidently AI drift evaluation metrics and updates Prometheus gauges."""
    if "input_dataset_drift" in payload:
        EVIDENTLY_INPUT_DATASET_DRIFT.set(1.0 if payload["input_dataset_drift"] else 0.0)
    if "input_drift_share" in payload:
        EVIDENTLY_INPUT_DRIFT_SHARE.set(float(payload["input_drift_share"]))
    if "input_drifted_features_count" in payload:
        EVIDENTLY_INPUT_DRIFTED_FEATURES_COUNT.set(int(payload["input_drifted_features_count"]))
    if "output_prediction_drift" in payload:
        EVIDENTLY_OUTPUT_PREDICTION_DRIFT.set(1.0 if payload["output_prediction_drift"] else 0.0)
    if "output_prediction_drift_score" in payload:
        EVIDENTLY_OUTPUT_PREDICTION_DRIFT_SCORE.set(float(payload["output_prediction_drift_score"]))

    if "feature_drift_scores" in payload and isinstance(payload["feature_drift_scores"], dict):
        for col_name, score in payload["feature_drift_scores"].items():
            EVIDENTLY_FEATURE_DRIFT_SCORE.labels(column_name=col_name).set(float(score))

    return {"status": "Drift metrics successfully ingested into Prometheus"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=False)
