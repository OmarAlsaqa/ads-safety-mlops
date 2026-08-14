import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure AWS S3 local endpoint for MLflow artifact retrieval
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:4566")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "AirQualityClassifier"
MODEL_ALIAS = "champion"

# Prometheus Operational Metrics
PREDICTION_COUNT = Counter(
    "iaq_predictions_total",
    "Total number of IAQ classification predictions served",
    ["risk_level"]
)

PREDICTION_LATENCY = Histogram(
    "iaq_prediction_latency_seconds",
    "Time spent processing IAQ prediction request in seconds"
)

# ------------------------------------------------------------------------------
# Evidently AI Data Drift Telemetry Gauges (Normalized & Separated Input/Output)
# ------------------------------------------------------------------------------
# Legacy aliases for backward compatibility
EVIDENTLY_DATASET_DRIFT = Gauge(
    "evidently_dataset_drift",
    "Dataset drift status (1 if dataset drift detected, 0 otherwise)"
)
EVIDENTLY_DRIFT_SHARE = Gauge(
    "evidently_drift_share",
    "Share of drifted features across monitored dataset (0.0 to 1.0)"
)
EVIDENTLY_DRIFTED_FEATURES_COUNT = Gauge(
    "evidently_number_of_drifted_columns",
    "Number of drifted features in current production window"
)

# Input Feature Drift Gauges
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
EVIDENTLY_FEATURE_DRIFT_STATUS = Gauge(
    "evidently_column_drift_status",
    "Drift status per feature (1 if drifted, 0 otherwise)",
    ["column_name"]
)

# Output Target / Prediction Drift Gauges
EVIDENTLY_OUTPUT_PREDICTION_DRIFT = Gauge(
    "evidently_output_prediction_drift",
    "Output target / prediction drift status (1 if prediction drift detected, 0 otherwise)"
)
EVIDENTLY_OUTPUT_PREDICTION_DRIFT_SCORE = Gauge(
    "evidently_output_prediction_drift_score",
    "Output target / prediction drift score / p-value"
)

# Global model store
model_store: Dict[str, Any] = {}

CLASS_MAP = {
    0: "Good / Low Risk",
    1: "Moderate Risk",
    2: "High Risk / Unhealthy"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager:
    Loads champion model from MLflow Model Registry on startup (with fallback to local models/model.pkl).
    """
    print(f"Connecting to MLflow Tracking Server at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        print(f"Loading champion model from MLflow Registry: {model_uri}...")
        model_store["model"] = mlflow.pyfunc.load_model(model_uri)
        model_store["source"] = f"MLflow Registry ({model_uri})"
        print("✅ Champion model loaded successfully from MLflow Registry!")
    except Exception as e:
        print(f"⚠️ Failed to load model from MLflow Registry ({e}). Attempting local fallback (models/model.pkl)...")
        local_path = "models/model.pkl"
        if os.path.exists(local_path):
            model_store["model"] = joblib.load(local_path)
            model_store["source"] = f"Local File ({local_path})"
            print("✅ Model loaded successfully from local file!")
        else:
            print("❌ Error: No trained model found locally or in MLflow Registry!")
            model_store["model"] = None
            model_store["source"] = "None"
            
    yield
    print("Shutting down FastAPI Model Serving Server...")

app = FastAPI(
    title="Indoor Air Quality (IAQ) Safety Classification API",
    description="Production REST API for real-time indoor air quality risk classification and Prometheus telemetry monitoring.",
    version="1.0.0",
    lifespan=lifespan
)

class SensorInput(BaseModel):
    """
    Pydantic Schema for Indoor Air Quality Sensor Readings.
    Supports standard snake_case keys and exact dataset column aliases.
    """
    temperature: float = Field(..., alias="Temperature (C)", ge=10.0, le=50.0, description="Temperature in Celsius")
    humidity: float = Field(..., alias="Humidity (%)", ge=0.0, le=100.0, description="Relative Humidity percentage")
    pressure: float = Field(..., alias="Pressure (hPa)", ge=800.0, le=1100.0, description="Barometric Pressure in hPa")
    gas_resistance: float = Field(..., alias="Gas Resistance (Ohms)", ge=0.0, description="Gas Sensor Resistance in Ohms")
    pm2_5: float = Field(..., alias="PM2.5", ge=0.0, description="Particulate Matter PM2.5 in ug/m3")
    tvoc: float = Field(..., alias="TVOC (ppb)", ge=0.0, description="Total Volatile Organic Compounds in ppb")
    eco2: float = Field(..., alias="eCO2 (ppm)", ge=300.0, description="Equivalent CO2 in ppm")
    voc_index: float = Field(..., alias="VOC Index", ge=0.0, le=500.0, description="VOC Index reading")
    mq135: float = Field(..., alias="MQ135 Value", ge=0.0, description="MQ135 Gas Sensor raw reading")
    voltage: float = Field(..., alias="Voltage", ge=0.0, le=5.0, description="Analog Sensor Output Voltage")
    ppm: float = Field(..., alias="PPM", ge=0.0, description="PPM Concentration reading")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Temperature (C)": 24.5,
                "Humidity (%)": 42.1,
                "Pressure (hPa)": 901.2,
                "Gas Resistance (Ohms)": 2500000.0,
                "PM2.5": 8.5,
                "TVOC (ppb)": 150.0,
                "eCO2 (ppm)": 420.0,
                "VOC Index": 85.0,
                "MQ135 Value": 165.0,
                "Voltage": 0.52,
                "PPM": 310.0
            }
        }
    )

class PredictionResponse(BaseModel):
    iaq_class: int = Field(..., description="Derived Risk Class (0: Good, 1: Moderate, 2: High Risk)")
    risk_level: str = Field(..., description="Human readable IAQ safety risk level")
    confidence: Optional[float] = Field(None, description="Prediction probability confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probability distribution")
    model_source: str = Field(..., description="Active MLflow model source")

def _update_evidently_metrics():
    """Reads latest Evidently drift telemetry summary and updates Prometheus Gauges."""
    possible_paths = [
        "/app/docs/reports/data_drift_summary.json",
        "docs/reports/data_drift_summary.json",
        os.path.join(os.path.dirname(__file__), "..", "docs", "reports", "data_drift_summary.json")
    ]
    summary_path = None
    for p in possible_paths:
        if os.path.exists(p):
            summary_path = p
            break

    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            
            for m in data.get("metrics", []):
                metric_name = str(m.get("metric", ""))
                if "DatasetDriftMetric" in metric_name:
                    res = m.get("result", {})
                    ds_drift = 1.0 if res.get("dataset_drift") else 0.0
                    d_share = float(res.get("share_of_drifted_columns", 0.0))
                    d_cnt = float(res.get("number_of_drifted_columns", 0))

                    EVIDENTLY_DATASET_DRIFT.set(ds_drift)
                    EVIDENTLY_DRIFT_SHARE.set(d_share)
                    EVIDENTLY_DRIFTED_FEATURES_COUNT.set(d_cnt)

                    EVIDENTLY_INPUT_DATASET_DRIFT.set(ds_drift)
                    EVIDENTLY_INPUT_DRIFT_SHARE.set(d_share)
                    EVIDENTLY_INPUT_DRIFTED_FEATURES_COUNT.set(d_cnt)
                
                elif "DataDriftTable" in metric_name:
                    drift_by_cols = m.get("result", {}).get("drift_by_columns", {})
                    for col_name, col_data in drift_by_cols.items():
                        raw_score = float(col_data.get("drift_score", 0.0))
                        detected = 1.0 if col_data.get("drift_detected") else 0.0
                        
                        # Clean score normalization:
                        # If score is p-value <= 1.0: 1.0 - p_val (0.0 = no drift, 1.0 = high drift confidence)
                        # If score is raw distance > 1.0: cap to [0.0, 1.0]
                        if raw_score <= 1.0:
                            norm_score = round(1.0 - raw_score, 4) if detected else round(raw_score, 4)
                        else:
                            norm_score = min(1.0, round(raw_score / 10.0, 4)) if detected else 0.0

                        EVIDENTLY_FEATURE_DRIFT_SCORE.labels(column_name=col_name).set(norm_score)
                        EVIDENTLY_FEATURE_DRIFT_STATUS.labels(column_name=col_name).set(detected)

                        # Separate Output Target / Prediction Drift
                        if col_name in ["IAQ_Class", "prediction", "target"]:
                            EVIDENTLY_OUTPUT_PREDICTION_DRIFT.set(detected)
                            EVIDENTLY_OUTPUT_PREDICTION_DRIFT_SCORE.set(norm_score)

        except Exception as e:
            print(f"Notice: Error updating Evidently drift metrics for Prometheus: {e}")

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "IAQ Safety Classification API",
        "status": "online",
        "model_loaded": model_store.get("model") is not None,
        "model_source": model_store.get("source", "Unknown")
    }

@app.get("/health", tags=["Health"])
def healthcheck():
    if model_store.get("model") is None:
        raise HTTPException(
            status_code=status.HTTP_530_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {
        "status": "healthy",
        "model_source": model_store.get("source")
    }

@app.get("/metrics", tags=["Telemetry"])
def metrics():
    """Prometheus Metrics Endpoint for Grafana scraping."""
    _update_evidently_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _prepare_features(sensor_data: SensorInput) -> pd.DataFrame:
    """Formats Pydantic model into DataFrame matching model training column order."""
    data_dict = {
        "Temperature (C)": sensor_data.temperature,
        "Humidity (%)": sensor_data.humidity,
        "Pressure (hPa)": sensor_data.pressure,
        "Gas Resistance (Ohms)": sensor_data.gas_resistance,
        "PM2.5": sensor_data.pm2_5,
        "TVOC (ppb)": sensor_data.tvoc,
        "eCO2 (ppm)": sensor_data.eco2,
        "VOC Index": sensor_data.voc_index,
        "MQ135 Value": sensor_data.mq135,
        "Voltage": sensor_data.voltage,
        "PPM": sensor_data.ppm
    }
    return pd.DataFrame([data_dict])

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: SensorInput):
    """
    Real-time single sample prediction endpoint.
    Returns predicted IAQ_Class label, human readable risk tier, and probabilities.
    """
    model = model_store.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or loaded."
        )

    start_time = time.time()
    try:
        input_df = _prepare_features(payload)
        
        # Predict using MLflow PyFunc or Scikit-learn model
        if hasattr(model, "predict_proba"):
            preds = model.predict(input_df)
            probs = model.predict_proba(input_df)[0]
            pred_class = int(preds[0])
            confidence = float(probs[pred_class])
            prob_dict = {CLASS_MAP[i]: float(probs[i]) for i in range(len(probs))}
        else:
            # PyFunc wrapper return DataFrame or ndarray
            raw_preds = model.predict(input_df)
            if isinstance(raw_preds, pd.DataFrame):
                pred_class = int(raw_preds.iloc[0, 0])
            else:
                pred_class = int(raw_preds[0])
            confidence = 1.0
            prob_dict = {CLASS_MAP[pred_class]: 1.0}

        risk_tier = CLASS_MAP.get(pred_class, "Unknown")
        latency = time.time() - start_time

        # Update Prometheus Metrics
        PREDICTION_COUNT.labels(risk_level=risk_tier).inc()
        PREDICTION_LATENCY.observe(latency)

        return PredictionResponse(
            iaq_class=pred_class,
            risk_level=risk_tier,
            confidence=confidence,
            probabilities=prob_dict,
            model_source=model_store.get("source", "Unknown")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/batch-predict", tags=["Inference"])
def batch_predict(payloads: List[SensorInput]):
    """Batch inference endpoint for multiple sensor reading samples."""
    model = model_store.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or loaded."
        )

    start_time = time.time()
    try:
        dfs = [_prepare_features(item) for item in payloads]
        batch_df = pd.concat(dfs, ignore_index=True)
        
        preds = model.predict(batch_df)
        if hasattr(preds, "tolist"):
            pred_classes = preds.tolist()
        else:
            pred_classes = list(preds)

        results = []
        for p_cls in pred_classes:
            p_int = int(p_cls)
            r_tier = CLASS_MAP.get(p_int, "Unknown")
            PREDICTION_COUNT.labels(risk_level=r_tier).inc()
            results.append({
                "iaq_class": p_int,
                "risk_level": r_tier
            })

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        return {
            "total_samples": len(payloads),
            "predictions": results,
            "latency_seconds": round(latency, 4)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
