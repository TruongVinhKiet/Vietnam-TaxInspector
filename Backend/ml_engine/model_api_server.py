"""
model_api_server.py – Microservice Model Inference API (Container: tax-model-server)
======================================================================================
Lightweight FastAPI server dedicated to DL model inference.
Runs in an isolated container with its own memory budget.

This service exposes internal-only endpoints for model prediction.
The main API server (tax-api-server) calls these endpoints via
Docker internal network: http://tax-model-server:8001/predict/<model>

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │           tax-model-server (Port 8001)              │
    │                                                     │
    │  GET  /health           → Container health check    │
    │  GET  /models/status    → All model cache status    │
    │  POST /predict/vae      → VAE anomaly inference     │
    │  POST /predict/transformer → Delinquency forecast   │
    │  POST /predict/gnn      → GNN fraud detection       │
    │  POST /predict/hetero-gnn → HeteroGNN/HGT risk      │
    │  POST /preload          → Warm-up model cache       │
    └─────────────────────────────────────────────────────┘

Design Decisions:
    - Separate process = isolated RAM budget (no OOM on main API)
    - Singleton ModelServingGateway handles LRU + caching internally
    - CPU-only inference (no GPU dependency for portability)
    - JSON in/out for maximum interop (upgrade path to gRPC/Triton)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ml_engine.model_serving import ModelServingGateway, get_model_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("model-server")

app = FastAPI(
    title="TaxInspector Model Inference Server",
    description="Internal microservice for DL model inference (VAE, Transformer, GNN, HeteroGNN)",
    version="1.0.0",
    docs_url="/docs",
)


# ════════════════════════════════════════════════════════════════
#  Request / Response Schemas
# ════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Generic prediction request with feature vector."""
    features: list[float] = Field(..., description="Input feature vector")
    config: dict[str, Any] = Field(default_factory=dict, description="Optional model config override")
    node_type: str | None = Field(default=None, description="Optional HeteroGNN node type")
    entity_id: str | None = Field(default=None, description="Optional entity identifier for audit traces")
    graph_context: dict[str, Any] = Field(default_factory=dict, description="Optional graph context")


class VAEPredictResponse(BaseModel):
    """VAE anomaly detection response."""
    reconstruction_error: float
    is_anomaly: bool
    anomaly_threshold: float
    latent_mu: list[float]
    inference_ms: float


class TransformerPredictResponse(BaseModel):
    """Delinquency Transformer prediction response."""
    prob_30d: float
    prob_60d: float
    prob_90d: float
    risk_level: str
    inference_ms: float


class GNNPredictResponse(BaseModel):
    """GNN fraud score response."""
    fraud_probability: float
    risk_level: str
    inference_ms: float


class HeteroGNNPredictResponse(BaseModel):
    """Heterogeneous GNN fraud score response."""
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float
    risk_level: str
    node_type_scores: dict[str, float]
    inference_ms: float
    model_version: str


class PreloadRequest(BaseModel):
    """Request to preload models into cache."""
    model_config = ConfigDict(protected_namespaces=())

    model_names: list[str] | None = None


def _risk_level(probability: float) -> str:
    return (
        "critical" if probability > 0.85
        else "high" if probability > 0.65
        else "medium" if probability > 0.4
        else "low"
    )


def _fit_feature_vector(features: list[float], expected_dim: int) -> torch.Tensor:
    values = list(features or [])
    if len(values) < expected_dim:
        values = values + [0.0] * (expected_dim - len(values))
    if len(values) > expected_dim:
        values = values[:expected_dim]
    return torch.tensor([values], dtype=torch.float32)


def _projection_input_dim(projection: Any, fallback: int) -> int:
    weight = getattr(projection, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return fallback


def _predict_hetero_single_vector(model: Any, features: list[float], node_type: str) -> dict[str, float]:
    """
    HGT normally needs a full heterogeneous graph. For the model-server API we
    expose a backward-compatible single-vector adapter that uses the trained
    per-node input projection and classifier heads. Full graph inference stays
    in the OSINT/graph pipelines.
    """
    node_types = list(getattr(model, "node_types", ()) or ())
    projections = getattr(model, "input_projections", None)
    classifiers = getattr(model, "classifiers", None)
    if not node_types or projections is None or classifiers is None:
        raw = float(np.mean(features or [0.0]))
        probability = 1.0 / (1.0 + np.exp(-raw))
        return {node_type: float(probability)}

    scores: dict[str, float] = {}
    with torch.no_grad():
        for current_type in node_types:
            if current_type not in projections or current_type not in classifiers:
                continue
            projection = projections[current_type]
            classifier = classifiers[current_type]
            expected_dim = _projection_input_dim(projection, len(features or []))
            x = _fit_feature_vector(features, expected_dim)
            hidden = projection(x)
            logits = classifier(hidden)
            score = torch.sigmoid(logits).reshape(-1)[0].item()
            scores[current_type] = round(float(score), 4)

    if node_type not in scores and scores:
        scores[node_type] = next(iter(scores.values()))
    return scores or {node_type: 0.5}


# ════════════════════════════════════════════════════════════════
#  Health & Status Endpoints
# ════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Container health check for Docker/K8s."""
    return {
        "status": "healthy",
        "service": "tax-model-server",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/models/status")
def get_models_status():
    """Get status of all registered models in the gateway."""
    gateway = get_model_gateway()
    return gateway.get_status()


@app.post("/preload")
def preload_models(request: PreloadRequest):
    """Pre-load models into memory for reduced first-request latency."""
    gateway = get_model_gateway()
    gateway.preload(request.model_names)
    return gateway.get_status()


# ════════════════════════════════════════════════════════════════
#  Prediction Endpoints
# ════════════════════════════════════════════════════════════════

@app.post("/predict/vae", response_model=VAEPredictResponse)
def predict_vae(request: PredictRequest):
    """
    Run VAE anomaly detection on a feature vector.
    
    Expected input: 16-dim feature vector (normalized).
    Returns: reconstruction error, anomaly flag, latent space.
    """
    t0 = time.perf_counter()
    gateway = get_model_gateway()
    model = gateway.get_model("vae")

    if model is None:
        raise HTTPException(status_code=503, detail="VAE model not loaded")

    try:
        x = torch.tensor([request.features], dtype=torch.float32)
        with torch.no_grad():
            x_recon, mu, logvar = model(x)
            recon_error = torch.mean((x - x_recon) ** 2).item()

        threshold = request.config.get("anomaly_threshold", 0.65)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return VAEPredictResponse(
            reconstruction_error=round(recon_error, 6),
            is_anomaly=recon_error > threshold,
            anomaly_threshold=threshold,
            latent_mu=mu[0].tolist(),
            inference_ms=round(inference_ms, 2),
        )
    except Exception as exc:
        logger.error("VAE prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/transformer", response_model=TransformerPredictResponse)
def predict_transformer(request: PredictRequest):
    """
    Run Delinquency Transformer prediction.
    
    Expected input: Flattened sequence tensor.
    Returns: 30/60/90-day delinquency probabilities.
    """
    t0 = time.perf_counter()
    gateway = get_model_gateway()
    model = gateway.get_model("transformer")

    if model is None:
        raise HTTPException(status_code=503, detail="Transformer model not loaded")

    try:
        feature_dim = request.config.get("feature_dim", 8)
        features = np.array(request.features, dtype=np.float32)
        seq_len = len(features) // feature_dim
        x = torch.tensor(features.reshape(1, seq_len, feature_dim), dtype=torch.float32)
        mask = torch.ones(1, seq_len, dtype=torch.bool)

        with torch.no_grad():
            out_30, out_60, out_90 = model(x, mask)
            prob_30 = torch.softmax(out_30, dim=1)[0, 1].item()
            prob_60 = torch.softmax(out_60, dim=1)[0, 1].item()
            prob_90 = torch.softmax(out_90, dim=1)[0, 1].item()

        max_prob = max(prob_30, prob_60, prob_90)
        risk_level = (
            "critical" if max_prob > 0.8
            else "high" if max_prob > 0.6
            else "medium" if max_prob > 0.4
            else "low"
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return TransformerPredictResponse(
            prob_30d=round(prob_30, 4),
            prob_60d=round(prob_60, 4),
            prob_90d=round(prob_90, 4),
            risk_level=risk_level,
            inference_ms=round(inference_ms, 2),
        )
    except Exception as exc:
        logger.error("Transformer prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/gnn", response_model=GNNPredictResponse)
def predict_gnn(request: PredictRequest):
    """
    Run GNN fraud score prediction.
    
    Expected input: Node feature vector.
    Returns: Fraud probability and risk level.
    """
    t0 = time.perf_counter()
    gateway = get_model_gateway()
    model = gateway.get_model("gnn")

    if model is None:
        raise HTTPException(status_code=503, detail="GNN model not loaded")

    try:
        x = torch.tensor([request.features], dtype=torch.float32)

        with torch.no_grad():
            # Simplified — full GNN requires graph structure
            # In production, this would receive adjacency data too
            if hasattr(model, 'forward_single'):
                output = model.forward_single(x)
            else:
                output = torch.tensor([[0.5]])  # Fallback

            fraud_prob = output[0, 0].item() if output.dim() > 1 else output.item()

        risk_level = _risk_level(fraud_prob)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return GNNPredictResponse(
            fraud_probability=round(fraud_prob, 4),
            risk_level=risk_level,
            inference_ms=round(inference_ms, 2),
        )
    except Exception as exc:
        logger.error("GNN prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ════════════════════════════════════════════════════════════════
#  Entrypoint
# ════════════════════════════════════════════════════════════════

@app.post("/predict/hetero-gnn", response_model=HeteroGNNPredictResponse)
def predict_hetero_gnn(request: PredictRequest):
    """
    Run Heterogeneous Graph Transformer risk prediction.

    Full HGT inference normally consumes a heterogeneous graph snapshot. This
    endpoint keeps the model-server contract simple by accepting the same
    feature-vector request as other DL endpoints and using the trained per-node
    projection/classifier adapter for low-latency single-entity scoring.
    """
    t0 = time.perf_counter()
    gateway = get_model_gateway()
    model = gateway.get_model("hetero_gnn")

    if model is None:
        raise HTTPException(status_code=503, detail="HeteroGNN model not loaded")

    try:
        node_type = str(request.node_type or request.config.get("node_type") or "company")
        node_type_scores = _predict_hetero_single_vector(model, request.features, node_type)
        fraud_prob = float(node_type_scores.get(node_type, next(iter(node_type_scores.values()), 0.5)))
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return HeteroGNNPredictResponse(
            fraud_probability=round(fraud_prob, 4),
            risk_level=_risk_level(fraud_prob),
            node_type_scores=node_type_scores,
            inference_ms=round(inference_ms, 2),
            model_version="hetero-gnn-hgt-v1",
        )
    except Exception as exc:
        logger.error("HeteroGNN prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(
        "ml_engine.model_api_server:app",
        host="0.0.0.0",
        port=8001,
        workers=1,   # Single worker — models share memory via singleton
        log_level="info",
    )
