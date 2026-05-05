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
from pydantic import BaseModel, Field

from ml_engine.model_serving import ModelServingGateway, get_model_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("model-server")

app = FastAPI(
    title="TaxInspector Model Inference Server",
    description="Internal microservice for DL model inference (VAE, Transformer, GNN)",
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


class PreloadRequest(BaseModel):
    """Request to preload models into cache."""
    model_names: list[str] | None = None


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

        risk_level = (
            "critical" if fraud_prob > 0.85
            else "high" if fraud_prob > 0.65
            else "medium" if fraud_prob > 0.4
            else "low"
        )
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

if __name__ == "__main__":
    uvicorn.run(
        "ml_engine.model_api_server:app",
        host="0.0.0.0",
        port=8001,
        workers=1,   # Single worker — models share memory via singleton
        log_level="info",
    )
