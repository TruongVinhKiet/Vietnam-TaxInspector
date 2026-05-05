"""
model_serving.py – Centralized Model Serving Gateway (Phase B1)
================================================================
Singleton model registry with lazy loading, LRU eviction, and
Redis-backed metadata tracking to eliminate redundant torch.load()
calls across the multi-agent pipeline.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              ModelServingGateway (Singleton)         │
    │                                                     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
    │  │   VAE    │  │Transformer│  │   GAT    │  ...     │
    │  │ (cached) │  │ (cached)  │  │ (cached) │          │
    │  └──────────┘  └──────────┘  └──────────┘          │
    │                                                     │
    │  LRU Eviction | Thread-Safe | Memory Budget         │
    │  Redis Metadata Tracking (model versions, health)   │
    └─────────────────────────────────────────────────────┘
         ↑                ↑                ↑
    tax_agent_tools   vae_anomaly   temporal_transformer
    (no more direct torch.load!)

Design Decisions:
    - Thread-safe via threading.Lock for concurrent FastAPI requests
    - LRU eviction: when MAX_CACHED_MODELS reached, evict least-recently-used
    - Redis metadata: track model load times, access counts, versions
    - Graceful fallback: works without Redis (in-memory only mode)
    - Each model loaded ONCE per process lifetime (until evicted)

Reference:
    Eliminates 6 redundant torch.load() calls found in:
    - vae_anomaly.py L423
    - temporal_transformer.py L457
    - tax_agent_tools.py L1174, L1368
    - gnn_model.py L649
    - hetero_gnn_model.py L338
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# Redis connection (lazy-loaded)
_redis_client = None
_redis_init_attempted = False


def _get_redis():
    """Get Redis client with lazy initialization and graceful fallback."""
    global _redis_client, _redis_init_attempted
    if _redis_init_attempted:
        return _redis_client
    _redis_init_attempted = True
    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        _redis_client.ping()
        logger.info("[ModelServing] Redis connected: %s", url)
    except Exception as exc:
        logger.warning("[ModelServing] Redis unavailable, using in-memory only: %s", exc)
        _redis_client = None
    return _redis_client


# ════════════════════════════════════════════════════════════════
#  Model Registration Metadata
# ════════════════════════════════════════════════════════════════

@dataclass
class ModelMeta:
    """Metadata for a registered model."""
    name: str
    model_path: str
    config_path: str | None = None
    factory: Any = None          # Callable that creates nn.Module from config
    input_dim_key: str = ""      # Config key for input dim
    version: str = "1.0"
    load_count: int = 0
    last_loaded: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    memory_bytes: int = 0


# ════════════════════════════════════════════════════════════════
#  Singleton Model Serving Gateway
# ════════════════════════════════════════════════════════════════

class ModelServingGateway:
    """
    Centralized model loading with singleton cache + LRU eviction.

    Usage:
        gateway = ModelServingGateway.instance()
        model = gateway.get_model("vae")          # Loads once, cached forever
        model2 = gateway.get_model("transformer")  # Same — single load
        
        # Or with explicit config override:
        model = gateway.get_model("vae", config={"input_dim": 16})
    """

    _instance: Optional["ModelServingGateway"] = None
    _instance_lock = threading.Lock()

    # Max models to keep in memory simultaneously
    MAX_CACHED_MODELS = int(os.getenv("MODEL_SERVING_MAX_CACHE", "6"))
    REDIS_KEY_PREFIX = "taxinspector:model_serving:"

    def __init__(self):
        self._models: dict[str, nn.Module] = {}
        self._meta: dict[str, ModelMeta] = {}
        self._lock = threading.Lock()
        self._registered: dict[str, ModelMeta] = {}
        self._register_default_models()
        logger.info(
            "[ModelServing] Gateway initialized (max_cache=%d, registered=%d)",
            self.MAX_CACHED_MODELS,
            len(self._registered),
        )

    @classmethod
    def instance(cls) -> "ModelServingGateway":
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.unload_all()
            cls._instance = None

    # ─── Model Registration ──────────────────────────────────────

    def _register_default_models(self):
        """Register all known DL models in the system."""

        # 1. VAE Anomaly Detection
        self._registered["vae"] = ModelMeta(
            name="vae",
            model_path=str(MODEL_DIR / "vae_anomaly.pt"),
            config_path=str(MODEL_DIR / "vae_anomaly_config.json"),
            factory=self._factory_vae,
            version="1.0",
        )

        # 2. Temporal Transformer (Delinquency)
        self._registered["transformer"] = ModelMeta(
            name="transformer",
            model_path=str(MODEL_DIR / "temporal_transformer.pt"),
            config_path=str(MODEL_DIR / "temporal_transformer_config.json"),
            factory=self._factory_transformer,
            version="1.0",
        )

        # 3. GNN (TaxFraudGAT)
        self._registered["gnn"] = ModelMeta(
            name="gnn",
            model_path=str(MODEL_DIR / "gat_model.pt"),
            config_path=str(MODEL_DIR / "gat_config.json"),
            factory=self._factory_gnn,
            version="1.0",
        )

        # 4. HeteroGNN
        self._registered["hetero_gnn"] = ModelMeta(
            name="hetero_gnn",
            model_path=str(MODEL_DIR / "hetero_gnn.pt"),
            config_path=str(MODEL_DIR / "hetero_gnn_config.json"),
            factory=self._factory_hetero_gnn,
            version="1.0",
        )

    # ─── Model Factories ─────────────────────────────────────────

    @staticmethod
    def _factory_vae(config: dict) -> nn.Module:
        from ml_engine.vae_anomaly import TransactionVAE
        return TransactionVAE(
            input_dim=config.get("input_dim", 16),
            latent_dim=config.get("latent_dim", 8),
        )

    @staticmethod
    def _factory_transformer(config: dict) -> nn.Module:
        from ml_engine.temporal_transformer import DelinquencyTransformer
        return DelinquencyTransformer(
            feature_dim=config.get("feature_dim", 8),
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 3),
        )

    @staticmethod
    def _factory_gnn(config: dict) -> nn.Module:
        from ml_engine.gnn_model import TaxFraudGAT
        return TaxFraudGAT(
            node_feat_dim=config.get("node_feat_dim", 22),
            edge_feat_dim=config.get("edge_feat_dim", 13),
        )

    @staticmethod
    def _factory_hetero_gnn(config: dict) -> nn.Module:
        from ml_engine.hetero_gnn_model import HeteroTaxGNN
        return HeteroTaxGNN(
            node_feat_dim=config.get("node_feat_dim", 22),
            edge_feat_dim=config.get("edge_feat_dim", 13),
        )

    # ─── Core API ─────────────────────────────────────────────────

    def get_model(
        self,
        model_name: str,
        *,
        config: dict | None = None,
    ) -> nn.Module | None:
        """
        Get a model by name. Loads from disk if not cached.
        
        Args:
            model_name: Registered model name (e.g., 'vae', 'transformer', 'gnn')
            config: Optional config override (otherwise loaded from config_path)
            
        Returns:
            nn.Module in eval() mode, or None if model file not found.
        """
        with self._lock:
            # Cache hit
            if model_name in self._models:
                meta = self._meta[model_name]
                meta.last_accessed = time.time()
                meta.access_count += 1
                self._update_redis_access(model_name, meta)
                logger.debug(
                    "[ModelServing] Cache HIT: %s (access #%d)",
                    model_name, meta.access_count,
                )
                return self._models[model_name]

            # Cache miss — load model
            return self._load_model_locked(model_name, config)

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in memory."""
        return model_name in self._models

    def get_status(self) -> dict[str, Any]:
        """Get status of all registered models."""
        with self._lock:
            status = {
                "max_cached": self.MAX_CACHED_MODELS,
                "currently_loaded": len(self._models),
                "registered": len(self._registered),
                "models": {},
            }
            for name, reg in self._registered.items():
                meta = self._meta.get(name)
                model_exists = Path(reg.model_path).exists()
                status["models"][name] = {
                    "registered": True,
                    "file_exists": model_exists,
                    "loaded": name in self._models,
                    "version": reg.version,
                    "load_count": meta.load_count if meta else 0,
                    "access_count": meta.access_count if meta else 0,
                    "last_accessed": meta.last_accessed if meta else None,
                    "memory_bytes": meta.memory_bytes if meta else 0,
                }
            return status

    def unload_model(self, model_name: str) -> bool:
        """Explicitly unload a model from memory."""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                logger.info("[ModelServing] Unloaded model: %s", model_name)
                self._update_redis_unload(model_name)
                return True
            return False

    def unload_all(self):
        """Unload all models from memory."""
        with self._lock:
            names = list(self._models.keys())
            self._models.clear()
            logger.info("[ModelServing] Unloaded all models: %s", names)

    def preload(self, model_names: list[str] | None = None):
        """Pre-load models at startup for reduced first-request latency."""
        targets = model_names or list(self._registered.keys())
        loaded = []
        for name in targets:
            reg = self._registered.get(name)
            if reg and Path(reg.model_path).exists():
                model = self.get_model(name)
                if model is not None:
                    loaded.append(name)
        logger.info("[ModelServing] Pre-loaded %d models: %s", len(loaded), loaded)

    # ─── Internal Loading Logic ───────────────────────────────────

    def _load_model_locked(
        self,
        model_name: str,
        config_override: dict | None = None,
    ) -> nn.Module | None:
        """Load a model (must be called with self._lock held)."""
        reg = self._registered.get(model_name)
        if reg is None:
            logger.error("[ModelServing] Unknown model: %s", model_name)
            return None

        model_path = Path(reg.model_path)
        if not model_path.exists():
            logger.warning(
                "[ModelServing] Model file not found: %s (%s)",
                model_name, model_path,
            )
            return None

        # Evict LRU if at capacity
        if len(self._models) >= self.MAX_CACHED_MODELS:
            self._evict_lru()

        # Load config
        config = config_override or {}
        if not config and reg.config_path and Path(reg.config_path).exists():
            try:
                with open(reg.config_path) as f:
                    config = json.load(f)
            except Exception as exc:
                logger.warning(
                    "[ModelServing] Config load failed for %s: %s",
                    model_name, exc,
                )

        # Create model via factory
        t0 = time.perf_counter()
        try:
            model = reg.factory(config)
            state_dict = torch.load(
                str(model_path),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as exc:
            logger.error(
                "[ModelServing] Failed to load model %s: %s",
                model_name, exc,
            )
            return None

        load_time_ms = (time.perf_counter() - t0) * 1000.0

        # Estimate memory usage
        memory_bytes = sum(
            p.nelement() * p.element_size()
            for p in model.parameters()
        )

        # Update metadata
        now = time.time()
        meta = self._meta.get(model_name, ModelMeta(
            name=model_name,
            model_path=str(model_path),
            config_path=reg.config_path,
        ))
        meta.load_count += 1
        meta.last_loaded = now
        meta.last_accessed = now
        meta.access_count += 1
        meta.memory_bytes = memory_bytes
        self._meta[model_name] = meta

        # Cache model
        self._models[model_name] = model

        # Report to Redis
        self._update_redis_load(model_name, meta, load_time_ms)

        logger.info(
            "[ModelServing] ✓ Loaded %s in %.1fms (params=%s, memory=%.1fKB, cached=%d/%d)",
            model_name,
            load_time_ms,
            sum(p.nelement() for p in model.parameters()),
            memory_bytes / 1024.0,
            len(self._models),
            self.MAX_CACHED_MODELS,
        )

        return model

    def _evict_lru(self):
        """Evict the least-recently-used model from cache."""
        if not self._meta:
            return

        # Find LRU among currently loaded models
        loaded_meta = {
            name: meta
            for name, meta in self._meta.items()
            if name in self._models
        }
        if not loaded_meta:
            return

        lru_name = min(loaded_meta, key=lambda n: loaded_meta[n].last_accessed)
        lru_meta = loaded_meta[lru_name]

        del self._models[lru_name]
        self._update_redis_unload(lru_name)

        logger.info(
            "[ModelServing] Evicted LRU model: %s (last_accessed=%.0fs ago, access_count=%d)",
            lru_name,
            time.time() - lru_meta.last_accessed,
            lru_meta.access_count,
        )

    # ─── Redis Tracking ───────────────────────────────────────────

    def _update_redis_load(self, model_name: str, meta: ModelMeta, load_time_ms: float):
        r = _get_redis()
        if r is None:
            return
        try:
            key = f"{self.REDIS_KEY_PREFIX}{model_name}"
            r.hset(key, mapping={
                "status": "loaded",
                "load_count": meta.load_count,
                "last_loaded": meta.last_loaded,
                "load_time_ms": round(load_time_ms, 1),
                "memory_bytes": meta.memory_bytes,
                "access_count": meta.access_count,
                "version": meta.version,
            })
            r.expire(key, 86400)  # TTL: 24h
        except Exception as exc:
            logger.debug("[ModelServing] Redis update failed: %s", exc)

    def _update_redis_access(self, model_name: str, meta: ModelMeta):
        r = _get_redis()
        if r is None:
            return
        try:
            key = f"{self.REDIS_KEY_PREFIX}{model_name}"
            r.hset(key, mapping={
                "last_accessed": meta.last_accessed,
                "access_count": meta.access_count,
            })
        except Exception:
            pass

    def _update_redis_unload(self, model_name: str):
        r = _get_redis()
        if r is None:
            return
        try:
            key = f"{self.REDIS_KEY_PREFIX}{model_name}"
            r.hset(key, "status", "evicted")
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
#  Convenience function for use across the codebase
# ════════════════════════════════════════════════════════════════

def get_model_gateway() -> ModelServingGateway:
    """Get the singleton ModelServingGateway instance."""
    return ModelServingGateway.instance()
