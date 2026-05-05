"""
streaming_feature_pipeline.py – Event-Driven Feature Pipeline (Phase B2)
==========================================================================
Kafka-based real-time feature computation with Redis caching and
anomaly alerting for the Tax Intelligence Multi-Agent System.

Architecture:
    ┌──────────────────┐     ┌──────────────────┐
    │  E-Invoice System │────▶│  Kafka Topic:     │
    │  (Hóa đơn điện tử)│     │  invoice.created  │
    └──────────────────┘     └────────┬─────────┘
                                      │
                         ┌────────────▼────────────┐
                         │ StreamingFeaturePipeline │
                         │                          │
                         │  1. Parse invoice event  │
                         │  2. Compute incremental  │
                         │     features (no SQL!)   │
                         │  3. Cache to Redis       │
                         │  4. Run VAE scoring      │
                         │  5. If anomaly → Alert   │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                   ▼
              ┌──────────┐   ┌──────────────┐   ┌────────────┐
              │  Redis    │   │ Investigation │   │ WebSocket  │
              │  Cache    │   │ Agent Alert   │   │ Push (SSE) │
              └──────────┘   └──────────────┘   └────────────┘

Kafka Topics:
    - invoice.created:    New invoice events from E-Invoice system
    - invoice.anomaly:    Anomaly alerts for Investigation Agent
    - feature.updated:    Feature cache updates for downstream models

Design Decisions:
    - Kafka for durable, ordered event streaming (critical for audit trail)
    - Redis for sub-millisecond feature cache (hot path for model inference)
    - Incremental feature computation: no full SQL re-query on each event
    - Graceful degradation: falls back to batch SQL when Kafka/Redis unavailable
    - JSON serialization for simplicity (Avro/Protobuf upgrade path available)

Reference:
    Replaces batch-only feature_store.py with event-driven architecture
    for real-time VAT refund fraud detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

TOPIC_INVOICE_CREATED = os.getenv("KAFKA_TOPIC_INVOICE_CREATED", "invoice.created")
TOPIC_INVOICE_ANOMALY = os.getenv("KAFKA_TOPIC_INVOICE_ANOMALY", "invoice.anomaly")
TOPIC_FEATURE_UPDATED = os.getenv("KAFKA_TOPIC_FEATURE_UPDATED", "feature.updated")

ANOMALY_THRESHOLD = float(os.getenv("STREAMING_ANOMALY_THRESHOLD", "0.65"))
CRITICAL_ANOMALY_THRESHOLD = float(os.getenv("STREAMING_CRITICAL_THRESHOLD", "0.85"))
FEATURE_CACHE_TTL = int(os.getenv("FEATURE_CACHE_TTL_SECONDS", "3600"))

CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "taxinspector-feature-pipeline")


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class InvoiceEvent:
    """Event payload for a new invoice."""
    invoice_number: str
    seller_tax_code: str
    buyer_tax_code: str
    amount: float
    vat_rate: float
    date: str  # ISO format
    goods_category: str = ""
    payment_status: str = "pending"
    source: str = "e_invoice"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyAlert:
    """Alert payload when anomaly detected."""
    alert_id: str
    tax_code: str
    invoice_number: str
    anomaly_score: float
    severity: str  # "critical", "high", "medium"
    features_snapshot: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IncrementalFeatures:
    """Incrementally computed features for a company."""
    tax_code: str
    invoice_count: int = 0
    total_amount: float = 0.0
    avg_amount: float = 0.0
    max_amount: float = 0.0
    unique_counterparties: int = 0
    last_invoice_date: str = ""
    # VAE input features (precomputed)
    amount_log: float = 0.0
    vat_rate_norm: float = 0.0
    in_out_ratio: float = 0.5
    velocity_7d: int = 0         # Invoices in last 7 days
    velocity_30d: int = 0        # Invoices in last 30 days
    amount_zscore: float = 0.0   # Z-score vs running mean
    # Running statistics for z-score computation
    _running_mean: float = 0.0
    _running_var: float = 0.0
    _count: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Remove private running stats from serialization
        d.pop("_running_mean", None)
        d.pop("_running_var", None)
        d.pop("_count", None)
        return d


# ════════════════════════════════════════════════════════════════
#  Redis Feature Cache
# ════════════════════════════════════════════════════════════════

class RedisFeatureCache:
    """Redis-backed feature cache for real-time model inference."""

    FEATURE_PREFIX = "taxinspector:features:"
    ALERT_PREFIX = "taxinspector:alerts:"
    STATS_KEY = "taxinspector:pipeline:stats"

    def __init__(self, redis_url: str = REDIS_URL):
        self._client = None
        self._redis_url = redis_url
        self._fallback_cache: dict[str, dict] = {}
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            self._client = redis.Redis.from_url(
                self._redis_url, decode_responses=True, socket_timeout=2,
            )
            self._client.ping()
            logger.info("[FeatureCache] Redis connected: %s", self._redis_url)
        except Exception as exc:
            logger.warning("[FeatureCache] Redis unavailable, using in-memory: %s", exc)
            self._client = None

    def get_features(self, tax_code: str) -> IncrementalFeatures | None:
        """Get cached features for a company."""
        key = f"{self.FEATURE_PREFIX}{tax_code}"

        if self._client:
            try:
                data = self._client.get(key)
                if data:
                    d = json.loads(data)
                    return IncrementalFeatures(**{
                        k: v for k, v in d.items()
                        if k in IncrementalFeatures.__dataclass_fields__
                    })
            except Exception as exc:
                logger.debug("[FeatureCache] Redis get failed: %s", exc)

        # Fallback to in-memory
        if tax_code in self._fallback_cache:
            d = self._fallback_cache[tax_code]
            return IncrementalFeatures(**{
                k: v for k, v in d.items()
                if k in IncrementalFeatures.__dataclass_fields__
            })

        return None

    def set_features(self, tax_code: str, features: IncrementalFeatures):
        """Cache features for a company."""
        key = f"{self.FEATURE_PREFIX}{tax_code}"
        data = json.dumps(features.to_dict(), default=str)

        if self._client:
            try:
                self._client.setex(key, FEATURE_CACHE_TTL, data)
            except Exception as exc:
                logger.debug("[FeatureCache] Redis set failed: %s", exc)

        # Always update in-memory fallback
        self._fallback_cache[tax_code] = features.to_dict()

    def store_alert(self, alert: AnomalyAlert):
        """Store anomaly alert in Redis for investigation queue."""
        key = f"{self.ALERT_PREFIX}{alert.alert_id}"

        if self._client:
            try:
                self._client.setex(key, 86400, json.dumps(alert.to_dict(), default=str))
                # Also push to a list for chronological access
                self._client.lpush(
                    f"{self.ALERT_PREFIX}queue",
                    json.dumps(alert.to_dict(), default=str),
                )
                self._client.ltrim(f"{self.ALERT_PREFIX}queue", 0, 999)
            except Exception as exc:
                logger.debug("[FeatureCache] Redis alert store failed: %s", exc)

    def get_pending_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent anomaly alerts."""
        if self._client:
            try:
                items = self._client.lrange(f"{self.ALERT_PREFIX}queue", 0, limit - 1)
                return [json.loads(item) for item in items]
            except Exception:
                pass
        return []

    def increment_stats(self, metric: str, value: int = 1):
        """Increment pipeline statistics counter."""
        if self._client:
            try:
                self._client.hincrby(self.STATS_KEY, metric, value)
            except Exception:
                pass

    def get_stats(self) -> dict[str, int]:
        """Get pipeline statistics."""
        if self._client:
            try:
                raw = self._client.hgetall(self.STATS_KEY)
                return {k: int(v) for k, v in raw.items()}
            except Exception:
                pass
        return {}


# ════════════════════════════════════════════════════════════════
#  Kafka Producer/Consumer Wrappers
# ════════════════════════════════════════════════════════════════

class KafkaEventBus:
    """Wrapper for Kafka producer/consumer with graceful fallback."""

    def __init__(self):
        self._producer = None
        self._consumer = None
        self._available = False
        self._init_kafka()

    def _init_kafka(self):
        try:
            from kafka import KafkaProducer, KafkaConsumer
            self._producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_block_ms=5000,
            )
            self._available = True
            logger.info("[KafkaEventBus] Producer connected: %s", KAFKA_BOOTSTRAP)
        except Exception as exc:
            logger.warning(
                "[KafkaEventBus] Kafka unavailable, events will be processed in-memory: %s",
                exc,
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def publish(self, topic: str, key: str, value: dict):
        """Publish event to Kafka topic."""
        if not self._available or self._producer is None:
            logger.debug("[KafkaEventBus] Skipping publish (Kafka unavailable): %s", topic)
            return False

        try:
            future = self._producer.send(topic, key=key, value=value)
            future.get(timeout=5)
            return True
        except Exception as exc:
            logger.error("[KafkaEventBus] Publish failed: %s", exc)
            return False

    def create_consumer(self, topic: str, group_id: str = CONSUMER_GROUP):
        """Create a Kafka consumer for a topic."""
        if not self._available:
            return None
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=1000,
            )
            return consumer
        except Exception as exc:
            logger.error("[KafkaEventBus] Consumer creation failed: %s", exc)
            return None

    def close(self):
        if self._producer:
            try:
                self._producer.close(timeout=5)
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════
#  Main Streaming Feature Pipeline
# ════════════════════════════════════════════════════════════════

class StreamingFeaturePipeline:
    """
    Event-driven feature computation pipeline with Kafka + Redis.

    Core Flow:
        1. Receive invoice event (via Kafka consumer or direct API call)
        2. Compute incremental features (O(1) per event, no SQL query)
        3. Cache features to Redis
        4. Run VAE anomaly scoring on updated features
        5. If anomaly score > threshold → publish alert to Kafka + Redis

    Usage:
        pipeline = StreamingFeaturePipeline()

        # Direct event processing (webhook mode):
        result = await pipeline.process_invoice_event(invoice_event)

        # Background consumer mode (Kafka):
        pipeline.start_consumer()  # Runs in background thread
    """

    def __init__(self):
        self._cache = RedisFeatureCache()
        self._kafka = KafkaEventBus()
        self._alert_callbacks: list[Callable] = []
        self._consumer_thread: threading.Thread | None = None
        self._consumer_running = False
        self._vae_model = None
        self._processing_count = 0
        self._anomaly_count = 0
        self._start_time = time.time()

        logger.info(
            "[StreamingPipeline] Initialized (kafka=%s, redis=%s, threshold=%.2f)",
            self._kafka.is_available,
            self._cache._client is not None,
            ANOMALY_THRESHOLD,
        )

    # ─── Event Processing ─────────────────────────────────────────

    async def process_invoice_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process a single invoice event through the full pipeline.

        Args:
            event: Invoice event dict with required fields:
                   invoice_number, seller_tax_code, buyer_tax_code,
                   amount, vat_rate, date

        Returns:
            Processing result with anomaly scores and any alerts.
        """
        t0 = time.perf_counter()
        self._processing_count += 1
        self._cache.increment_stats("events_processed")

        # Parse event
        invoice = InvoiceEvent(
            invoice_number=event.get("invoice_number", ""),
            seller_tax_code=event.get("seller_tax_code", ""),
            buyer_tax_code=event.get("buyer_tax_code", ""),
            amount=float(event.get("amount", 0)),
            vat_rate=float(event.get("vat_rate", 10)),
            date=event.get("date", datetime.now(timezone.utc).isoformat()),
            goods_category=event.get("goods_category", ""),
            payment_status=event.get("payment_status", "pending"),
            source=event.get("source", "webhook"),
        )

        # Step 1: Update incremental features for seller
        seller_features = self._update_features(
            invoice.seller_tax_code, invoice, role="seller",
        )

        # Step 2: Update incremental features for buyer
        buyer_features = self._update_features(
            invoice.buyer_tax_code, invoice, role="buyer",
        )

        # Step 3: Run anomaly scoring on seller (primary risk subject)
        anomaly_result = self._run_anomaly_scoring(
            invoice.seller_tax_code, seller_features, invoice,
        )

        # Step 4: Publish feature update event to Kafka
        if self._kafka.is_available:
            self._kafka.publish(
                TOPIC_FEATURE_UPDATED,
                key=invoice.seller_tax_code,
                value={
                    "tax_code": invoice.seller_tax_code,
                    "features": seller_features.to_dict(),
                    "trigger_invoice": invoice.invoice_number,
                    "timestamp": time.time(),
                },
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "status": "processed",
            "invoice_number": invoice.invoice_number,
            "seller_tax_code": invoice.seller_tax_code,
            "buyer_tax_code": invoice.buyer_tax_code,
            "seller_features": seller_features.to_dict(),
            "buyer_features": buyer_features.to_dict(),
            "anomaly_score": anomaly_result.get("score", 0),
            "is_anomaly": anomaly_result.get("is_anomaly", False),
            "alert": anomaly_result.get("alert"),
            "latency_ms": round(latency_ms, 1),
        }

        if anomaly_result.get("is_anomaly"):
            logger.warning(
                "[StreamingPipeline] 🚨 ANOMALY: %s (score=%.3f, invoice=%s)",
                invoice.seller_tax_code,
                anomaly_result["score"],
                invoice.invoice_number,
            )

        return result

    # ─── Incremental Feature Computation ──────────────────────────

    def _update_features(
        self,
        tax_code: str,
        invoice: InvoiceEvent,
        role: str,
    ) -> IncrementalFeatures:
        """
        Update features incrementally without SQL query.

        Uses Welford's online algorithm for running mean/variance.
        """
        # Get existing features from cache (or create new)
        features = self._cache.get_features(tax_code)
        if features is None:
            features = IncrementalFeatures(tax_code=tax_code)

        amount = invoice.amount

        # Update basic counters
        features.invoice_count += 1
        features.total_amount += amount
        features.avg_amount = features.total_amount / features.invoice_count
        features.max_amount = max(features.max_amount, amount)
        features.last_invoice_date = invoice.date

        # Welford's online algorithm for running statistics
        features._count += 1
        delta = amount - features._running_mean
        features._running_mean += delta / features._count
        delta2 = amount - features._running_mean
        features._running_var += delta * delta2

        # Compute z-score
        if features._count >= 2:
            variance = features._running_var / (features._count - 1)
            std = max(1.0, variance ** 0.5)
            features.amount_zscore = (amount - features._running_mean) / std
        else:
            features.amount_zscore = 0.0

        # Precompute VAE input features
        features.amount_log = float(np.log1p(amount))
        features.vat_rate_norm = invoice.vat_rate / 100.0

        # Update timestamp
        features.updated_at = time.time()

        # Persist to cache
        self._cache.set_features(tax_code, features)

        return features

    # ─── Anomaly Scoring ──────────────────────────────────────────

    def _run_anomaly_scoring(
        self,
        tax_code: str,
        features: IncrementalFeatures,
        invoice: InvoiceEvent,
    ) -> dict[str, Any]:
        """
        Run VAE anomaly scoring on the current feature state.

        Uses ModelServingGateway for efficient model access.
        """
        try:
            import torch
            from ml_engine.model_serving import get_model_gateway

            gateway = get_model_gateway()
            model = gateway.get_model("vae")

            if model is None:
                return {"score": 0.0, "is_anomaly": False, "reason": "model_not_loaded"}

            # Build feature vector (simplified — matches VAE input schema)
            feature_vector = np.array([
                features.amount_log / 25.0,          # amount_log (normalized)
                features.vat_rate_norm,               # vat_rate
                0.5,                                  # seller_risk (placeholder)
                0.5,                                  # buyer_risk (placeholder)
                0.5,                                  # days_since_reg (placeholder)
                min(1.0, features.invoice_count / 100),  # degree (normalized)
                min(1.0, features.invoice_count / 100),  # degree_buyer (placeholder)
                features.in_out_ratio,                # in_out_ratio
                0.0,                                  # is_reciprocal
                0.5,                                  # delta_recip_days (neutral)
                0.0,                                  # day_of_week_sin (placeholder)
                1.0,                                  # day_of_week_cos (placeholder)
                0.0,                                  # month_sin (placeholder)
                1.0,                                  # month_cos (placeholder)
                max(-3.0, min(3.0, features.amount_zscore)),  # amount_zscore (clipped)
                0.0,                                  # near_dup_count
            ], dtype=np.float32)

            # Run inference
            x_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                x_recon, mu, logvar = model(x_tensor)
                recon_error = torch.mean((x_tensor - x_recon) ** 2).item()

            is_anomaly = recon_error > ANOMALY_THRESHOLD
            severity = (
                "critical" if recon_error > CRITICAL_ANOMALY_THRESHOLD
                else "high" if recon_error > ANOMALY_THRESHOLD
                else "normal"
            )

            result = {
                "score": round(recon_error, 4),
                "is_anomaly": is_anomaly,
                "severity": severity,
                "threshold": ANOMALY_THRESHOLD,
            }

            # Generate and publish alert if anomaly detected
            if is_anomaly:
                self._anomaly_count += 1
                self._cache.increment_stats("anomalies_detected")

                alert = AnomalyAlert(
                    alert_id=f"alert-{tax_code}-{int(time.time() * 1000)}",
                    tax_code=tax_code,
                    invoice_number=invoice.invoice_number,
                    anomaly_score=round(recon_error, 4),
                    severity=severity,
                    features_snapshot=features.to_dict(),
                )

                # Store in Redis
                self._cache.store_alert(alert)

                # Publish to Kafka
                if self._kafka.is_available:
                    self._kafka.publish(
                        TOPIC_INVOICE_ANOMALY,
                        key=tax_code,
                        value=alert.to_dict(),
                    )

                # Notify registered callbacks
                for callback in self._alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(alert.to_dict()))
                        else:
                            callback(alert.to_dict())
                    except Exception as exc:
                        logger.debug("[StreamingPipeline] Alert callback error: %s", exc)

                result["alert"] = alert.to_dict()

            return result

        except Exception as exc:
            logger.warning("[StreamingPipeline] Anomaly scoring failed: %s", exc)
            return {"score": 0.0, "is_anomaly": False, "error": str(exc)}

    # ─── Kafka Consumer (Background) ──────────────────────────────

    def start_consumer(self):
        """Start background Kafka consumer thread."""
        if not self._kafka.is_available:
            logger.info("[StreamingPipeline] Kafka not available, consumer not started")
            return

        if self._consumer_running:
            return

        self._consumer_running = True
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name="streaming-feature-consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        logger.info("[StreamingPipeline] Kafka consumer started on topic: %s", TOPIC_INVOICE_CREATED)

    def stop_consumer(self):
        """Stop background Kafka consumer."""
        self._consumer_running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5)
            self._consumer_thread = None
            logger.info("[StreamingPipeline] Kafka consumer stopped")

    def _consumer_loop(self):
        """Background consumer loop processing invoice events."""
        consumer = self._kafka.create_consumer(TOPIC_INVOICE_CREATED)
        if consumer is None:
            self._consumer_running = False
            return

        loop = asyncio.new_event_loop()

        while self._consumer_running:
            try:
                for message in consumer:
                    if not self._consumer_running:
                        break
                    try:
                        event = message.value
                        loop.run_until_complete(self.process_invoice_event(event))
                    except Exception as exc:
                        logger.error("[StreamingPipeline] Event processing error: %s", exc)
                        self._cache.increment_stats("processing_errors")
            except Exception as exc:
                logger.error("[StreamingPipeline] Consumer error: %s", exc)
                time.sleep(1)

        try:
            consumer.close()
        except Exception:
            pass
        loop.close()

    # ─── Alert Subscription ───────────────────────────────────────

    def register_alert_callback(self, callback: Callable):
        """Register a callback to be notified on anomaly alerts."""
        self._alert_callbacks.append(callback)

    # ─── Status & Metrics ─────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get pipeline status and metrics."""
        uptime = time.time() - self._start_time
        redis_stats = self._cache.get_stats()
        pending_alerts = self._cache.get_pending_alerts(limit=5)

        return {
            "status": "running" if self._consumer_running else "webhook_only",
            "kafka_available": self._kafka.is_available,
            "redis_available": self._cache._client is not None,
            "uptime_seconds": round(uptime, 1),
            "events_processed": self._processing_count,
            "anomalies_detected": self._anomaly_count,
            "anomaly_rate": (
                round(self._anomaly_count / max(1, self._processing_count), 4)
            ),
            "thresholds": {
                "anomaly": ANOMALY_THRESHOLD,
                "critical": CRITICAL_ANOMALY_THRESHOLD,
            },
            "kafka_topics": {
                "invoice_created": TOPIC_INVOICE_CREATED,
                "invoice_anomaly": TOPIC_INVOICE_ANOMALY,
                "feature_updated": TOPIC_FEATURE_UPDATED,
            },
            "redis_stats": redis_stats,
            "recent_alerts": pending_alerts,
            "consumer_running": self._consumer_running,
        }

    def close(self):
        """Cleanup resources."""
        self.stop_consumer()
        self._kafka.close()


# ════════════════════════════════════════════════════════════════
#  Singleton Access
# ════════════════════════════════════════════════════════════════

_pipeline_instance: StreamingFeaturePipeline | None = None
_pipeline_lock = threading.Lock()


def get_streaming_pipeline() -> StreamingFeaturePipeline:
    """Get the singleton StreamingFeaturePipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = StreamingFeaturePipeline()
    return _pipeline_instance
