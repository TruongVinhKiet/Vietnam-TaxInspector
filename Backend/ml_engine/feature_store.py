from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Redis cache TTL for company features (seconds)
FEATURE_CACHE_TTL = int(os.getenv("FEATURE_CACHE_TTL_SECONDS", "1800"))
FEATURE_CACHE_PREFIX = "taxinspector:feature_store:"

# Redis client (lazy-loaded, shared with streaming pipeline)
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
        logger.info("[FeatureStore] Redis cache connected: %s", url)
    except Exception as exc:
        logger.warning("[FeatureStore] Redis unavailable, no caching: %s", exc)
        _redis_client = None
    return _redis_client


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SnapshotKey:
    entity_type: str
    entity_id: str
    as_of_date: date


class FeatureStore:
    """
    Leakage-safe point-in-time feature snapshots with Redis cache layer.
    All builders must respect: only data <= as_of_date.
    
    Redis cache: build_company_snapshot results are cached with configurable TTL.
    Falls back to direct SQL when Redis is unavailable.
    """

    def __init__(self, db: Session):
        self.db = db

    def _cache_get(self, key: str) -> Dict[str, Any] | None:
        """Try to get features from Redis cache."""
        r = _get_redis()
        if r is None:
            return None
        try:
            data = r.get(f"{FEATURE_CACHE_PREFIX}{key}")
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    def _cache_set(self, key: str, features: Dict[str, Any]):
        """Store features in Redis cache."""
        r = _get_redis()
        if r is None:
            return
        try:
            r.setex(
                f"{FEATURE_CACHE_PREFIX}{key}",
                FEATURE_CACHE_TTL,
                json.dumps(features, default=str),
            )
        except Exception:
            pass

    def ensure_feature_set(self, *, name: str, version: str, owner: str = "system", description: str = "") -> int:
        row = self.db.execute(
            text(
                "SELECT id FROM feature_sets WHERE name = :name AND version = :version"
            ),
            {"name": name, "version": version},
        ).fetchone()
        if row:
            return int(row[0])
        inserted = self.db.execute(
            text(
                "INSERT INTO feature_sets (name, version, owner, description) "
                "VALUES (:name, :version, :owner, :description) RETURNING id"
            ),
            {"name": name, "version": version, "owner": owner, "description": description},
        ).fetchone()
        self.db.commit()
        return int(inserted[0])

    def upsert_snapshot(
        self,
        *,
        feature_set_id: int,
        key: SnapshotKey,
        features: Dict[str, Any],
        source_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        source_hash = _hash_payload(source_payload or {"features": features})
        features_json = json.dumps(features, default=str)
        self.db.execute(
            text(
                "INSERT INTO feature_snapshots (entity_type, entity_id, as_of_date, feature_set_id, features_json, source_hash) "
                "VALUES (:entity_type, :entity_id, :as_of_date, :feature_set_id, CAST(:features_json AS jsonb), :source_hash) "
                "ON CONFLICT DO NOTHING"
            ),
            {
                "entity_type": key.entity_type,
                "entity_id": key.entity_id,
                "as_of_date": key.as_of_date,
                "feature_set_id": feature_set_id,
                "features_json": features_json,
                "source_hash": source_hash,
            },
        )
        self.db.commit()

    def build_company_snapshot(self, *, tax_code: str, as_of_date: date) -> Dict[str, Any]:
        # Check Redis cache first (O(1) vs 3 SQL queries)
        cache_key = f"company:{tax_code}:{as_of_date}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        company = self.db.execute(
            text(
                "SELECT tax_code, industry, risk_score, is_active, registration_date "
                "FROM companies WHERE tax_code = :tax_code"
            ),
            {"tax_code": tax_code},
        ).fetchone()
        if not company:
            return {"available": False, "reason": "company_not_found"}

        inv_stats = self.db.execute(
            text(
                "SELECT "
                "COUNT(*) AS n_inv, "
                "COALESCE(SUM(amount), 0) AS sum_amount, "
                "COALESCE(AVG(amount), 0) AS avg_amount, "
                "COUNT(DISTINCT buyer_tax_code) AS n_buyers, "
                "COUNT(DISTINCT seller_tax_code) AS n_sellers "
                "FROM invoices "
                "WHERE date <= :as_of_date AND (seller_tax_code = :tax_code OR buyer_tax_code = :tax_code)"
            ),
            {"as_of_date": as_of_date, "tax_code": tax_code},
        ).fetchone()

        payment_stats = self.db.execute(
            text(
                "SELECT "
                "COUNT(*) AS n_payments, "
                "COALESCE(SUM(amount_paid), 0) AS sum_paid, "
                "COALESCE(SUM(amount_due), 0) AS sum_due "
                "FROM tax_payments "
                "WHERE due_date <= :as_of_date AND tax_code = :tax_code"
            ),
            {"as_of_date": as_of_date, "tax_code": tax_code},
        ).fetchone()

        features = {
            "available": True,
            "tax_code": company[0],
            "industry": company[1],
            "base_risk_score": float(company[2] or 0.0),
            "is_active": bool(company[3]) if company[3] is not None else True,
            "registration_date": str(company[4]) if company[4] else None,
            "inv_count": int(inv_stats[0] or 0),
            "inv_sum_amount": float(inv_stats[1] or 0),
            "inv_avg_amount": float(inv_stats[2] or 0),
            "inv_unique_buyers": int(inv_stats[3] or 0),
            "inv_unique_sellers": int(inv_stats[4] or 0),
            "payment_count": int(payment_stats[0] or 0),
            "payment_sum_paid": float(payment_stats[1] or 0),
            "payment_sum_due": float(payment_stats[2] or 0),
        }

        # Cache the result for subsequent requests
        self._cache_set(cache_key, features)

        return features

    def build_invoice_snapshot(self, *, invoice_number: str, as_of_date: date) -> Dict[str, Any]:
        inv = self.db.execute(
            text(
                "SELECT invoice_number, seller_tax_code, buyer_tax_code, amount, vat_rate, date, goods_category, payment_status, is_adjustment "
                "FROM invoices WHERE invoice_number = :invoice_number"
            ),
            {"invoice_number": invoice_number},
        ).fetchone()
        if not inv:
            return {"available": False, "reason": "invoice_not_found"}

        # strict point-in-time: if invoice date is after as_of_date, snapshot should mark unavailable
        if inv[5] and inv[5] > as_of_date:
            return {"available": False, "reason": "invoice_after_as_of_date"}

        lifecycle = self.db.execute(
            text(
                "SELECT COUNT(*) AS n_events "
                "FROM invoice_lifecycle_events e "
                "JOIN invoices i ON i.id = e.invoice_id "
                "WHERE i.invoice_number = :invoice_number AND e.event_time::date <= :as_of_date"
            ),
            {"invoice_number": invoice_number, "as_of_date": as_of_date},
        ).fetchone()

        features = {
            "available": True,
            "invoice_number": inv[0],
            "seller_tax_code": inv[1],
            "buyer_tax_code": inv[2],
            "amount": float(inv[3] or 0),
            "vat_rate": float(inv[4] or 0),
            "date": str(inv[5]) if inv[5] else None,
            "goods_category": inv[6],
            "payment_status": inv[7],
            "is_adjustment": bool(inv[8]) if inv[8] is not None else False,
            "lifecycle_event_count": int(lifecycle[0] or 0),
        }
        return features

