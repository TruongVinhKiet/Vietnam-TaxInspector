from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _ensure_schema(db) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS graph_snapshots (
            id SERIAL PRIMARY KEY,
            snapshot_id VARCHAR(120) NOT NULL UNIQUE,
            graph_family VARCHAR(80) NOT NULL,
            as_of_timestamp TIMESTAMP NOT NULL,
            extraction_policy_json JSONB,
            seed_set_json JSONB,
            source_tables_json JSONB,
            resolution_ruleset_version VARCHAR(80),
            lineage_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS graph_snapshot_nodes (
            id SERIAL PRIMARY KEY,
            snapshot_id VARCHAR(120) NOT NULL,
            node_id VARCHAR(120) NOT NULL,
            node_type VARCHAR(40) NOT NULL,
            feature_payload JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS graph_snapshot_edges (
            id SERIAL PRIMARY KEY,
            snapshot_id VARCHAR(120) NOT NULL,
            edge_id VARCHAR(120) NOT NULL,
            edge_type VARCHAR(40) NOT NULL,
            src_node_id VARCHAR(120) NOT NULL,
            dst_node_id VARCHAR(120) NOT NULL,
            attributes_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    for sql in statements:
        db.execute(text(sql))
    db.commit()


def build_snapshot(as_of_timestamp: datetime, graph_family: str = "osint_heterograph") -> dict[str, Any]:
    with SessionLocal() as db:
        _ensure_schema(db)
        registry = ModelRegistryService(db)

        nodes = db.execute(
            text(
                """
                SELECT
                    gn.node_id,
                    gn.node_type,
                    gn.risk_score,
                    gn.country_code,
                    gn.attributes_json
                FROM graph_nodes gn
                WHERE EXISTS (
                    SELECT 1
                    FROM graph_node_versions gnv
                    WHERE gnv.node_id = gn.node_id
                      AND gnv.valid_from <= :as_of_ts
                      AND (gnv.valid_to IS NULL OR gnv.valid_to > :as_of_ts)
                )
                """
            ),
            {"as_of_ts": as_of_timestamp},
        ).mappings().all()

        edges = db.execute(
            text(
                """
                SELECT edge_id, edge_type, src_node_id, dst_node_id, attributes_json
                FROM graph_edges
                WHERE COALESCE(valid_from, observed_at, TIMESTAMP '1900-01-01') <= :as_of_ts
                  AND (valid_to IS NULL OR valid_to > :as_of_ts)
                """
            ),
            {"as_of_ts": as_of_timestamp},
        ).mappings().all()

        seed_set = {
            "node_types": sorted({str(row["node_type"]) for row in nodes}),
            "edge_types": sorted({str(row["edge_type"]) for row in edges}),
        }
        source_tables = [
            "graph_nodes",
            "graph_edges",
            "graph_node_versions",
            "graph_labels",
        ]
        extraction_policy = {
            "include_node_if_valid": True,
            "include_edge_if_valid": True,
            "point_in_time": as_of_timestamp.isoformat(),
            "require_versioned_nodes": True,
        }
        lineage_hash = _hash_payload(
            {
                "graph_family": graph_family,
                "as_of_timestamp": as_of_timestamp.isoformat(),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "seed_set": seed_set,
            }
        )
        snapshot_id = f"{graph_family}:{as_of_timestamp.strftime('%Y%m%dT%H%M%S')}"

        db.execute(text("DELETE FROM graph_snapshot_nodes WHERE snapshot_id = :snapshot_id"), {"snapshot_id": snapshot_id})
        db.execute(text("DELETE FROM graph_snapshot_edges WHERE snapshot_id = :snapshot_id"), {"snapshot_id": snapshot_id})
        db.execute(text("DELETE FROM graph_snapshots WHERE snapshot_id = :snapshot_id"), {"snapshot_id": snapshot_id})
        db.execute(
            text(
                """
                INSERT INTO graph_snapshots (
                    snapshot_id, graph_family, as_of_timestamp, extraction_policy_json, seed_set_json,
                    source_tables_json, resolution_ruleset_version, lineage_hash
                )
                VALUES (
                    :snapshot_id, :graph_family, :as_of_timestamp, CAST(:extraction_policy_json AS jsonb),
                    CAST(:seed_set_json AS jsonb), CAST(:source_tables_json AS jsonb),
                    :resolution_ruleset_version, :lineage_hash
                )
                """
            ),
            {
                "snapshot_id": snapshot_id,
                "graph_family": graph_family,
                "as_of_timestamp": as_of_timestamp,
                "extraction_policy_json": json.dumps(extraction_policy, default=str),
                "seed_set_json": json.dumps(seed_set, default=str),
                "source_tables_json": json.dumps(source_tables, default=str),
                "resolution_ruleset_version": "osint-readiness-v1",
                "lineage_hash": lineage_hash,
            },
        )
        for row in nodes:
            feature_payload = {
                "risk_score": float(row.get("risk_score") or 0.0),
                "country_code": row.get("country_code"),
                "attributes": row.get("attributes_json") or {},
            }
            db.execute(
                text(
                    """
                    INSERT INTO graph_snapshot_nodes (snapshot_id, node_id, node_type, feature_payload)
                    VALUES (:snapshot_id, :node_id, :node_type, CAST(:feature_payload AS jsonb))
                    """
                ),
                {
                    "snapshot_id": snapshot_id,
                    "node_id": row["node_id"],
                    "node_type": row["node_type"],
                    "feature_payload": json.dumps(feature_payload, default=str),
                },
            )
        for row in edges:
            db.execute(
                text(
                    """
                    INSERT INTO graph_snapshot_edges (snapshot_id, edge_id, edge_type, src_node_id, dst_node_id, attributes_json)
                    VALUES (:snapshot_id, :edge_id, :edge_type, :src_node_id, :dst_node_id, CAST(:attributes_json AS jsonb))
                    """
                ),
                {
                    "snapshot_id": snapshot_id,
                    "edge_id": row["edge_id"],
                    "edge_type": row["edge_type"],
                    "src_node_id": row["src_node_id"],
                    "dst_node_id": row["dst_node_id"],
                    "attributes_json": json.dumps(row.get("attributes_json") or {}, default=str),
                },
            )
        db.commit()

        dataset_version = f"{graph_family}-snapshot-{as_of_timestamp.strftime('%Y%m%dT%H%M%S')}"
        dataset_version_id = registry.register_dataset_version(
            dataset_key="osint_graph_snapshot",
            dataset_version=dataset_version,
            entity_type="graph_snapshot",
            row_count=len(nodes) + len(edges),
            source_tables=source_tables,
            filters={"as_of_timestamp": as_of_timestamp.isoformat()},
            data_hash=lineage_hash,
            created_by="build_osint_graph_snapshot",
        )
        return {
            "snapshot_id": snapshot_id,
            "dataset_version": dataset_version,
            "dataset_version_id": dataset_version_id,
            "graph_family": graph_family,
            "as_of_timestamp": as_of_timestamp.isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "seed_set": seed_set,
            "lineage_hash": lineage_hash,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"))
    parser.add_argument("--graph-family", default="osint_heterograph")
    args = parser.parse_args()
    as_of_timestamp = datetime.fromisoformat(args.as_of)
    result = build_snapshot(as_of_timestamp=as_of_timestamp, graph_family=args.graph_family)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
