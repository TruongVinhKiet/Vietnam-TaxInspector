from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


LABEL_NAME_MAP = {
    "fraud_confirmed": "subgraph_case_confirmed",
    "fraud_rejected": "osint_false_positive",
    "shell_confirmed": "shell_network_confirmed",
    "shell_company_confirmed": "shell_network_confirmed",
    "offshore_reviewed": "offshore_structure_reviewed",
    "related_party_abuse": "related_party_abuse_confirmed",
    "ubo_risk": "ubo_risk_confirmed",
}

POSITIVE_LABELS = {
    "ubo_risk_confirmed",
    "related_party_abuse_confirmed",
    "shell_network_confirmed",
    "offshore_structure_reviewed",
    "edge_suspicious_confirmed",
    "subgraph_case_confirmed",
}


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _ensure_schema(db) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS graph_labels (
            id SERIAL PRIMARY KEY,
            entity_type VARCHAR(30) NOT NULL,
            entity_id VARCHAR(120) NOT NULL,
            label_name VARCHAR(80) NOT NULL,
            label_value VARCHAR(40) NOT NULL,
            trust_tier VARCHAR(20) NOT NULL,
            label_source VARCHAR(80),
            annotator VARCHAR(120),
            confidence DOUBLE PRECISION,
            valid_from TIMESTAMP,
            valid_to TIMESTAMP,
            evidence_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_graph_labels_entity ON graph_labels (entity_type, entity_id, label_name)",
        "CREATE INDEX IF NOT EXISTS idx_graph_labels_tier ON graph_labels (trust_tier, label_name)",
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id SERIAL PRIMARY KEY,
            node_id VARCHAR(120) NOT NULL UNIQUE,
            node_type VARCHAR(40) NOT NULL,
            native_table VARCHAR(80),
            native_key VARCHAR(120),
            display_name VARCHAR(255),
            status VARCHAR(30),
            risk_score DOUBLE PRECISION,
            country_code VARCHAR(40),
            attributes_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type)",
        """
        CREATE TABLE IF NOT EXISTS graph_edges (
            id SERIAL PRIMARY KEY,
            edge_id VARCHAR(120) NOT NULL UNIQUE,
            src_node_id VARCHAR(120) NOT NULL,
            dst_node_id VARCHAR(120) NOT NULL,
            edge_type VARCHAR(40) NOT NULL,
            directed BOOLEAN NOT NULL DEFAULT TRUE,
            weight DOUBLE PRECISION,
            confidence DOUBLE PRECISION,
            valid_from TIMESTAMP,
            valid_to TIMESTAMP,
            observed_at TIMESTAMP,
            attributes_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges (edge_type)",
        """
        CREATE TABLE IF NOT EXISTS graph_edge_evidence (
            id SERIAL PRIMARY KEY,
            edge_id VARCHAR(120) NOT NULL,
            source_type VARCHAR(80),
            source_ref VARCHAR(160),
            source_url VARCHAR(400),
            observed_at TIMESTAMP,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            extractor_version VARCHAR(80),
            confidence DOUBLE PRECISION,
            raw_payload_json JSONB
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_graph_edge_evidence_edge ON graph_edge_evidence (edge_id)",
        """
        CREATE TABLE IF NOT EXISTS graph_node_versions (
            id SERIAL PRIMARY KEY,
            node_id VARCHAR(120) NOT NULL,
            valid_from TIMESTAMP NOT NULL,
            valid_to TIMESTAMP,
            attributes_json JSONB,
            source_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    for sql in statements:
        db.execute(text(sql))
    db.commit()


def _label_name(raw: str | None) -> str:
    normalized = str(raw or "").strip().lower()
    return LABEL_NAME_MAP.get(normalized, "ubo_risk_confirmed")


def _confidence_score(raw: str | None) -> float:
    mapping = {"low": 0.45, "medium": 0.7, "high": 0.9}
    return float(mapping.get(str(raw or "").strip().lower(), 0.6))


def _trust_tier(row: dict[str, Any], label_name: str) -> str:
    origin = str(row.get("label_origin") or "").lower()
    decision = str(row.get("decision") or "").lower()
    outcome = str(row.get("outcome_status") or "").lower()
    if outcome in {"recovered", "partial_recovered", "dismissed"} or decision in {"penalize", "dismiss"}:
        return "gold"
    if origin in {"manual_inspector", "case_review", "graph_investigator"}:
        return "silver" if label_name == "osint_false_positive" else "gold"
    return "silver"


def _upsert_node(db, *, node_id: str, node_type: str, native_table: str, native_key: str, display_name: str | None, status: str | None, risk_score: float | None, country_code: str | None, attributes: dict[str, Any]) -> None:
    db.execute(
        text(
            """
            INSERT INTO graph_nodes (
                node_id, node_type, native_table, native_key, display_name, status, risk_score, country_code, attributes_json
            )
            VALUES (
                :node_id, :node_type, :native_table, :native_key, :display_name, :status, :risk_score, :country_code, CAST(:attributes_json AS jsonb)
            )
            ON CONFLICT (node_id) DO UPDATE
            SET node_type = EXCLUDED.node_type,
                native_table = EXCLUDED.native_table,
                native_key = EXCLUDED.native_key,
                display_name = EXCLUDED.display_name,
                status = EXCLUDED.status,
                risk_score = EXCLUDED.risk_score,
                country_code = EXCLUDED.country_code,
                attributes_json = EXCLUDED.attributes_json,
                updated_at = CURRENT_TIMESTAMP
            """
        ),
        {
            "node_id": node_id,
            "node_type": node_type,
            "native_table": native_table,
            "native_key": native_key,
            "display_name": display_name,
            "status": status,
            "risk_score": risk_score,
            "country_code": country_code,
            "attributes_json": json.dumps(attributes, default=str),
        },
    )


def _upsert_edge(db, *, edge_id: str, src_node_id: str, dst_node_id: str, edge_type: str, directed: bool, weight: float | None, confidence: float | None, valid_from: Any, valid_to: Any, observed_at: Any, attributes: dict[str, Any], evidence: dict[str, Any]) -> None:
    db.execute(
        text(
            """
            INSERT INTO graph_edges (
                edge_id, src_node_id, dst_node_id, edge_type, directed, weight, confidence, valid_from, valid_to, observed_at, attributes_json
            )
            VALUES (
                :edge_id, :src_node_id, :dst_node_id, :edge_type, :directed, :weight, :confidence, :valid_from, :valid_to, :observed_at, CAST(:attributes_json AS jsonb)
            )
            ON CONFLICT (edge_id) DO UPDATE
            SET src_node_id = EXCLUDED.src_node_id,
                dst_node_id = EXCLUDED.dst_node_id,
                edge_type = EXCLUDED.edge_type,
                directed = EXCLUDED.directed,
                weight = EXCLUDED.weight,
                confidence = EXCLUDED.confidence,
                valid_from = EXCLUDED.valid_from,
                valid_to = EXCLUDED.valid_to,
                observed_at = EXCLUDED.observed_at,
                attributes_json = EXCLUDED.attributes_json
            """
        ),
        {
            "edge_id": edge_id,
            "src_node_id": src_node_id,
            "dst_node_id": dst_node_id,
            "edge_type": edge_type,
            "directed": directed,
            "weight": weight,
            "confidence": confidence,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "observed_at": observed_at,
            "attributes_json": json.dumps(attributes, default=str),
        },
    )
    db.execute(text("DELETE FROM graph_edge_evidence WHERE edge_id = :edge_id"), {"edge_id": edge_id})
    db.execute(
        text(
            """
            INSERT INTO graph_edge_evidence (
                edge_id, source_type, source_ref, observed_at, extractor_version, confidence, raw_payload_json
            )
            VALUES (
                :edge_id, :source_type, :source_ref, :observed_at, :extractor_version, :confidence, CAST(:raw_payload_json AS jsonb)
            )
            """
        ),
        {
            "edge_id": edge_id,
            "source_type": evidence.get("source_type"),
            "source_ref": evidence.get("source_ref"),
            "observed_at": observed_at,
            "extractor_version": evidence.get("extractor_version"),
            "confidence": confidence,
            "raw_payload_json": json.dumps(evidence, default=str),
        },
    )


def _record_node_version(db, *, node_id: str, valid_from: Any, valid_to: Any, attributes: dict[str, Any]) -> None:
    source_hash = _hash_payload({"node_id": node_id, "valid_from": valid_from, "valid_to": valid_to, "attributes": attributes})
    db.execute(
        text(
            """
            DELETE FROM graph_node_versions
            WHERE node_id = :node_id AND valid_from = :valid_from
            """
        ),
        {"node_id": node_id, "valid_from": valid_from},
    )
    db.execute(
        text(
            """
            INSERT INTO graph_node_versions (node_id, valid_from, valid_to, attributes_json, source_hash)
            VALUES (:node_id, :valid_from, :valid_to, CAST(:attributes_json AS jsonb), :source_hash)
            """
        ),
        {
            "node_id": node_id,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "attributes_json": json.dumps(attributes, default=str),
            "source_hash": source_hash,
        },
    )


def materialize_graph_labels(db, registry: ModelRegistryService) -> dict[str, Any]:
    label_rows = db.execute(
        text(
            """
            SELECT
                tax_code,
                inspector_id,
                label_type,
                confidence,
                label_origin,
                evidence_summary,
                decision,
                decision_date,
                amount_recovered,
                outcome_status,
                created_at
            FROM inspector_labels
            ORDER BY created_at DESC
            """
        )
    ).mappings().all()

    ownership_rows = db.execute(
        text(
            """
            SELECT
                id,
                parent_tax_code,
                child_tax_code,
                relationship_type,
                ownership_percent,
                verified,
                effective_date,
                end_date,
                data_source
            FROM ownership_links
            """
        )
    ).mappings().all()
    bootstrap_negative_rows = db.execute(
        text(
            """
            SELECT
                c.tax_code,
                COALESCE(oe.entity_code, c.tax_code) AS entity_code,
                COALESCE(oe.risk_score, c.risk_score, 0) AS risk_score,
                COUNT(ol.id) AS link_count
            FROM companies c
            LEFT JOIN offshore_entities oe ON oe.proxy_tax_code = c.tax_code
            LEFT JOIN ownership_links ol
                ON ol.parent_tax_code = c.tax_code
                OR ol.child_tax_code = c.tax_code
            WHERE c.tax_code IS NOT NULL
              AND (
                    c.industry = 'Offshore Entity'
                    OR COALESCE(oe.entity_code, '') <> ''
                    OR COALESCE(c.risk_score, 0) <= 20
                  )
            GROUP BY c.tax_code, COALESCE(oe.entity_code, c.tax_code), COALESCE(oe.risk_score, c.risk_score, 0)
            HAVING COALESCE(oe.risk_score, c.risk_score, 0) < 20
               AND COUNT(ol.id) <= 1
            """
        )
    ).mappings().all()

    db.execute(text("DELETE FROM graph_labels"))
    inserted = 0
    positives = 0
    negatives = 0

    for row in label_rows:
        label_name = _label_name(row.get("label_type"))
        label_value = "negative" if label_name == "osint_false_positive" else "positive"
        trust_tier = _trust_tier(dict(row), label_name)
        confidence = _confidence_score(row.get("confidence"))
        db.execute(
            text(
                """
                INSERT INTO graph_labels (
                    entity_type, entity_id, label_name, label_value, trust_tier, label_source, annotator, confidence,
                    valid_from, valid_to, evidence_json
                )
                VALUES (
                    'company', :entity_id, :label_name, :label_value, :trust_tier, :label_source, :annotator, :confidence,
                    :valid_from, NULL, CAST(:evidence_json AS jsonb)
                )
                """
            ),
            {
                "entity_id": row["tax_code"],
                "label_name": label_name,
                "label_value": label_value,
                "trust_tier": trust_tier,
                "label_source": row.get("label_origin") or "inspector_labels",
                "annotator": f"inspector:{row['inspector_id']}" if row.get("inspector_id") else "inspector:unknown",
                "confidence": confidence,
                "valid_from": row.get("decision_date") or row.get("created_at"),
                "evidence_json": json.dumps(
                    {
                        "source_table": "inspector_labels",
                        "decision": row.get("decision"),
                        "outcome_status": row.get("outcome_status"),
                        "amount_recovered": float(row.get("amount_recovered") or 0.0),
                        "evidence_summary": row.get("evidence_summary"),
                    },
                    default=str,
                ),
            },
        )
        inserted += 1
        if label_value == "positive":
            positives += 1
        else:
            negatives += 1

    for row in ownership_rows:
        suspicious = (
            bool(row.get("verified") is False)
            or float(row.get("ownership_percent") or 0.0) >= 75.0
            or str(row.get("relationship_type") or "").lower() in {"nominee", "proxy", "ultimate_beneficiary"}
        )
        if not suspicious:
            continue
        db.execute(
            text(
                """
                INSERT INTO graph_labels (
                    entity_type, entity_id, label_name, label_value, trust_tier, label_source, annotator, confidence,
                    valid_from, valid_to, evidence_json
                )
                VALUES (
                    'edge', :entity_id, 'edge_suspicious_confirmed', 'positive', 'bronze', :label_source, 'rule_engine',
                    :confidence, :valid_from, :valid_to, CAST(:evidence_json AS jsonb)
                )
                """
            ),
            {
                "entity_id": f"ownership:{row['id']}",
                "label_source": row.get("data_source") or "ownership_links",
                "confidence": 0.55 if row.get("verified") is False else 0.65,
                "valid_from": row.get("effective_date"),
                "valid_to": row.get("end_date"),
                "evidence_json": json.dumps(
                    {
                        "source_table": "ownership_links",
                        "parent_tax_code": row.get("parent_tax_code"),
                        "child_tax_code": row.get("child_tax_code"),
                        "relationship_type": row.get("relationship_type"),
                        "ownership_percent": float(row.get("ownership_percent") or 0.0),
                    },
                    default=str,
                ),
            },
        )
        inserted += 1
        positives += 1

    labeled_entities = {
        str(row["tax_code"])
        for row in label_rows
        if row.get("tax_code")
    }
    for row in bootstrap_negative_rows:
        tax_code = str(row.get("tax_code") or "")
        if not tax_code or tax_code in labeled_entities:
            continue
        db.execute(
            text(
                """
                INSERT INTO graph_labels (
                    entity_type, entity_id, label_name, label_value, trust_tier, label_source, annotator, confidence,
                    valid_from, valid_to, evidence_json
                )
                VALUES (
                    'company', :entity_id, 'osint_false_positive', 'negative', 'bronze', 'offshore_low_risk_bootstrap',
                    'rule_engine', :confidence, :valid_from, NULL, CAST(:evidence_json AS jsonb)
                )
                """
            ),
            {
                "entity_id": tax_code,
                "confidence": 0.4,
                # Backdate so temporal snapshots can see negatives for benchmarking.
                "valid_from": datetime.utcnow() - timedelta(days=365 * 5),
                "evidence_json": json.dumps(
                    {
                        "source_table": "offshore_entities",
                        "entity_code": row.get("entity_code"),
                        "risk_score": float(row.get("risk_score") or 0.0),
                        "link_count": int(row.get("link_count") or 0),
                    },
                    default=str,
                ),
            },
        )
        inserted += 1
        negatives += 1

    # ────────────────────────────────────────────────────────────
    # Label projection: edge/subgraph/offshore -> company nodes
    # ────────────────────────────────────────────────────────────
    # Goal: ensure the company node training split has both classes under temporal snapshots.
    #
    # Strategy (bronze/silver only):
    # - Offshore high-risk proxy companies -> positive
    # - Suspicious ownership edges -> positive for connected companies
    # - Cluster propagation (1-2 hops) over ownership + high-score alias edges

    backdate = datetime.utcnow() - timedelta(days=365 * 5)
    already_company_labeled = set(
        db.execute(text("SELECT entity_id FROM graph_labels WHERE entity_type = 'company'")).scalars().all()
    )

    # 1) Offshore high-risk proxy -> company positive
    offshore_proxy_rows = db.execute(
        text(
            """
            SELECT
                oe.proxy_tax_code AS tax_code,
                oe.entity_code,
                oe.country,
                COALESCE(oe.risk_score, 0) AS risk_score,
                COALESCE(oe.jurisdiction_risk_weight, 0.5) AS juris_w
            FROM offshore_entities oe
            WHERE oe.proxy_tax_code IS NOT NULL
              AND (COALESCE(oe.risk_score, 0) >= 70 OR COALESCE(oe.jurisdiction_risk_weight, 0.5) >= 0.75)
            """
        )
    ).mappings().all()
    for row in offshore_proxy_rows:
        tax_code = str(row.get("tax_code") or "")
        if not tax_code or tax_code in already_company_labeled:
            continue
        db.execute(
            text(
                """
                INSERT INTO graph_labels (
                    entity_type, entity_id, label_name, label_value, trust_tier, label_source, annotator, confidence,
                    valid_from, valid_to, evidence_json
                )
                VALUES (
                    'company', :entity_id, 'offshore_structure_reviewed', 'positive', 'bronze',
                    'offshore_entities', 'projection_engine', :confidence, :valid_from, NULL, CAST(:evidence_json AS jsonb)
                )
                """
            ),
            {
                "entity_id": tax_code,
                "confidence": 0.55,
                "valid_from": backdate,
                "evidence_json": json.dumps(
                    {
                        "projection_rule": "offshore_high_risk_proxy",
                        "entity_code": row.get("entity_code"),
                        "country": row.get("country"),
                        "risk_score": float(row.get("risk_score") or 0.0),
                        "jurisdiction_risk_weight": float(row.get("juris_w") or 0.0),
                    },
                    default=str,
                ),
            },
        )
        already_company_labeled.add(tax_code)
        inserted += 1
        positives += 1

    # 2) Ownership suspicious edges -> company positive (both endpoints)
    suspicious_links = db.execute(
        text(
            """
            SELECT id, parent_tax_code, child_tax_code, relationship_type, ownership_percent, verified, effective_date, end_date
            FROM ownership_links
            WHERE (verified IS FALSE)
               OR (COALESCE(ownership_percent, 0) >= 75.0)
               OR (LOWER(COALESCE(relationship_type, '')) IN ('nominee', 'proxy', 'ultimate_beneficiary'))
            """
        )
    ).mappings().all()
    for row in suspicious_links:
        for tc in (row.get("parent_tax_code"), row.get("child_tax_code")):
            tax_code = str(tc or "")
            if not tax_code or tax_code in already_company_labeled:
                continue
            db.execute(
                text(
                    """
                    INSERT INTO graph_labels (
                        entity_type, entity_id, label_name, label_value, trust_tier, label_source, annotator, confidence,
                        valid_from, valid_to, evidence_json
                    )
                    VALUES (
                        'company', :entity_id, 'ubo_risk_confirmed', 'positive', 'bronze',
                        'ownership_links', 'projection_engine', :confidence, :valid_from, :valid_to, CAST(:evidence_json AS jsonb)
                    )
                    """
                ),
                {
                    "entity_id": tax_code,
                    "confidence": 0.5,
                    "valid_from": row.get("effective_date") or backdate,
                    "valid_to": row.get("end_date"),
                    "evidence_json": json.dumps(
                        {
                            "projection_rule": "suspicious_ownership_edge",
                            "ownership_id": row.get("id"),
                            "parent_tax_code": row.get("parent_tax_code"),
                            "child_tax_code": row.get("child_tax_code"),
                            "relationship_type": row.get("relationship_type"),
                            "ownership_percent": float(row.get("ownership_percent") or 0.0),
                            "verified": bool(row.get("verified")) if row.get("verified") is not None else None,
                        },
                        default=str,
                    ),
                },
            )
            already_company_labeled.add(tax_code)
            inserted += 1
            positives += 1

    # 3) Ownership-control subgraph scoring with evidence quality (provenance/confidence)
    # Build adjacency from ownership_links and strong alias edges.
    adj: dict[str, set[str]] = {}
    def _add_edge(a: str, b: str) -> None:
        if not a or not b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for row in db.execute(text("SELECT parent_tax_code, child_tax_code FROM ownership_links")).fetchall():
        _add_edge(str(row[0] or ""), str(row[1] or ""))
    for row in db.execute(
        text(
            """
            SELECT src_tax_code, dst_tax_code
            FROM entity_alias_edges
            WHERE COALESCE(score, 0) >= 0.90
            """
        )
    ).fetchall():
        _add_edge(str(row[0] or ""), str(row[1] or ""))

    highrisk_offshore_proxies = {str(r.get("tax_code") or "") for r in offshore_proxy_rows if r.get("tax_code")}
    suspicious_endpoints = set()
    for r in suspicious_links:
        if r.get("parent_tax_code"):
            suspicious_endpoints.add(str(r["parent_tax_code"]))
        if r.get("child_tax_code"):
            suspicious_endpoints.add(str(r["child_tax_code"]))

    alias_deg: dict[str, int] = {}
    for row in db.execute(
        text(
            """
            SELECT src_tax_code, dst_tax_code
            FROM entity_alias_edges
            WHERE COALESCE(score, 0) >= 0.90
            """
        )
    ).fetchall():
        a, b = str(row[0] or ""), str(row[1] or "")
        if a:
            alias_deg[a] = alias_deg.get(a, 0) + 1
        if b:
            alias_deg[b] = alias_deg.get(b, 0) + 1

    phoenix_deg: dict[str, int] = {}
    for row in db.execute(text("SELECT old_tax_code, new_tax_code FROM phoenix_candidates")).fetchall():
        a, b = str(row[0] or ""), str(row[1] or "")
        if a:
            phoenix_deg[a] = phoenix_deg.get(a, 0) + 1
        if b:
            phoenix_deg[b] = phoenix_deg.get(b, 0) + 1

    # Evidence quality map from graph_edge_evidence + graph_edges endpoint projection.
    evidence_quality_by_company: dict[str, float] = {}
    evidence_rows = db.execute(
        text(
            """
            SELECT
                ge.src_node_id,
                ge.dst_node_id,
                ge.edge_type,
                gee.source_type,
                COALESCE(gee.confidence, ge.confidence, 0.4) AS conf
            FROM graph_edges ge
            LEFT JOIN graph_edge_evidence gee ON gee.edge_id = ge.edge_id
            WHERE ge.edge_type IN ('owns', 'controls', 'alias', 'phoenix_successor')
            """
        )
    ).mappings().all()
    source_weight = {
        "ownership_links": 1.0,
        "entity_alias_edges": 0.85,
        "phoenix_candidates": 0.8,
        "invoices": 0.75,
        "default": 0.6,
    }
    for row in evidence_rows:
        conf = float(row.get("conf") or 0.4)
        st = str(row.get("source_type") or "default")
        w = source_weight.get(st, source_weight["default"])
        quality = max(0.0, min(1.0, conf * w))
        for node_ref in (row.get("src_node_id"), row.get("dst_node_id")):
            node_ref = str(node_ref or "")
            if node_ref.startswith("company:"):
                tc = node_ref.split(":", 1)[1]
                evidence_quality_by_company[tc] = max(evidence_quality_by_company.get(tc, 0.0), quality)

    # Candidate set: unlabeled companies that are in the 2-hop neighborhood of offshore proxies or suspicious endpoints.
    seeds = set(highrisk_offshore_proxies) | set(list(suspicious_endpoints)[:5000])
    candidates: set[str] = set()
    for seed in list(seeds)[:8000]:
        for n1 in adj.get(seed, set()):
            candidates.add(n1)
            for n2 in adj.get(n1, set()):
                candidates.add(n2)

    def _score_company(tc: str) -> tuple[float, dict[str, float]]:
        is_offshore_proxy = 1.0 if tc in highrisk_offshore_proxies else 0.0
        neighbor_offshore = 0.0
        for nb in adj.get(tc, set()):
            if nb in highrisk_offshore_proxies:
                neighbor_offshore = 1.0
                break
        susp = 1.0 if tc in suspicious_endpoints else 0.0
        a_deg = float(alias_deg.get(tc, 0))
        p_deg = float(phoenix_deg.get(tc, 0))
        evidence_q = float(evidence_quality_by_company.get(tc, 0.25))
        # Weighted score with light saturation and evidence adjustment.
        score = (
            1.20 * is_offshore_proxy
            + 0.75 * neighbor_offshore
            + 0.55 * susp
            + 0.15 * min(5.0, a_deg)
            + 0.10 * min(5.0, p_deg)
            + 0.60 * evidence_q
        )
        return float(score), {
            "is_offshore_proxy": is_offshore_proxy,
            "neighbor_offshore": neighbor_offshore,
            "suspicious_endpoint": susp,
            "alias_deg": a_deg,
            "phoenix_deg": p_deg,
            "evidence_quality": evidence_q,
        }

    scored: list[tuple[str, float, dict[str, float]]] = []
    for tc in candidates:
        tc = str(tc or "")
        if not tc or tc in already_company_labeled:
            continue
        score, components = _score_company(tc)
        if score >= 1.10:
            scored.append((tc, score, components))

    scored.sort(key=lambda x: (-x[1], x[0]))
    cap_total = 20000
    projected = 0
    for tc, score, components in scored:
        if projected >= cap_total:
            break
        confidence = min(0.85, 0.30 + 0.12 * score + 0.20 * float(components.get("evidence_quality", 0.0)))
        db.execute(
            text(
                """
                INSERT INTO graph_labels (
                    entity_type, entity_id, label_name, label_value, trust_tier, label_source, annotator, confidence,
                    valid_from, valid_to, evidence_json
                )
                VALUES (
                    'company', :entity_id, 'shell_network_confirmed', 'positive', 'bronze',
                    'subgraph_scoring', 'projection_engine', :confidence, :valid_from, NULL, CAST(:evidence_json AS jsonb)
                )
                """
            ),
            {
                "entity_id": tc,
                "confidence": confidence,
                "valid_from": backdate,
                "evidence_json": json.dumps(
                    {
                        "projection_rule": "ownership_control_subgraph_scoring",
                        "score": score,
                        "components": components,
                    },
                    default=str,
                ),
            },
        )
        already_company_labeled.add(tc)
        inserted += 1
        positives += 1
        projected += 1

    db.commit()
    label_version = f"osint-graph-labels-{date.today().isoformat()}"
    label_hash = _hash_payload({"rows": inserted, "positives": positives, "negatives": negatives})
    label_version_id = registry.register_label_version(
        label_key="osint_graph_supervision",
        label_version=label_version,
        entity_type="graph_entity",
        label_source="graph_labels",
        positive_count=positives,
        negative_count=negatives,
        label_hash=label_hash,
        notes="Gold/silver/bronze graph supervision materialized from inspector labels and ownership heuristics.",
    )
    return {
        "label_version": label_version,
        "label_version_id": label_version_id,
        "rows": inserted,
        "positives": positives,
        "negatives": negatives,
    }


def materialize_graph_topology(db, registry: ModelRegistryService) -> dict[str, Any]:
    companies = db.execute(
        text(
            """
            SELECT tax_code, name, industry, is_active, risk_score, registration_date,
                   COALESCE(NULLIF(country_inferred, ''), province, 'Vietnam') AS country_code
            FROM companies
            """
        )
    ).mappings().all()
    offshore_entities = db.execute(
        text(
            """
            SELECT entity_code, proxy_tax_code, name, country, risk_score, entity_type, status, registration_date
            FROM offshore_entities
            """
        )
    ).mappings().all()
    ownership_links = db.execute(
        text(
            """
            SELECT id, parent_tax_code, child_tax_code, relationship_type, ownership_percent,
                   person_id, person_name, effective_date, end_date, verified, data_source, created_at
            FROM ownership_links
            """
        )
    ).mappings().all()
    alias_edges = db.execute(
        text(
            """
            SELECT id, src_tax_code, dst_tax_code, edge_type, score, evidence_json, created_at
            FROM entity_alias_edges
            """
        )
    ).mappings().all()
    phoenix_edges = db.execute(
        text(
            """
            SELECT id, old_tax_code, new_tax_code, score, signals_json, as_of_date, created_at
            FROM phoenix_candidates
            """
        )
    ).mappings().all()
    invoice_edges = db.execute(
        text(
            """
            SELECT invoice_number, seller_tax_code, buyer_tax_code, amount, vat_rate, date, payment_status
            FROM invoices
            WHERE seller_tax_code IS NOT NULL AND buyer_tax_code IS NOT NULL
            """
        )
    ).mappings().all()

    node_count = 0
    edge_count = 0
    db.execute(text("DELETE FROM graph_edge_evidence"))
    db.execute(text("DELETE FROM graph_edges"))
    db.execute(text("DELETE FROM graph_node_versions"))
    db.execute(text("DELETE FROM graph_nodes"))

    for row in companies:
        attrs = {
            "industry": row.get("industry"),
            "registration_date": row.get("registration_date"),
            "is_active": row.get("is_active"),
        }
        _upsert_node(
            db,
            node_id=f"company:{row['tax_code']}",
            node_type="company",
            native_table="companies",
            native_key=str(row["tax_code"]),
            display_name=row.get("name"),
            status="active" if row.get("is_active", True) else "inactive",
            risk_score=float(row.get("risk_score") or 0.0),
            country_code=row.get("country_code"),
            attributes=attrs,
        )
        _record_node_version(
            db,
            node_id=f"company:{row['tax_code']}",
            valid_from=row.get("registration_date") or datetime(2000, 1, 1),
            valid_to=None,
            attributes=attrs,
        )
        node_count += 1

    for row in offshore_entities:
        attrs = {
            "proxy_tax_code": row.get("proxy_tax_code"),
            "entity_type": row.get("entity_type"),
            "registration_date": row.get("registration_date"),
        }
        _upsert_node(
            db,
            node_id=f"offshore:{row['entity_code']}",
            node_type="offshore_entity",
            native_table="offshore_entities",
            native_key=str(row["entity_code"]),
            display_name=row.get("name"),
            status=row.get("status"),
            risk_score=float(row.get("risk_score") or 0.0),
            country_code=row.get("country"),
            attributes=attrs,
        )
        _record_node_version(
            db,
            node_id=f"offshore:{row['entity_code']}",
            valid_from=row.get("registration_date") or datetime(2000, 1, 1),
            valid_to=None,
            attributes=attrs,
        )
        node_count += 1

    seen_people: set[str] = set()
    for row in ownership_links:
        person_id = row.get("person_id") or row.get("person_name")
        if person_id:
            node_id = f"person:{person_id}"
            if node_id not in seen_people:
                _upsert_node(
                    db,
                    node_id=node_id,
                    node_type="person",
                    native_table="ownership_links",
                    native_key=str(person_id),
                    display_name=row.get("person_name") or str(person_id),
                    status="observed",
                    risk_score=None,
                    country_code=None,
                    attributes={"person_name": row.get("person_name"), "person_id": row.get("person_id")},
                )
                _record_node_version(
                    db,
                    node_id=node_id,
                    valid_from=row.get("effective_date") or row.get("created_at") or datetime(2000, 1, 1),
                    valid_to=row.get("end_date"),
                    attributes={"person_name": row.get("person_name"), "person_id": row.get("person_id")},
                )
                seen_people.add(node_id)
                node_count += 1

        parent_node = f"company:{row['parent_tax_code']}"
        child_node = f"company:{row['child_tax_code']}"
        if any(str(row.get("parent_tax_code") or "").startswith(prefix) for prefix in ("OFF", "OS")):
            parent_node = f"offshore:{row['parent_tax_code']}"
        _upsert_edge(
            db,
            edge_id=f"ownership:{row['id']}",
            src_node_id=parent_node,
            dst_node_id=child_node,
            edge_type="owns",
            directed=True,
            weight=float(row.get("ownership_percent") or 0.0) / 100.0,
            confidence=0.9 if row.get("verified") else 0.6,
            valid_from=row.get("effective_date"),
            valid_to=row.get("end_date"),
            observed_at=row.get("created_at"),
            attributes={
                "relationship_type": row.get("relationship_type"),
                "ownership_percent": float(row.get("ownership_percent") or 0.0),
                "verified": row.get("verified"),
            },
            evidence={
                "source_type": row.get("data_source") or "ownership_links",
                "source_ref": f"ownership_links:{row['id']}",
                "extractor_version": "osint-readiness-v1",
            },
        )
        edge_count += 1
        if person_id:
            _upsert_edge(
                db,
                edge_id=f"controls:{row['id']}",
                src_node_id=f"person:{person_id}",
                dst_node_id=child_node,
                edge_type="controls",
                directed=True,
                weight=float(row.get("ownership_percent") or 0.0) / 100.0,
                confidence=0.75,
                valid_from=row.get("effective_date"),
                valid_to=row.get("end_date"),
                observed_at=row.get("created_at"),
                attributes={"relationship_type": row.get("relationship_type")},
                evidence={
                    "source_type": row.get("data_source") or "ownership_links",
                    "source_ref": f"ownership_links:{row['id']}",
                    "extractor_version": "osint-readiness-v1",
                },
            )
            edge_count += 1

    for row in alias_edges:
        _upsert_edge(
            db,
            edge_id=f"alias:{row['id']}",
            src_node_id=f"company:{row['src_tax_code']}",
            dst_node_id=f"company:{row['dst_tax_code']}",
            edge_type="alias",
            directed=False,
            weight=float(row.get("score") or 0.0),
            confidence=float(row.get("score") or 0.0),
            valid_from=row.get("created_at"),
            valid_to=None,
            observed_at=row.get("created_at"),
            attributes={"edge_type": row.get("edge_type")},
            evidence={
                "source_type": "entity_alias_edges",
                "source_ref": f"entity_alias_edges:{row['id']}",
                "extractor_version": "osint-readiness-v1",
                "evidence_json": row.get("evidence_json"),
            },
        )
        edge_count += 1

    for row in phoenix_edges:
        _upsert_edge(
            db,
            edge_id=f"phoenix:{row['id']}",
            src_node_id=f"company:{row['old_tax_code']}",
            dst_node_id=f"company:{row['new_tax_code']}",
            edge_type="phoenix_successor",
            directed=True,
            weight=float(row.get("score") or 0.0),
            confidence=float(row.get("score") or 0.0),
            valid_from=row.get("as_of_date"),
            valid_to=None,
            observed_at=row.get("created_at"),
            attributes={"signals": row.get("signals_json")},
            evidence={
                "source_type": "phoenix_candidates",
                "source_ref": f"phoenix_candidates:{row['id']}",
                "extractor_version": "osint-readiness-v1",
            },
        )
        edge_count += 1

    for row in invoice_edges:
        _upsert_edge(
            db,
            edge_id=f"invoice:{row['invoice_number']}",
            src_node_id=f"company:{row['seller_tax_code']}",
            dst_node_id=f"company:{row['buyer_tax_code']}",
            edge_type="issued_invoice_to",
            directed=True,
            weight=float(row.get("amount") or 0.0),
            confidence=0.95,
            valid_from=row.get("date"),
            valid_to=None,
            observed_at=row.get("date"),
            attributes={
                "amount": float(row.get("amount") or 0.0),
                "vat_rate": float(row.get("vat_rate") or 0.0),
                "payment_status": row.get("payment_status"),
            },
            evidence={
                "source_type": "invoices",
                "source_ref": f"invoices:{row['invoice_number']}",
                "extractor_version": "osint-readiness-v1",
            },
        )
        edge_count += 1

    db.commit()
    dataset_payload = {
        "companies": len(companies),
        "offshore_entities": len(offshore_entities),
        "ownership_links": len(ownership_links),
        "alias_edges": len(alias_edges),
        "phoenix_edges": len(phoenix_edges),
        "invoice_edges": len(invoice_edges),
    }
    dataset_version = f"osint-canonical-graph-{date.today().isoformat()}"
    dataset_version_id = registry.register_dataset_version(
        dataset_key="osint_canonical_graph",
        dataset_version=dataset_version,
        entity_type="graph",
        row_count=node_count + edge_count,
        source_tables=[
            "companies",
            "offshore_entities",
            "ownership_links",
            "entity_alias_edges",
            "phoenix_candidates",
            "invoices",
        ],
        filters={"materialized_at": datetime.utcnow().isoformat()},
        data_hash=_hash_payload(dataset_payload),
        created_by="osint_graph_readiness",
    )
    return {
        "dataset_version": dataset_version,
        "dataset_version_id": dataset_version_id,
        "nodes": node_count,
        "edges": edge_count,
    }


def main() -> None:
    with SessionLocal() as db:
        _ensure_schema(db)
        registry = ModelRegistryService(db)
        experiment_id = registry.ensure_experiment(
            experiment_key="osint_heterograph_readiness",
            model_name="osint_heterograph_readiness",
            owner="ml-platform",
            objective="graph_readiness",
            metadata={
                "primary_metric": "readiness_score",
                "description": "Readiness materialization for OSINT hetero-graph labels and canonical topology.",
            },
        )
        run_id = registry.start_training_run(
            model_name="osint_heterograph_readiness",
            experiment_id=experiment_id,
            model_version=f"readiness-{date.today().isoformat()}",
            code_hash=_hash_payload({"script": "materialize_osint_graph_readiness"}),
            hyperparams={"mode": "materialize"},
        )
        # Build topology/evidence first so label projection can consume fresh edge evidence quality.
        topology_result = materialize_graph_topology(db, registry)
        label_result = materialize_graph_labels(db, registry)
        metrics = {
            "label_rows": label_result["rows"],
            "graph_nodes": topology_result["nodes"],
            "graph_edges": topology_result["edges"],
            "readiness_score": round(
                min(1.0, 0.35 + 0.25 * bool(label_result["rows"]) + 0.2 * bool(topology_result["nodes"]) + 0.2 * bool(topology_result["edges"])),
                4,
            ),
        }
        registry.complete_training_run(
            run_id=run_id,
            status="completed",
            metrics=metrics,
            artifacts={
                "label_result": label_result,
                "topology_result": topology_result,
            },
        )
        print(json.dumps({"labels": label_result, "topology": topology_result, "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
