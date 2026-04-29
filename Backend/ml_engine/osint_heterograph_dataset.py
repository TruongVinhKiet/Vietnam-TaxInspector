from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sqlalchemy import text

try:
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    HeteroData = Any  # type: ignore[assignment]
    PYG_AVAILABLE = False


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
DEFAULT_NODE_TYPES = ("company", "offshore_entity", "person")
DEFAULT_EDGE_TYPES = (
    "owns",
    "controls",
    "alias",
    "phoenix_successor",
    "issued_invoice_to",
)
COUNTRY_RISK = {
    "vietnam": 0.1,
    "unknown": 0.3,
    "cayman islands": 1.0,
    "british virgin islands (bvi)": 1.0,
    "panama": 1.0,
    "seychelles": 0.9,
    "bahamas": 0.9,
    "cyprus": 0.75,
    "hong kong": 0.55,
    "singapore": 0.45,
}
INDUSTRY_BUCKETS = {
    "offshore entity": 1.0,
    "financial": 0.8,
    "real estate": 0.7,
    "construction": 0.55,
    "trading": 0.45,
    "manufacturing": 0.35,
    "technology": 0.25,
}
FEATURE_NAMES = [
    "risk_score_norm",
    "country_risk",
    "industry_bucket",
    "node_type_company",
    "node_type_offshore",
    "node_type_person",
    "status_active",
    "age_norm",
    "in_degree_norm",
    "out_degree_norm",
    "owns_out_norm",
    "controls_out_norm",
    "alias_degree_norm",
    "phoenix_out_norm",
    "invoice_flow_log_norm",
]

def _cluster_partition(cluster_id: str) -> str:
    """
    Deterministic partition for entity-disjoint cluster split.
    """
    if not cluster_id or cluster_id == "none":
        return "none"
    h = _sha({"cluster_id": cluster_id})
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _compute_offshore_cluster_map(
    company_rows: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    offshore_root_risk: dict[str, float],
    max_hops: int = 2,
) -> dict[str, str]:
    """
    Build offshore-rooted clusters from the snapshot topology.
    Cluster root = offshore node_id (e.g. offshore:OE123).
    Assign each company: tax_code to a root if reachable within max_hops via owns/controls/alias/phoenix edges.
    """
    company_nodes = {str(r.get("node_id")) for r in company_rows if r.get("node_id")}
    allowed_edge_types = {"owns", "alias", "phoenix_successor"}

    adj: dict[str, set[str]] = {}
    def _add(a: str, b: str) -> None:
        if not a or not b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for e in edges:
        et = str(e.get("edge_type") or "")
        if et not in allowed_edge_types:
            continue
        src = str(e.get("src_node_id") or "")
        dst = str(e.get("dst_node_id") or "")
        _add(src, dst)

    # Offshore roots are company proxies from offshore_entities.proxy_tax_code.
    offshore_roots = {node_id for node_id in offshore_root_risk.keys() if node_id in company_nodes}

    # Multi-source BFS from offshore roots with hop cap.
    company_to_root: dict[str, str] = {}
    ordered_roots = sorted(offshore_roots, key=lambda node_id: (-offshore_root_risk.get(node_id, 0.0), node_id))
    for root in ordered_roots:
        frontier = {root}
        visited = {root}
        for hop in range(1, max_hops + 1):
            nxt = set()
            for node in frontier:
                for nb in adj.get(node, set()):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    nxt.add(nb)
            for node in nxt:
                if node in company_nodes and node not in company_to_root:
                    company_to_root[node] = root
            frontier = nxt

    # Companies not connected to any offshore root -> none.
    for c in company_nodes:
        company_to_root.setdefault(c, "none")
    return company_to_root


@dataclass
class SnapshotGraphArtifacts:
    snapshot_id: str
    as_of_timestamp: datetime
    node_types: list[str]
    edge_types: list[str]
    node_index_by_type: dict[str, dict[str, int]]
    labeled_company_nodes: list[str]
    label_map: dict[str, int]
    trust_map: dict[str, str]
    slice_maps: dict[str, dict[str, str]]
    hetero_data: Any | None
    homo_x: torch.Tensor
    homo_edge_index: torch.Tensor
    homo_edge_type: torch.Tensor
    homo_node_type: torch.Tensor
    homo_company_mask: torch.Tensor
    homo_company_labels: torch.Tensor
    homo_company_node_ids: list[str]
    homo_company_global_indices: list[int]
    feature_names: list[str]
    lineage_hash: str


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_node_id(entity_type: str, entity_id: str) -> str:
    raw = str(entity_id or "").strip()
    if ":" in raw:
        return raw
    mapping = {
        "company": "company",
        "offshore_entity": "offshore",
        "person": "person",
        "edge": "edge",
        "subgraph": "subgraph",
    }
    prefix = mapping.get(str(entity_type or "").strip().lower(), str(entity_type or "entity"))
    return f"{prefix}:{raw}"


def _country_risk(country: Any) -> float:
    key = str(country or "unknown").strip().lower()
    if key in COUNTRY_RISK:
        return float(COUNTRY_RISK[key])
    if key and key != "vietnam":
        return 0.5
    return 0.1


def _industry_bucket(industry: Any) -> float:
    text_value = str(industry or "").strip().lower()
    for key, value in INDUSTRY_BUCKETS.items():
        if key in text_value:
            return float(value)
    return 0.2 if text_value else 0.0


def _age_norm(registration_date: Any, as_of_timestamp: datetime) -> float:
    if not registration_date:
        return 0.0
    if isinstance(registration_date, str):
        try:
            registration_date = datetime.fromisoformat(registration_date).date()
        except Exception:
            return 0.0
    try:
        age_days = max(0, (as_of_timestamp.date() - registration_date).days)
    except Exception:
        return 0.0
    return min(1.0, age_days / 3650.0)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except Exception:
        return float(default)


def list_snapshots(db, graph_family: str = "osint_heterograph") -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT snapshot_id, as_of_timestamp, lineage_hash
            FROM graph_snapshots
            WHERE graph_family = :graph_family
            ORDER BY as_of_timestamp ASC
            """
        ),
        {"graph_family": graph_family},
    ).mappings().all()
    return [dict(row) for row in rows]


def load_latest_benchmark_contract(db, graph_family: str = "osint_heterograph") -> dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT benchmark_key, baseline_models, split_strategy, metric_contract, slice_contract, promotion_gate
            FROM graph_benchmark_specs
            WHERE graph_family = :graph_family
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"graph_family": graph_family},
    ).mappings().first()
    return dict(row) if row else {}


def temporal_split(snapshot_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    if len(snapshot_rows) >= 3:
        train = [row["snapshot_id"] for row in snapshot_rows[:-2]]
        val = [snapshot_rows[-2]["snapshot_id"]]
        test = [snapshot_rows[-1]["snapshot_id"]]
    elif len(snapshot_rows) == 2:
        train = [snapshot_rows[0]["snapshot_id"]]
        val = [snapshot_rows[1]["snapshot_id"]]
        test = [snapshot_rows[1]["snapshot_id"]]
    elif len(snapshot_rows) == 1:
        train = [snapshot_rows[0]["snapshot_id"]]
        val = [snapshot_rows[0]["snapshot_id"]]
        test = [snapshot_rows[0]["snapshot_id"]]
    else:
        train, val, test = [], [], []
    return {"train": train, "val": val, "test": test}


def _load_snapshot_records(db, snapshot_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = db.execute(
        text(
            """
            SELECT snapshot_id, as_of_timestamp, seed_set_json, extraction_policy_json, lineage_hash
            FROM graph_snapshots
            WHERE snapshot_id = :snapshot_id
            """
        ),
        {"snapshot_id": snapshot_id},
    ).mappings().first()
    if snapshot is None:
        raise ValueError(f"Snapshot not found: {snapshot_id}")
    nodes = db.execute(
        text(
            """
            SELECT node_id, node_type, feature_payload
            FROM graph_snapshot_nodes
            WHERE snapshot_id = :snapshot_id
            ORDER BY node_type, node_id
            """
        ),
        {"snapshot_id": snapshot_id},
    ).mappings().all()
    edges = db.execute(
        text(
            """
            SELECT edge_id, edge_type, src_node_id, dst_node_id, attributes_json
            FROM graph_snapshot_edges
            WHERE snapshot_id = :snapshot_id
            ORDER BY edge_type, edge_id
            """
        ),
        {"snapshot_id": snapshot_id},
    ).mappings().all()
    return dict(snapshot), [dict(row) for row in nodes], [dict(row) for row in edges]


def _query_labels(db, *, as_of_timestamp: datetime | None) -> list[dict[str, Any]]:
    if as_of_timestamp is not None:
        rows = db.execute(
            text(
                """
                SELECT entity_type, entity_id, label_name, label_value, trust_tier
                FROM graph_labels
                WHERE (valid_from IS NULL OR valid_from <= :as_of_ts)
                  AND (valid_to IS NULL OR valid_to > :as_of_ts)
                ORDER BY
                  CASE trust_tier WHEN 'gold' THEN 3 WHEN 'silver' THEN 2 ELSE 1 END DESC,
                  created_at DESC
                """
            ),
            {"as_of_ts": as_of_timestamp},
        ).mappings().all()
        return [dict(row) for row in rows]
    rows = db.execute(
        text(
            """
            SELECT entity_type, entity_id, label_name, label_value, trust_tier
            FROM graph_labels
            ORDER BY
              CASE trust_tier WHEN 'gold' THEN 3 WHEN 'silver' THEN 2 ELSE 1 END DESC,
              created_at DESC
            """
        )
    ).mappings().all()
    return [dict(row) for row in rows]


def _load_labels(db, as_of_timestamp: datetime, candidate_node_ids: set[str] | None = None) -> tuple[dict[str, int], dict[str, str]]:
    rows = _query_labels(db, as_of_timestamp=as_of_timestamp)
    if not rows:
        rows = _query_labels(db, as_of_timestamp=None)
    label_map: dict[str, int] = {}
    trust_map: dict[str, str] = {}
    for row in rows:
        if str(row.get("entity_type")) != "company":
            continue
        node_id = _normalize_node_id(str(row.get("entity_type")), str(row.get("entity_id")))
        if node_id in label_map:
            continue
        label_value = 0 if str(row.get("label_value")) == "negative" else 1
        label_map[node_id] = label_value
        trust_map[node_id] = str(row.get("trust_tier") or "bronze")
    if candidate_node_ids is not None and not any(node_id in candidate_node_ids for node_id in label_map):
        rows = _query_labels(db, as_of_timestamp=None)
        label_map = {}
        trust_map = {}
        for row in rows:
            if str(row.get("entity_type")) != "company":
                continue
            node_id = _normalize_node_id(str(row.get("entity_type")), str(row.get("entity_id")))
            if node_id in label_map:
                continue
            label_value = 0 if str(row.get("label_value")) == "negative" else 1
            label_map[node_id] = label_value
            trust_map[node_id] = str(row.get("trust_tier") or "bronze")
    return label_map, trust_map


def _compute_structural_maps(node_ids: list[str], edges: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    in_degree = {node_id: 0.0 for node_id in node_ids}
    out_degree = {node_id: 0.0 for node_id in node_ids}
    by_type: dict[str, dict[str, float]] = {}
    amount_flow = {node_id: 0.0 for node_id in node_ids}
    for edge in edges:
        src = str(edge["src_node_id"])
        dst = str(edge["dst_node_id"])
        edge_type = str(edge["edge_type"])
        attrs = _safe_json(edge.get("attributes_json"))
        out_degree[src] = out_degree.get(src, 0.0) + 1.0
        in_degree[dst] = in_degree.get(dst, 0.0) + 1.0
        bucket = by_type.setdefault(edge_type, {})
        bucket[src] = bucket.get(src, 0.0) + 1.0
        bucket[dst] = bucket.get(dst, 0.0) + (0.0 if edge_type in {"owns", "controls", "phoenix_successor"} else 1.0)
        if edge_type == "issued_invoice_to":
            amt = _coerce_float(attrs.get("amount"), 0.0)
            amount_flow[src] = amount_flow.get(src, 0.0) + amt
            amount_flow[dst] = amount_flow.get(dst, 0.0) + amt
    max_in = max(in_degree.values(), default=1.0) or 1.0
    max_out = max(out_degree.values(), default=1.0) or 1.0
    max_amount = max(amount_flow.values(), default=1.0) or 1.0
    maps: dict[str, dict[str, float]] = {
        "in_degree_norm": {k: v / max_in for k, v in in_degree.items()},
        "out_degree_norm": {k: v / max_out for k, v in out_degree.items()},
        "invoice_flow_log_norm": {
            k: min(1.0, math.log1p(v) / max(1.0, math.log1p(max_amount)))
            for k, v in amount_flow.items()
        },
    }
    for edge_type in DEFAULT_EDGE_TYPES:
        values = by_type.get(edge_type, {})
        max_value = max(values.values(), default=1.0) or 1.0
        maps[f"{edge_type}_norm"] = {
            node_id: values.get(node_id, 0.0) / max_value
            for node_id in node_ids
        }
    return maps


def _build_node_feature(
    node_type: str,
    feature_payload: dict[str, Any],
    structural_maps: dict[str, dict[str, float]],
    node_id: str,
    as_of_timestamp: datetime,
) -> list[float]:
    attrs = _safe_json(feature_payload.get("attributes"))
    risk_score = _coerce_float(feature_payload.get("risk_score"), 0.0)
    country_code = feature_payload.get("country_code") or attrs.get("country") or "unknown"
    status = str(attrs.get("status") or attrs.get("is_active") or "").lower()
    features = [
        min(1.0, risk_score / 100.0),
        _country_risk(country_code),
        _industry_bucket(attrs.get("industry") or attrs.get("entity_type")),
        1.0 if node_type == "company" else 0.0,
        1.0 if node_type == "offshore_entity" else 0.0,
        1.0 if node_type == "person" else 0.0,
        1.0 if status in {"active", "true", "1"} or attrs.get("is_active") is True else 0.0,
        _age_norm(attrs.get("registration_date"), as_of_timestamp),
        structural_maps["in_degree_norm"].get(node_id, 0.0),
        structural_maps["out_degree_norm"].get(node_id, 0.0),
        structural_maps.get("owns_norm", {}).get(node_id, 0.0),
        structural_maps.get("controls_norm", {}).get(node_id, 0.0),
        structural_maps.get("alias_norm", {}).get(node_id, 0.0),
        structural_maps.get("phoenix_successor_norm", {}).get(node_id, 0.0),
        structural_maps.get("invoice_flow_log_norm", {}).get(node_id, 0.0),
    ]
    return [float(x) for x in features]


def build_snapshot_graph(db, snapshot_id: str) -> SnapshotGraphArtifacts:
    snapshot, nodes, edges = _load_snapshot_records(db, snapshot_id)
    as_of_timestamp = snapshot["as_of_timestamp"]
    if isinstance(as_of_timestamp, str):
        as_of_timestamp = datetime.fromisoformat(as_of_timestamp)
    node_types = sorted({str(row["node_type"]) for row in nodes}) or list(DEFAULT_NODE_TYPES)
    edge_types = sorted({str(row["edge_type"]) for row in edges}) or list(DEFAULT_EDGE_TYPES)
    candidate_node_ids = {str(row["node_id"]) for row in nodes if str(row["node_type"]) == "company"}
    label_map, trust_map = _load_labels(db, as_of_timestamp, candidate_node_ids=candidate_node_ids)
    node_ids = [str(row["node_id"]) for row in nodes]
    structural_maps = _compute_structural_maps(node_ids, edges)
    node_rows_by_type: dict[str, list[dict[str, Any]]] = {node_type: [] for node_type in node_types}
    for row in nodes:
        node_rows_by_type[str(row["node_type"])].append(row)

    offshore_rows = db.execute(
        text(
            """
            SELECT
                proxy_tax_code,
                COALESCE(jurisdiction_risk_weight, 0.5) AS jurisdiction_risk_weight,
                COALESCE(risk_score, 0) AS risk_score
            FROM offshore_entities
            WHERE proxy_tax_code IS NOT NULL
            """
        )
    ).mappings().all()
    offshore_root_risk: dict[str, float] = {}
    for row in offshore_rows:
        proxy_node = f"company:{str(row.get('proxy_tax_code') or '').strip()}"
        if proxy_node == "company:":
            continue
        root_risk = (
            0.6 * min(1.0, _coerce_float(row.get("jurisdiction_risk_weight"), 0.5))
            + 0.4 * min(1.0, _coerce_float(row.get("risk_score"), 0.0) / 100.0)
        )
        offshore_root_risk[proxy_node] = float(root_risk)

    offshore_cluster_map = _compute_offshore_cluster_map(
        node_rows_by_type.get("company", []),
        edges,
        offshore_root_risk=offshore_root_risk,
        max_hops=2,
    )

    node_index_by_type: dict[str, dict[str, int]] = {}
    feature_tensors: dict[str, torch.Tensor] = {}
    for node_type, rows in node_rows_by_type.items():
        rows_sorted = sorted(rows, key=lambda item: str(item["node_id"]))
        node_index_by_type[node_type] = {
            str(row["node_id"]): idx for idx, row in enumerate(rows_sorted)
        }
        features = [
            _build_node_feature(
                node_type=node_type,
                feature_payload=_safe_json(row.get("feature_payload")),
                structural_maps=structural_maps,
                node_id=str(row["node_id"]),
                as_of_timestamp=as_of_timestamp,
            )
            for row in rows_sorted
        ]
        if features:
            feature_tensors[node_type] = torch.tensor(features, dtype=torch.float32)
        else:
            feature_tensors[node_type] = torch.zeros((0, len(FEATURE_NAMES)), dtype=torch.float32)

    hetero_data = None
    if PYG_AVAILABLE:
        hetero_data = HeteroData()
        for node_type, tensor in feature_tensors.items():
            hetero_data[node_type].x = tensor
            hetero_data[node_type].node_id = list(node_index_by_type[node_type].keys())

    company_node_ids = list(node_index_by_type.get("company", {}).keys())
    company_labels = torch.tensor(
        [label_map.get(node_id, -1) for node_id in company_node_ids],
        dtype=torch.long,
    )
    company_mask = company_labels >= 0
    trust_values = [trust_map.get(node_id, "bronze") for node_id in company_node_ids]
    slice_maps = {
        "jurisdiction": {},
        "ownership_degree_band": {},
        "cold_start": {},
        "proxy_link_quality": {},
        "offshore_cluster": {},
        "cluster_partition": {},
    }
    for node_id in company_node_ids:
        idx = node_index_by_type["company"][node_id]
        x = feature_tensors["company"][idx]
        slice_maps["jurisdiction"][node_id] = (
            "high_risk" if float(x[1]) >= 0.75 else "medium_risk" if float(x[1]) >= 0.45 else "low_risk"
        )
        out_degree = float(x[9])
        slice_maps["ownership_degree_band"][node_id] = ">=3" if out_degree >= 0.5 else "1-2" if out_degree > 0 else "0"
        slice_maps["cold_start"][node_id] = "true" if float(x[7]) <= 0.05 else "false"
        trust = trust_map.get(node_id, "bronze")
        slice_maps["proxy_link_quality"][node_id] = "weak" if trust == "bronze" else "strong"
        cluster_id = offshore_cluster_map.get(node_id, "none")
        slice_maps["offshore_cluster"][node_id] = cluster_id
        slice_maps["cluster_partition"][node_id] = _cluster_partition(cluster_id)

    homo_features: list[torch.Tensor] = []
    homo_node_type: list[int] = []
    global_offsets: dict[str, int] = {}
    offset = 0
    for type_idx, node_type in enumerate(node_types):
        tensor = feature_tensors[node_type]
        global_offsets[node_type] = offset
        homo_features.append(tensor)
        homo_node_type.extend([type_idx] * tensor.shape[0])
        offset += tensor.shape[0]
    homo_x = torch.cat(homo_features, dim=0) if homo_features else torch.zeros((0, len(FEATURE_NAMES)))
    homo_node_type_tensor = torch.tensor(homo_node_type, dtype=torch.long)

    homo_edge_pairs: list[list[int]] = [[], []]
    homo_edge_type_values: list[int] = []
    edge_type_to_idx = {edge_type: idx for idx, edge_type in enumerate(edge_types)}
    for edge in edges:
        src_node = str(edge["src_node_id"])
        dst_node = str(edge["dst_node_id"])
        src_type = src_node.split(":", 1)[0]
        dst_type = dst_node.split(":", 1)[0]
        src_type = "offshore_entity" if src_type == "offshore" else src_type
        dst_type = "offshore_entity" if dst_type == "offshore" else dst_type
        if src_type not in global_offsets or dst_type not in global_offsets:
            continue
        if src_node not in node_index_by_type.get(src_type, {}) or dst_node not in node_index_by_type.get(dst_type, {}):
            continue
        src_idx = global_offsets[src_type] + node_index_by_type[src_type][src_node]
        dst_idx = global_offsets[dst_type] + node_index_by_type[dst_type][dst_node]
        homo_edge_pairs[0].append(src_idx)
        homo_edge_pairs[1].append(dst_idx)
        homo_edge_type_values.append(edge_type_to_idx[str(edge["edge_type"])])
        if PYG_AVAILABLE:
            triplet = (src_type, str(edge["edge_type"]), dst_type)
            edge_index = hetero_data[triplet].get("edge_index")
            append_edge = torch.tensor([[node_index_by_type[src_type][src_node]], [node_index_by_type[dst_type][dst_node]]], dtype=torch.long)
            hetero_data[triplet].edge_index = append_edge if edge_index is None else torch.cat([edge_index, append_edge], dim=1)

    homo_edge_index = (
        torch.tensor(homo_edge_pairs, dtype=torch.long)
        if homo_edge_pairs[0]
        else torch.zeros((2, 0), dtype=torch.long)
    )
    homo_edge_type_tensor = (
        torch.tensor(homo_edge_type_values, dtype=torch.long)
        if homo_edge_type_values
        else torch.zeros((0,), dtype=torch.long)
    )

    homo_company_mask = torch.zeros((homo_x.shape[0],), dtype=torch.bool)
    homo_company_labels = torch.full((homo_x.shape[0],), -1, dtype=torch.long)
    homo_company_global_indices: list[int] = []
    for node_id, local_idx in node_index_by_type.get("company", {}).items():
        global_idx = global_offsets["company"] + local_idx
        if label_map.get(node_id) is not None:
            homo_company_mask[global_idx] = True
            homo_company_labels[global_idx] = int(label_map[node_id])
        homo_company_global_indices.append(global_idx)

    lineage_hash = _sha(
        {
            "snapshot_id": snapshot_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "label_count": int(company_mask.sum().item()),
            "trust_counts": {
                "gold": trust_values.count("gold"),
                "silver": trust_values.count("silver"),
                "bronze": trust_values.count("bronze"),
            },
        }
    )
    return SnapshotGraphArtifacts(
        snapshot_id=snapshot_id,
        as_of_timestamp=as_of_timestamp,
        node_types=node_types,
        edge_types=edge_types,
        node_index_by_type=node_index_by_type,
        labeled_company_nodes=[node_id for node_id in company_node_ids if node_id in label_map],
        label_map=label_map,
        trust_map=trust_map,
        slice_maps=slice_maps,
        hetero_data=hetero_data,
        homo_x=homo_x,
        homo_edge_index=homo_edge_index,
        homo_edge_type=homo_edge_type_tensor,
        homo_node_type=homo_node_type_tensor,
        homo_company_mask=homo_company_mask,
        homo_company_labels=homo_company_labels,
        homo_company_node_ids=company_node_ids,
        homo_company_global_indices=homo_company_global_indices,
        feature_names=list(FEATURE_NAMES),
        lineage_hash=lineage_hash,
    )


def save_dataset_manifest(manifest: dict[str, Any], filename: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    return path
