from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sqlalchemy import text

try:
    from torch_geometric.nn import HGTConv, Linear, RGCNConv
    PYG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    PYG_AVAILABLE = False

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(ENV_PATH)

from app.database import SessionLocal
from app.scripts.build_osint_graph_snapshot import build_snapshot
from ml_engine.model_registry import ModelRegistryService
from ml_engine.osint_heterograph_dataset import (
    PYG_AVAILABLE as DATASET_PYG_AVAILABLE,
    build_snapshot_graph,
    list_snapshots,
    load_latest_benchmark_contract,
    save_dataset_manifest,
    temporal_split,
)


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_VERSION = f"osint-heterograph-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


@dataclass
class EvalResult:
    model_name: str
    model_version: str
    probs: np.ndarray
    labels: np.ndarray
    node_ids: list[str]
    metrics: dict[str, float]
    calibration_bins: list[dict[str, Any]]
    slice_metrics: list[dict[str, Any]]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_monthly_snapshots(db, months: int = 6, graph_family: str = "osint_heterograph") -> list[dict[str, Any]]:
    existing = list_snapshots(db, graph_family=graph_family)
    target_dates = []
    now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for step in range(max(1, int(months))):
        target_dates.append(now - timedelta(days=30 * (months - step - 1)))
    existing_keys = {row["snapshot_id"] for row in existing}
    for ts in target_dates:
        snapshot_id = f"{graph_family}:{ts.strftime('%Y%m%dT%H%M%S')}"
        if snapshot_id in existing_keys:
            continue
        build_snapshot(as_of_timestamp=ts, graph_family=graph_family)
    return list_snapshots(db, graph_family=graph_family)


def _extract_company_arrays(artifacts) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, dict[str, str]]]:
    company_idx = artifacts.node_index_by_type.get("company", {})
    if not company_idx:
        return np.zeros((0, len(artifacts.feature_names))), np.zeros((0,)), [], artifacts.slice_maps
    rows = []
    labels = []
    node_ids = []
    company_x = artifacts.hetero_data["company"].x if artifacts.hetero_data is not None else None
    for node_id, idx in company_idx.items():
        label = artifacts.label_map.get(node_id)
        if label is None:
            continue
        node_ids.append(node_id)
        labels.append(label)
        if company_x is not None:
            rows.append(company_x[idx].detach().cpu().numpy())
        else:
            global_idx = idx
            rows.append(artifacts.homo_x[global_idx].detach().cpu().numpy())
    if not rows:
        return np.zeros((0, len(artifacts.feature_names))), np.zeros((0,)), [], artifacts.slice_maps
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), node_ids, artifacts.slice_maps


def _filter_by_cluster_partition(
    x: np.ndarray,
    y: np.ndarray,
    node_ids: list[str],
    slice_maps: dict[str, dict[str, str]],
    allow_partitions: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mapping = slice_maps.get("cluster_partition") or {}
    keep = [idx for idx, node_id in enumerate(node_ids) if mapping.get(node_id, "none") in allow_partitions]
    if not keep:
        return np.zeros((0, x.shape[1])), np.zeros((0,), dtype=int), []
    return x[keep], y[keep], [node_ids[i] for i in keep]


def _precision_at_k(labels: np.ndarray, probs: np.ndarray, k: int) -> float:
    if labels.size == 0:
        return 0.0
    k = max(1, min(int(k), len(labels)))
    order = np.argsort(-probs)[:k]
    return float(np.mean(labels[order]))


def _recall_at_k(labels: np.ndarray, probs: np.ndarray, k: int) -> float:
    positives = max(1, int(labels.sum()))
    k = max(1, min(int(k), len(labels)))
    order = np.argsort(-probs)[:k]
    return float(labels[order].sum() / positives)


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = 5) -> tuple[float, list[dict[str, Any]]]:
    if labels.size == 0:
        return 0.0, []
    calibration_rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for idx in range(bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx == bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if not mask.any():
            calibration_rows.append(
                {
                    "bin_label": f"{lower:.1f}-{upper:.1f}",
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "predicted_mean": 0.0,
                    "observed_rate": 0.0,
                    "sample_size": 0,
                }
            )
            continue
        predicted_mean = float(probs[mask].mean())
        observed_rate = float(labels[mask].mean())
        sample_size = int(mask.sum())
        weight = sample_size / len(labels)
        ece += abs(predicted_mean - observed_rate) * weight
        calibration_rows.append(
            {
                "bin_label": f"{lower:.1f}-{upper:.1f}",
                "lower_bound": lower,
                "upper_bound": upper,
                "predicted_mean": predicted_mean,
                "observed_rate": observed_rate,
                "sample_size": sample_size,
            }
        )
    return float(ece), calibration_rows


def _compute_slice_metrics(labels: np.ndarray, probs: np.ndarray, node_ids: list[str], slice_maps: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if labels.size == 0:
        return rows
    for slice_name, mapping in slice_maps.items():
        buckets: dict[str, list[int]] = {}
        for idx, node_id in enumerate(node_ids):
            value = mapping.get(node_id, "unknown")
            buckets.setdefault(value, []).append(idx)
        for slice_value, idxs in buckets.items():
            if len(idxs) == 0:
                continue
            slice_labels = labels[idxs]
            slice_probs = probs[idxs]
            metric_value = float(slice_labels.mean())
            if len(np.unique(slice_labels)) >= 2:
                metric_value = float(average_precision_score(slice_labels, slice_probs))
            rows.append(
                {
                    "slice_name": slice_name,
                    "slice_value": slice_value,
                    "metric_name": "pr_auc_slice",
                    "metric_value": metric_value,
                    "sample_size": len(idxs),
                }
            )
    return rows


def _evaluate_predictions(model_name: str, model_version: str, labels: np.ndarray, probs: np.ndarray, node_ids: list[str], slice_maps: dict[str, dict[str, str]]) -> EvalResult:
    if labels.size == 0:
        metrics = {
            "auc": 0.0,
            "pr_auc": 0.0,
            "precision_at_50": 0.0,
            "precision_at_100": 0.0,
            "recall_at_100": 0.0,
            "topn_review_yield": 0.0,
            "ece": 0.0,
            "brier_score": 0.0,
            "score_std_30d": 0.0,
        }
        return EvalResult(model_name, model_version, probs, labels, node_ids, metrics, [], [])
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) >= 2 else 0.5
    pr_auc = float(average_precision_score(labels, probs)) if len(np.unique(labels)) >= 2 else float(labels.mean())
    precision_50 = _precision_at_k(labels, probs, 50)
    precision_100 = _precision_at_k(labels, probs, 100)
    recall_100 = _recall_at_k(labels, probs, 100)
    ece_value, calibration_rows = _ece(labels, probs, bins=5)
    metrics = {
        "auc": auc,
        "pr_auc": pr_auc,
        "precision_at_50": precision_50,
        "precision_at_100": precision_100,
        "recall_at_100": recall_100,
        "topn_review_yield": precision_100,
        "ece": float(ece_value),
        "brier_score": float(np.mean((probs - labels) ** 2)),
        "score_std_30d": float(np.std(probs)),
    }
    slice_metrics = _compute_slice_metrics(labels, probs, node_ids, slice_maps)
    return EvalResult(model_name, model_version, probs, labels, node_ids, metrics, calibration_rows, slice_metrics)


class HGTNodeClassifier(nn.Module):
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], in_channels: int, hidden_channels: int = 64, heads: int = 2):
        super().__init__()
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric is required for HGTNodeClassifier.")
        node_types, _ = metadata
        self.lin_dict = nn.ModuleDict({
            node_type: Linear(in_channels, hidden_channels) for node_type in node_types
        })
        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
        self.conv2 = HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](data[node_type].x))
            for node_type in data.node_types
        }
        x0 = x_dict
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        # HGTConv may omit types with no effective messages; keep prior embeddings.
        for node_type in data.node_types:
            if node_type not in x_dict:
                x_dict[node_type] = x0[node_type]
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x1 = x_dict
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        for node_type in data.node_types:
            if node_type not in x_dict:
                x_dict[node_type] = x1[node_type]
        return self.classifier(x_dict["company"]).squeeze(-1)


class RGCNNodeClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int):
        super().__init__()
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric is required for RGCNNodeClassifier.")
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return self.classifier(x).squeeze(-1)


def _train_hgt(train_artifacts, val_artifacts, epochs: int = 30, lr: float = 1e-3) -> tuple[Any, EvalResult]:
    if train_artifacts.hetero_data is None or not PYG_AVAILABLE or not DATASET_PYG_AVAILABLE:
        raise RuntimeError("PyG hetero graph support unavailable.")
    model = HGTNodeClassifier(
        metadata=train_artifacts.hetero_data.metadata(),
        in_channels=train_artifacts.hetero_data["company"].x.shape[1],
        hidden_channels=64,
        heads=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_part = train_artifacts.slice_maps.get("cluster_partition", {})
    train_labels = torch.tensor(
        [train_artifacts.label_map.get(node_id, -1) for node_id in train_artifacts.homo_company_node_ids],
        dtype=torch.float32,
    )
    train_mask = torch.tensor(
        [
            (train_labels[i].item() >= 0)
            and (train_part.get(node_id, "none") in {"train", "val"})
            for i, node_id in enumerate(train_artifacts.homo_company_node_ids)
        ],
        dtype=torch.bool,
    )
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(train_artifacts.hetero_data)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], train_labels[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_artifacts.hetero_data)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    val_part = val_artifacts.slice_maps.get("cluster_partition", {})
    val_labels = np.asarray(
        [val_artifacts.label_map.get(node_id, -1) for node_id in val_artifacts.homo_company_node_ids],
        dtype=int,
    )
    mask = np.asarray(
        [(val_labels[i] >= 0) and (val_part.get(node_id, "none") == "test") for i, node_id in enumerate(val_artifacts.homo_company_node_ids)],
        dtype=bool,
    )
    result = _evaluate_predictions(
        model_name="osint_hgt",
        model_version=MODEL_VERSION,
        labels=val_labels[mask],
        probs=probs[mask],
        node_ids=[node_id for idx, node_id in enumerate(val_artifacts.homo_company_node_ids) if mask[idx]],
        slice_maps=val_artifacts.slice_maps,
    )
    return model, result


def _train_rgcn(train_artifacts, val_artifacts, epochs: int = 30, lr: float = 1e-3) -> tuple[Any, EvalResult]:
    if not PYG_AVAILABLE:
        raise RuntimeError("torch_geometric is required for RGCN.")
    model = RGCNNodeClassifier(
        in_channels=train_artifacts.homo_x.shape[1],
        hidden_channels=64,
        num_relations=max(1, len(train_artifacts.edge_types)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_part = train_artifacts.slice_maps.get("cluster_partition", {})
    train_labels = train_artifacts.homo_company_labels.float()
    train_mask = torch.zeros((train_artifacts.homo_x.shape[0],), dtype=torch.bool)
    for node_id, global_idx in zip(train_artifacts.homo_company_node_ids, train_artifacts.homo_company_global_indices):
        if global_idx >= train_mask.shape[0]:
            continue
        if not bool(train_artifacts.homo_company_mask[global_idx].item()):
            continue
        if train_part.get(node_id, "none") in {"train", "val"}:
            train_mask[global_idx] = True
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(train_artifacts.homo_x, train_artifacts.homo_edge_index, train_artifacts.homo_edge_type)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], train_labels[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_artifacts.homo_x, val_artifacts.homo_edge_index, val_artifacts.homo_edge_type)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = val_artifacts.homo_company_labels.detach().cpu().numpy()
    val_part = val_artifacts.slice_maps.get("cluster_partition", {})
    mask = np.zeros((len(labels),), dtype=bool)
    node_ids: list[str] = []
    probs_company: list[float] = []
    labels_company: list[int] = []
    for node_id, global_idx in zip(val_artifacts.homo_company_node_ids, val_artifacts.homo_company_global_indices):
        if global_idx >= len(labels):
            continue
        if labels[global_idx] < 0:
            continue
        if val_part.get(node_id, "none") != "test":
            continue
        node_ids.append(node_id)
        probs_company.append(float(probs[global_idx]))
        labels_company.append(int(labels[global_idx]))
    result = _evaluate_predictions(
        model_name="osint_rgcn",
        model_version=MODEL_VERSION,
        labels=np.asarray(labels_company, dtype=int),
        probs=np.asarray(probs_company, dtype=float),
        node_ids=node_ids,
        slice_maps=val_artifacts.slice_maps,
    )
    return model, result


def _train_baseline_logreg(train_artifacts, val_artifacts) -> tuple[Any, EvalResult]:
    x_train, y_train, train_node_ids, _ = _extract_company_arrays(train_artifacts)
    x_val, y_val, node_ids, slice_maps = _extract_company_arrays(val_artifacts)
    x_train, y_train, _ = _filter_by_cluster_partition(
        x_train, y_train, train_node_ids, train_artifacts.slice_maps, {"train", "val"}
    )
    x_val, y_val, node_ids = _filter_by_cluster_partition(x_val, y_val, node_ids, slice_maps, {"test"})
    if x_train.shape[0] == 0:
        raise RuntimeError("No labeled company samples available for heterograph baseline training.")
    if len(np.unique(y_train)) < 2:
        prior = float(np.mean(y_train)) if y_train.size else 0.0
        clf = {"model_type": "constant_prior", "positive_rate": prior}
        probs = np.full(shape=(len(y_val),), fill_value=prior, dtype=float)
    else:
        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        clf.fit(x_train, y_train)
        probs = clf.predict_proba(x_val)[:, 1]
    result = _evaluate_predictions(
        model_name="osint_graph_tabular_baseline",
        model_version=MODEL_VERSION,
        labels=y_val,
        probs=probs,
        node_ids=node_ids,
        slice_maps=slice_maps,
    )
    return clf, result


def _persist_eval(db, result: EvalResult, window_start: datetime, window_end: datetime) -> None:
    db.execute(
        text("DELETE FROM evaluation_slices WHERE model_name = :model_name AND model_version = :model_version"),
        {"model_name": result.model_name, "model_version": result.model_version},
    )
    db.execute(
        text("DELETE FROM calibration_bins WHERE model_name = :model_name AND model_version = :model_version"),
        {"model_name": result.model_name, "model_version": result.model_version},
    )
    for row in result.slice_metrics:
        db.execute(
            text(
                """
                INSERT INTO evaluation_slices (
                    model_name, model_version, slice_name, slice_value, metric_name, metric_value, sample_size, window_start, window_end, details
                )
                VALUES (
                    :model_name, :model_version, :slice_name, :slice_value, :metric_name, :metric_value, :sample_size, :window_start, :window_end, CAST(:details AS jsonb)
                )
                """
            ),
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "slice_name": row["slice_name"],
                "slice_value": row["slice_value"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "sample_size": row["sample_size"],
                "window_start": window_start,
                "window_end": window_end,
                "details": json.dumps({"stage": "heterograph_training"}, default=str),
            },
        )
    for row in result.calibration_bins:
        db.execute(
            text(
                """
                INSERT INTO calibration_bins (
                    model_name, model_version, bin_label, lower_bound, upper_bound, predicted_mean, observed_rate, sample_size
                )
                VALUES (
                    :model_name, :model_version, :bin_label, :lower_bound, :upper_bound, :predicted_mean, :observed_rate, :sample_size
                )
                """
            ),
            {
                "model_name": result.model_name,
                "model_version": result.model_version,
                **row,
            },
        )
    db.commit()


def _champion_decision(baseline: EvalResult, challenger: EvalResult, promotion_gate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pr_lift = challenger.metrics["pr_auc"] - baseline.metrics["pr_auc"]
    precision_lift = challenger.metrics["precision_at_100"] - baseline.metrics["precision_at_100"]
    ece_regression = challenger.metrics["ece"] - baseline.metrics["ece"]
    decision = "hold"
    if (
        pr_lift >= float(promotion_gate.get("minimum_pr_auc_lift", 0.03))
        and precision_lift >= float(promotion_gate.get("minimum_precision_at_100_lift", 0.05))
        and ece_regression <= float(promotion_gate.get("max_ece_regression", 0.02))
    ):
        decision = "promote"
    summary = {
        "baseline": baseline.metrics,
        "challenger": challenger.metrics,
        "pr_auc_lift": pr_lift,
        "precision_at_100_lift": precision_lift,
        "ece_regression": ece_regression,
    }
    return decision, summary


def train_osint_heterograph(months: int = 6, epochs: int = 30, preferred_model: str = "both") -> dict[str, Any]:
    with SessionLocal() as db:
        snapshots = _ensure_monthly_snapshots(db, months=months)
        split = temporal_split(snapshots)
        if not split["train"] or not split["val"]:
            raise RuntimeError("Need at least one snapshot to train OSINT heterograph.")

        benchmark_contract = load_latest_benchmark_contract(db)
        promotion_gate = benchmark_contract.get("promotion_gate") or {}
        if isinstance(promotion_gate, str):
            promotion_gate = json.loads(promotion_gate)

        train_artifacts = build_snapshot_graph(db, split["train"][-1])
        val_artifacts = build_snapshot_graph(db, split["val"][-1])
        test_artifacts = build_snapshot_graph(db, split["test"][-1])

        registry = ModelRegistryService(db)
        experiment_id = registry.ensure_experiment(
            experiment_key="osint_heterograph_training",
            model_name="osint_heterograph",
            objective="heterograph_node_classification",
            owner="ml-platform",
            metadata={
                "preferred_model": preferred_model,
                "months": months,
                "epochs": epochs,
            },
        )
        dataset_manifest = {
            "snapshots": split,
            "train_lineage_hash": train_artifacts.lineage_hash,
            "val_lineage_hash": val_artifacts.lineage_hash,
            "test_lineage_hash": test_artifacts.lineage_hash,
            "feature_names": train_artifacts.feature_names,
        }
        manifest_path = save_dataset_manifest(dataset_manifest, "osint_heterograph_dataset_manifest.json")
        dataset_version = f"osint-heterograph-dataset-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        dataset_version_id = registry.register_dataset_version(
            dataset_key="osint_heterograph_training_dataset",
            dataset_version=dataset_version,
            entity_type="graph_snapshot",
            row_count=len(train_artifacts.homo_company_node_ids) + len(val_artifacts.homo_company_node_ids) + len(test_artifacts.homo_company_node_ids),
            source_tables=["graph_snapshots", "graph_snapshot_nodes", "graph_snapshot_edges", "graph_labels"],
            filters={"split": split},
            data_hash=dataset_manifest["train_lineage_hash"],
            created_by="train_osint_heterograph",
        )
        label_version_row = db.execute(
            text(
                """
                SELECT id, label_version
                FROM label_versions
                WHERE label_key = 'osint_graph_supervision'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
        ).fetchone()
        label_version_id = int(label_version_row[0]) if label_version_row else None
        run_id = registry.start_training_run(
            model_name="osint_heterograph",
            experiment_id=experiment_id,
            model_version=MODEL_VERSION,
            dataset_version_id=dataset_version_id,
            label_version_id=label_version_id,
            seed=42,
            code_hash=dataset_manifest["val_lineage_hash"],
            hyperparams={"months": months, "epochs": epochs, "preferred_model": preferred_model},
        )

        baseline_model, baseline_eval = _train_baseline_logreg(train_artifacts, test_artifacts)
        _persist_eval(db, baseline_eval, train_artifacts.as_of_timestamp, test_artifacts.as_of_timestamp)
        train_company_labels = np.asarray(
            [train_artifacts.label_map.get(node_id, -1) for node_id in train_artifacts.homo_company_node_ids],
            dtype=int,
        )
        train_labeled_mask = train_company_labels >= 0
        can_train_graph = len(np.unique(train_company_labels[train_labeled_mask])) >= 2 if train_labeled_mask.any() else False

        model_results: dict[str, Any] = {
            "baseline": baseline_eval.metrics,
            "split": split,
            "dataset_version": dataset_version,
            "dataset_manifest_path": str(manifest_path),
            "graph_training_ready": bool(can_train_graph),
        }
        best_name = "baseline"
        best_eval = baseline_eval
        artifacts: dict[str, str] = {"dataset_manifest": str(manifest_path)}

        if preferred_model in {"both", "rgcn"} and PYG_AVAILABLE and DATASET_PYG_AVAILABLE and can_train_graph:
            rgcn_model, rgcn_eval = _train_rgcn(train_artifacts, test_artifacts, epochs=epochs)
            rgcn_path = MODEL_DIR / "osint_rgcn_model.pt"
            torch.save(rgcn_model.state_dict(), rgcn_path)
            artifacts["rgcn_model"] = str(rgcn_path)
            _persist_eval(db, rgcn_eval, train_artifacts.as_of_timestamp, test_artifacts.as_of_timestamp)
            model_results["rgcn"] = rgcn_eval.metrics
            if rgcn_eval.metrics["pr_auc"] > best_eval.metrics["pr_auc"]:
                best_name = "rgcn"
                best_eval = rgcn_eval

        if preferred_model in {"both", "hgt"} and PYG_AVAILABLE and DATASET_PYG_AVAILABLE and train_artifacts.hetero_data is not None and can_train_graph:
            hgt_model, hgt_eval = _train_hgt(train_artifacts, test_artifacts, epochs=epochs)
            hgt_path = MODEL_DIR / "osint_hgt_model.pt"
            torch.save(hgt_model.state_dict(), hgt_path)
            artifacts["hgt_model"] = str(hgt_path)
            _persist_eval(db, hgt_eval, train_artifacts.as_of_timestamp, test_artifacts.as_of_timestamp)
            model_results["hgt"] = hgt_eval.metrics
            if hgt_eval.metrics["pr_auc"] > best_eval.metrics["pr_auc"]:
                best_name = "hgt"
                best_eval = hgt_eval
        if not can_train_graph:
            model_results["graph_training_blocker"] = "Training split currently has one class only; baseline persisted, graph challengers skipped."

        baseline_path = MODEL_DIR / "osint_graph_tabular_baseline.joblib"
        joblib.dump(baseline_model, baseline_path)
        artifacts["baseline_model"] = str(baseline_path)

        decision, summary = _champion_decision(baseline_eval, best_eval, promotion_gate)
        db.execute(
            text(
                """
                INSERT INTO champion_challenger_results (
                    model_name, champion_version, challenger_version, decision, metric_summary, notes
                )
                VALUES (
                    'osint_heterograph', 'osint_graph_tabular_baseline', :challenger_version, :decision,
                    CAST(:metric_summary AS jsonb), :notes
                )
                """
            ),
            {
                "challenger_version": MODEL_VERSION,
                "decision": decision,
                "metric_summary": json.dumps(summary, default=str),
                "notes": f"Selected best model family: {best_name}",
            },
        )
        db.commit()

        registry.complete_training_run(
            run_id=run_id,
            status="completed",
            metrics={
                "best_model_family": best_name,
                "decision": decision,
                **best_eval.metrics,
            },
            artifacts=artifacts,
        )
        registry.register_rollout(
            model_name="osint_heterograph",
            model_version=MODEL_VERSION,
            environment="staging",
            rollout_type="shadow",
            status="planned" if decision == "hold" else "approved",
            notes=f"Heterograph training complete. Best model={best_name}.",
            metadata={"decision": decision, "summary": summary, "split": split},
        )

        report = {
            "model_version": MODEL_VERSION,
            "best_model_family": best_name,
            "decision": decision,
            "summary": summary,
            "results": model_results,
            "artifacts": artifacts,
        }
        report_path = MODEL_DIR / "osint_heterograph_training_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--preferred-model", choices=["both", "rgcn", "hgt"], default="both")
    args = parser.parse_args()
    result = train_osint_heterograph(
        months=max(1, int(args.months)),
        epochs=max(1, int(args.epochs)),
        preferred_model=args.preferred_model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
