"""Retrain macro digital-twin models from reviewed scenario data.

This module keeps the production data path explicit:

1. crawled/live data enters the review queue with provenance and deduplication;
2. a human reviewer approves or rejects it;
3. approved rows and approved text-scenario memories become training signals.

The script intentionally does not crawl or call LLMs. It only consumes local
canonical/reviewed data and writes auditable model artifacts.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from ml_engine.macro_event_ingest import REVIEW_QUEUE_PATH
from ml_engine.macro_scenario_engine import (
    ScenarioParams,
    compute_scenario,
    load_events,
    load_provinces,
)


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "data"
MODEL_DIR = BACKEND_DIR / "data" / "models"
MEMORY_PATH = DATA_DIR / "macro_text_scenario_memory.jsonl"
TAX_POLICY_SCENARIOS_PATH = DATA_DIR / "tax_policy_scenarios.json"

EVENT_MODEL_PATH = MODEL_DIR / "macro_event_impact_model.joblib"
PROVINCE_MODEL_PATH = MODEL_DIR / "macro_province_response_model.joblib"
REPORT_PATH = MODEL_DIR / "macro_retrain_report.json"
DATASET_PREVIEW_PATH = MODEL_DIR / "macro_retrain_dataset_preview.jsonl"

MODEL_VERSION = "macro-digital-twin-reviewed-v1"

EVENT_TYPES = [
    "policy",
    "trade_war",
    "natural_disaster",
    "pandemic",
    "financial_crisis",
    "trade_agreement",
    "growth",
    "geopolitics",
    "infrastructure_shock",
    "sanction",
    "war",
    "unknown",
]

SEVERITIES = ["low", "medium", "high", "extreme"]

EVENT_TARGETS = [
    "gdp_delta_pct",
    "tax_rate_delta",
    "compliance_delta",
    "unemployment_delta",
    "fdi_delta_pct",
    "tax_revenue_delta_pct",
]

PROVINCE_TARGETS = [
    "delta_revenue_pct",
    "risk_score",
    "projected_compliance",
    "projected_unemployment",
    "delta_gdp_pct",
]

KEYWORD_FLAGS = [
    ("tariff", ("thuế quan", "áp thuế", "đánh thuế", "tariff", "export duty")),
    ("sanction", ("cấm vận", "trừng phạt", "sanction", "embargo")),
    ("pandemic", ("đại dịch", "dịch bệnh", "pandemic", "covid")),
    ("war", ("chiến tranh", "xung đột", "war", "conflict")),
    ("disaster", ("bão", "lũ", "hạn hán", "thiên tai", "disaster")),
    ("financial", ("lãi suất", "tỷ giá", "ngân hàng", "khủng hoảng tài chính")),
    ("investment", ("fdi", "đầu tư", "nâng hạng", "chuỗi cung ứng")),
    ("tax_policy", ("vat", "gtgt", "tndn", "thuế suất", "giảm thuế", "tăng thuế")),
]


@dataclass
class TrainingSourceRow:
    source: str
    text: str
    event_type: str
    severity: str
    affected_provinces: List[str]
    affected_sectors: List[str]
    target: Dict[str, float]
    trust_weight: float = 1.0


def run_retrain(
    *,
    min_samples: int = 3000,
    seed: int = 42,
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train both text-impact and province-response models."""
    rng = np.random.default_rng(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    source_rows, source_counts = collect_training_sources()
    if not source_rows:
        raise RuntimeError("No canonical or reviewed macro scenario data found.")

    event_X, event_y, event_meta = build_event_training_matrix(
        source_rows,
        min_samples=max(300, int(min_samples)),
        rng=rng,
    )
    province_X, province_y, province_meta = build_province_training_matrix(
        source_rows,
        min_samples=max(300, int(min_samples)),
        rng=rng,
    )

    event_report, event_bundle = _train_regressor_bundle(
        event_X,
        event_y,
        feature_names=event_feature_names(),
        target_names=EVENT_TARGETS,
        seed=seed,
        model_name="macro_event_impact_model",
    )
    province_report, province_bundle = _train_regressor_bundle(
        province_X,
        province_y,
        feature_names=province_feature_names(),
        target_names=PROVINCE_TARGETS,
        seed=seed,
        model_name="macro_province_response_model",
    )

    report = {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
        "source_counts": source_counts,
        "dataset": {
            "review_policy": "canonical events + human-approved memories/review-queue rows only",
            "source_rows": len(source_rows),
            "event_samples": int(event_X.shape[0]),
            "province_samples": int(province_X.shape[0]),
            "event_source_mix": _count_values([m["source"] for m in event_meta]),
            "province_source_mix": _count_values([m["source"] for m in province_meta]),
        },
        "models": {
            "event_impact": event_report,
            "province_response": province_report,
        },
        "artifacts": {
            "event_impact_model": str(EVENT_MODEL_PATH),
            "province_response_model": str(PROVINCE_MODEL_PATH),
            "report": str(REPORT_PATH),
            "dataset_preview": str(DATASET_PREVIEW_PATH),
        },
        "next_training_gate": {
            "crawl": "python Backend/scripts/crawl_macro_news.py --max-per-feed 20 --use-llm",
            "review": "Approve provenance/dedup queue rows before training.",
            "retrain": "python Backend/scripts/retrain_macro_from_reviewed_data.py --min-samples 5000",
        },
    }

    if write_artifacts:
        event_bundle.update({
            "model_version": MODEL_VERSION,
            "artifact_role": "text_scenario_to_macro_parameters",
            "created_at": report["generated_at"],
            "source_counts": source_counts,
        })
        province_bundle.update({
            "model_version": MODEL_VERSION,
            "artifact_role": "province_response_to_macro_parameters",
            "created_at": report["generated_at"],
            "source_counts": source_counts,
        })
        joblib.dump(event_bundle, EVENT_MODEL_PATH)
        joblib.dump(province_bundle, PROVINCE_MODEL_PATH)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_dataset_preview(source_rows, event_meta, province_meta)

    return report


def collect_training_sources() -> Tuple[List[TrainingSourceRow], Dict[str, int]]:
    rows: List[TrainingSourceRow] = []
    counts = {
        "canonical_events": 0,
        "approved_text_memories": 0,
        "approved_review_queue": 0,
        "tax_policy_scenarios": 0,
        "rejected_or_pending_review_queue": 0,
        "rejected_or_low_rating_memories": 0,
    }

    for event in load_events():
        row = _source_from_canonical_event(event)
        if row:
            rows.append(row)
            counts["canonical_events"] += 1

    for memory in _read_jsonl(MEMORY_PATH):
        if memory.get("review_status") == "approved" and float(memory.get("rating") or 0.0) >= 4.0:
            row = _source_from_text_memory(memory)
            if row:
                rows.append(row)
                counts["approved_text_memories"] += 1
        else:
            counts["rejected_or_low_rating_memories"] += 1

    for queued in _read_jsonl(REVIEW_QUEUE_PATH):
        if queued.get("review_status") == "approved":
            row = _source_from_review_queue(queued)
            if row:
                rows.append(row)
                counts["approved_review_queue"] += 1
        else:
            counts["rejected_or_pending_review_queue"] += 1

    for scenario in _read_json(TAX_POLICY_SCENARIOS_PATH):
        row = _source_from_canonical_event(scenario)
        if row:
            row.source = "tax_policy_scenario"
            rows.append(row)
            counts["tax_policy_scenarios"] += 1

    return rows, counts


def build_event_training_matrix(
    source_rows: Sequence[TrainingSourceRow],
    *,
    min_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    weighted_rows: List[TrainingSourceRow] = []
    for row in source_rows:
        repeats = max(1, min(8, int(round(float(row.trust_weight)))))
        weighted_rows.extend([row] * repeats)

    X: List[List[float]] = []
    y: List[List[float]] = []
    meta: List[Dict[str, Any]] = []
    for idx in range(max(int(min_samples), len(weighted_rows))):
        row = weighted_rows[idx % len(weighted_rows)] if idx < len(weighted_rows) else rng.choice(weighted_rows)
        jitter = 0.0 if idx < len(weighted_rows) else float(rng.normal(0.0, 0.045))
        target = _jitter_target(row.target, row.severity, jitter)
        X.append(encode_event_features(
            text=row.text,
            event_type=row.event_type,
            severity=row.severity,
            affected_provinces=row.affected_provinces,
            affected_sectors=row.affected_sectors,
            trust_weight=row.trust_weight,
        ))
        y.append([float(target.get(name, 0.0)) for name in EVENT_TARGETS])
        meta.append({"source": row.source, "event_type": row.event_type, "severity": row.severity})
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), meta


def build_province_training_matrix(
    source_rows: Sequence[TrainingSourceRow],
    *,
    min_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    provinces = load_provinces()
    if not provinces:
        raise RuntimeError("No province baselines available for macro response training.")

    weighted_rows: List[TrainingSourceRow] = []
    for row in source_rows:
        repeats = max(1, min(8, int(round(float(row.trust_weight)))))
        weighted_rows.extend([row] * repeats)

    X: List[List[float]] = []
    y: List[List[float]] = []
    meta: List[Dict[str, Any]] = []
    target_samples = max(int(min_samples), len(provinces) * min(10, len(weighted_rows)))

    for idx in range(target_samples):
        row = weighted_rows[idx % len(weighted_rows)] if idx < len(weighted_rows) else rng.choice(weighted_rows)
        province = provinces[idx % len(provinces)] if idx < len(provinces) else rng.choice(provinces)
        jitter = 0.0 if idx < len(weighted_rows) else float(rng.normal(0.0, 0.04))
        impact = _jitter_target(row.target, row.severity, jitter)
        params = ScenarioParams(
            gdp_delta_pct=float(impact.get("gdp_delta_pct", 0.0)),
            tax_rate_delta=float(impact.get("tax_rate_delta", 0.0)),
            compliance_delta=float(impact.get("compliance_delta", 0.0)),
            unemployment_delta=float(impact.get("unemployment_delta", 0.0)),
            fdi_delta_pct=float(impact.get("fdi_delta_pct", 0.0)),
        )
        try:
            result = compute_scenario(str(province.get("province_code")), params)
        except Exception:
            continue
        X.append(encode_province_features(province, impact, horizon_years=int(rng.choice([1, 5, 10, 20]))))
        y.append([
            float(result.delta_revenue_pct),
            _risk_to_score(result.projected_risk),
            float(result.projected_compliance),
            float(result.projected_unemployment),
            float(result.delta_gdp_pct),
        ])
        meta.append({
            "source": row.source,
            "province_code": province.get("province_code"),
            "event_type": row.event_type,
        })
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), meta


def predict_scenario_from_trained_model(text: str, *, min_confidence: float = 0.55) -> Optional[Dict[str, Any]]:
    """Predict macro parameters locally from the reviewed scenario model."""
    if not EVENT_MODEL_PATH.exists():
        return None
    meta = infer_event_metadata(text)
    if meta["event_type"] == "unknown" and meta["keyword_score"] < 0.2:
        return None
    try:
        bundle = joblib.load(EVENT_MODEL_PATH)
        model = bundle["model"]
        X = np.asarray([encode_event_features(
            text=text,
            event_type=meta["event_type"],
            severity=meta["severity"],
            affected_provinces=[],
            affected_sectors=meta["affected_sectors"],
            trust_weight=3.5,
        )], dtype=float)
        pred = model.predict(X)[0]
    except Exception:
        return None

    values = {name: float(pred[idx]) for idx, name in enumerate(EVENT_TARGETS)}
    params = {
        "gdp_delta_pct": _clip(values.get("gdp_delta_pct", 0.0), -18.0, 14.0),
        "tax_rate_delta": _clip(values.get("tax_rate_delta", 0.0), -0.08, 0.08),
        "compliance_delta": _clip(values.get("compliance_delta", 0.0), -0.15, 0.15),
        "unemployment_delta": _clip(values.get("unemployment_delta", 0.0), -3.0, 7.0),
        "fdi_delta_pct": _clip(values.get("fdi_delta_pct", 0.0), -55.0, 35.0),
    }
    confidence = min(0.82, max(min_confidence, 0.52 + 0.16 * meta["keyword_score"] + 0.04 * len(meta["affected_sectors"])))
    if confidence < min_confidence:
        return None

    return {
        "scenario_title": meta["title"],
        "event_type": meta["event_type"],
        "severity": meta["severity"],
        "affected_provinces": [],
        "affected_sectors": meta["affected_sectors"],
        "macro_parameters": params,
        "candidate_events": [
            {
                "headline": meta["title"],
                "summary": "Mô hình đã học từ sự kiện lịch sử và kịch bản đã được duyệt để ước lượng tác động vĩ mô sơ bộ.",
                "probability": round(confidence, 3),
                "impact_level": meta["severity"],
            }
        ],
        "reasoning_brief": "Dự đoán cục bộ từ macro_event_impact_model, cần human review trước khi dùng làm dữ liệu học lâu dài.",
        "confidence": round(confidence, 3),
        "llm_provider": "local_reviewed_model",
    }


def event_feature_names() -> List[str]:
    names = [
        "text_len_log",
        "affected_province_count_log",
        "affected_sector_count",
        "trust_weight",
        "keyword_score",
    ]
    names += [f"event_type__{name}" for name in EVENT_TYPES]
    names += [f"severity__{name}" for name in SEVERITIES]
    names += [f"keyword__{name}" for name, _ in KEYWORD_FLAGS]
    return names


def province_feature_names() -> List[str]:
    return [
        "province_gdp_log",
        "province_population_log",
        "province_tax_revenue_log",
        "province_enterprise_log",
        "baseline_compliance",
        "baseline_unemployment",
        "baseline_fdi_log",
        "region_bucket",
        "horizon_years",
        "scenario_gdp_delta_pct",
        "scenario_tax_rate_delta",
        "scenario_compliance_delta",
        "scenario_unemployment_delta",
        "scenario_fdi_delta_pct",
        "scenario_tax_revenue_delta_pct",
    ]


def encode_event_features(
    *,
    text: str,
    event_type: str,
    severity: str,
    affected_provinces: Sequence[str],
    affected_sectors: Sequence[str],
    trust_weight: float,
) -> List[float]:
    normalized = normalize_text(text)
    keyword_values = [_contains_any(normalized, terms) for _, terms in KEYWORD_FLAGS]
    keyword_score = sum(keyword_values) / max(1, len(keyword_values))
    row: List[float] = [
        math.log1p(len(normalized)),
        math.log1p(len(affected_provinces)),
        float(len(affected_sectors)),
        float(trust_weight),
        float(keyword_score),
    ]
    row += [1.0 if event_type == name else 0.0 for name in EVENT_TYPES]
    row += [1.0 if severity == name else 0.0 for name in SEVERITIES]
    row += [float(v) for v in keyword_values]
    return row


def encode_province_features(province: Dict[str, Any], impact: Dict[str, float], *, horizon_years: int) -> List[float]:
    return [
        math.log1p(float(province.get("gdp_billion_vnd") or 0.0)),
        math.log1p(float(province.get("population") or 0.0)),
        math.log1p(float(province.get("tax_revenue_billion_vnd") or 0.0)),
        math.log1p(float(province.get("num_enterprises") or 0.0)),
        float(province.get("compliance_rate") or 0.0),
        float(province.get("unemployment_rate") or 0.0),
        math.log1p(float(province.get("fdi_billion_usd") or 0.0)),
        _stable_bucket(province.get("region"), 19) / 18.0,
        float(horizon_years),
        float(impact.get("gdp_delta_pct", 0.0)),
        float(impact.get("tax_rate_delta", 0.0)),
        float(impact.get("compliance_delta", 0.0)),
        float(impact.get("unemployment_delta", 0.0)),
        float(impact.get("fdi_delta_pct", 0.0)),
        float(impact.get("tax_revenue_delta_pct", 0.0)),
    ]


def infer_event_metadata(text: str) -> Dict[str, Any]:
    normalized = normalize_text(text)
    event_type = "unknown"
    severity = "medium"
    sectors: List[str] = []

    if _contains_any(normalized, KEYWORD_FLAGS[0][1]):
        event_type = "trade_war"
        sectors.extend(["Xuất nhập khẩu", "Sản xuất", "Logistics"])
        severity = "high" if _extract_first_number(normalized) >= 25 else "medium"
    if _contains_any(normalized, KEYWORD_FLAGS[1][1]):
        event_type = "sanction"
        sectors.extend(["Ngân hàng", "Xuất nhập khẩu", "Năng lượng"])
        severity = "extreme"
    if _contains_any(normalized, KEYWORD_FLAGS[2][1]):
        event_type = "pandemic"
        sectors.extend(["Du lịch", "Y tế", "Logistics", "Dịch vụ"])
        severity = "extreme"
    if _contains_any(normalized, KEYWORD_FLAGS[3][1]):
        event_type = "war" if event_type == "unknown" else event_type
        sectors.extend(["Năng lượng", "Logistics", "Xuất nhập khẩu"])
        severity = "extreme"
    if _contains_any(normalized, KEYWORD_FLAGS[4][1]):
        event_type = "natural_disaster" if event_type == "unknown" else event_type
        sectors.extend(["Nông nghiệp", "Logistics", "Xây dựng"])
        severity = "high"
    if _contains_any(normalized, KEYWORD_FLAGS[5][1]) and event_type == "unknown":
        event_type = "financial_crisis"
        sectors.extend(["Tài chính", "Bất động sản", "Sản xuất"])
        severity = "high"
    if _contains_any(normalized, KEYWORD_FLAGS[6][1]) and event_type == "unknown":
        event_type = "growth"
        sectors.extend(["Công nghệ", "Sản xuất", "Dịch vụ"])
        severity = "medium"
    if _contains_any(normalized, KEYWORD_FLAGS[7][1]) and event_type == "unknown":
        event_type = "policy"
        sectors.extend(["Thương mại", "Dịch vụ", "Sản xuất"])
        severity = "medium"

    keyword_score = sum(_contains_any(normalized, terms) for _, terms in KEYWORD_FLAGS) / max(1, len(KEYWORD_FLAGS))
    title = _make_title(text, event_type)
    return {
        "event_type": event_type,
        "severity": severity,
        "affected_sectors": sorted(set(sectors)),
        "keyword_score": float(keyword_score),
        "title": title,
    }


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s.%+-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _source_from_canonical_event(event: Dict[str, Any]) -> Optional[TrainingSourceRow]:
    title = str(event.get("event_name_vi") or event.get("event_name") or event.get("event_key") or "")
    description = str(event.get("description_vi") or event.get("description") or "")
    if not title:
        return None
    gdp = float(event.get("impact_gdp_pct") or 0.0)
    tax = float(event.get("impact_tax_revenue_pct") or 0.0)
    unemp = float(event.get("impact_unemployment_pct") or 0.0)
    fdi = float(event.get("impact_fdi_pct") or 0.0)
    return TrainingSourceRow(
        source="canonical_event",
        text=f"{title}. {description}",
        event_type=_safe_event_type(event.get("event_type")),
        severity=_safe_severity(event.get("severity")),
        affected_provinces=[str(x) for x in (event.get("affected_provinces") or [])],
        affected_sectors=[str(x) for x in (event.get("affected_sectors") or [])],
        target={
            "gdp_delta_pct": gdp,
            "tax_rate_delta": float(event.get("tax_rate_delta") or event.get("policy_tax_rate_delta") or 0.0),
            "compliance_delta": float(event.get("compliance_delta") or _estimate_compliance_delta(gdp, tax)),
            "unemployment_delta": unemp,
            "fdi_delta_pct": fdi,
            "tax_revenue_delta_pct": tax,
        },
        trust_weight=3.5,
    )


def _source_from_text_memory(memory: Dict[str, Any]) -> Optional[TrainingSourceRow]:
    payload = dict(memory.get("payload") or {})
    params = dict(payload.get("macro_parameters") or {})
    text = str(memory.get("scenario_text") or payload.get("scenario_title") or "")
    if not text or not params:
        return None
    rating = max(4.0, min(5.0, float(memory.get("rating") or 4.0)))
    gdp = float(params.get("gdp_delta_pct") or 0.0)
    tax_rate = float(params.get("tax_rate_delta") or 0.0)
    compliance = float(params.get("compliance_delta") or 0.0)
    unemp = float(params.get("unemployment_delta") or 0.0)
    fdi = float(params.get("fdi_delta_pct") or 0.0)
    tax_delta = float(params.get("tax_revenue_delta_pct") or (gdp + compliance * 100.0 * 0.45 + tax_rate * 100.0 * 0.55 - unemp * 0.35))
    return TrainingSourceRow(
        source="approved_text_memory",
        text=text,
        event_type=_safe_event_type(payload.get("event_type")),
        severity=_safe_severity(payload.get("severity")),
        affected_provinces=[str(x) for x in (payload.get("affected_provinces") or [])],
        affected_sectors=[str(x) for x in (payload.get("affected_sectors") or [])],
        target={
            "gdp_delta_pct": gdp,
            "tax_rate_delta": tax_rate,
            "compliance_delta": compliance,
            "unemployment_delta": unemp,
            "fdi_delta_pct": fdi,
            "tax_revenue_delta_pct": tax_delta,
        },
        trust_weight=rating,
    )


def _source_from_review_queue(row: Dict[str, Any]) -> Optional[TrainingSourceRow]:
    candidate = dict(row.get("candidate") or {})
    hints = dict(candidate.get("impact_hints") or {})
    title = str(candidate.get("title") or "")
    description = str(candidate.get("description") or "")
    if not title:
        return None
    gdp = float(hints.get("gdp_delta_pct") or hints.get("impact_gdp_pct") or 0.0)
    tax = float(hints.get("tax_revenue_delta_pct") or hints.get("impact_tax_revenue_pct") or 0.0)
    unemp = float(hints.get("unemployment_delta") or hints.get("impact_unemployment_pct") or 0.0)
    fdi = float(hints.get("fdi_delta_pct") or hints.get("impact_fdi_pct") or 0.0)
    compliance = float(hints.get("compliance_delta") or _estimate_compliance_delta(gdp, tax))
    return TrainingSourceRow(
        source="approved_review_queue",
        text=f"{title}. {description}",
        event_type=_safe_event_type(candidate.get("event_type")),
        severity=_safe_severity(candidate.get("severity") or hints.get("severity")),
        affected_provinces=[str(x) for x in (candidate.get("affected_provinces") or [])],
        affected_sectors=[str(x) for x in (candidate.get("affected_sectors") or [])],
        target={
            "gdp_delta_pct": gdp,
            "tax_rate_delta": float(hints.get("tax_rate_delta") or 0.0),
            "compliance_delta": compliance,
            "unemployment_delta": unemp,
            "fdi_delta_pct": fdi,
            "tax_revenue_delta_pct": tax or (gdp + compliance * 45.0 - unemp * 0.35),
        },
        trust_weight=float(row.get("review_rating") or 4.5),
    )


def _train_regressor_bundle(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: List[str],
    target_names: List[str],
    seed: int,
    model_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=seed)
    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=14,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    per_target = {}
    r2_values: List[float] = []
    for idx, target in enumerate(target_names):
        variance = float(np.var(y_test[:, idx]))
        target_r2 = None if variance < 1e-10 else float(r2_score(y_test[:, idx], preds[:, idx]))
        if target_r2 is not None:
            r2_values.append(target_r2)
        per_target[target] = {
            "mae": round(float(mean_absolute_error(y_test[:, idx], preds[:, idx])), 5),
            "r2": None if target_r2 is None else round(target_r2, 5),
            "constant_target": variance < 1e-10,
        }
    report = {
        "model_name": model_name,
        "model_type": "sklearn_random_forest_regressor",
        "samples": int(X.shape[0]),
        "features": len(feature_names),
        "targets": target_names,
        "metrics": {
            "mean_mae": round(float(mean_absolute_error(y_test, preds)), 5),
            "mean_r2": round(float(sum(r2_values) / max(1, len(r2_values))), 5),
            "per_target": per_target,
        },
    }
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
        "report": report,
    }
    return report, bundle


def _jitter_target(target: Dict[str, float], severity: str, jitter: float) -> Dict[str, float]:
    scale = {
        "low": 0.35,
        "medium": 0.55,
        "high": 0.8,
        "extreme": 1.05,
    }.get(severity, 0.55)
    return {
        "gdp_delta_pct": _clip(float(target.get("gdp_delta_pct", 0.0)) + jitter * scale * 3.0, -18.0, 14.0),
        "tax_rate_delta": _clip(float(target.get("tax_rate_delta", 0.0)), -0.08, 0.08),
        "compliance_delta": _clip(float(target.get("compliance_delta", 0.0)) + jitter * 0.015, -0.15, 0.15),
        "unemployment_delta": _clip(float(target.get("unemployment_delta", 0.0)) - jitter * scale * 1.2, -3.0, 7.0),
        "fdi_delta_pct": _clip(float(target.get("fdi_delta_pct", 0.0)) + jitter * scale * 8.0, -55.0, 35.0),
        "tax_revenue_delta_pct": _clip(float(target.get("tax_revenue_delta_pct", 0.0)) + jitter * scale * 4.0, -30.0, 24.0),
    }


def _write_dataset_preview(
    source_rows: Sequence[TrainingSourceRow],
    event_meta: Sequence[Dict[str, Any]],
    province_meta: Sequence[Dict[str, Any]],
) -> None:
    preview = []
    for row in source_rows[:30]:
        preview.append({
            "kind": "source",
            "source": row.source,
            "event_type": row.event_type,
            "severity": row.severity,
            "text": row.text[:240],
            "target": row.target,
        })
    preview.append({"kind": "event_meta_sample", "rows": list(event_meta[:20])})
    preview.append({"kind": "province_meta_sample", "rows": list(province_meta[:20])})
    with DATASET_PREVIEW_PATH.open("w", encoding="utf-8") as fh:
        for item in preview:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _read_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else list(parsed.get("items") or [])


def _contains_any(normalized_text: str, terms: Iterable[str]) -> int:
    return int(any(normalize_text(term) in normalized_text for term in terms))


def _extract_first_number(text: str) -> float:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0).replace(",", "."))


def _make_title(text: str, event_type: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) > 76:
        clean = clean[:73].rstrip() + "..."
    if clean:
        return clean[0].upper() + clean[1:]
    return f"Kịch bản {event_type}"


def _safe_event_type(value: Any) -> str:
    text = str(value or "unknown").strip()
    return text if text in EVENT_TYPES else "unknown"


def _safe_severity(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    return text if text in SEVERITIES else "medium"


def _estimate_compliance_delta(gdp_delta_pct: float, tax_revenue_delta_pct: float) -> float:
    return _clip((float(tax_revenue_delta_pct) - float(gdp_delta_pct)) / 180.0, -0.12, 0.10)


def _risk_to_score(level: str) -> float:
    return {"low": 0.18, "medium": 0.55, "high": 0.88}.get(str(level or "").lower(), 0.5)


def _stable_bucket(value: Any, modulo: int) -> int:
    text = normalize_text(value)
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
    return total % max(1, modulo)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _count_values(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out
