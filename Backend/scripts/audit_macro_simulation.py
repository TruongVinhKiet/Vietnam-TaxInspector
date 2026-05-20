"""
Audit the Vietnam Digital Twin macro simulation assets.

This script is intentionally lightweight: it does not train models or require a
database. It validates that the offline assets and deterministic scenario engine
are production-ready enough for the frontend to render and for demos to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
REPORT_DIR = BACKEND / "reports"
DEFAULT_BOUNDARY_VERSION = "vn_34_2025"
EXPECTED_PROVINCE_COUNT = 34
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_scenario_engine import (  # noqa: E402
    ScenarioParams,
    build_vietnam_geojson,
    compute_scenario,
    get_events_for_province,
    load_events,
    load_provinces,
)
from ml_engine.admin_boundary_manager import audit_boundary_readiness  # noqa: E402
from ml_engine.macro_event_ingest import build_ingest_status  # noqa: E402


REQUIRED_PROVINCE_FIELDS = {
    "province_code",
    "province_name",
    "region",
    "gdp_billion_vnd",
    "tax_revenue_billion_vnd",
    "lat",
    "lng",
}

REQUIRED_EVENT_FIELDS = {
    "event_key",
    "event_name_vi",
    "event_type",
    "start_date",
    "impact_gdp_pct",
    "impact_tax_revenue_pct",
}


def _missing_fields(rows: List[Dict[str, Any]], required: set[str], sample_limit: int = 20) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = sorted(field for field in required if field not in row or row.get(field) in (None, ""))
        if missing:
            findings.append({
                "index": idx,
                "key": row.get("province_code") or row.get("event_key") or idx,
                "missing": missing,
            })
            if len(findings) >= sample_limit:
                break
    return findings


def _province_event_coverage(provinces: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    weak: List[Dict[str, Any]] = []
    for province in provinces:
        code = str(province.get("province_code") or "")
        events = get_events_for_province(code, limit=80)
        counts[code] = len(events)
        if len(events) < 5:
            weak.append({
                "province_code": code,
                "province_name": province.get("province_name"),
                "event_count": len(events),
            })
    values = list(counts.values())
    return {
        "min_events_per_province": min(values) if values else 0,
        "max_events_per_province": max(values) if values else 0,
        "avg_events_per_province": round(sum(values) / max(len(values), 1), 2),
        "weak_coverage": weak,
    }


def _scenario_smoke_tests() -> List[Dict[str, Any]]:
    probes = [
        ("VN34-HN", ScenarioParams(gdp_delta_pct=2.5, compliance_delta=0.03)),
        ("VN34-HCM", ScenarioParams(gdp_delta_pct=-4.0, tax_rate_delta=-0.01, compliance_delta=-0.02)),
        ("VN34-DN", ScenarioParams(gdp_delta_pct=-6.0, compliance_delta=-0.04)),
        ("VN34-CT", ScenarioParams(gdp_delta_pct=3.0, tax_rate_delta=0.005, compliance_delta=0.02)),
    ]
    rows: List[Dict[str, Any]] = []
    for province_code, params in probes:
        try:
            result = compute_scenario(province_code, params)
            rows.append({
                "province_code": province_code,
                "ok": True,
                "projected_risk": result.projected_risk,
                "delta_revenue_pct": result.delta_revenue_pct,
                "confidence_score": result.confidence_score,
                "uncertainty_pct": result.uncertainty_band_revenue.get("uncertainty_pct"),
                "related_event_count": len(result.related_events),
            })
        except Exception as exc:
            rows.append({"province_code": province_code, "ok": False, "error": str(exc)})
    return rows


def build_report(*, production: bool = False) -> Dict[str, Any]:
    provinces = load_provinces(DEFAULT_BOUNDARY_VERSION)
    events = load_events()
    event_types = Counter(str(event.get("event_type") or "unknown") for event in events)
    geojson = build_vietnam_geojson(DEFAULT_BOUNDARY_VERSION)
    smoke = _scenario_smoke_tests()
    boundary = audit_boundary_readiness(production=production)
    ingest_status = build_ingest_status()
    frontend_findings = _frontend_map_findings()

    hard_failures: List[str] = []
    if len(provinces) != EXPECTED_PROVINCE_COUNT:
        hard_failures.append(f"Expected {EXPECTED_PROVINCE_COUNT} provinces for {DEFAULT_BOUNDARY_VERSION}, found {len(provinces)}")
    if len(events) < 100:
        hard_failures.append(f"Expected at least 100 macro events, found {len(events)}")
    if len(geojson.get("features", [])) != len(provinces):
        hard_failures.append("GeoJSON fallback feature count does not match province count")
    if any(not row.get("ok") for row in smoke):
        hard_failures.append("One or more scenario smoke tests failed")

    province_missing = _missing_fields(provinces, REQUIRED_PROVINCE_FIELDS)
    event_missing = _missing_fields(events, REQUIRED_EVENT_FIELDS)
    if province_missing:
        hard_failures.append("Province records have missing required fields")
    if event_missing:
        hard_failures.append("Event records have missing required fields")

    coverage = _province_event_coverage(provinces)
    if coverage["weak_coverage"]:
        hard_failures.append("Some provinces have weak event coverage")
    hard_failures.extend(boundary.get("failures", []))
    if production and boundary.get("active_version") != DEFAULT_BOUNDARY_VERSION:
        hard_failures.append(f"Production active boundary must be `{DEFAULT_BOUNDARY_VERSION}`, found `{boundary.get('active_version')}`")
    if frontend_findings["legacy_geojson_reads"]:
        hard_failures.append("Simulation frontend map files still read legacy ../json/vietnam.json.")
    if frontend_findings["random_mock_risk"]:
        hard_failures.append("Simulation frontend map files still use random/mock risk generation.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not hard_failures else "needs_attention",
        "hard_failures": hard_failures,
        "warnings": boundary.get("warnings", []),
        "province_count": len(provinces),
        "event_count": len(events),
        "event_types": dict(event_types.most_common()),
        "province_missing_fields": province_missing,
        "event_missing_fields": event_missing,
        "event_coverage": coverage,
        "frontend_map_findings": frontend_findings,
        "geojson": {
            "feature_count": len(geojson.get("features", [])),
            "source": geojson.get("metadata", {}).get("source"),
            "boundary_precision": geojson.get("metadata", {}).get("boundary_precision"),
            "boundary_version": geojson.get("metadata", {}).get("boundary_version"),
            "boundary_status": geojson.get("metadata", {}).get("boundary_status"),
        },
        "boundary_readiness": boundary,
        "event_ingest_status": ingest_status,
        "scenario_smoke_tests": smoke,
        "recommendations": _recommendations(hard_failures, coverage),
    }


def _recommendations(hard_failures: List[str], coverage: Dict[str, Any]) -> List[str]:
    recommendations = []
    if coverage.get("weak_coverage"):
        recommendations.append("Add province-specific events for weakly covered provinces before claiming high-resolution provincial forecasting.")
    recommendations.append("Load reviewed official GeoJSON for `vn_34_2025` before production map/legal-boundary use.")
    recommendations.append("Attach real-time crawlers through `macro_event_ingest`; keep new events in review queue before model use.")
    recommendations.append("Retrain macro models only after event provenance and label quality pass this audit.")
    if not hard_failures:
        recommendations.insert(0, "Current offline Digital Twin assets are sufficient for demo and deterministic scenario simulation.")
    return recommendations


def _frontend_map_findings() -> Dict[str, Any]:
    files = [
        ROOT / "Frontend" / "js" / "vietnam_map.js",
        ROOT / "Frontend" / "js" / "vietnam_3d_map.js",
        ROOT / "Frontend" / "js" / "vietnam_echarts_map.js",
    ]
    legacy_reads = []
    random_mock = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "../json/vietnam.json" in text:
            legacy_reads.append(str(path.relative_to(ROOT)))
        if "Math.random" in text or "fakeStats" in text:
            random_mock.append(str(path.relative_to(ROOT)))
    return {
        "legacy_geojson_reads": legacy_reads,
        "random_mock_risk": random_mock,
        "required_adapter": "Frontend/js/macro_map_data.js",
        "adapter_exists": (ROOT / "Frontend" / "js" / "macro_map_data.js").exists(),
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Macro Simulation Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Provinces: {report['province_count']}",
        f"- Historical events: {report['event_count']}",
        f"- GeoJSON features: {report['geojson']['feature_count']}",
        "",
        "## Event Types",
        "",
    ]
    for event_type, count in report["event_types"].items():
        lines.append(f"- `{event_type}`: {count}")
    lines.extend(["", "## Smoke Tests", ""])
    for row in report["scenario_smoke_tests"]:
        if row.get("ok"):
            lines.append(
                f"- `{row['province_code']}`: risk={row['projected_risk']}, "
                f"delta_revenue={row['delta_revenue_pct']}%, confidence={row['confidence_score']}"
            )
        else:
            lines.append(f"- `{row['province_code']}`: FAILED - {row.get('error')}")
    lines.extend(["", "## Findings", ""])
    if report["hard_failures"]:
        for failure in report["hard_failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- No hard failures.")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(["", "## Boundary Readiness", ""])
    lines.append(f"- Active version: `{report['boundary_readiness']['active_version']}`")
    lines.append(f"- Production target: `{report['boundary_readiness']['production_target_version']}`")
    lines.append(f"- Current boundary status: `{report['geojson'].get('boundary_status')}`")
    lines.extend(["", "## Event Ingest Queue", ""])
    lines.append(f"- Pending review: {report['event_ingest_status']['pending_review']}")
    lines.append(f"- Total queued rows: {report['event_ingest_status']['total']}")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in report["recommendations"]:
        lines.append(f"- {recommendation}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit macro simulation data and engine readiness.")
    parser.add_argument("--json-out", default=str(REPORT_DIR / "macro_simulation_audit.json"))
    parser.add_argument("--md-out", default=str(REPORT_DIR / "macro_simulation_audit.md"))
    parser.add_argument("--production", action="store_true", help="Fail if production boundary/data governance is not ready.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(production=args.production)

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Macro simulation audit status: {report['status']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
