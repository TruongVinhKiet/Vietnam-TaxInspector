"""Build reviewed 34-unit Vietnam GeoJSON and aggregate economic profiles.

This script derives the 2025 34-unit analytical boundary from the legacy
63-province GeoJSON bundled with the project. It uses Shapely union to dissolve
old provincial polygons into new units and writes provenance metadata so the UI
can render a real 34-feature polygon map instead of centroid tiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from shapely.geometry import mapping, shape  # type: ignore
from shapely.ops import unary_union  # type: ignore

from ml_engine.admin_reorg_2025 import (  # noqa: E402
    VN34_SOURCE_REFS,
    VN34_UNITS,
    normalize_admin_name,
    unit_for_legacy_name,
)


DATA_DIR = BACKEND / "data" / "data"
FRONTEND_GEOJSON = ROOT / "Frontend" / "json" / "vietnam.json"
OUTPUT_GEOJSON = DATA_DIR / "vietnam_admin_boundaries_34_2025_official.geojson"
OUTPUT_PROVINCES = DATA_DIR / "vietnam_provinces_34_2025.json"


def build_vn34_boundaries(
    *,
    source_geojson: Path = FRONTEND_GEOJSON,
    output_geojson: Path = OUTPUT_GEOJSON,
    output_profiles: Path = OUTPUT_PROVINCES,
) -> Dict[str, Any]:
    source = json.loads(source_geojson.read_text(encoding="utf-8"))
    features = source.get("features") or []
    if len(features) < 60:
        raise RuntimeError(f"Expected legacy 63-unit GeoJSON, got {len(features)} features.")

    grouped: Dict[str, Dict[str, Any]] = {
        unit["code"]: {"unit": unit, "features": [], "legacy_names": []}
        for unit in VN34_UNITS
    }
    unmatched: List[str] = []

    for feature in features:
        props = feature.get("properties") or {}
        old_name = props.get("name") or props.get("Ten") or props.get("province_name")
        unit = unit_for_legacy_name(old_name)
        if not unit:
            unmatched.append(str(old_name))
            continue
        grouped[unit["code"]]["features"].append(feature)
        grouped[unit["code"]]["legacy_names"].append(str(old_name))

    empty_units = [row["unit"]["name"] for row in grouped.values() if not row["features"]]
    if unmatched or empty_units:
        raise RuntimeError(f"Cannot map all legacy provinces. unmatched={unmatched}; empty_units={empty_units}")

    out_features = []
    for row in grouped.values():
        unit = row["unit"]
        geometries = []
        for feature in row["features"]:
            geom = shape(feature["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            geometries.append(geom)
        merged = unary_union(geometries)
        out_features.append({
            "type": "Feature",
            "properties": {
                "province_code": unit["code"],
                "province_name": unit["name"],
                "name": unit["name"],
                "unit_type": unit["type"],
                "member_names": unit["members"],
                "legacy_geojson_names": sorted(row["legacy_names"]),
                "area_km2_official_ref": unit.get("area_km2"),
                "population_official_ref": unit.get("official_population"),
                "source": "derived_from_63_legacy_geojson_shapely_union",
            },
            "geometry": mapping(merged),
        })

    out_features.sort(key=lambda feature: feature["properties"]["province_name"])
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "boundary_version": "vn_34_2025",
            "boundary_status": "official_or_reviewed_geojson",
            "feature_count": len(out_features),
            "expected_unit_count": 34,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "derivation": "Dissolved legacy 63-province polygons into 34 units using 2025 merger mapping.",
            "legal_note": "Analytical reviewed boundary derived from project GeoJSON; replace with government-published survey-grade GeoJSON when available.",
            "source_refs": VN34_SOURCE_REFS,
        },
        "features": out_features,
    }
    output_geojson.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")

    profiles = build_vn34_profiles()
    output_profiles.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "geojson_path": str(output_geojson),
        "profile_path": str(output_profiles),
        "feature_count": len(out_features),
        "profile_count": len(profiles),
    }


def build_vn34_profiles() -> List[Dict[str, Any]]:
    legacy_profiles = json.loads((DATA_DIR / "vietnam_provinces.json").read_text(encoding="utf-8"))
    by_name = {normalize_admin_name(row.get("province_name")): row for row in legacy_profiles}
    profiles: List[Dict[str, Any]] = []

    for unit in VN34_UNITS:
        members = [by_name[normalize_admin_name(name)] for name in unit["members"]]
        population = sum(float(row.get("population") or 0) for row in members)
        gdp = sum(float(row.get("gdp_billion_vnd") or 0) for row in members)
        revenue = sum(float(row.get("tax_revenue_billion_vnd") or 0) for row in members)
        enterprises = sum(float(row.get("num_enterprises") or 0) for row in members)
        fdi = sum(float(row.get("fdi_billion_usd") or 0) for row in members)
        lat = sum(float(row.get("lat") or 0) * float(row.get("population") or 0) for row in members) / max(population, 1)
        lng = sum(float(row.get("lng") or 0) * float(row.get("population") or 0) for row in members) / max(population, 1)
        compliance = sum(float(row.get("compliance_rate") or 0) * float(row.get("tax_revenue_billion_vnd") or 0) for row in members) / max(revenue, 1)
        unemployment = sum(float(row.get("unemployment_rate") or 0) * float(row.get("population") or 0) for row in members) / max(population, 1)
        sectors = sorted({sector for row in members for sector in (row.get("dominant_sectors") or [])})
        risk_level = "high" if any(row.get("risk_level") == "high" for row in members) else "medium" if any(row.get("risk_level") == "medium" for row in members) else "low"
        profiles.append({
            "province_code": unit["code"],
            "province_name": unit["name"],
            "unit_type": unit["type"],
            "member_codes": [str(row.get("province_code")) for row in members],
            "member_names": [row.get("province_name") for row in members],
            "region": _dominant_region(members),
            "gdp_billion_vnd": round(gdp, 2),
            "population": int(round(unit.get("official_population") or population)),
            "population_baseline_sum": int(round(population)),
            "area_km2_official_ref": unit.get("area_km2"),
            "num_enterprises": int(round(enterprises)),
            "tax_revenue_billion_vnd": round(revenue, 2),
            "fdi_billion_usd": round(fdi, 3),
            "unemployment_rate": round(unemployment, 3),
            "dominant_sectors": sectors[:8],
            "compliance_rate": round(compliance, 4),
            "risk_level": risk_level,
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "data_year": 2025,
            "source": "aggregated_from_legacy_taxinspector_profiles_plus_vn34_official_population_refs",
        })
    return sorted(profiles, key=lambda item: item["province_name"])


def _dominant_region(rows: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        region = str(row.get("region") or "")
        counts[region] = counts.get(region, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build vn_34_2025 reviewed GeoJSON from legacy 63-unit source.")
    parser.add_argument("--source-geojson", type=Path, default=FRONTEND_GEOJSON)
    parser.add_argument("--output-geojson", type=Path, default=OUTPUT_GEOJSON)
    parser.add_argument("--output-profiles", type=Path, default=OUTPUT_PROVINCES)
    args = parser.parse_args()

    result = build_vn34_boundaries(
        source_geojson=args.source_geojson,
        output_geojson=args.output_geojson,
        output_profiles=args.output_profiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
