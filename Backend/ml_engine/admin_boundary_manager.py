"""
Versioned Vietnam administrative boundary loader.

Production rule: never pretend centroid tiles are official boundaries. The
loader can keep legacy dashboards usable, but the metadata always declares
whether the response is a reviewed GeoJSON file or a visual fallback.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"
MANIFEST_PATH = DATA_DIR / "admin_boundaries_manifest.json"
PROVINCES_PATH = DATA_DIR / "vietnam_provinces.json"


DEFAULT_MANIFEST: Dict[str, Any] = {
    "active_version": "vn_34_2025",
    "production_target_version": "vn_34_2025",
    "versions": {
        "vn_63_legacy": {
            "label": "Vietnam 63 provincial-level units - legacy analytical baseline",
            "expected_unit_count": 63,
            "geojson_path": "vietnam_admin_boundaries_63_official.geojson",
            "fallback": "centroid_tile",
            "status": "reviewed_legacy_geojson_loaded",
            "source_name": "TaxInspector restored 63-province GeoJSON asset enriched with province codes",
        },
        "vn_34_2025": {
            "label": "Vietnam 34 provincial-level units after 2025 administrative reorganization",
            "expected_unit_count": 34,
            "geojson_path": "vietnam_admin_boundaries_34_2025_official.geojson",
            "fallback": "none",
            "status": "awaiting_official_geojson",
        },
    },
}


def load_boundary_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_MANIFEST


def list_boundary_versions() -> Dict[str, Any]:
    manifest = load_boundary_manifest()
    versions = manifest.get("versions") or {}
    enriched = {}
    for key, config in versions.items():
        path = _resolve_geojson_path(config)
        exists = bool(path and path.exists())
        feature_count = _count_geojson_features(path) if exists else 0
        enriched[key] = {
            **config,
            "geojson_exists": exists,
            "geojson_feature_count": feature_count,
            "geojson_sha256": _sha256_file(path) if exists else None,
            "readiness": _readiness_for_version(config, exists, feature_count),
        }
    return {
        "active_version": manifest.get("active_version", "vn_34_2025"),
        "production_target_version": manifest.get("production_target_version", "vn_34_2025"),
        "versions": enriched,
        "source_refs": manifest.get("source_refs", []),
    }


def load_boundary_geojson(
    boundary_version: Optional[str] = None,
    *,
    allow_centroid_fallback: bool = True,
) -> Dict[str, Any]:
    manifest = load_boundary_manifest()
    version = boundary_version or manifest.get("active_version") or "vn_34_2025"
    versions = manifest.get("versions") or {}
    config = versions.get(version)
    if not config:
        raise ValueError(f"Unknown boundary version: {version}")

    geojson_path = _resolve_geojson_path(config)
    if geojson_path and geojson_path.exists():
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        _validate_geojson(geojson)
        feature_count = len(geojson.get("features") or [])
        geojson["metadata"] = {
            **(geojson.get("metadata") or {}),
            "boundary_version": version,
            "boundary_status": "official_or_reviewed_geojson",
            "expected_unit_count": config.get("expected_unit_count"),
            "feature_count": feature_count,
            "source_name": config.get("source_name"),
            "source_url": config.get("source_url"),
            "sha256": _sha256_file(geojson_path),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
        return geojson

    if allow_centroid_fallback and config.get("fallback") == "centroid_tile":
        return build_centroid_tile_geojson(
            version=version,
            expected_unit_count=int(config.get("expected_unit_count") or 63),
            source_name=config.get("source_name") or "TaxInspector synthetic province baseline",
        )

    return {
        "type": "FeatureCollection",
        "metadata": {
            "boundary_version": version,
            "boundary_status": "missing_official_boundary",
            "expected_unit_count": config.get("expected_unit_count"),
            "feature_count": 0,
            "source_name": config.get("source_name"),
            "source_url": config.get("source_url"),
            "message": "Official/reviewed GeoJSON is required before this boundary version can be rendered.",
        },
        "features": [],
    }


def build_centroid_tile_geojson(
    *,
    version: str = "vn_63_legacy",
    expected_unit_count: int = 63,
    source_name: str = "TaxInspector synthetic province baseline",
) -> Dict[str, Any]:
    provinces = _load_provinces()
    features: List[Dict[str, Any]] = []
    for province in provinces:
        lat = float(province.get("lat") or 0.0)
        lng = float(province.get("lng") or 0.0)
        if not lat or not lng:
            continue
        delta_lat = 0.18
        delta_lng = 0.22
        coordinates = [[
            [lng - delta_lng, lat - delta_lat],
            [lng + delta_lng, lat - delta_lat],
            [lng + delta_lng, lat + delta_lat],
            [lng - delta_lng, lat + delta_lat],
            [lng - delta_lng, lat - delta_lat],
        ]]
        features.append({
            "type": "Feature",
            "properties": {
                "province_code": str(province.get("province_code") or ""),
                "Ten": province.get("province_name", ""),
                "name": province.get("province_name", ""),
                "risk_level": province.get("risk_level", "medium"),
                "source": "centroid_tile_fallback",
            },
            "geometry": {"type": "Polygon", "coordinates": coordinates},
        })
    return {
        "type": "FeatureCollection",
        "metadata": {
            "boundary_version": version,
            "boundary_status": "centroid_tile_fallback",
            "source": source_name,
            "feature_count": len(features),
            "expected_unit_count": expected_unit_count,
            "boundary_precision": "centroid_tile_not_official_boundary",
            "message": "Visual fallback only. Not valid for legal boundary analysis.",
        },
        "features": features,
    }


def audit_boundary_readiness(*, production: bool = False) -> Dict[str, Any]:
    versions = list_boundary_versions()
    active = versions["active_version"]
    target = versions["production_target_version"]
    active_info = versions["versions"].get(active, {})
    target_info = versions["versions"].get(target, {})

    warnings: List[str] = []
    failures: List[str] = []
    if active_info.get("readiness") != "ready":
        warnings.append(f"Active boundary `{active}` is not official/reviewed; readiness={active_info.get('readiness')}.")
    if target_info.get("readiness") != "ready":
        message = f"Production target boundary `{target}` is not ready; load reviewed GeoJSON before production."
        if production:
            failures.append(message)
        else:
            warnings.append(message)

    return {
        "status": "pass" if not failures else "fail",
        "active_version": active,
        "production_target_version": target,
        "warnings": warnings,
        "failures": failures,
        "versions": versions["versions"],
        "source_refs": versions.get("source_refs", []),
    }


def _load_provinces() -> List[Dict[str, Any]]:
    if not PROVINCES_PATH.exists():
        return []
    return json.loads(PROVINCES_PATH.read_text(encoding="utf-8"))


def _resolve_geojson_path(config: Dict[str, Any]) -> Optional[Path]:
    raw = config.get("geojson_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else DATA_DIR / path


def _validate_geojson(geojson: Dict[str, Any]) -> None:
    if geojson.get("type") != "FeatureCollection" or not isinstance(geojson.get("features"), list):
        raise ValueError("Boundary file must be a GeoJSON FeatureCollection.")


def _count_geojson_features(path: Optional[Path]) -> int:
    if not path:
        return 0
    try:
        geojson = json.loads(path.read_text(encoding="utf-8"))
        _validate_geojson(geojson)
        return len(geojson.get("features") or [])
    except Exception:
        return 0


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readiness_for_version(config: Dict[str, Any], exists: bool, feature_count: int) -> str:
    expected = int(config.get("expected_unit_count") or 0)
    if exists and (not expected or feature_count == expected):
        return "ready"
    if exists and expected and feature_count != expected:
        return "feature_count_mismatch"
    if config.get("fallback") == "centroid_tile":
        return "fallback_available"
    return "missing_official_geojson"
