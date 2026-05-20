"""Load macro_timeseries_vietnam.json into governed DB time-series tables.

The JSON file contains observed national World Bank/IMF-style series and
province-level baseline-anchored estimates. This loader preserves that source
quality split and also materializes the 2025 34-unit administrative view by
aggregating legacy member provinces with provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from Backend.app.database import engine  # noqa: E402
from ml_engine.macro_scenario_engine import load_provinces  # noqa: E402


DATA_PATH = BACKEND / "data" / "data" / "macro_timeseries_vietnam.json"


NATIONAL_INDICATORS = {
    "population": ("Dân số", "people"),
    "gdp_current_usd": ("GDP hiện hành USD", "USD"),
    "gdp_growth_pct": ("Tăng trưởng GDP", "pct"),
    "cpi_inflation_pct": ("CPI/Inflation", "pct"),
    "unemployment_pct": ("Thất nghiệp", "pct"),
    "fdi_net_inflows_pct_gdp": ("FDI net inflows / GDP", "pct_gdp"),
}

PROVINCE_INDICATORS = {
    "population": ("Dân số tỉnh", "people"),
    "grdp_billion_vnd_est": ("GRDP tỉnh ước lượng", "billion_vnd"),
    "cpi_inflation_pct": ("CPI/Inflation", "pct"),
    "unemployment_pct_est": ("Thất nghiệp tỉnh ước lượng", "pct"),
    "fdi_billion_usd_est": ("FDI tỉnh ước lượng", "billion_usd"),
}


def load_payload(path: Path = DATA_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_observations(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    generated_at = payload.get("generated_at")
    province_panel = list(payload.get("province_panel") or [])

    for row in payload.get("national") or []:
        year = int(row["year"])
        for key, (label, unit) in NATIONAL_INDICATORS.items():
            value = row.get(key)
            if value is None:
                continue
            source_key = "world_bank_indicators_api"
            if key in {"gdp_growth_pct", "cpi_inflation_pct", "unemployment_pct"}:
                source_key = "world_bank_or_imf_best_available"
            yield {
                "boundary_version": "national",
                "province_code": "__national__",
                "indicator_key": key,
                "indicator_label": label,
                "year": year,
                "quarter": 0,
                "value_num": float(value),
                "unit": unit,
                "is_observed": True,
                "source_key": source_key,
                "source_quality": "official",
                "provenance_json": {
                    "generated_at": generated_at,
                    "country_code": payload.get("country_code"),
                },
            }

    for row in province_panel:
        yield from _province_indicator_rows(
            row,
            boundary_version="vn_63_legacy",
            source_quality="reviewed_estimate",
            provenance_json={
                "generated_at": generated_at,
                "province_name": row.get("province_name"),
                "source": row.get("source"),
            },
        )

    by_code_year = {
        (str(row.get("province_code") or ""), int(row["year"])): row
        for row in province_panel
        if row.get("province_code") and row.get("year") is not None
    }
    years = sorted({year for _, year in by_code_year})
    for province in load_provinces("vn_34_2025"):
        code = str(province.get("province_code") or "")
        member_codes = [str(item) for item in province.get("member_codes") or [] if item]
        if not code or not member_codes:
            continue
        for year in years:
            parts = [by_code_year[(member, year)] for member in member_codes if (member, year) in by_code_year]
            if not parts:
                continue
            cpi_values = [float(x.get("cpi_inflation_pct")) for x in parts if x.get("cpi_inflation_pct") is not None]
            unemp_values = [float(x.get("unemployment_pct_est")) for x in parts if x.get("unemployment_pct_est") is not None]
            aggregated = {
                "province_code": code,
                "province_name": province.get("province_name"),
                "year": year,
                "population": sum(float(x.get("population") or 0.0) for x in parts),
                "grdp_billion_vnd_est": sum(float(x.get("grdp_billion_vnd_est") or 0.0) for x in parts),
                "cpi_inflation_pct": sum(cpi_values) / max(1, len(cpi_values)) if cpi_values else None,
                "unemployment_pct_est": sum(unemp_values) / max(1, len(unemp_values)) if unemp_values else None,
                "fdi_billion_usd_est": sum(float(x.get("fdi_billion_usd_est") or 0.0) for x in parts),
            }
            yield from _province_indicator_rows(
                aggregated,
                boundary_version="vn_34_2025",
                source_quality="reviewed_estimate",
                provenance_json={
                    "generated_at": generated_at,
                    "province_name": province.get("province_name"),
                    "member_codes": member_codes,
                    "source": "aggregated_from_legacy_member_province_estimates",
                },
            )


def _province_indicator_rows(
    row: Dict[str, Any],
    *,
    boundary_version: str,
    source_quality: str,
    provenance_json: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    year = int(row["year"])
    for key, (label, unit) in PROVINCE_INDICATORS.items():
        value = row.get(key)
        if value is None:
            continue
        yield {
            "boundary_version": boundary_version,
            "province_code": str(row.get("province_code") or ""),
            "indicator_key": key,
            "indicator_label": label,
            "year": year,
            "quarter": 0,
            "value_num": float(value),
            "unit": unit,
            "is_observed": False,
            "source_key": "gso_province_yearbook_pending_review",
            "source_quality": source_quality,
            "provenance_json": provenance_json,
        }


def ensure_sources(conn) -> None:
    rows = [
        (
            "world_bank_indicators_api",
            "World Bank Indicators API",
            "https://api.worldbank.org/v2/country/VNM/indicator/{indicator}?format=json",
            "official",
            "national",
            "approved",
        ),
        (
            "world_bank_or_imf_best_available",
            "World Bank / IMF best available national macro series",
            "https://www.imf.org/external/datamapper/api/",
            "official",
            "national",
            "approved",
        ),
        (
            "gso_province_yearbook_pending_review",
            "GSO provincial statistical yearbooks",
            "https://www.gso.gov.vn/",
            "reviewed_estimate",
            "province",
            "needs_more_source",
        ),
    ]
    for row in rows:
        conn.execute(text("""
            INSERT INTO macro_data_sources (
                source_key, source_name, source_url, source_type, observed_level, review_status
            )
            VALUES (:source_key, :source_name, :source_url, :source_type, :observed_level, :review_status)
            ON CONFLICT (source_key) DO UPDATE SET
                source_name = EXCLUDED.source_name,
                source_url = EXCLUDED.source_url,
                source_type = EXCLUDED.source_type,
                observed_level = EXCLUDED.observed_level,
                review_status = EXCLUDED.review_status,
                updated_at = NOW()
        """), {
            "source_key": row[0],
            "source_name": row[1],
            "source_url": row[2],
            "source_type": row[3],
            "observed_level": row[4],
            "review_status": row[5],
        })


def upsert_observations(rows: List[Dict[str, Any]], *, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "rows": len(rows)}
    with engine.begin() as conn:
        ensure_sources(conn)
        for row in rows:
            conn.execute(text("""
                INSERT INTO macro_timeseries_observations (
                    boundary_version, province_code, indicator_key, indicator_label,
                    year, quarter, value_num, unit, is_observed, source_key,
                    source_quality, provenance_json
                )
                VALUES (
                    :boundary_version, :province_code, :indicator_key, :indicator_label,
                    :year, :quarter, :value_num, :unit, :is_observed, :source_key,
                    :source_quality, CAST(:provenance_json AS JSONB)
                )
                ON CONFLICT (boundary_version, province_code, indicator_key, year, quarter, source_key)
                DO UPDATE SET
                    value_num = EXCLUDED.value_num,
                    indicator_label = EXCLUDED.indicator_label,
                    unit = EXCLUDED.unit,
                    is_observed = EXCLUDED.is_observed,
                    source_quality = EXCLUDED.source_quality,
                    provenance_json = EXCLUDED.provenance_json,
                    observed_at = NOW()
            """), {
                **row,
                "provenance_json": json.dumps(row["provenance_json"], ensure_ascii=False),
            })
    return {"status": "loaded", "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Load macro time-series JSON into DB governance tables.")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = load_payload(args.input)
    rows = list(iter_observations(payload))
    result = upsert_observations(rows, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
