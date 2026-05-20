"""Fetch trusted macro time-series and create province-level analytical panels.

National series are fetched from World Bank and IMF public APIs when network is
available. Provincial panels are transparent estimates anchored to the local
province baseline because GSO province time-series are often published in
tables/PDFs rather than a stable public JSON API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_scenario_engine import load_provinces  # noqa: E402


DATA_DIR = BACKEND / "data" / "data"
OUTPUT = DATA_DIR / "macro_timeseries_vietnam.json"

WORLD_BANK_INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "cpi_inflation_pct": "FP.CPI.TOTL.ZG",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "fdi_net_inflows_pct_gdp": "BX.KLT.DINV.WD.GD.ZS",
}

IMF_DATAMAPPER_INDICATORS = {
    "real_gdp_growth_pct": "NGDP_RPCH",
    "consumer_price_inflation_pct": "PCPIPCH",
    "unemployment_rate_pct": "LUR",
}


def fetch_macro_timeseries(*, start_year: int = 2015, end_year: int = 2025, timeout: int = 25) -> Dict[str, Any]:
    wb = {key: _fetch_world_bank(indicator, start_year, end_year, timeout=timeout) for key, indicator in WORLD_BANK_INDICATORS.items()}
    imf = {key: _fetch_imf_datamapper(indicator, start_year, end_year, timeout=timeout) for key, indicator in IMF_DATAMAPPER_INDICATORS.items()}
    national = _merge_national_series(wb, imf, start_year, end_year)
    provinces = build_province_timeseries(national, start_year=start_year, end_year=end_year)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "country": "Vietnam",
        "country_code": "VNM",
        "year_range": [start_year, end_year],
        "sources": [
            {
                "name": "World Bank Indicators API",
                "url": "https://api.worldbank.org/v2/country/VNM/indicator/{indicator}?format=json",
                "fields": list(WORLD_BANK_INDICATORS.keys()),
            },
            {
                "name": "IMF DataMapper API",
                "url": "https://www.imf.org/external/datamapper/api/v1/{indicator}/VNM",
                "fields": list(IMF_DATAMAPPER_INDICATORS.keys()),
            },
            {
                "name": "GSO provincial statistical yearbooks",
                "url": "https://www.gso.gov.vn/",
                "fields": ["province_population", "province_grdp", "province_fdi"],
                "note": "Province panel is estimated from local province baselines until reviewed GSO table extraction is ingested.",
            },
        ],
        "national": national,
        "province_panel": provinces,
        "quality": {
            "world_bank_series_loaded": {k: bool(v) for k, v in wb.items()},
            "imf_series_loaded": {k: bool(v) for k, v in imf.items()},
            "province_method": "baseline_anchored_estimate_pending_gso_table_review",
        },
    }


def build_province_timeseries(national: List[Dict[str, Any]], *, start_year: int, end_year: int) -> List[Dict[str, Any]]:
    provinces = load_provinces()
    by_year = {int(row["year"]): row for row in national}
    rows: List[Dict[str, Any]] = []
    for province in provinces:
        base_year = int(province.get("data_year") or 2024)
        base_pop = float(province.get("population") or 0.0)
        base_gdp = float(province.get("gdp_billion_vnd") or 0.0)
        base_fdi = float(province.get("fdi_billion_usd") or 0.0)
        base_unemp = float(province.get("unemployment_rate") or 2.3)
        population_growth = _region_population_growth(str(province.get("region") or ""))
        for year in range(start_year, end_year + 1):
            national_row = by_year.get(year, {})
            years_from_base = year - base_year
            gdp_growth = float(national_row.get("gdp_growth_pct") or 6.0) / 100.0
            province_gdp_factor = (1.0 + gdp_growth) ** years_from_base if years_from_base >= 0 else (1.0 + gdp_growth) ** years_from_base
            rows.append({
                "province_code": province.get("province_code"),
                "province_name": province.get("province_name"),
                "year": year,
                "population": int(round(base_pop * ((1.0 + population_growth) ** years_from_base))),
                "grdp_billion_vnd_est": round(base_gdp * province_gdp_factor, 2),
                "cpi_inflation_pct": national_row.get("cpi_inflation_pct"),
                "unemployment_pct_est": round(max(0.2, base_unemp + float(national_row.get("unemployment_pct_delta_from_2024") or 0.0)), 3),
                "fdi_billion_usd_est": round(max(0.0, base_fdi * (1.0 + float(national_row.get("fdi_net_inflows_pct_gdp") or 0.0) / 100.0 * 0.08) ** years_from_base), 3),
                "source": "province_estimate_from_local_baseline_and_national_wb_imf_trend",
            })
    return rows


def _fetch_world_bank(indicator: str, start_year: int, end_year: int, *, timeout: int) -> Dict[int, float]:
    url = (
        f"https://api.worldbank.org/v2/country/VNM/indicator/{indicator}"
        f"?format=json&per_page=120&date={start_year}:{end_year}"
    )
    data = _http_json(url, timeout=timeout)
    rows = data[1] if isinstance(data, list) and len(data) > 1 else []
    out: Dict[int, float] = {}
    for item in rows:
        value = item.get("value")
        if value is None:
            continue
        out[int(item["date"])] = float(value)
    return out


def _fetch_imf_datamapper(indicator: str, start_year: int, end_year: int, *, timeout: int) -> Dict[int, float]:
    url = f"https://www.imf.org/external/datamapper/api/v1/{indicator}/VNM"
    data = _http_json(url, timeout=timeout)
    raw = (((data.get("values") or {}).get(indicator) or {}).get("VNM") or {})
    out: Dict[int, float] = {}
    for year, value in raw.items():
        try:
            year_i = int(year)
        except Exception:
            continue
        if start_year <= year_i <= end_year and value is not None:
            out[year_i] = float(value)
    return out


def _merge_national_series(wb: Dict[str, Dict[int, float]], imf: Dict[str, Dict[int, float]], start_year: int, end_year: int) -> List[Dict[str, Any]]:
    rows = []
    unemp_2024 = wb.get("unemployment_pct", {}).get(2024) or imf.get("unemployment_rate_pct", {}).get(2024) or 2.3
    for year in range(start_year, end_year + 1):
        gdp_growth = wb.get("gdp_growth_pct", {}).get(year)
        if gdp_growth is None:
            gdp_growth = imf.get("real_gdp_growth_pct", {}).get(year)
        inflation = wb.get("cpi_inflation_pct", {}).get(year)
        if inflation is None:
            inflation = imf.get("consumer_price_inflation_pct", {}).get(year)
        unemployment = wb.get("unemployment_pct", {}).get(year)
        if unemployment is None:
            unemployment = imf.get("unemployment_rate_pct", {}).get(year)
        rows.append({
            "year": year,
            "population": _round_or_none(wb.get("population", {}).get(year), 0),
            "gdp_current_usd": _round_or_none(wb.get("gdp_current_usd", {}).get(year), 2),
            "gdp_growth_pct": _round_or_none(gdp_growth, 3),
            "cpi_inflation_pct": _round_or_none(inflation, 3),
            "unemployment_pct": _round_or_none(unemployment, 3),
            "unemployment_pct_delta_from_2024": _round_or_none((unemployment or unemp_2024) - unemp_2024, 3),
            "fdi_net_inflows_pct_gdp": _round_or_none(wb.get("fdi_net_inflows_pct_gdp", {}).get(year), 3),
        })
    return rows


def _http_json(url: str, *, timeout: int) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "TaxInspector/1.0 research crawler"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def _region_population_growth(region: str) -> float:
    text = region.lower()
    if "đông nam" in text:
        return 0.011
    if "tây nguyên" in text:
        return 0.010
    if "cửu long" in text:
        return 0.0015
    if "trung du" in text or "núi" in text:
        return 0.006
    return 0.0045


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch WB/IMF macro series and build province panel estimates.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    data = fetch_macro_timeseries(start_year=args.start_year, end_year=args.end_year)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "path": str(args.output),
        "national_rows": len(data["national"]),
        "province_rows": len(data["province_panel"]),
        "quality": data["quality"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

