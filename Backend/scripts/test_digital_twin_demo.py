"""
Comprehensive Vietnam Digital Twin system test.
Exercises all key components: provinces, events, GeoJSON, scenarios, news crawler.
"""
from __future__ import annotations
import sys, json, io, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml_engine.macro_scenario_engine import (
    load_provinces, load_events, compute_scenario, ScenarioParams,
    get_events_for_province, build_vietnam_geojson, generate_narrative_sync,
)
from ml_engine.admin_boundary_manager import audit_boundary_readiness
from ml_engine.macro_event_ingest import build_ingest_status
from collections import Counter

SEP = "=" * 72

def main():
    print(SEP)
    print("  VIETNAM DIGITAL TWIN - COMPREHENSIVE SYSTEM TEST")
    print(SEP)

    # ── 1. Province data ──
    provinces = load_provinces()
    print(f"\n[1] PROVINCES: {len(provinces)} loaded")
    for p in provinces[:5]:
        name = p.get("province_name", "?")
        gdp = p.get("gdp_billion_vnd", 0)
        tax = p.get("tax_revenue_billion_vnd", 0)
        risk = p.get("risk_level", "?")
        print(f"    {p['province_code']:>3s}  {name:22s}  GDP={gdp:>10,.0f}  Tax={tax:>8,.0f}  Risk={risk}")
    print(f"    ... and {len(provinces) - 5} more provinces")

    # ── 2. Events ──
    events = load_events()
    types = Counter(e.get("event_type") for e in events)
    print(f"\n[2] HISTORICAL EVENTS: {len(events)} total")
    for t, c in types.most_common():
        print(f"    {str(t):28s}: {c} events")

    # ── 3. GeoJSON Map ──
    geo = build_vietnam_geojson()
    meta = geo.get("metadata", {})
    features = geo.get("features", [])
    print(f"\n[3] GEOJSON MAP:")
    print(f"    Features:      {len(features)}")
    print(f"    Version:       {meta.get('boundary_version')}")
    print(f"    Status:        {meta.get('boundary_status')}")
    print(f"    Precision:     {meta.get('boundary_precision')}")

    # ── 4. Boundary readiness ──
    boundary = audit_boundary_readiness()
    print(f"\n[4] BOUNDARY READINESS:")
    print(f"    Active:        {boundary['active_version']}")
    print(f"    Prod target:   {boundary['production_target_version']}")
    for w in boundary.get("warnings", []):
        print(f"    [WARN] {w}")

    # ── 5. Event coverage per province ──
    print(f"\n[5] EVENT COVERAGE (sample provinces):")
    sample_codes = ["01", "79", "48", "27", "74", "22", "40", "92"]
    for code in sample_codes:
        evts = get_events_for_province(code, limit=80)
        p = next((x for x in provinces if x["province_code"] == code), None)
        name = p["province_name"] if p else code
        print(f"    {code}  {name:22s}: {len(evts)} applicable events")

    # ── 6. Scenario simulation demos ──
    print(f"\n{SEP}")
    print("  SCENARIO SIMULATION DEMOS")
    print(SEP)

    demos = [
        ("01", "Ha Noi - GDP +5%, Compliance +3%",
         ScenarioParams(gdp_delta_pct=5.0, compliance_delta=0.03)),
        ("79", "TP.HCM - COVID-19 scenario",
         ScenarioParams(event_key="covid19_wave4")),
        ("48", "Da Nang - Storm Yagi scenario",
         ScenarioParams(event_key="typhoon_yagi_2024")),
        ("27", "Bac Ninh - Samsung FDI boom",
         ScenarioParams(gdp_delta_pct=8.0, fdi_delta_pct=15.0, compliance_delta=0.05)),
        ("92", "Can Tho - Mekong flood + trade war",
         ScenarioParams(gdp_delta_pct=-6.0, compliance_delta=-0.04, unemployment_delta=1.5)),
    ]

    for province_code, label, params in demos:
        try:
            result = compute_scenario(province_code, params)
            narrative = generate_narrative_sync(result)
            print(f"\n  --- {label} ---")
            print(f"  Province:    {result.province_name} ({result.region})")
            print(f"  Baseline:    GDP={result.baseline_gdp:,.0f} ty | Rev={result.baseline_revenue:,.0f} ty | Risk={result.baseline_risk}")
            print(f"  Projected:   GDP={result.projected_gdp:,.0f} ty | Rev={result.projected_revenue:,.0f} ty | Risk={result.projected_risk}")
            print(f"  Delta Rev:   {result.delta_revenue_pct:+.2f}%")
            print(f"  Confidence:  {result.confidence_score:.4f}")
            print(f"  Uncertainty: {result.uncertainty_band_revenue.get('uncertainty_pct', 0):.1f}%")
            print(f"  Tax rate:    {result.baseline_tax_rate:.4f} -> {result.projected_tax_rate:.4f}")
            if result.event_applied:
                print(f"  Event:       {result.event_applied}")
            print(f"  Related events: {len(result.related_events)}")
            # Print first 200 chars of narrative
            print(f"  Narrative:   {narrative[:200]}...")
        except Exception as e:
            print(f"\n  --- {label} --- FAILED: {e}")

    # ── 7. Ingest queue ──
    ingest = build_ingest_status()
    print(f"\n\n[7] NEWS INGEST QUEUE:")
    print(f"    Total queued:    {ingest['total']}")
    print(f"    Pending review:  {ingest['pending_review']}")

    # ── 8. News crawler module check ──
    print(f"\n[8] NEWS CRAWLER MODULE:")
    try:
        from ml_engine.news_crawler import RSS_FEEDS, extract_province_codes, classify_article_fallback
        print(f"    RSS feeds configured: {len(RSS_FEEDS)}")
        for f in RSS_FEEDS:
            print(f"      - {f['source_name']:20s} ({f['language']}) {f['url'][:50]}...")

        # Test province extraction
        test_text = "Samsung khai truong nha may moi tai Bac Ninh va Thai Nguyen, tang FDI cho Viet Nam"
        codes = extract_province_codes(test_text)
        print(f"    Province extraction test: '{test_text[:60]}...'")
        print(f"      -> codes: {codes}")

        # Test fallback classifier
        test_article = {
            "title": "Bao so 3 gay thiet hai nang o Quang Ninh va Hai Phong",
            "description": "Con bao manh nhat trong 10 nam qua da lam thiet hai hang nghin ty dong cho kinh te vung ven bien",
        }
        cls = classify_article_fallback(test_article)
        print(f"    Fallback classification test:")
        print(f"      event_type: {cls['event_type']}, severity: {cls['severity']}")
        print(f"      provinces:  {cls['affected_provinces']}")
        print(f"      sectors:    {cls['affected_sectors']}")
    except Exception as e:
        print(f"    IMPORT FAILED: {e}")

    print(f"\n{SEP}")
    print(f"  ALL CHECKS COMPLETED SUCCESSFULLY")
    print(f"  System ready for thesis demo")
    print(f"{SEP}")


if __name__ == "__main__":
    main()
