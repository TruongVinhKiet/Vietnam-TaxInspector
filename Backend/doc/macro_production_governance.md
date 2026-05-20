# Macro Simulation Production Governance

This note documents the production path for the Vietnam Digital Twin map and
real-time macro event ingestion.

## Boundary Versions

Boundary metadata lives in:

- `Backend/data/data/admin_boundaries_manifest.json`

The legacy analytical baseline remains `vn_63_legacy` for backwards-compatible
experiments. The production map target is `vn_34_2025`, reflecting the 2025
provincial-level administrative reorganization.

Production checklist:

1. Build or replace the reviewed GeoJSON at
   `Backend/data/data/vietnam_admin_boundaries_34_2025_official.geojson`.
2. Ensure it is a GeoJSON `FeatureCollection` with exactly 34 features.
3. Rebuild the current reviewed analytical boundary from the 63-unit source:

```bash
python Backend/scripts/build_vn34_boundaries.py
```

4. Run:

```bash
python Backend/scripts/audit_macro_simulation.py --production
```

The generated file is a Shapely-dissolved analytical boundary with provenance.
Replace it with a government-published survey-grade GeoJSON when such a file is
available for legal boundary analysis.

## Macro Event Ingest

External news/API/crawler items must enter the review queue first:

```bash
python Backend/scripts/crawl_macro_news.py --max-per-feed 10 --dry-run
python Backend/scripts/crawl_macro_news.py --max-per-feed 10 --use-llm
python Backend/scripts/review_macro_event_queue.py --status
python Backend/scripts/review_macro_event_queue.py --auto-approve-trusted --limit 25
python Backend/scripts/ingest_macro_events.py --input events.jsonl --source-name "VNA crawler" --dry-run
python Backend/scripts/ingest_macro_events.py --input events.jsonl --source-name "VNA crawler"
python Backend/scripts/ingest_macro_events.py --status
```

The ingest pipeline records provenance, source URL, fingerprint, duplicate
reason, and review status. New events are not added to
`historical_economic_events.json` until reviewed and approved.

## Natural-Language Scenario Memory

Future what-if text is interpreted through:

- Approved memory lookup first.
- Local reviewed-data model lookup when trained.
- LLM provider fallback from environment variables.
- Deterministic rule fallback when no LLM is available.
- Human approval/rating before any result becomes reusable memory.

Provider priority:

1. `OPENROUTER_API_KEY`
2. `GEMINI_API_KEY`
3. `GITHUB_MODELS_TOKEN` or `GITHUB_TOKEN`
4. `GROQ_API_KEY`
5. `COHERE_API_KEY`

APIs:

- `POST /api/simulation/text-scenario/interpret`
- `POST /api/simulation/text-scenario/feedback`
- `GET /api/simulation/text-scenario/memory/status`

## Reviewed-Data Retraining

The macro digital twin has a separate reviewed-data retrain step. It consumes:

- canonical `historical_economic_events.json`;
- approved text scenario memory rows with rating >= 4;
- approved rows in `macro_event_review_queue.jsonl`.

It does not crawl, call LLMs, or train from pending review rows.

```bash
python Backend/scripts/add_tax_policy_scenarios.py
python Backend/scripts/retrain_macro_from_reviewed_data.py --min-samples 5000
```

Artifacts:

- `Backend/data/models/macro_event_impact_model.joblib`
- `Backend/data/models/macro_province_response_model.joblib`
- `Backend/data/models/macro_retrain_report.json`
- `Backend/data/models/macro_retrain_dataset_preview.jsonl`

## Macro Time-Series Data

National macro series can be refreshed from World Bank and IMF public APIs. The
province panel is explicitly marked as a baseline-anchored estimate until
reviewed GSO province tables are ingested.

```bash
python Backend/scripts/crawl_macro_timeseries.py --start-year 2015 --end-year 2025
```

Output:

- `Backend/data/data/macro_timeseries_vietnam.json`

The text scenario interpreter uses the local reviewed model before LLM fallback
when it recognizes a macro shock family. API status:

- `GET /api/simulation/macro-retrain/status`

## API Surfaces

- `GET /api/simulation/boundary-versions`
- `GET /api/simulation/geojson-vietnam?boundary_version=vn_63_legacy`
- `GET /api/simulation/geojson-vietnam?boundary_version=vn_34_2025`
- `GET /api/simulation/event-ingest/status`
- `POST /api/simulation/text-scenario/interpret`
- `POST /api/simulation/text-scenario/feedback`
- `GET /api/simulation/macro-retrain/status`

## Database Migration

Apply:

- `Database/migrations/v4_macro_production_governance.sql`

It creates `admin_boundary_versions` and `macro_event_ingest_queue` for
PostgreSQL-backed governance and auditability.
