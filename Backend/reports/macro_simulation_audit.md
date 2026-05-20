# Macro Simulation Audit

- Status: **pass**
- Provinces: 34
- Historical events: 168
- GeoJSON features: 34

## Event Types

- `growth`: 81
- `financial_crisis`: 19
- `natural_disaster`: 17
- `policy`: 17
- `geopolitics`: 11
- `trade_agreement`: 9
- `pandemic`: 7
- `trade_war`: 5
- `infrastructure_shock`: 2

## Smoke Tests

- `VN34-HN`: risk=low, delta_revenue=5.99%, confidence=0.791
- `VN34-HCM`: risk=low, delta_revenue=-9.45%, confidence=0.7876
- `VN34-DN`: risk=medium, delta_revenue=-10.39%, confidence=0.7882
- `VN34-CT`: risk=low, delta_revenue=10.58%, confidence=0.7898

## Findings

- No hard failures.

## Boundary Readiness

- Active version: `vn_34_2025`
- Production target: `vn_34_2025`
- Current boundary status: `official_or_reviewed_geojson`

## Event Ingest Queue

- Pending review: 0
- Total queued rows: 19

## Recommendations

- Current offline Digital Twin assets are sufficient for demo and deterministic scenario simulation.
- Load reviewed official GeoJSON for `vn_34_2025` before production map/legal-boundary use.
- Attach real-time crawlers through `macro_event_ingest`; keep new events in review queue before model use.
- Retrain macro models only after event provenance and label quality pass this audit.
