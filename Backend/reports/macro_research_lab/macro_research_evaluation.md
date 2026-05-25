# Macro-Fiscal Digital Twin Research Evaluation

- Generated at: `2026-05-22T10:38:25.094839Z`
- Boundary version: `vn_34_2025`
- Province coverage: `34/34`
- Historical events: `168`
- Data fingerprint: `902409b2a12e69bd`

## Forecast Backtest Proxy

- Sample provinces: `8`
- MAE proxy mean: `0.055`
- Mean interval width: `11.908%`

## Causal Merger Probe

- Province: `Cà Mau`
- DiD proxy: `0.988%`
- Placebo p-value proxy: `1.0`

## Required Ablations

- `baseline_elasticity`: required=True
- `plus_news_embeddings`: required=True
- `plus_spatial_graph`: required=True
- `plus_causal_policy_features`: required=True

## Acceptance Targets

- `beat_baseline_on_targets`: >=2/3
- `interval_coverage`: 85-95%
- `placebo_false_signal_rate`: <=10%
