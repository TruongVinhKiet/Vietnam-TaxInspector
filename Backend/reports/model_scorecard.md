# TaxInspector Model Scorecard

- Generated at: `2026-04-28T14:50:37Z`

## Summary

This scorecard standardizes maturity across **Data / ModelQuality / Serving / Monitoring / Governance** on a 1–5 scale (higher is more production-ready).

## Catalog

### `delinquency` — Collections/Compliance

- **purpose**: Predict late/overdue payment risk (P30/P60/P90) for companies.
- **model_type**: LightGBMClassifier (tabular) + feature engineer; DL benchmark exists (GRU).
- **training_paths**: `Backend/ml_engine/train_delinquency.py`, `Backend/ml_engine/benchmark_delinquency_sequence.py`
- **serving_paths**: `Backend/app/routers/delinquency.py`, `Backend/ml_engine/delinquency_model.py`
- **materializers**: `-`
- **output_tables**: `delinquency_predictions`
- **api_endpoints**: `/api/delinquency`, `/api/delinquency/{tax_code}`, `/api/delinquency/predict-batch`
- **fallback**: Has baseline fallback logic in pipeline; DL is benchmark-only.
- **maturity**:
  - **data**: **4/5** — Uses tax_payments history; label logic exists; still needs adjudicated ground-truth for edge cases.
  - **model_quality**: **4/5** — Strong tabular baseline + calibration hooks; DL benchmark provides upgrade path.
  - **serving**: **4/5** — FastAPI endpoints + DB materialization table.
  - **monitoring**: **4/5** — Quality/drift tables available; needs automated alert thresholds by horizon.
  - **governance**: **4/5** — ModelRegistry + training runs supported; rollout gating can be strengthened.
- **key_gaps**:
  - Automated retrain triggers by drift
  - Formal horizon-specific calibration gates

### `vat_refund_risk` — Compliance/Fraud

- **purpose**: Score VAT refund cases for fraud/abuse risk.
- **model_type**: RandomForestClassifier (tabular) + optional calibration.
- **training_paths**: `Backend/ml_engine/train_vat_refund.py`
- **serving_paths**: `Backend/app/routers/vat_refund.py`
- **materializers**: `Backend/app/scripts/train_vat_refund_case_risk.py`
- **output_tables**: `vat_refund_predictions`
- **api_endpoints**: `/api/vat-refund/cases`, `/api/vat-refund/cases/{case_id}/risk`
- **fallback**: Table-driven serving; fallback not explicitly exposed but can be added.
- **maturity**:
  - **data**: **3/5** — Depends on case outcome labeling quality; needs stronger adverse-outcome ground-truth.
  - **model_quality**: **3/5** — Solid classical baseline; could benefit from calibrated GBM and richer fraud signals.
  - **serving**: **4/5** — Predictions persisted; endpoints consume consistently.
  - **monitoring**: **3/5** — Slices/calibration possible but not consistently enforced.
  - **governance**: **3/5** — Registry exists; needs champion/challenger + rollback gates.
- **key_gaps**:
  - Outcome label pipeline + adjudication
  - Cost-sensitive thresholds tied to audit capacity

### `invoice_risk` — Fraud

- **purpose**: Score invoices for suspicious behavior (duplication, abnormal patterns).
- **model_type**: RandomForestClassifier (learned) + heuristics.
- **training_paths**: `Backend/ml_engine/train_invoice_risk_model.py`
- **serving_paths**: `Backend/app/routers/invoice_risk.py`, `Backend/ml_engine/invoice_risk_model.py`
- **materializers**: `-`
- **output_tables**: `invoice_risk_predictions`
- **api_endpoints**: `/api/invoice/{invoice_number}/risk`, `/api/invoice/risk`
- **fallback**: Has heuristic fallback when learned artifact missing.
- **maturity**:
  - **data**: **3/5** — Pseudo-label heavy; needs investigation outcomes / confirmed fraud labels.
  - **model_quality**: **3/5** — Good baseline; missing sequence/graph context in main model.
  - **serving**: **4/5** — Online scoring endpoints + audit logging.
  - **monitoring**: **3/5** — Needs consistent calibration and slice monitoring (sector/size).
  - **governance**: **4/5** — Inference audit exists; rollout gates can be formalized.
- **key_gaps**:
  - Ground-truth outcomes
  - Graph-aware invoice ring detection model (GNN) in production path

### `transfer_pricing_mispricing` — TransferPricing

- **purpose**: Detect mispricing/outlier behavior in related-party trade pricing.
- **model_type**: RandomForestClassifier (learned-first) + z-score baseline fallback.
- **training_paths**: `Backend/ml_engine/train_transfer_pricing_model.py`
- **serving_paths**: `Backend/app/routers/transfer_pricing.py`
- **materializers**: `-`
- **output_tables**: `mispricing_predictions`
- **api_endpoints**: `/api/transfer-pricing/score`, `/api/transfer-pricing/mispricing`
- **fallback**: Yes (baseline z-score).
- **maturity**:
  - **data**: **2/5** — Needs richer comparables, customs data, and adjudicated TP adjustments.
  - **model_quality**: **3/5** — Strong baseline; could add quantile regression and monotonic constraints.
  - **serving**: **4/5** — Router supports learned-first fallback.
  - **monitoring**: **3/5** — Needs slice monitoring by HS code / partner country / industry.
  - **governance**: **4/5** — Training run + registry integrated.
- **key_gaps**:
  - Comparable price corpus + label adjudication
  - Explanation templates for legal defensibility

### `ops_uplift_audit_collections` — Ops/Collections

- **purpose**: Audit selection, next-best-action, and expected collection uplift.
- **model_type**: RandomForestClassifier + RandomForestRegressor + hybrid materialized fusion.
- **training_paths**: `Backend/ml_engine/train_ops_uplift_models.py`
- **serving_paths**: `Backend/app/routers/audit_selection.py`, `Backend/app/routers/collections.py`, `Backend/app/routers/case_triage.py`
- **materializers**: `Backend/app/scripts/materialize_ops_predictions.py`, `Backend/app/scripts/materialize_phase60_models.py`
- **output_tables**: `audit_selection_predictions`, `case_triage_predictions`, `nba_predictions`, `entity_risk_fusion_predictions`
- **api_endpoints**: `/api/audit/shortlist`, `/api/case-triage/queue`, `/api/collections/next-best-action`, `/api/collections/fusion-overview`
- **fallback**: Yes (hybrid formulas if learned artifacts missing).
- **maturity**:
  - **data**: **3/5** — Needs causal/experimental design for uplift (policy changes, selection bias).
  - **model_quality**: **3/5** — Good start; can move to CATE/uplift trees + off-policy evaluation.
  - **serving**: **4/5** — Batch materialization stable; online feature service could improve freshness.
  - **monitoring**: **4/5** — Rollout tracking and eval tables present; add automated gate enforcement.
  - **governance**: **4/5** — Strong lineage + rollout metadata; phase60 lineage fixed.
- **key_gaps**:
  - True uplift validation (OPE, RCT-like)
  - Streaming updates for rapid risk shifts

### `osint_graph` — OSINT/GraphFraud

- **purpose**: Ownership/UBO risk inference including offshore/proxy/phoenix patterns.
- **model_type**: Tabular (XGBoost/GB) + graph heuristic + GNN (RGCN/HGT) benchmark.
- **training_paths**: `Backend/ml_engine/train_osint.py`, `Backend/ml_engine/train_osint_heterograph.py`
- **serving_paths**: `Backend/app/routers/osint.py`, `Backend/app/routers/graph.py`
- **materializers**: `Backend/app/scripts/materialize_osint_graph_readiness.py`, `Backend/app/scripts/build_osint_graph_snapshot.py`, `Backend/app/scripts/bootstrap_osint_graph_benchmark.py`
- **output_tables**: `graph_nodes`, `graph_edges`, `graph_edge_evidence`, `graph_snapshots`, `graph_labels`, `graph_benchmark_specs`
- **api_endpoints**: `/api/osint/*`, `/api/graph/*`
- **fallback**: Yes (baseline + heuristic + label projection).
- **maturity**:
  - **data**: **3/5** — Evidence provenance modeled; still needs validated labels and ongoing OSINT ingestion.
  - **model_quality**: **3/5** — GNN harness exists; needs operational serving + backtesting against investigations.
  - **serving**: **3/5** — Graph APIs exist; deep model serving not yet a standard prod path.
  - **monitoring**: **4/5** — Benchmark contract + eval/calibration tables exist.
  - **governance**: **4/5** — Good snapshot/versioning + champion/challenger scaffolding.
- **key_gaps**:
  - Real-time OSINT updates
  - GNN inference service + latency/rollback tooling

### `macro_simulation_hypothesis` — Macro/Forecasting

- **purpose**: 1/5/10y macro simulation with probabilistic/regime-aware residuals + narrative generation.
- **model_type**: Hybrid baseline + residual ML + constraint engine + local narrative expansion.
- **training_paths**: `Backend/ml_engine/train_simulation.py`
- **serving_paths**: `Backend/app/routers/simulation.py`
- **materializers**: `-`
- **output_tables**: `macro_hypothesis_runs`, `macro_hypothesis_outputs`, `macro_external_signals`
- **api_endpoints**: `/api/simulation/*`
- **fallback**: Yes (baseline elasticity model).
- **maturity**:
  - **data**: **3/5** — External signals modeled; still limited by historical horizon length/noise.
  - **model_quality**: **4/5** — Advanced features implemented (probabilistic, regime-aware, backtesting).
  - **serving**: **4/5** — Comprehensive API suite; outputs materialized.
  - **monitoring**: **3/5** — Needs forecast calibration monitoring by horizon in standard dashboard.
  - **governance**: **4/5** — Run registry + constraints audit logs exist.
- **key_gaps**:
  - Longer history + true observed/synthetic separation
  - Forecast governance dashboards

### `tax_agent_router` — TaxAgent

- **purpose**: Offline/internal tax assistant: intent routing + retrieval + policy gating + trace ledger.
- **model_type**: Rule intent + hash-TF embedding retrieval + template synthesis + policy thresholds.
- **training_paths**: `Backend/app/scripts/ingest_tax_knowledge.py`
- **serving_paths**: `Backend/app/routers/tax_agent.py`
- **materializers**: `-`
- **output_tables**: `knowledge_documents`, `knowledge_document_versions`, `knowledge_chunks`, `knowledge_chunk_embeddings`, `retrieval_logs`, `agent_sessions`, `agent_turns`, `agent_tool_calls`, `agent_decision_traces`, `policy_rules`, `policy_execution_logs`, `agent_eval_suites`, `agent_eval_runs`
- **api_endpoints**: `/api/tax-agent/chat`
- **fallback**: Rules are primary; missing learned intent/rerank/synthesis components.
- **maturity**:
  - **data**: **3/5** — KB versioning exists; missing relevance/grounding labels and adjudication.
  - **model_quality**: **2/5** — Routing/retrieval/synthesis are prototype-grade; needs ML upgrades.
  - **serving**: **3/5** — Endpoint works; retrieval currently scans limited rows and scores in Python.
  - **monitoring**: **4/5** — Trace logs exist; needs eval harness + dashboards.
  - **governance**: **3/5** — Policy logs exist; missing policy version history and prompt registry.
- **key_gaps**:
  - Intent classifier + calibration + OOD
  - Hybrid retrieval + ANN + reranker
  - Grounded synthesis model (local) and citation faithfulness gates
  - Red-team and safety regression suite
