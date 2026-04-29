from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ModelScorecard:
    model_key: str
    domain: str
    purpose: str
    model_type: str
    training_paths: list[str]
    serving_paths: list[str]
    materializers: list[str]
    output_tables: list[str]
    api_endpoints: list[str]
    fallback: str
    maturity: dict[str, dict[str, object]]  # {dimension: {score:int, notes:str}}
    key_gaps: list[str]


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def build_catalog() -> list[ModelScorecard]:
    # NOTE: This is a curated catalog (not auto-discovered) to keep it stable and reviewable.
    # Update this list whenever new train/serve scripts are added.
    return [
        ModelScorecard(
            model_key="delinquency",
            domain="Collections/Compliance",
            purpose="Predict late/overdue payment risk (P30/P60/P90) for companies.",
            model_type="LightGBMClassifier (tabular) + feature engineer; DL benchmark exists (GRU).",
            training_paths=[
                "Backend/ml_engine/train_delinquency.py",
                "Backend/ml_engine/benchmark_delinquency_sequence.py",
            ],
            serving_paths=["Backend/app/routers/delinquency.py", "Backend/ml_engine/delinquency_model.py"],
            materializers=[],
            output_tables=["delinquency_predictions"],
            api_endpoints=["/api/delinquency", "/api/delinquency/{tax_code}", "/api/delinquency/predict-batch"],
            fallback="Has baseline fallback logic in pipeline; DL is benchmark-only.",
            maturity={
                "data": {"score": 4, "notes": "Uses tax_payments history; label logic exists; still needs adjudicated ground-truth for edge cases."},
                "model_quality": {"score": 4, "notes": "Strong tabular baseline + calibration hooks; DL benchmark provides upgrade path."},
                "serving": {"score": 4, "notes": "FastAPI endpoints + DB materialization table."},
                "monitoring": {"score": 4, "notes": "Quality/drift tables available; needs automated alert thresholds by horizon."},
                "governance": {"score": 4, "notes": "ModelRegistry + training runs supported; rollout gating can be strengthened."},
            },
            key_gaps=["Automated retrain triggers by drift", "Formal horizon-specific calibration gates"],
        ),
        ModelScorecard(
            model_key="vat_refund_risk",
            domain="Compliance/Fraud",
            purpose="Score VAT refund cases for fraud/abuse risk.",
            model_type="RandomForestClassifier (tabular) + optional calibration.",
            training_paths=["Backend/ml_engine/train_vat_refund.py"],
            serving_paths=["Backend/app/routers/vat_refund.py"],
            materializers=["Backend/app/scripts/train_vat_refund_case_risk.py"],
            output_tables=["vat_refund_predictions"],
            api_endpoints=["/api/vat-refund/cases", "/api/vat-refund/cases/{case_id}/risk"],
            fallback="Table-driven serving; fallback not explicitly exposed but can be added.",
            maturity={
                "data": {"score": 3, "notes": "Depends on case outcome labeling quality; needs stronger adverse-outcome ground-truth."},
                "model_quality": {"score": 3, "notes": "Solid classical baseline; could benefit from calibrated GBM and richer fraud signals."},
                "serving": {"score": 4, "notes": "Predictions persisted; endpoints consume consistently."},
                "monitoring": {"score": 3, "notes": "Slices/calibration possible but not consistently enforced."},
                "governance": {"score": 3, "notes": "Registry exists; needs champion/challenger + rollback gates."},
            },
            key_gaps=["Outcome label pipeline + adjudication", "Cost-sensitive thresholds tied to audit capacity"],
        ),
        ModelScorecard(
            model_key="invoice_risk",
            domain="Fraud",
            purpose="Score invoices for suspicious behavior (duplication, abnormal patterns).",
            model_type="RandomForestClassifier (learned) + heuristics.",
            training_paths=["Backend/ml_engine/train_invoice_risk_model.py"],
            serving_paths=["Backend/app/routers/invoice_risk.py", "Backend/ml_engine/invoice_risk_model.py"],
            materializers=[],
            output_tables=["invoice_risk_predictions"],
            api_endpoints=["/api/invoice/{invoice_number}/risk", "/api/invoice/risk"],
            fallback="Has heuristic fallback when learned artifact missing.",
            maturity={
                "data": {"score": 3, "notes": "Pseudo-label heavy; needs investigation outcomes / confirmed fraud labels."},
                "model_quality": {"score": 3, "notes": "Good baseline; missing sequence/graph context in main model."},
                "serving": {"score": 4, "notes": "Online scoring endpoints + audit logging."},
                "monitoring": {"score": 3, "notes": "Needs consistent calibration and slice monitoring (sector/size)." },
                "governance": {"score": 4, "notes": "Inference audit exists; rollout gates can be formalized."},
            },
            key_gaps=["Ground-truth outcomes", "Graph-aware invoice ring detection model (GNN) in production path"],
        ),
        ModelScorecard(
            model_key="transfer_pricing_mispricing",
            domain="TransferPricing",
            purpose="Detect mispricing/outlier behavior in related-party trade pricing.",
            model_type="RandomForestClassifier (learned-first) + z-score baseline fallback.",
            training_paths=["Backend/ml_engine/train_transfer_pricing_model.py"],
            serving_paths=["Backend/app/routers/transfer_pricing.py"],
            materializers=[],
            output_tables=["mispricing_predictions"],
            api_endpoints=["/api/transfer-pricing/score", "/api/transfer-pricing/mispricing"],
            fallback="Yes (baseline z-score).",
            maturity={
                "data": {"score": 2, "notes": "Needs richer comparables, customs data, and adjudicated TP adjustments."},
                "model_quality": {"score": 3, "notes": "Strong baseline; could add quantile regression and monotonic constraints."},
                "serving": {"score": 4, "notes": "Router supports learned-first fallback."},
                "monitoring": {"score": 3, "notes": "Needs slice monitoring by HS code / partner country / industry."},
                "governance": {"score": 4, "notes": "Training run + registry integrated."},
            },
            key_gaps=["Comparable price corpus + label adjudication", "Explanation templates for legal defensibility"],
        ),
        ModelScorecard(
            model_key="ops_uplift_audit_collections",
            domain="Ops/Collections",
            purpose="Audit selection, next-best-action, and expected collection uplift.",
            model_type="RandomForestClassifier + RandomForestRegressor + hybrid materialized fusion.",
            training_paths=["Backend/ml_engine/train_ops_uplift_models.py"],
            serving_paths=["Backend/app/routers/audit_selection.py", "Backend/app/routers/collections.py", "Backend/app/routers/case_triage.py"],
            materializers=["Backend/app/scripts/materialize_ops_predictions.py", "Backend/app/scripts/materialize_phase60_models.py"],
            output_tables=["audit_selection_predictions", "case_triage_predictions", "nba_predictions", "entity_risk_fusion_predictions"],
            api_endpoints=["/api/audit/shortlist", "/api/case-triage/queue", "/api/collections/next-best-action", "/api/collections/fusion-overview"],
            fallback="Yes (hybrid formulas if learned artifacts missing).",
            maturity={
                "data": {"score": 3, "notes": "Needs causal/experimental design for uplift (policy changes, selection bias)." },
                "model_quality": {"score": 3, "notes": "Good start; can move to CATE/uplift trees + off-policy evaluation."},
                "serving": {"score": 4, "notes": "Batch materialization stable; online feature service could improve freshness."},
                "monitoring": {"score": 4, "notes": "Rollout tracking and eval tables present; add automated gate enforcement."},
                "governance": {"score": 4, "notes": "Strong lineage + rollout metadata; phase60 lineage fixed."},
            },
            key_gaps=["True uplift validation (OPE, RCT-like)", "Streaming updates for rapid risk shifts"],
        ),
        ModelScorecard(
            model_key="osint_graph",
            domain="OSINT/GraphFraud",
            purpose="Ownership/UBO risk inference including offshore/proxy/phoenix patterns.",
            model_type="Tabular (XGBoost/GB) + graph heuristic + GNN (RGCN/HGT) benchmark.",
            training_paths=["Backend/ml_engine/train_osint.py", "Backend/ml_engine/train_osint_heterograph.py"],
            serving_paths=["Backend/app/routers/osint.py", "Backend/app/routers/graph.py"],
            materializers=[
                "Backend/app/scripts/materialize_osint_graph_readiness.py",
                "Backend/app/scripts/build_osint_graph_snapshot.py",
                "Backend/app/scripts/bootstrap_osint_graph_benchmark.py",
            ],
            output_tables=[
                "graph_nodes",
                "graph_edges",
                "graph_edge_evidence",
                "graph_snapshots",
                "graph_labels",
                "graph_benchmark_specs",
            ],
            api_endpoints=["/api/osint/*", "/api/graph/*"],
            fallback="Yes (baseline + heuristic + label projection).",
            maturity={
                "data": {"score": 3, "notes": "Evidence provenance modeled; still needs validated labels and ongoing OSINT ingestion."},
                "model_quality": {"score": 3, "notes": "GNN harness exists; needs operational serving + backtesting against investigations."},
                "serving": {"score": 3, "notes": "Graph APIs exist; deep model serving not yet a standard prod path."},
                "monitoring": {"score": 4, "notes": "Benchmark contract + eval/calibration tables exist."},
                "governance": {"score": 4, "notes": "Good snapshot/versioning + champion/challenger scaffolding."},
            },
            key_gaps=["Real-time OSINT updates", "GNN inference service + latency/rollback tooling"],
        ),
        ModelScorecard(
            model_key="macro_simulation_hypothesis",
            domain="Macro/Forecasting",
            purpose="1/5/10y macro simulation with probabilistic/regime-aware residuals + narrative generation.",
            model_type="Hybrid baseline + residual ML + constraint engine + local narrative expansion.",
            training_paths=["Backend/ml_engine/train_simulation.py"],
            serving_paths=["Backend/app/routers/simulation.py"],
            materializers=[],
            output_tables=["macro_hypothesis_runs", "macro_hypothesis_outputs", "macro_external_signals"],
            api_endpoints=["/api/simulation/*"],
            fallback="Yes (baseline elasticity model).",
            maturity={
                "data": {"score": 3, "notes": "External signals modeled; still limited by historical horizon length/noise."},
                "model_quality": {"score": 4, "notes": "Advanced features implemented (probabilistic, regime-aware, backtesting)."},
                "serving": {"score": 4, "notes": "Comprehensive API suite; outputs materialized."},
                "monitoring": {"score": 3, "notes": "Needs forecast calibration monitoring by horizon in standard dashboard."},
                "governance": {"score": 4, "notes": "Run registry + constraints audit logs exist."},
            },
            key_gaps=["Longer history + true observed/synthetic separation", "Forecast governance dashboards"],
        ),
        ModelScorecard(
            model_key="tax_agent_router",
            domain="TaxAgent",
            purpose="Offline/internal tax assistant: intent routing + retrieval + policy gating + trace ledger.",
            model_type="Rule intent + hash-TF embedding retrieval + template synthesis + policy thresholds.",
            training_paths=["Backend/app/scripts/ingest_tax_knowledge.py"],
            serving_paths=["Backend/app/routers/tax_agent.py"],
            materializers=[],
            output_tables=[
                "knowledge_documents",
                "knowledge_document_versions",
                "knowledge_chunks",
                "knowledge_chunk_embeddings",
                "retrieval_logs",
                "agent_sessions",
                "agent_turns",
                "agent_tool_calls",
                "agent_decision_traces",
                "policy_rules",
                "policy_execution_logs",
                "agent_eval_suites",
                "agent_eval_runs",
            ],
            api_endpoints=["/api/tax-agent/chat"],
            fallback="Rules are primary; missing learned intent/rerank/synthesis components.",
            maturity={
                "data": {"score": 3, "notes": "KB versioning exists; missing relevance/grounding labels and adjudication."},
                "model_quality": {"score": 2, "notes": "Routing/retrieval/synthesis are prototype-grade; needs ML upgrades."},
                "serving": {"score": 3, "notes": "Endpoint works; retrieval currently scans limited rows and scores in Python."},
                "monitoring": {"score": 4, "notes": "Trace logs exist; needs eval harness + dashboards."},
                "governance": {"score": 3, "notes": "Policy logs exist; missing policy version history and prompt registry."},
            },
            key_gaps=[
                "Intent classifier + calibration + OOD",
                "Hybrid retrieval + ANN + reranker",
                "Grounded synthesis model (local) and citation faithfulness gates",
                "Red-team and safety regression suite",
            ],
        ),
    ]


def _to_markdown(cards: list[ModelScorecard]) -> str:
    lines: list[str] = []
    lines.append(f"# TaxInspector Model Scorecard\n")
    lines.append(f"- Generated at: `{_now_iso()}`\n")
    lines.append("## Summary\n")
    lines.append(
        "This scorecard standardizes maturity across **Data / ModelQuality / Serving / Monitoring / Governance** "
        "on a 1–5 scale (higher is more production-ready).\n"
    )
    lines.append("## Catalog\n")
    for c in cards:
        lines.append(f"### `{c.model_key}` — {c.domain}\n")
        lines.append(f"- **purpose**: {c.purpose}")
        lines.append(f"- **model_type**: {c.model_type}")
        lines.append(f"- **training_paths**: {', '.join(f'`{p}`' for p in c.training_paths) if c.training_paths else '`-`'}")
        lines.append(f"- **serving_paths**: {', '.join(f'`{p}`' for p in c.serving_paths) if c.serving_paths else '`-`'}")
        lines.append(f"- **materializers**: {', '.join(f'`{p}`' for p in c.materializers) if c.materializers else '`-`'}")
        lines.append(f"- **output_tables**: {', '.join(f'`{t}`' for t in c.output_tables) if c.output_tables else '`-`'}")
        lines.append(f"- **api_endpoints**: {', '.join(f'`{e}`' for e in c.api_endpoints) if c.api_endpoints else '`-`'}")
        lines.append(f"- **fallback**: {c.fallback}")
        lines.append("- **maturity**:")
        for dim in ["data", "model_quality", "serving", "monitoring", "governance"]:
            entry = c.maturity.get(dim, {"score": 1, "notes": ""})
            lines.append(f"  - **{dim}**: **{entry.get('score', 1)}/5** — {entry.get('notes','')}")
        if c.key_gaps:
            lines.append("- **key_gaps**:")
            for g in c.key_gaps:
                lines.append(f"  - {g}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    cards = build_catalog()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "Backend" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "model_scorecard.md"
    json_path = out_dir / "model_scorecard.json"

    md_path.write_text(_to_markdown(cards), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(c) for c in cards], indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[OK] wrote: {md_path}")
    print(f"[OK] wrote: {json_path}")


if __name__ == "__main__":
    main()

