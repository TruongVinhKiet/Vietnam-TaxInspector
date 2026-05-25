"""Authoritative model inventory for TaxInspector.

The README and infrastructure endpoints should describe models from this
lightweight catalog instead of maintaining separate hand-written lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelInventoryEntry:
    model_key: str
    display_name: str
    family: str
    artifacts: tuple[str, ...] = ()
    training_scripts: tuple[str, ...] = ()
    serving_surface: tuple[str, ...] = ()
    agent_tools: tuple[str, ...] = ()
    ui_pages: tuple[str, ...] = ()
    fallback: str = ""
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def readiness(self) -> dict[str, Any]:
        checked = []
        existing = []
        missing = []
        for artifact in self.artifacts:
            if any(ch in artifact for ch in "*?"):
                checked.append({"path": artifact, "exists": None, "wildcard": True})
                continue
            path = REPO_ROOT / artifact
            exists = path.exists()
            checked.append({"path": artifact, "exists": exists})
            if exists:
                existing.append(artifact)
            else:
                missing.append(artifact)
        return {
            "artifact_count": len(self.artifacts),
            "existing_count": len(existing),
            "missing_count": len(missing),
            "ready": not self.artifacts or len(missing) == 0,
            "checked": checked,
        }

    def to_dict(self, *, include_readiness: bool = True) -> dict[str, Any]:
        payload = {
            "model_key": self.model_key,
            "display_name": self.display_name,
            "family": self.family,
            "artifact_paths": list(self.artifacts),
            "training_scripts": list(self.training_scripts),
            "serving_surface": list(self.serving_surface),
            "agent_tools": list(self.agent_tools),
            "ui_pages": list(self.ui_pages),
            "fallback": self.fallback,
            "notes": self.notes,
            "tags": list(self.tags),
        }
        if include_readiness:
            payload["readiness_check"] = self.readiness()
        return payload


MODEL_INVENTORY: tuple[ModelInventoryEntry, ...] = (
    ModelInventoryEntry(
        model_key="fraud-hybrid-v2",
        display_name="Fraud hybrid risk scoring",
        family="Fraud",
        artifacts=(
            "Backend/data/models/xgboost_model.joblib",
            "Backend/data/models/isolation_forest.joblib",
            "Backend/data/models/fraud_calibrator.joblib",
            "Backend/data/models/shap_background.joblib",
            "Backend/data/models/fraud_model_manifest.json",
        ),
        training_scripts=("Backend/ml_engine/train_model.py",),
        serving_surface=("/api/ai/*", "ml_engine.pipeline.TaxFraudPipeline"),
        agent_tools=("company_risk_lookup", "top_n_risky_companies"),
        ui_pages=("Frontend/pages/fraud.html", "Frontend/pages/agent.html"),
        fallback="fraud-hybrid-legacy / rules if artifacts are unavailable",
        tags=("xgboost", "isolation_forest", "calibration", "xai"),
    ),
    ModelInventoryEntry(
        model_key="delinquency-temporal-v1",
        display_name="Delinquency temporal risk",
        family="Delinquency",
        artifacts=(
            "Backend/data/models/delinquency_lgbm.joblib",
            "Backend/data/models/delinquency_config.json",
            "Backend/data/models/delinquency_quality_report.json",
        ),
        training_scripts=("Backend/ml_engine/train_delinquency.py",),
        serving_surface=("/api/delinquency/*", "ml_engine.delinquency_model"),
        agent_tools=("delinquency_check",),
        ui_pages=("Frontend/pages/delinquency.html", "Frontend/pages/agent.html"),
        fallback="statistical delinquency baseline",
        tags=("lightgbm", "tabular", "collections"),
    ),
    ModelInventoryEntry(
        model_key="temporal-transformer-v1",
        display_name="Temporal Transformer delinquency deep model",
        family="Delinquency",
        artifacts=(
            "Backend/data/models/temporal_transformer.pt",
            "Backend/data/models/temporal_transformer_config.json",
        ),
        training_scripts=("Backend/ml_engine/train_temporal_transformer.py",),
        serving_surface=("/predict/transformer", "ModelServingGateway:transformer"),
        agent_tools=("temporal_delinquency_deep",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="delinquency-temporal-v1",
        tags=("torch", "sequence"),
    ),
    ModelInventoryEntry(
        model_key="vae-anomaly-v1",
        display_name="VAE transaction anomaly detector",
        family="Fraud",
        artifacts=(
            "Backend/data/models/vae_anomaly.pt",
            "Backend/data/models/vae_anomaly_config.json",
            "Backend/data/models/vae_anomaly_scaler.json",
        ),
        training_scripts=("Backend/ml_engine/train_vae_anomaly.py",),
        serving_surface=("/predict/vae", "ModelServingGateway:vae"),
        agent_tools=("vae_anomaly_scan",),
        ui_pages=("Frontend/pages/agent.html", "Frontend/pages/fraud.html"),
        fallback="rule anomaly score",
        tags=("torch", "anomaly"),
    ),
    ModelInventoryEntry(
        model_key="gnn-gat-v1",
        display_name="VAT graph GAT fraud model",
        family="Graph/Fraud",
        artifacts=("Backend/data/models/gat_model.pt", "Backend/data/models/gat_config.json"),
        training_scripts=("Backend/app/scripts/train_gnn.py",),
        serving_surface=("/predict/gnn", "ModelServingGateway:gnn", "/api/graph/*"),
        agent_tools=("gnn_analysis", "motif_detection", "ownership_analysis"),
        ui_pages=("Frontend/pages/graph.html", "Frontend/pages/agent.html"),
        fallback="NetworkX graph heuristics",
        tags=("torch", "gnn", "gat"),
    ),
    ModelInventoryEntry(
        model_key="hetero-gnn-hgt-v1",
        display_name="Heterogeneous Graph Transformer",
        family="Graph/OSINT",
        artifacts=("Backend/data/models/hgt_model.pt", "Backend/data/models/hgt_config.json"),
        training_scripts=("Backend/ml_engine/train_hetero_gnn.py", "Backend/ml_engine/train_osint_heterograph.py"),
        serving_surface=("/predict/hetero-gnn", "ModelServingGateway:hetero_gnn", "/api/osint/*"),
        agent_tools=("hetero_gnn_risk", "ownership_analysis"),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="OSINT tabular/rule baseline",
        tags=("torch", "hgt", "heterogeneous_graph"),
    ),
    ModelInventoryEntry(
        model_key="vat-refund-v1",
        display_name="VAT refund case risk",
        family="VAT",
        artifacts=("Backend/data/models/vat_refund_model.joblib", "Backend/data/models/vat_refund_calibrator.joblib"),
        training_scripts=("Backend/ml_engine/train_vat_refund.py", "Backend/app/scripts/train_vat_refund_case_risk.py"),
        serving_surface=("/api/vat-refund/*", "/api/ai/*"),
        agent_tools=("vat_refund_risk",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="vat-refund-heuristic",
        tags=("random_forest", "calibration"),
    ),
    ModelInventoryEntry(
        model_key="audit-value-v1",
        display_name="Audit value / recoverability",
        family="Audit",
        artifacts=("Backend/data/models/audit_value_model.joblib", "Backend/data/models/audit_value_calibrator.joblib"),
        training_scripts=("Backend/ml_engine/train_audit_value.py",),
        serving_surface=("/api/ai/*",),
        agent_tools=("audit_selection",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="audit-value-heuristic",
        tags=("random_forest", "audit"),
    ),
    ModelInventoryEntry(
        model_key="invoice-risk-v1",
        display_name="Invoice risk scorer",
        family="VAT/Fraud",
        artifacts=("Backend/data/models/invoice_risk_model.joblib", "Backend/data/models/invoice_risk_config.json"),
        training_scripts=("Backend/ml_engine/train_invoice_risk_model.py", "Backend/app/scripts/train_invoice_risk.py"),
        serving_surface=("/api/invoice/*", "ml_engine.invoice_risk_model.InvoiceRiskScorer"),
        agent_tools=("invoice_risk_scan",),
        ui_pages=("Frontend/pages/agent.html", "Frontend/pages/graph.html"),
        fallback="invoice-risk-heuristic-v1",
        tags=("random_forest", "heuristic"),
    ),
    ModelInventoryEntry(
        model_key="transfer-pricing-v1",
        display_name="Transfer pricing mispricing",
        family="TransferPricing",
        artifacts=("Backend/data/models/transfer_pricing_model.joblib", "Backend/data/models/transfer_pricing_model_meta.json"),
        training_scripts=("Backend/ml_engine/train_transfer_pricing_model.py",),
        serving_surface=("/api/transfer-pricing/*",),
        agent_tools=("transfer_pricing_check",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="z-score baseline",
        tags=("random_forest", "mispricing"),
    ),
    ModelInventoryEntry(
        model_key="macro-simulation-v1",
        display_name="Macro simulation legacy compatibility model",
        family="Macro",
        artifacts=("Backend/data/models/simulation_lgbm.joblib", "Backend/data/models/simulation_config.json"),
        training_scripts=("Backend/ml_engine/train_simulation.py", "Backend/app/scripts/train_macro_hypothesis.py"),
        serving_surface=("/api/simulation/*",),
        agent_tools=("macro_forecast", "revenue_forecast"),
        ui_pages=("Frontend/pages/simulation.html", "Frontend/pages/agent.html"),
        fallback="macro-ensemble-v2 / baseline elasticity",
        notes="Backward-compatible model key retained for existing UI and agent contracts.",
        tags=("lightgbm", "scenario", "legacy"),
    ),
    ModelInventoryEntry(
        model_key="macro-ensemble-v2",
        display_name="Macro-Fiscal ensemble forecast lab",
        family="Macro",
        artifacts=(
            "Backend/data/models/simulation_lgbm.joblib",
            "Backend/data/models/simulation_config.json",
            "Backend/reports/macro_research_lab/macro_research_evaluation.json",
        ),
        training_scripts=(
            "Backend/ml_engine/train_simulation.py",
            "Backend/scripts/run_macro_research_evaluation.py",
        ),
        serving_surface=(
            "/api/simulation/forecast/run",
            "/api/simulation/research/state",
            "/api/simulation/data-quality",
        ),
        agent_tools=("macro_forecast", "revenue_forecast"),
        ui_pages=("Frontend/pages/simulation.html", "Frontend/pages/agent.html"),
        fallback="macro-simulation-v1 elasticity/ridge baseline",
        notes="Hybrid research endpoint with fan chart uncertainty, model cards and approved-source policy.",
        tags=("lightgbm", "tft_ready", "nbeats_ready", "conformal", "research_lab"),
    ),
    ModelInventoryEntry(
        model_key="macro-shock-graph-v1",
        display_name="Spatio-temporal macro shock propagation graph",
        family="Macro/Graph",
        artifacts=(),
        training_scripts=("Backend/ml_engine/macro_research_lab.py",),
        serving_surface=("/api/simulation/shock-propagation/run",),
        agent_tools=("macro_forecast",),
        ui_pages=("Frontend/pages/simulation.html",),
        fallback="distance/economic-similarity diffusion",
        notes="STGCN-style contract for shock propagation across 34/63 province graphs.",
        tags=("stgcn_ready", "graph", "shock_propagation", "spatial"),
    ),
    ModelInventoryEntry(
        model_key="macro-causal-merger-v1",
        display_name="Causal merger and policy impact lab",
        family="Macro/Causal",
        artifacts=(),
        training_scripts=("Backend/ml_engine/macro_research_lab.py",),
        serving_surface=("/api/simulation/causal/merger-effect",),
        agent_tools=("macro_forecast",),
        ui_pages=("Frontend/pages/simulation.html",),
        fallback="synthetic-control proxy / event-study fallback",
        notes="CausalImpact/Synthetic-Control inspired contract for merger and tax-policy evaluation.",
        tags=("synthetic_control", "did", "placebo", "causal"),
    ),
    ModelInventoryEntry(
        model_key="revenue-forecast-v1",
        display_name="Revenue forecast",
        family="Forecasting",
        artifacts=(),
        training_scripts=("Backend/ml_engine/revenue_forecast_model.py",),
        serving_surface=("ml_engine.revenue_forecast_model",),
        agent_tools=("revenue_forecast",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="seasonal/statistical forecast",
        tags=("forecasting",),
    ),
    ModelInventoryEntry(
        model_key="causal-uplift-v1",
        display_name="Collections causal uplift / NBA",
        family="Collections",
        artifacts=(
            "Backend/data/models/uplift_model_treated.joblib",
            "Backend/data/models/uplift_model_control.joblib",
            "Backend/data/models/uplift_propensity.joblib",
        ),
        training_scripts=("Backend/ml_engine/train_ops_uplift_models.py", "Backend/app/scripts/train_ops_uplift_models.py"),
        serving_surface=("/api/collections/*", "ml_engine.causal_uplift_model"),
        agent_tools=("causal_uplift_recommend",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="policy/rule next-best-action",
        tags=("uplift", "t_learner"),
    ),
    ModelInventoryEntry(
        model_key="audit-selection-v1",
        display_name="Audit selection learned model",
        family="Audit",
        artifacts=("Backend/data/models/audit_selection_learned_model.joblib",),
        training_scripts=("Backend/ml_engine/train_ops_uplift_models.py",),
        serving_surface=("/api/audit/*", "/api/case-triage/*"),
        agent_tools=("audit_selection",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="hybrid priority formula",
        tags=("ranking", "ops"),
    ),
    ModelInventoryEntry(
        model_key="osint-risk-v1",
        display_name="OSINT/offshore risk classifier",
        family="OSINT",
        artifacts=("Backend/data/models/osint_risk_model.joblib", "Backend/data/models/osint_config.json"),
        training_scripts=("Backend/ml_engine/train_osint.py",),
        serving_surface=("/api/osint/*",),
        agent_tools=("ownership_analysis", "hetero_gnn_risk"),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="ownership graph heuristics",
        tags=("osint", "offshore"),
    ),
    ModelInventoryEntry(
        model_key="nlp-red-flag-v1",
        display_name="NLP red flag detector",
        family="NLP",
        artifacts=(),
        training_scripts=("Backend/ml_engine/nlp_red_flag_detector.py",),
        serving_surface=("/api/ml/redflag/*", "ml_engine.nlp_red_flag_detector"),
        agent_tools=("nlp_red_flag_scan",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="keyword/rule red flags",
        tags=("nlp", "rules"),
    ),
    ModelInventoryEntry(
        model_key="entity-resolution-v1",
        display_name="Entity resolution / Siamese Bi-Encoder",
        family="EntityResolution",
        artifacts=(),
        training_scripts=("Backend/app/scripts/generate_entity_resolution_data.py",),
        serving_surface=("/api/ml/entity/*", "/api/entity-resolution/*"),
        agent_tools=("entity_resolution_check",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="name/tax-code similarity rules",
        tags=("entity_resolution", "bi_encoder"),
    ),
    ModelInventoryEntry(
        model_key="ocr-document-v1",
        display_name="Document OCR and table extraction",
        family="OCR",
        artifacts=("Backend/data/models/table_transformer",),
        training_scripts=("Backend/app/scripts/download_table_models.py",),
        serving_surface=("/api/ml/ocr/*", "ml_engine.document_ocr_engine.DocumentOCREngine"),
        agent_tools=("ocr_document_process",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="PaddleOCR -> EasyOCR -> Tesseract -> regex_only",
        tags=("paddleocr", "easyocr", "tesseract", "table_transformer"),
    ),
    ModelInventoryEntry(
        model_key="tax-agent-intent-v1",
        display_name="Tax Agent intent classifier",
        family="TaxAgent",
        artifacts=("Backend/data/models/tax_agent_intent_model.joblib", "Backend/data/models/tax_agent_intent_vectorizer.joblib"),
        training_scripts=("Backend/ml_engine/train_tax_agent_intent.py",),
        serving_surface=("ml_engine.tax_agent_intent_model", "/api/tax-agent/chat/v2"),
        agent_tools=(),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="rule/regex intent router",
        tags=("intent", "routing"),
    ),
    ModelInventoryEntry(
        model_key="tax-agent-rag-v1",
        display_name="Tax Agent RAG, reranker, embeddings, GraphRAG",
        family="TaxAgent/RAG",
        artifacts=("Backend/data/models/embeddings", "Backend/data/models/reranker"),
        training_scripts=("Backend/app/scripts/ingest_tax_knowledge.py",),
        serving_surface=("/api/tax-agent/chat/v2", "ml_engine.tax_agent_graphrag"),
        agent_tools=("knowledge_search",),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="BM25 + hash-TF dense retrieval",
        tags=("pgvector", "bm25", "cross_encoder", "graphrag"),
    ),
    ModelInventoryEntry(
        model_key="tax-agent-llm-v1",
        display_name="Local Tax Agent LLM / LoRA adapter",
        family="TaxAgent/LLM",
        artifacts=("Backend/data/models/tax_llm_lora",),
        training_scripts=("Backend/run_llm_training.py", "Backend/ml_engine/tax_agent_llm_data_pipeline.py"),
        serving_surface=("ml_engine.tax_agent_llm_model", "/api/tax-agent/chat/v2"),
        agent_tools=(),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="template synthesis",
        tags=("local_llm", "lora"),
    ),
    ModelInventoryEntry(
        model_key="dpo-rlhf-v1",
        display_name="DPO/RLHF evaluator and preference trainer",
        family="Governance",
        artifacts=(),
        training_scripts=("Backend/ml_engine/rlhf_dpo_trainer.py",),
        serving_surface=("/api/tax-agent/feedback", "/api/tax-agent/dpo/*"),
        agent_tools=(),
        ui_pages=("Frontend/pages/agent.html",),
        fallback="feedback logging only",
        tags=("dpo", "rlhf", "evaluation"),
    ),
)


REQUIRED_MODEL_KEYS: tuple[str, ...] = tuple(entry.model_key for entry in MODEL_INVENTORY)


def get_model_inventory(*, include_readiness: bool = True) -> list[dict[str, Any]]:
    return [entry.to_dict(include_readiness=include_readiness) for entry in MODEL_INVENTORY]


def get_model_inventory_payload(*, include_readiness: bool = True) -> dict[str, Any]:
    models = get_model_inventory(include_readiness=include_readiness)
    families = sorted({model["family"] for model in models})
    return {
        "schema_version": "model-inventory-v1",
        "count": len(models),
        "families": families,
        "required_model_keys": list(REQUIRED_MODEL_KEYS),
        "models": models,
    }
