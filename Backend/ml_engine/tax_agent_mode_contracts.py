"""
Shared mode contracts for TaxInspector multi-agent chat.

This module is intentionally small and dependency-light so routing,
orchestration, API serialization, and frontend contract tests can use the
same source of truth instead of drifting mode-by-mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AGENT_RESPONSE_SCHEMA_VERSION = "agent-chat-v2.1"
MODE_CONTRACT_VERSION = "mode-contract-v1.0"


@dataclass(frozen=True)
class ModeContract:
    mode: str
    label: str
    answer_contract: str
    allowed_tools: set[str] | None = None
    selected_model_bundle: list[str] = field(default_factory=list)
    required_visualization_keys: list[str] = field(default_factory=list)
    workspace_panel: str | None = None
    canonical_upload_handlers: dict[str, str] = field(default_factory=dict)


class AgentModeContractRegistry:
    """Authoritative mode profile registry for chat routing and UI payloads."""

    _CONTRACTS: dict[str, ModeContract] = {
        "legal": ModeContract(
            mode="legal",
            label="Phap ly",
            answer_contract="legal_consultation",
            allowed_tools={"knowledge_search"},
            selected_model_bundle=["GraphRAG", "legal_verifier", "citation_grounding"],
            required_visualization_keys=["knowledge_graph"],
            workspace_panel="legal",
            canonical_upload_handlers={"document": "legal_document", "ocr_invoice": "legal_document"},
        ),
        "fraud": ModeContract(
            mode="fraud",
            label="Gian lan",
            answer_contract="fraud_analysis",
            allowed_tools={
                "company_risk_lookup",
                "top_n_risky_companies",
                "invoice_risk_scan",
                "gnn_analysis",
                "hetero_gnn_risk",
                "vae_anomaly_scan",
                "motif_detection",
                "ownership_analysis",
                "nlp_red_flag_scan",
                "ring_scoring",
                "entity_resolution_check",
            },
            selected_model_bundle=["TaxFraudPipeline", "XGBoost", "IsolationForest", "VAE", "SHAP_XAI"],
            required_visualization_keys=["fraud", "risk_gauge", "batch_summary"],
            workspace_panel="fraud",
            canonical_upload_handlers={"risk_csv": "risk_batch"},
        ),
        "vat": ModeContract(
            mode="vat",
            label="VAT & Hoa don",
            answer_contract="vat_graph",
            allowed_tools={
                "company_risk_lookup",
                "invoice_risk_scan",
                "vat_refund_risk",
                "vae_anomaly_scan",
                "nlp_red_flag_scan",
                "gnn_analysis",
                "motif_detection",
                "ownership_analysis",
                "ring_scoring",
                "entity_resolution_check",
            },
            selected_model_bundle=["VAT_GNN", "invoice_risk", "ring_motif_detection", "OCR_invoice"],
            required_visualization_keys=["vat", "vat_graph_batch", "ocr_extraction"],
            workspace_panel="vat",
            canonical_upload_handlers={"vat_graph_csv": "vat_graph", "ocr_invoice": "ocr_invoice"},
        ),
        "delinquency": ModeContract(
            mode="delinquency",
            label="Du bao no",
            answer_contract="risk_profile",
            allowed_tools={
                "company_risk_lookup",
                "delinquency_check",
                "temporal_delinquency_deep",
                "causal_uplift_recommend",
                "revenue_forecast",
            },
            selected_model_bundle=["temporal_delinquency", "revenue_forecast", "causal_uplift"],
            required_visualization_keys=["delinquency_timeline", "uplift_actions"],
            workspace_panel="delinquency",
        ),
        "macro": ModeContract(
            mode="macro",
            label="Vi mo",
            answer_contract="risk_profile",
            allowed_tools={"macro_forecast", "revenue_forecast"},
            selected_model_bundle=["macro_forecast", "scenario_simulation"],
            required_visualization_keys=["macro_kpis"],
            workspace_panel="simulation",
        ),
        "full": ModeContract(
            mode="full",
            label="Auto",
            answer_contract="risk_profile",
            allowed_tools=None,
            selected_model_bundle=["dialogue_policy"],
            required_visualization_keys=[],
            workspace_panel=None,
        ),
    }

    @classmethod
    def get(cls, mode: str | None) -> ModeContract:
        return cls._CONTRACTS.get((mode or "full").lower(), cls._CONTRACTS["full"])

    @classmethod
    def domain_allowed_tools(cls) -> dict[str, set[str]]:
        return {
            mode: set(contract.allowed_tools or set())
            for mode, contract in cls._CONTRACTS.items()
            if mode != "full"
        }

    @classmethod
    def capability_registry(cls) -> dict[str, list[str]]:
        return {
            **{
                mode: list(contract.selected_model_bundle)
                for mode, contract in cls._CONTRACTS.items()
                if mode != "full"
            },
            "general": ["dialogue_policy"],
        }

    @classmethod
    def metadata(cls, mode: str | None) -> dict[str, Any]:
        contract = cls.get(mode)
        return {
            "schema_version": AGENT_RESPONSE_SCHEMA_VERSION,
            "mode_contract_version": MODE_CONTRACT_VERSION,
            "mode": contract.mode,
            "label": contract.label,
            "answer_contract": contract.answer_contract,
            "selected_model_bundle": list(contract.selected_model_bundle),
            "required_visualization_keys": list(contract.required_visualization_keys),
            "workspace_panel": contract.workspace_panel,
        }


def canonical_run_state(state: str | None) -> str:
    allowed = {"queued", "streaming", "finalized", "partial_error", "error", "cancelled"}
    value = (state or "finalized").lower()
    return value if value in allowed else "finalized"

