"""
Canonical tool contract for the TaxInspector multi-agent runtime.

This module is intentionally dependency-light.  It is the shared source of
truth for:
- LoRA/tool-calling prompts
- dataset generators
- contract tests
- backwards-compatible alias handling for older training data

Runtime-only orchestration actions such as debate/adjudication are not tools
and must not appear inside model <tool_call> JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalToolContract:
    name: str
    description: str
    intent: str
    default_mode: str
    required_args: tuple[str, ...] = ()
    optional_args: tuple[str, ...] = ()
    requires_tax_code: bool = False


CANONICAL_TOOL_CONTRACTS: tuple[CanonicalToolContract, ...] = (
    CanonicalToolContract(
        name="knowledge_search",
        description="Search tax-law RAG/GraphRAG evidence with citations.",
        intent="general_tax_query",
        default_mode="legal",
        required_args=("query",),
        optional_args=("intent", "top_k"),
    ),
    CanonicalToolContract(
        name="company_risk_lookup",
        description="Lookup the overall taxpayer/company risk profile.",
        intent="general_tax_query",
        default_mode="fraud",
        required_args=("tax_code",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="delinquency_check",
        description="Predict tax debt and late-payment risk.",
        intent="delinquency",
        default_mode="delinquency",
        required_args=("tax_code",),
        optional_args=("horizon_days",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="invoice_risk_scan",
        description="Scan invoice risk for a taxpayer or period.",
        intent="invoice_risk",
        default_mode="vat",
        required_args=("tax_code",),
        optional_args=("period",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="vat_refund_risk",
        description="Assess VAT refund case risk.",
        intent="vat_refund_risk",
        default_mode="vat",
        required_args=("tax_code",),
        optional_args=("period", "limit"),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="gnn_analysis",
        description="Analyze VAT transaction graph and GNN network risk.",
        intent="vat_network_analysis",
        default_mode="vat",
        required_args=("tax_code",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="motif_detection",
        description="Detect suspicious transaction motifs and cycles.",
        intent="vat_network_analysis",
        default_mode="vat",
        required_args=("tax_code",),
        optional_args=("max_hops",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="ring_scoring",
        description="Score closed VAT transaction rings.",
        intent="vat_network_analysis",
        default_mode="vat",
        required_args=("tax_code",),
        optional_args=("max_rings",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="ownership_analysis",
        description="Analyze ownership chains, UBOs and common controllers.",
        intent="osint_ownership",
        default_mode="fraud",
        required_args=("tax_code",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="temporal_delinquency_deep",
        description="Deep temporal model for delinquency forecasting.",
        intent="delinquency",
        default_mode="delinquency",
        required_args=("tax_code",),
        optional_args=("horizon_days",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="hetero_gnn_risk",
        description="Heterogeneous graph risk model for linked entities.",
        intent="vat_network_analysis",
        default_mode="fraud",
        required_args=("tax_code",),
        optional_args=("node_type",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="vae_anomaly_scan",
        description="VAE anomaly scan over invoice and transaction behavior.",
        intent="invoice_risk",
        default_mode="vat",
        required_args=("tax_code",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="causal_uplift_recommend",
        description="Recommend debt-collection action using causal uplift.",
        intent="delinquency",
        default_mode="delinquency",
        required_args=("tax_code",),
        optional_args=("objective",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="top_n_risky_companies",
        description="Return the top N riskiest companies.",
        intent="top_n_query",
        default_mode="fraud",
        required_args=("n",),
        optional_args=("sort_by", "mode"),
    ),
    CanonicalToolContract(
        name="company_name_search",
        description="Fuzzy search company records by name.",
        intent="company_name_lookup",
        default_mode="fraud",
        required_args=("name",),
        optional_args=("limit",),
    ),
    CanonicalToolContract(
        name="nlp_red_flag_scan",
        description="Scan invoice descriptions and text for NLP red flags.",
        intent="invoice_risk",
        default_mode="vat",
        required_args=("tax_code",),
        optional_args=("text",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="revenue_forecast",
        description="Forecast revenue and payment capacity.",
        intent="delinquency",
        default_mode="delinquency",
        required_args=("tax_code",),
        optional_args=("periods",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="entity_resolution_check",
        description="Detect duplicate or related entities.",
        intent="osint_ownership",
        default_mode="fraud",
        required_args=("tax_code",),
        optional_args=("query",),
        requires_tax_code=True,
    ),
    CanonicalToolContract(
        name="ocr_document_process",
        description="Extract structured data from invoice/document images.",
        intent="document_ocr",
        default_mode="vat",
        required_args=("file_path",),
        optional_args=("document_type", "language"),
    ),
    CanonicalToolContract(
        name="macro_forecast",
        description="Run macroeconomic tax-revenue scenario simulation.",
        intent="macro_forecast",
        default_mode="macro",
        required_args=("scenario",),
        optional_args=("action",),
    ),
)


CANONICAL_TOOL_NAMES: frozenset[str] = frozenset(
    contract.name for contract in CANONICAL_TOOL_CONTRACTS
)
TOOL_CONTRACT_BY_NAME: dict[str, CanonicalToolContract] = {
    contract.name: contract for contract in CANONICAL_TOOL_CONTRACTS
}
TOOL_TO_INTENT_MAP: dict[str, str] = {
    contract.name: contract.intent for contract in CANONICAL_TOOL_CONTRACTS
}

# Historical tool names from older scripts/model checkpoints.  The runtime may
# accept these and canonicalize them, but new datasets must not emit them.
DEPRECATED_TOOL_ALIASES: dict[str, str] = {
    "gnn_vat_fraud": "gnn_analysis",
    "run_hetero_gnn": "hetero_gnn_risk",
    "run_vae_anomaly": "vae_anomaly_scan",
    "predict_delinquency": "temporal_delinquency_deep",
    "causal_uplift_action": "causal_uplift_recommend",
    "query_legal_graphrag": "knowledge_search",
    "run_macro_simulation": "macro_forecast",
}

NON_TOOL_AGENT_ACTIONS: frozenset[str] = frozenset({
    "escalate_to_debate",
})

ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "company_name_search": {"query": "name"},
    "ocr_document_process": {"path": "file_path", "document": "file_path"},
}


def canonicalize_tool_name(name: str | None) -> str:
    """Return the canonical backend tool name for model/dataset output."""
    raw = (name or "").strip()
    return DEPRECATED_TOOL_ALIASES.get(raw, raw)


def is_canonical_tool(name: str | None) -> bool:
    """True when the name resolves to a real registered backend tool."""
    return canonicalize_tool_name(name) in CANONICAL_TOOL_NAMES


def validate_tool_call(payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any], str]:
    """
    Validate and canonicalize a model-produced tool call.

    Returns (ok, canonical_tool_name, arguments, reason).
    """
    if not isinstance(payload, dict):
        return False, "", {}, "tool_call_payload_not_object"
    raw_name = str(payload.get("name") or "").strip()
    if raw_name in NON_TOOL_AGENT_ACTIONS:
        return False, "", {}, "runtime_action_not_tool"
    name = canonicalize_tool_name(raw_name)
    if name not in CANONICAL_TOOL_NAMES:
        return False, name, {}, "unknown_tool"
    args = payload.get("arguments") or {}
    if not isinstance(args, dict):
        return False, name, {}, "arguments_not_object"
    arg_aliases = ARGUMENT_ALIASES.get(name, {})
    if arg_aliases:
        args = dict(args)
        for old_key, new_key in arg_aliases.items():
            if old_key in args and new_key not in args:
                args[new_key] = args[old_key]
    contract = TOOL_CONTRACT_BY_NAME[name]
    missing = [arg for arg in contract.required_args if arg not in args or args.get(arg) in ("", None)]
    if missing:
        return False, name, args, f"missing_required_args:{','.join(missing)}"
    return True, name, args, "ok"


def tool_prompt_lines() -> list[str]:
    """Compact tool list for SFT prompts and agentic LLM system prompts."""
    lines: list[str] = []
    for idx, contract in enumerate(CANONICAL_TOOL_CONTRACTS, 1):
        required = ", ".join(contract.required_args)
        optional = ", ".join(contract.optional_args)
        args = required
        if optional:
            args = f"{args}, {optional}" if args else optional
        lines.append(f"{idx}. {contract.name}({args}): {contract.description}")
    return lines
