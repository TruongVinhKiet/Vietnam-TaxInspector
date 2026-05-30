from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_agentic_llm import AgenticLLM  # noqa: E402
from ml_engine.tax_agent_tool_contracts import (  # noqa: E402
    CANONICAL_TOOL_NAMES,
    NON_TOOL_AGENT_ACTIONS,
    validate_tool_call,
)
from ml_engine.tax_agent_tools import build_default_registry  # noqa: E402
from scripts.generate_mega_agent_dataset_v4 import generate  # noqa: E402


def test_canonical_tool_contract_matches_default_registry() -> None:
    registry_names = set(build_default_registry().list_tool_names())

    assert registry_names == CANONICAL_TOOL_NAMES
    assert "gnn_vat_fraud" not in registry_names
    assert "escalate_to_debate" in NON_TOOL_AGENT_ACTIONS


def test_agentic_llm_parser_canonicalizes_legacy_tool_alias() -> None:
    decision = AgenticLLM()._parse_output(
        '<thought>Need graph analysis.</thought>\n'
        '<tool_call>{"name":"gnn_vat_fraud","arguments":{"tax_code":"0101234567"}}</tool_call>'
    )

    assert decision is not None
    assert decision.tool_name == "gnn_analysis"
    assert decision.mapped_intent == "vat_network_analysis"


def test_agentic_llm_parser_rejects_runtime_action_as_tool() -> None:
    decision = AgenticLLM()._parse_output(
        '<thought>Escalate.</thought>\n'
        '<tool_call>{"name":"escalate_to_debate","arguments":{"tax_code":"0101234567"}}</tool_call>'
    )

    assert decision is None


def test_v5_dataset_sample_has_only_canonical_tools_and_stable_splits(tmp_path) -> None:
    out = tmp_path / "agent_sample.jsonl"
    summary = generate(
        total_simple=40,
        total_legal=10,
        total_smalltalk=5,
        total_clarification=5,
        seed=11,
        output_path=out,
        write_latest_alias=False,
    )
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    emitted_tools: set[str] = set()

    for record in records:
        metadata = record.get("metadata") or {}
        assert metadata.get("split") in {"train", "dev", "test"}
        assert metadata.get("split_group")
        for message in record.get("messages", []):
            if message.get("role") != "assistant":
                continue
            match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", message.get("content", ""), re.DOTALL)
            if not match:
                continue
            payload = json.loads(match.group(1))
            ok, tool_name, _args, reason = validate_tool_call(payload)
            assert ok, reason
            emitted_tools.add(tool_name)

    assert emitted_tools <= CANONICAL_TOOL_NAMES
    assert "gnn_vat_fraud" not in emitted_tools
    assert "escalate_to_debate" not in emitted_tools
    assert Path(summary["manifest"]).exists()


def test_telemetry_js_does_not_redeclare_global_api_base() -> None:
    telemetry_js = (REPO_ROOT / "Frontend" / "js" / "telemetry.js").read_text(encoding="utf-8")

    assert not re.search(r"^\s*const\s+API_BASE\s*=", telemetry_js, re.MULTILINE)
    assert "apiUrl('/tax-agent/dpo/dry-run')" in telemetry_js
