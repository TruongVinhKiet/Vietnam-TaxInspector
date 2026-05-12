from __future__ import annotations

import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_mode_contracts import (  # noqa: E402
    AGENT_RESPONSE_SCHEMA_VERSION,
    MODE_CONTRACT_VERSION,
    AgentModeContractRegistry,
)
from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator  # noqa: E402
from ml_engine.tax_agent_tools import (  # noqa: E402
    ToolCallRequest,
    ToolCategory,
    ToolExecutor,
    ToolSpec,
    ToolStatus,
    ToolRegistry,
)
from ml_engine.tax_agent_turns import AgentTurnRepository  # noqa: E402


def test_mode_contract_registry_is_source_of_truth_for_core_modes() -> None:
    fraud = AgentModeContractRegistry.get("fraud")
    macro = AgentModeContractRegistry.get("macro")
    vat = AgentModeContractRegistry.get("vat")

    assert fraud.workspace_panel == "fraud"
    assert "risk_gauge" in fraud.required_visualization_keys
    assert macro.workspace_panel == "simulation"
    assert "vat_graph_csv" in vat.canonical_upload_handlers
    assert AgentModeContractRegistry.metadata("macro")["schema_version"] == AGENT_RESPONSE_SCHEMA_VERSION
    assert AgentModeContractRegistry.metadata("macro")["mode_contract_version"] == MODE_CONTRACT_VERSION


def test_prior_answer_fact_extraction_captures_summary_recommendation_and_ranked_subject() -> None:
    orch = TaxAgentOrchestrator()
    facts = orch._extract_prior_answer_facts(
        mode="fraud",
        intent="batch_analysis",
        answer="",
        summary="Batch co 2 doanh nghiep rui ro cao.",
        recommendations=["Uu tien thanh tra doanh nghiep diem cao."],
        analysis_blocks=[
            {
                "type": "fraud_analysis",
                "title": "Fraud batch",
                "summary": "Nhieu dau hieu bat thuong.",
                "metrics": {
                    "top_records": [
                        {"tax_code": "0100000000", "risk_score": 91.2},
                    ],
                },
            }
        ],
        tool_results={
            "_session_upload_row": {
                "tax_code": "0100000000",
                "source_filename": "risk.csv",
            }
        },
    )

    fact_types = {fact["fact_type"] for fact in facts}
    subjects = {fact.get("subject_key") for fact in facts}
    assert {"summary", "recommendation", "fraud_analysis", "ranked_subject", "session_upload_row"}.issubset(fact_types)
    assert "0100000000" in subjects


def test_tool_executor_returns_timeout_for_long_single_tool() -> None:
    def slow_tool(execution_context=None):
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            if execution_context is not None:
                execution_context.raise_if_cancelled()
            time.sleep(0.01)
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="slow_tool",
        description="slow test tool",
        category=ToolCategory.ANALYTICS,
        input_schema={},
        output_schema={},
        handler=slow_tool,
        timeout_seconds=0.02,
        max_retries=0,
        requires_db=False,
    ))
    executor = ToolExecutor(registry, max_workers=1)
    try:
        result = executor.execute_parallel([
            ToolCallRequest(tool_name="slow_tool", inputs={}, timeout_override=0.02)
        ])[0]
    finally:
        executor._executor.shutdown(wait=False, cancel_futures=True)

    assert result.status == ToolStatus.TIMEOUT
    assert result.metadata.get("request_id")


def test_turn_repository_allocates_pairs_without_duplicate_indices_sqlite() -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE agent_sessions (session_id TEXT PRIMARY KEY)"))
        conn.execute(text(
            """
            CREATE TABLE agent_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                message_text TEXT NOT NULL,
                normalized_intent TEXT,
                confidence FLOAT,
                citations_json TEXT
            )
            """
        ))
        conn.execute(text("INSERT INTO agent_sessions(session_id) VALUES ('s1')"))

    db = SessionLocal()
    try:
        repo = AgentTurnRepository(db)
        first = repo.allocate_turn_pair(session_id="s1", user_message="hello")
        second = repo.allocate_turn_pair(session_id="s1", user_message="next")
        db.commit()

        rows = db.execute(text(
            "SELECT turn_index, role FROM agent_turns WHERE session_id='s1' ORDER BY turn_index"
        )).fetchall()
    finally:
        db.close()

    assert (first.user_turn_index, first.assistant_turn_index) == (1, 2)
    assert (second.user_turn_index, second.assistant_turn_index) == (3, 4)
    assert [row[0] for row in rows] == [1, 2, 3, 4]
    assert len({row[0] for row in rows}) == 4
