from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_UI = REPO_ROOT / "Frontend" / "js" / "agent_ui.js"


def test_agent_ui_stream_state_machine_contract_is_present() -> None:
    source = AGENT_UI.read_text(encoding="utf-8")
    for token in ("queued", "streaming", "finalized", "partial_error", "error", "cancelled"):
        assert token in source
    assert "buildStreamRecoveryCard" in source
    assert "agent-retry-btn" in source
    assert "agent-regenerate-btn" in source
    assert "stream-cancel-btn" in source


def test_agent_ui_async_polling_cancel_and_workspace_contract_is_present() -> None:
    source = AGENT_UI.read_text(encoding="utf-8")
    assert "pollAsyncFileJob" in source
    assert "cancelAsyncFileJob" in source
    assert "async-cancel-btn" in source
    assert "renderModeWorkspace" in source
    assert "mode_workspace" in source

