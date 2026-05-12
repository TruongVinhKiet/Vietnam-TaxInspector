from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_async_file_job_processing_stale_timeout_becomes_terminal_error(monkeypatch) -> None:
    import app.routers.tax_agent as tax_agent_router

    job_id = "job-stale-contract"
    monkeypatch.setattr(tax_agent_router, "_ASYNC_FILE_JOB_STALE_SECONDS", 1)
    monkeypatch.setattr(tax_agent_router, "_load_async_job", lambda _job_id: None)
    monkeypatch.setattr(tax_agent_router, "_persist_async_job", lambda *_args, **_kwargs: None)

    with tax_agent_router._async_file_jobs_lock:
        tax_agent_router._async_file_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "phase": "analyze_attachment",
            "progress": 55.0,
            "created_at": time.time() - 10,
            "updated_at": time.time() - 10,
        }

    try:
        job = tax_agent_router._get_async_job(job_id)
    finally:
        with tax_agent_router._async_file_jobs_lock:
            tax_agent_router._async_file_jobs.pop(job_id, None)
            tax_agent_router._async_file_job_cancel_flags.pop(job_id, None)

    assert job is not None
    assert job["status"] == "error"
    assert job["phase"] == "stale_timeout"
    assert job["progress"] == 100.0
    assert job["error_detail"]["phase"] == "stale_timeout"

