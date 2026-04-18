"""
run_local_readiness_smoke.py

One-command local readiness smoke checker for TaxInspector.

Checks:
- Backend HTTP endpoints (monitoring + delinquency health)
- Critical model artifact presence
- Frontend page/script wiring and API_BASE sanity

Usage:
    python -m app.scripts.run_local_readiness_smoke --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import error, request


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _http_get(url: str, timeout_seconds: float) -> tuple[int, str]:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), body
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), body


def _http_get_json(url: str, timeout_seconds: float) -> tuple[int, Optional[dict[str, Any]], str]:
    status, body = _http_get(url, timeout_seconds)
    try:
        payload = json.loads(body)
    except Exception:
        return status, None, body
    if not isinstance(payload, dict):
        return status, None, body
    return status, payload, body


def _require_keys(payload: dict[str, Any], keys: Iterable[str]) -> list[str]:
    missing = []
    for key in keys:
        if key not in payload:
            missing.append(key)
    return missing


def _check_api(base_url: str, timeout_seconds: float) -> list[CheckResult]:
    base = base_url.rstrip("/")
    checks: list[CheckResult] = []

    docs_url = f"{base}/docs"
    try:
        status, body = _http_get(docs_url, timeout_seconds)
        ok = status == 200 and "swagger" in body.lower()
        detail = f"status={status}" if ok else f"status={status}, expected 200 with Swagger page"
        checks.append(CheckResult("backend_docs", ok, detail))
    except Exception as exc:
        checks.append(CheckResult("backend_docs", False, f"request failed: {exc}"))

    json_checks = [
        ("monitoring_health", "/api/monitoring/health", ("status",)),
        ("fraud_quality", "/api/monitoring/fraud_quality", ("status", "model_info")),
        ("graph_quality", "/api/monitoring/graph_quality", ("status", "gate_summary")),
        (
            "split_trigger_status",
            "/api/monitoring/split_trigger_status",
            ("ready", "schema_ready", "readiness_score", "track_status", "totals"),
        ),
        (
            "specialized_rollout_status",
            "/api/monitoring/specialized_rollout_status",
            ("available", "rollout_status", "summary", "artifacts"),
        ),
        (
            "delinquency_cache_health",
            "/api/delinquency/health/cache",
            ("status", "coverage", "freshness", "sources", "model_versions", "alerts"),
        ),
    ]

    for name, path, required_keys in json_checks:
        url = f"{base}{path}"
        try:
            status, payload, _raw = _http_get_json(url, timeout_seconds)
        except Exception as exc:
            checks.append(CheckResult(name, False, f"request failed: {exc}"))
            continue

        if status != 200:
            checks.append(CheckResult(name, False, f"status={status}, expected 200"))
            continue

        if payload is None:
            checks.append(CheckResult(name, False, "response is not valid JSON object"))
            continue

        missing = _require_keys(payload, required_keys)
        if missing:
            checks.append(CheckResult(name, False, f"missing keys: {', '.join(missing)}"))
            continue

        checks.append(CheckResult(name, True, "ok"))

    return checks


def _check_artifacts(models_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    required_files = [
        "xgboost_model.joblib",
        "isolation_forest.joblib",
        "delinquency_lgbm.joblib",
        "gat_model.pt",
        "audit_value_model.joblib",
        "vat_refund_model.joblib",
        "specialized_go_no_go_report.json",
        "specialized_pilot_report.json",
    ]

    for file_name in required_files:
        file_path = models_dir / file_name
        checks.append(
            CheckResult(
                name=f"artifact:{file_name}",
                ok=file_path.exists(),
                detail="present" if file_path.exists() else "missing",
            )
        )

    return checks


def _check_frontend(project_root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    frontend_dir = project_root / "Frontend"
    pages_dir = frontend_dir / "pages"
    js_dir = frontend_dir / "js"

    page_script_pairs = [
        ("login.html", "login.js"),
        ("dashboard.html", "dashboard.js"),
        ("fraud.html", "fraud.js"),
        ("delinquency.html", "delinquency.js"),
        ("delinquency-detail.html", "delinquency-detail.js"),
        ("graph.html", "graph.js"),
        ("profile.html", "profile.js"),
        ("reset-password.html", "reset-password.js"),
    ]

    for page_name, script_name in page_script_pairs:
        page_path = pages_dir / page_name
        script_path = js_dir / script_name

        if not page_path.exists():
            checks.append(CheckResult(f"frontend_page:{page_name}", False, "missing"))
            continue
        checks.append(CheckResult(f"frontend_page:{page_name}", True, "present"))

        if not script_path.exists():
            checks.append(CheckResult(f"frontend_script:{script_name}", False, "missing"))
            continue
        checks.append(CheckResult(f"frontend_script:{script_name}", True, "present"))

        page_text = page_path.read_text(encoding="utf-8", errors="replace")
        has_api_ref = "../js/api.js" in page_text
        has_script_ref = f"../js/{script_name}" in page_text

        checks.append(
            CheckResult(
                f"frontend_wiring:{page_name}",
                has_api_ref and has_script_ref,
                "ok" if (has_api_ref and has_script_ref) else "missing api.js or page script reference",
            )
        )

    api_js_path = js_dir / "api.js"
    if not api_js_path.exists():
        checks.append(CheckResult("frontend_api_base", False, "Frontend/js/api.js missing"))
        return checks

    api_text = api_js_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"const\s+API_BASE\s*=\s*\"([^\"]+)\"", api_text)
    if not match:
        checks.append(CheckResult("frontend_api_base", False, "API_BASE constant not found"))
        return checks

    api_base = match.group(1).strip()
    expected_hosts = ("http://localhost:8000/api", "http://127.0.0.1:8000/api")
    checks.append(
        CheckResult(
            "frontend_api_base",
            api_base in expected_hosts,
            f"value={api_base}",
        )
    )

    return checks


def _print_results(results: list[CheckResult]) -> None:
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        print(f"[{mark}] {result.name}: {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local readiness smoke checks")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend base URL (without trailing slash).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=8.0,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--skip-api-checks",
        action="store_true",
        help="Skip backend HTTP checks.",
    )
    parser.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="Skip artifact/frontend file checks.",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[2]
    project_root = backend_dir.parent
    models_dir = backend_dir / "data" / "models"

    all_results: list[CheckResult] = []

    if not args.skip_api_checks:
        all_results.extend(_check_api(base_url=args.base_url, timeout_seconds=args.timeout_seconds))

    if not args.skip_file_checks:
        all_results.extend(_check_artifacts(models_dir=models_dir))
        all_results.extend(_check_frontend(project_root=project_root))

    if not all_results:
        print("No checks selected.")
        return 2

    _print_results(all_results)

    failed = [result for result in all_results if not result.ok]
    passed = len(all_results) - len(failed)

    print("-" * 72)
    print(f"Summary: {passed}/{len(all_results)} passed, {len(failed)} failed")

    if failed:
        print("Readiness smoke check failed. Fix failed checks and run again.")
        return 1

    print("Readiness smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
