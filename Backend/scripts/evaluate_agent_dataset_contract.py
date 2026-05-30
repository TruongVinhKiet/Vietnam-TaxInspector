"""
Evaluate a TaxInspector agent SFT dataset for release readiness.

This is a lightweight contract/eval report, not a full model-quality benchmark.
It verifies that training records obey the backend tool contract before LoRA or
DPO training starts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "agent_ultimate_dataset_v4.jsonl"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_engine.tax_agent_tool_contracts import (  # noqa: E402
    CANONICAL_TOOL_NAMES,
    DEPRECATED_TOOL_ALIASES,
    NON_TOOL_AGENT_ACTIONS,
    validate_tool_call,
)


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"_decode_error": str(exc)}


def extract_tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in record.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        match = TOOL_CALL_RE.search(content)
        if not match:
            continue
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError as exc:
            calls.append({"_decode_error": str(exc), "_raw": match.group(1)[:200]})
    return calls


def evaluate(path: Path) -> dict[str, Any]:
    rows = 0
    decode_errors: list[dict[str, Any]] = []
    invalid_tool_calls: list[dict[str, Any]] = []
    splits: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    deprecated_hits: Counter[str] = Counter()

    for line_no, record in iter_records(path):
        rows += 1
        if "_decode_error" in record:
            decode_errors.append({"line": line_no, "error": record["_decode_error"]})
            continue

        metadata = record.get("metadata") or {}
        splits[str(metadata.get("split") or "missing")] += 1
        kinds[str(metadata.get("kind") or "unknown")] += 1

        for call in extract_tool_calls(record):
            if "_decode_error" in call:
                invalid_tool_calls.append({"line": line_no, "reason": call["_decode_error"]})
                continue
            raw_name = str(call.get("name") or "")
            if raw_name in DEPRECATED_TOOL_ALIASES or raw_name in NON_TOOL_AGENT_ACTIONS:
                deprecated_hits[raw_name] += 1
            ok, canonical_name, _args, reason = validate_tool_call(call)
            if ok:
                tools[canonical_name] += 1
            else:
                invalid_tool_calls.append({
                    "line": line_no,
                    "tool": raw_name,
                    "canonical_tool": canonical_name,
                    "reason": reason,
                })

    missing_tools = sorted(CANONICAL_TOOL_NAMES - set(tools))
    report = {
        "dataset": str(path),
        "total_records": rows,
        "decode_error_count": len(decode_errors),
        "invalid_tool_call_count": len(invalid_tool_calls),
        "deprecated_tool_call_count": sum(deprecated_hits.values()),
        "split_distribution": dict(sorted(splits.items())),
        "kind_distribution": dict(sorted(kinds.items())),
        "tool_distribution": dict(sorted(tools.items())),
        "missing_canonical_tools": missing_tools,
        "deprecated_tool_hits": dict(sorted(deprecated_hits.items())),
        "sample_errors": (decode_errors + invalid_tool_calls)[:20],
        "status": "pass" if not decode_errors and not invalid_tool_calls and not deprecated_hits else "fail",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TaxInspector agent dataset contracts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.input)
    output = args.output or args.input.with_suffix(args.input.suffix + ".eval_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report": str(output)}, ensure_ascii=False, indent=2))
    if args.fail_on_invalid and report["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
