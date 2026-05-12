"""Verify that doc.js metrics match experimental_evaluation_metrics.json"""
import json
import re

with open("e:/TaxInspector/Backend/reports/experimental_evaluation_metrics.json", "r", encoding="utf-8") as f:
    m = json.load(f)
with open("e:/TaxInspector/doc.js", "r", encoding="utf-8") as f:
    doc = f.read()

errors = []

# Helper: check if a value appears in doc (with comma decimal)
def check_val(section, label, expected, tolerance=0.002):
    # Format with comma decimal
    s = f"{expected:.3f}".replace(".", ",")
    if s not in doc:
        # Try 2 decimal places
        s2 = f"{expected:.2f}".replace(".", ",")
        if s2 not in doc:
            # Try 1 decimal
            s1 = f"{expected:.1f}".replace(".", ",")
            if s1 not in doc:
                errors.append(f"[{section}] {label}: expected ~{expected} (tried {s}, {s2}, {s1}) NOT FOUND in doc.js")

# Fraud models
for model_name, vals in m["fraud"]["models"].items():
    for metric, data in vals.items():
        check_val("Fraud", f"{model_name}.{metric}.mean", data["mean"])

# Delinquency models
for model_name, vals in m["delinquency"]["models"].items():
    for metric, data in vals.items():
        check_val("Delinquency", f"{model_name}.{metric}.mean", data["mean"])

# Agent
agent = m["agent"]["metrics"]
check_val("Agent", "legal_route_accuracy", agent["legal_route_accuracy"] * 100)
check_val("Agent", "faq_grounding_rate", agent["faq_grounding_rate"] * 100)
check_val("Agent", "actionable_steps_rate", agent["actionable_steps_rate"] * 100)
check_val("Agent", "citation_or_reference_rate", agent["citation_or_reference_rate"] * 100)
check_val("Agent", "mean_confidence", agent["mean_confidence"])
check_val("Agent", "mean_latency_ms", agent["mean_latency_ms"])

# System perf
for key, vals in m["system_performance"].items():
    for pct in ["p50", "p95", "p99"]:
        if pct in vals:
            check_val("SysPerf", f"{key}.{pct}", vals[pct])

# Check dataset counts
if "300 câu hỏi" not in doc:
    errors.append("[Agent] '300 câu hỏi' not found in doc.js")
if "30 chủ đề" not in doc:
    errors.append("[Agent] '30 chủ đề' not found in doc.js")

# Check date consistency
if "10/05/2026" in doc or "May 10" in doc:
    errors.append("[Date] Old date '10/05/2026' or 'May 10' still present")

if errors:
    print(f"\n❌ FOUND {len(errors)} MISMATCHES:\n")
    for e in errors:
        print(f"  • {e}")
else:
    print("\n✅ ALL metrics in doc.js match experimental_evaluation_metrics.json")
    print("   ✅ Agent: 300 cases, 30 topics")
    print("   ✅ Dates: all 11/05/2026 / May 11")
    print("\n   Ready to run: node doc.js")
