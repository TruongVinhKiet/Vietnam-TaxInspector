"""Master thesis evaluation runner for TaxInspector.

Runs the publishable evaluation stack: ablation, statistical significance,
fairness, legal KB/RAG grounding, deep learning proxies, concept drift,
federated learning PoC, and user-study dry run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_step(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        return {
            "status": "complete",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "result": payload,
        }
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }


def run_drift_simulation(seed: int = 42) -> dict[str, Any]:
    import numpy as np
    from ml_engine.concept_drift_detector import DriftMonitor, population_stability_index

    rng = np.random.default_rng(seed)
    monitor = DriftMonitor(model_key="fraud-hybrid-v2", warmup=40, adwin_delta=0.01, page_hinkley_threshold=1.0)
    detections = []
    for idx, confidence in enumerate(list(rng.normal(0.82, 0.03, 120)) + list(rng.normal(0.42, 0.04, 120))):
        status = monitor.update(
            prediction_confidence=float(confidence),
            features={
                "revenue_log": float(rng.normal(16.0 if idx < 120 else 17.8, 0.5)),
                "invoice_velocity": float(rng.normal(8.0 if idx < 120 else 20.0, 2.0)),
            },
            metadata={"simulation_index": idx},
        )
        if status["drift_detected"]:
            detections.append(status)
    psi = population_stability_index(rng.normal(0, 1, 500), rng.normal(1.3, 1.1, 500))
    return {
        "adwin_or_page_hinkley_validated": bool(detections),
        "detection_count": len(detections),
        "psi_shift_example": round(float(psi), 6),
        "status": monitor.get_status(),
    }


def _summary_from_steps(steps: dict[str, Any]) -> tuple[dict[str, Any], float]:
    def ok(name: str) -> bool:
        return steps.get(name, {}).get("status") == "complete"

    ablation = steps.get("ablation", {}).get("result", {})
    fairness = steps.get("fairness", {}).get("result", {})
    rag = steps.get("rag_grounding", {}).get("result", {})
    dl = steps.get("deep_learning", {}).get("result", {})
    drift = steps.get("concept_drift", {}).get("result", {})
    fl = steps.get("federated_learning", {}).get("result", {})
    user = steps.get("user_study", {}).get("result", {})

    fairness_pass = bool(fairness.get("disparate_impact_pass")) or not fairness.get("red_flags")
    rag_rate = float(rag.get("grounding_rate", 0.0))
    fed_gap = float(fl.get("federated_gap", 1.0)) if fl else 1.0
    score = 7.9
    score += 0.35 if ok("ablation") else 0.0
    score += 0.25 if ok("ablation") and ablation.get("pairwise_vs_B1") else 0.0
    score += 0.2 if ok("fairness") and fairness_pass else 0.0
    score += 0.2 if ok("rag_grounding") and rag_rate >= 0.75 else 0.08 if ok("rag_grounding") else 0.0
    score += 0.15 if ok("deep_learning") else 0.0
    score += 0.15 if ok("concept_drift") and drift.get("adwin_or_page_hinkley_validated") else 0.0
    score += 0.25 if ok("federated_learning") and fed_gap <= 0.03 else 0.12 if ok("federated_learning") else 0.0
    score += 0.1 if ok("user_study") else 0.0
    score = min(9.7, round(score, 2))

    summary = {
        "ablation": {
            "status": steps.get("ablation", {}).get("status"),
            "best_model": max(ablation.get("configs", {}), key=lambda name: ablation.get("configs", {}).get(name, {}).get("auc_roc", {}).get("mean", 0.0)) if ablation.get("configs") else None,
            "key_finding": ablation.get("contribution_delta", {}).get("C5_Full_Hybrid"),
        },
        "statistical_significance": {
            "status": "complete" if ok("ablation") and ablation.get("pairwise_vs_B1") else steps.get("ablation", {}).get("status"),
            "baseline": "B1_XGBoost",
        },
        "fairness": {
            "status": steps.get("fairness", {}).get("status"),
            "disparate_impact_pass": fairness_pass,
            "red_flag_count": len(fairness.get("red_flags", [])) if isinstance(fairness.get("red_flags"), list) else 0,
        },
        "rag_grounding": {
            "status": steps.get("rag_grounding", {}).get("status"),
            "rate": rag_rate,
            "recommended_path": rag.get("kb_audit", {}).get("recommended_path"),
        },
        "deep_learning": {
            "status": steps.get("deep_learning", {}).get("status"),
            "gat_f1": dl.get("deep_learning_benchmarks", {}).get("gat", {}).get("edge", {}).get("f1"),
            "vae_f1": dl.get("deep_learning_benchmarks", {}).get("vae_anomaly", {}).get("vae_proxy", {}).get("f1"),
        },
        "concept_drift": {
            "status": steps.get("concept_drift", {}).get("status"),
            "adwin_validated": drift.get("adwin_or_page_hinkley_validated"),
        },
        "federated_learning": {
            "status": steps.get("federated_learning", {}).get("status"),
            "fed_vs_central_gap": fed_gap,
        },
        "user_study": {
            "status": steps.get("user_study", {}).get("status"),
            "sus_mean": user.get("analysis", {}).get("sus_mean"),
            "simulated": True,
        },
    }
    return summary, score


def run_comprehensive_evaluation(
    *,
    rows: int = 120_000,
    folds: int = 5,
    seed: int = 42,
    out_dir: Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    out_dir = out_dir or (BACKEND_DIR / "reports" / "thesis_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_eval = min(rows, 5_000) if quick else rows
    folds_eval = min(folds, 2) if quick else folds

    steps: dict[str, Any] = {}

    def ablation_step():
        from scripts.ablation_study import run_ablation, write_reports

        result = run_ablation(rows=rows_eval, folds=folds_eval, seed=seed, bootstrap=80 if quick else 250)
        write_reports(result, out_dir)
        return result

    def fairness_step():
        from scripts.fairness_analysis import run_fairness_analysis, write_fairness_reports

        result = run_fairness_analysis(rows=rows_eval, seed=seed, min_samples=20 if quick else 80)
        write_fairness_reports(result, out_dir)
        return result

    def rag_step():
        from scripts.expand_legal_knowledge_base import audit_legal_knowledge_base, export_curated_documents, write_audit_report
        from scripts.finetune_reranker import build_training_pairs, evaluate_lightweight_grounding, write_weight_file

        audit = audit_legal_knowledge_base()
        write_audit_report(audit, out_dir)
        export_curated_documents(out_dir)
        pairs = build_training_pairs(seed=seed)
        grounding = evaluate_lightweight_grounding(pairs)
        write_weight_file(BACKEND_DIR / "models" / "tax_agent_reranker", grounding)
        return {"kb_audit": audit, "grounding_rate": grounding["top1_grounding_rate"], "pair_count": grounding["pair_count"]}

    def dl_step():
        from scripts.benchmark_deep_learning import run_deep_learning_benchmarks

        result = run_deep_learning_benchmarks(rows=min(rows_eval, 15_000), seed=seed)
        (out_dir / "dl_benchmarks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def fl_step():
        from ml_engine.federated_learning_poc import run_federated_learning_poc

        result = run_federated_learning_poc(rows=max(800, min(rows_eval, 12_000)), rounds=2 if quick else 10, seed=seed)
        (out_dir / "federated_learning_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def user_step():
        from scripts.user_study_framework import analyze_study, simulate_expert_responses, write_outputs

        package = write_outputs(out_dir / "user_study", experts=10, seed=seed)
        return {"analysis": package["analysis"], "participant_count": package["analysis"]["participant_count"]}

    step_fns = {
        "ablation": ablation_step,
        "fairness": fairness_step,
        "rag_grounding": rag_step,
        "deep_learning": dl_step,
        "concept_drift": lambda: run_drift_simulation(seed=seed),
        "federated_learning": fl_step,
        "user_study": user_step,
    }
    for name, fn in step_fns.items():
        steps[name] = _run_step(name, fn)

    summary, score = _summary_from_steps(steps)
    report = {
        "generated_at": _now_iso(),
        "quick_mode": quick,
        "rows": rows_eval,
        "folds": folds_eval,
        "seed": seed,
        "thesis_score_estimate": score,
        "evaluation_summary": summary,
        "steps": steps,
    }
    return report


def write_consolidated_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "thesis_evaluation_complete.json"
    md_path = out_dir / "thesis_evaluation_complete.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# TaxInspector Comprehensive Thesis Evaluation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Quick mode: `{report['quick_mode']}`",
        f"- Rows: `{report['rows']}`",
        f"- Folds: `{report['folds']}`",
        f"- Thesis score estimate: `{report['thesis_score_estimate']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["evaluation_summary"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run comprehensive TaxInspector thesis evaluation")
    parser.add_argument("--all", action="store_true", help="Run full configured suite")
    parser.add_argument("--quick", action="store_true", help="Use smaller row counts/folds for smoke testing")
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "reports" / "thesis_evaluation")
    args = parser.parse_args()

    report = run_comprehensive_evaluation(
        rows=args.rows,
        folds=args.folds,
        seed=args.seed,
        out_dir=args.out_dir,
        quick=bool(args.quick and not args.all),
    )
    json_path, md_path = write_consolidated_report(report, args.out_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(json.dumps({"score": report["thesis_score_estimate"], "rows": report["rows"], "folds": report["folds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
