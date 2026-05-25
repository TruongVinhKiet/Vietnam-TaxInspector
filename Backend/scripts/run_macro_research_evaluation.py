"""
Run thesis-oriented evaluation for the Macro-Fiscal Digital Twin Research Lab.

Outputs:
    Backend/reports/macro_research_lab/macro_research_evaluation.json
    Backend/reports/macro_research_lab/macro_research_evaluation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.macro_research_lab import (  # noqa: E402
    build_data_quality_report,
    build_macro_research_evaluation,
    build_research_state,
    run_causal_merger_effect,
    run_forecast_research,
    run_shock_propagation,
)


REPORT_DIR = BACKEND_DIR / "reports" / "macro_research_lab"


def _write_markdown(payload: dict, path: Path) -> None:
    quality = payload["data_quality"]
    eval_report = payload["evaluation"]
    causal = payload["causal_probe"]
    lines = [
        "# Macro-Fiscal Digital Twin Research Evaluation",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Boundary version: `{payload['boundary_version']}`",
        f"- Province coverage: `{quality['province_count']}/{quality['expected_provinces']}`",
        f"- Historical events: `{quality['historical_event_count']}`",
        f"- Data fingerprint: `{quality['data_fingerprint']}`",
        "",
        "## Forecast Backtest Proxy",
        "",
        f"- Sample provinces: `{eval_report['forecast_backtest']['sample_provinces']}`",
        f"- MAE proxy mean: `{eval_report['forecast_backtest']['mae_proxy_mean']}`",
        f"- Mean interval width: `{eval_report['forecast_backtest']['mean_interval_width_pct']}%`",
        "",
        "## Causal Merger Probe",
        "",
        f"- Province: `{causal['province_name']}`",
        f"- DiD proxy: `{causal['metrics']['difference_in_differences_pct']}%`",
        f"- Placebo p-value proxy: `{causal['metrics']['p_value_proxy']}`",
        "",
        "## Required Ablations",
        "",
    ]
    for item in eval_report["ablation_plan"]:
        lines.append(f"- `{item['config']}`: required={item['required']}")
    lines.extend([
        "",
        "## Acceptance Targets",
        "",
    ])
    for key, value in eval_report["acceptance_targets"].items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(boundary_version: str) -> dict:
    state = build_research_state(boundary_version=boundary_version)
    quality = build_data_quality_report(boundary_version=boundary_version)
    forecast_probe = run_forecast_research({
        "boundary_version": boundary_version,
        "province_code": "VN34-HCM" if boundary_version == "vn_34_2025" else "79",
        "horizon_quarters": 12,
        "scenario_params": {"gdp_delta_pct": -2.0, "compliance_delta": -0.02, "fdi_delta_pct": -4.0},
    })
    shock_probe = run_shock_propagation({
        "boundary_version": boundary_version,
        "source_province_code": forecast_probe["province_code"],
        "shock_strength_pct": -3.0,
        "horizon_quarters": 8,
    })
    causal_probe = run_causal_merger_effect({"province_code": "VN34-CM", "boundary_version": "vn_34_2025"})
    evaluation = build_macro_research_evaluation(boundary_version=boundary_version)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "boundary_version": boundary_version,
        "research_state": state,
        "data_quality": quality,
        "forecast_probe": forecast_probe,
        "shock_probe": shock_probe,
        "causal_probe": causal_probe,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-version", default="vn_34_2025", choices=["vn_34_2025", "vn_63_legacy"])
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = run(args.boundary_version)
    json_path = out_dir / "macro_research_evaluation.json"
    md_path = out_dir / "macro_research_evaluation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"[OK] Wrote {json_path}")
    print(f"[OK] Wrote {md_path}")


if __name__ == "__main__":
    main()
