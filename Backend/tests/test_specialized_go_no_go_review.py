import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.scripts import run_specialized_go_no_go_review as go_no_go


def _quality_pass_payload() -> dict:
    return {
        "acceptance_gates": {
            "overall_pass": True,
        }
    }


def _pilot_payload(*, vat_f1_delta: float = -0.03947) -> dict:
    return {
        "tracks": {
            "audit_value": {
                "samples_evaluated": 200,
                "model": {"accuracy": 0.605},
                "heuristic": {"accuracy": 0.625},
                "delta_model_minus_heuristic": {
                    "f1_delta": 0.242407,
                },
            },
            "vat_refund": {
                "samples_evaluated": 200,
                "model": {"accuracy": 0.935},
                "heuristic": {"accuracy": 0.98},
                "delta_model_minus_heuristic": {
                    "f1_delta": vat_f1_delta,
                },
            },
        }
    }


def _split_ready_payload() -> dict:
    return {
        "schema_ready": True,
        "ready": True,
        "readiness_score": 90.0,
        "critical_tracks": ["audit_value", "vat_refund"],
        "totals": {"enabled_rules": 4, "passed_rules": 4},
    }


def test_build_go_no_go_report_returns_conditional_go_without_stability():
    report = go_no_go.build_go_no_go_report(
        audit_quality_payload=_quality_pass_payload(),
        vat_quality_payload=_quality_pass_payload(),
        pilot_payload=_pilot_payload(),
        split_payload=_split_ready_payload(),
        history_rows=[],
        min_pilot_samples=200,
        audit_min_f1_delta=0.05,
        vat_min_f1_delta=-0.05,
        max_accuracy_drop=0.05,
        min_consecutive_hard_pass_runs=2,
    )

    assert report["summary"]["hard_gates_pass"] is True
    assert report["summary"]["split_gate_pass"] is True
    assert report["summary"]["stability_gate_pass"] is False
    assert report["decision"]["status"] == "conditional_go_continue_integrated_first"
    assert report["decision"]["go_live_phase_d"] is False


def test_build_go_no_go_report_returns_no_go_when_vat_delta_fails():
    report = go_no_go.build_go_no_go_report(
        audit_quality_payload=_quality_pass_payload(),
        vat_quality_payload=_quality_pass_payload(),
        pilot_payload=_pilot_payload(vat_f1_delta=-0.20),
        split_payload=_split_ready_payload(),
        history_rows=[{"hard_gates_pass": True}],
        min_pilot_samples=200,
        audit_min_f1_delta=0.05,
        vat_min_f1_delta=-0.05,
        max_accuracy_drop=0.05,
        min_consecutive_hard_pass_runs=1,
    )

    assert report["summary"]["hard_gates_pass"] is False
    assert report["decision"]["status"] == "no_go_tune_models_or_data"
    assert report["decision"]["go_live_phase_d"] is False


def test_build_go_no_go_report_returns_go_when_all_gates_pass():
    report = go_no_go.build_go_no_go_report(
        audit_quality_payload=_quality_pass_payload(),
        vat_quality_payload=_quality_pass_payload(),
        pilot_payload=_pilot_payload(),
        split_payload=_split_ready_payload(),
        history_rows=[{"hard_gates_pass": True}],
        min_pilot_samples=200,
        audit_min_f1_delta=0.05,
        vat_min_f1_delta=-0.05,
        max_accuracy_drop=0.05,
        min_consecutive_hard_pass_runs=2,
    )

    assert report["summary"]["hard_gates_pass"] is True
    assert report["summary"]["split_gate_pass"] is True
    assert report["summary"]["stability_gate_pass"] is True
    assert report["decision"]["status"] == "go_phase_d_candidate"
    assert report["decision"]["go_live_phase_d"] is True
