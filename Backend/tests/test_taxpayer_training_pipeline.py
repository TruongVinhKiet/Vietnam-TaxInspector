import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.train_taxpayer_models import run_training


def test_taxpayer_training_pipeline_sandbox_outputs(tmp_path: Path) -> None:
    manifest = run_training(
        source="synthetic",
        sample_size=420,
        out_dir=tmp_path,
        seed=7,
        trials=1,
        max_retries=1,
        fast=True,
        task_filter={
            "taxpayer_invoice_risk",
            "taxpayer_expense_deductibility",
            "taxpayer_revenue_forecast",
            "taxpayer_next_best_action_uplift",
            "taxpayer_reconciliation_ranker",
        },
        register_models=True,
        write_drift_baseline=True,
    )

    assert manifest["summary"]["trained_model_count"] == 5
    assert manifest["summary"]["sandbox_count"] >= 1
    assert manifest["summary"]["prod_candidate_count"] == 0
    assert Path(manifest["registry_path"]).exists()
    assert Path(manifest["drift_baseline_path"]).exists()

    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["production_policy"]["prod_requires_database_source"] is True

    for model in manifest["models"].values():
        assert Path(model["artifact_path"]).exists()
        assert Path(model["quality_report_path"]).exists()
        assert Path(model["model_card_path"]).exists()
        assert model["metrics"]
        assert "overall_pass" in model["acceptance_gates"]
