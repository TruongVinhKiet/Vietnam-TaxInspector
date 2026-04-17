import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.pipeline import TaxFraudPipeline
from ml_engine.feature_engineering import TaxFeatureEngineer


def _minimal_raw_frame():
    return pd.DataFrame(
        [
            {
                "tax_code": "0101234567",
                "year": 2024,
                "revenue": 1200000,
                "total_expenses": 930000,
                "net_profit": 270000,
                "vat_input": 89000,
                "vat_output": 120000,
            }
        ]
    )


def test_coerce_raw_input_adds_default_columns():
    pipeline = TaxFraudPipeline(model_dir=".")
    frame = _minimal_raw_frame()

    out = pipeline._coerce_raw_input_dataframe(frame, context="test")

    assert out["company_name"].iloc[0] == ""
    assert out["industry"].iloc[0] == "Unknown"
    assert out["industry_avg_profit_margin"].iloc[0] == pytest.approx(0.08)
    assert out["cost_of_goods"].iloc[0] == pytest.approx(930000 * 0.75)
    assert out["operating_expenses"].iloc[0] == pytest.approx(930000 * 0.25)


def test_coerce_raw_input_rejects_missing_required_columns():
    pipeline = TaxFraudPipeline(model_dir=".")
    frame = pd.DataFrame(
        [
            {
                "tax_code": "0101234567",
                "year": 2024,
                "revenue": 1200000,
                "total_expenses": 930000,
                "net_profit": 270000,
                "vat_input": 89000,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        pipeline._coerce_raw_input_dataframe(frame, context="test")


def test_validate_feature_frame_rejects_non_finite_values():
    pipeline = TaxFraudPipeline(model_dir=".")

    frame = pd.DataFrame([{name: 0.1 for name in TaxFeatureEngineer.FEATURE_COLS}])
    frame.loc[0, "f3_vat_structure"] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        pipeline._validate_feature_frame(frame, context="test")


def test_coerce_raw_input_normalizes_tax_code_strings():
    pipeline = TaxFraudPipeline(model_dir=".")
    frame = pd.DataFrame(
        [
            {
                "tax_code": 200004979.0,
                "year": 2024,
                "revenue": 1200000,
                "total_expenses": 930000,
                "net_profit": 270000,
                "vat_input": 89000,
                "vat_output": 120000,
            },
            {
                "tax_code": "0200004979 ",
                "year": 2023,
                "revenue": 1100000,
                "total_expenses": 880000,
                "net_profit": 220000,
                "vat_input": 81000,
                "vat_output": 110000,
            },
        ]
    )

    out = pipeline._coerce_raw_input_dataframe(frame, context="test")

    assert out["tax_code"].tolist() == ["200004979", "0200004979"]
