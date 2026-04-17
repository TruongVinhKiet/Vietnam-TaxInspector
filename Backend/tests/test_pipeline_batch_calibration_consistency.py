import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.feature_engineering import TaxFeatureEngineer
from ml_engine.pipeline import TaxFraudPipeline


class _StubFeatureEngineer:
    FEATURE_COLS = TaxFeatureEngineer.FEATURE_COLS

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for idx, col in enumerate(self.FEATURE_COLS):
            if col not in out.columns:
                out[col] = 0.1 + idx * 0.01
        return out

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.FEATURE_COLS].to_numpy(dtype=float)

    def generate_red_flags(self, row):
        return []


class _StubXgbModel:
    def predict_proba(self, X):
        probs = np.full((len(X), 2), 0.0, dtype=float)
        probs[:, 1] = 0.8
        probs[:, 0] = 0.2
        return probs


class _StubIsoModel:
    def decision_function(self, X):
        return np.zeros(len(X), dtype=float)


def test_batch_year_trend_uses_calibrated_probabilities():
    pipeline = TaxFraudPipeline(model_dir=".")
    pipeline._loaded = True
    pipeline.feature_engineer = _StubFeatureEngineer()
    pipeline.xgboost_model = _StubXgbModel()
    pipeline.isolation_forest = _StubIsoModel()
    pipeline._calibrate_fraud_probs = lambda probs: np.full(len(probs), 0.2, dtype=float)
    pipeline.get_serving_metadata = lambda: {
        "model_version": "fraud-test",
        "calibration_method": "test",
    }

    frame = pd.DataFrame(
        [
            {
                "tax_code": "0100000001",
                "year": 2023,
                "revenue": 1_000_000,
                "total_expenses": 800_000,
                "net_profit": 200_000,
                "vat_input": 70_000,
                "vat_output": 100_000,
            },
            {
                "tax_code": "0100000001",
                "year": 2024,
                "revenue": 1_200_000,
                "total_expenses": 900_000,
                "net_profit": 300_000,
                "vat_input": 85_000,
                "vat_output": 120_000,
            },
            {
                "tax_code": "0100000002",
                "year": 2023,
                "revenue": 900_000,
                "total_expenses": 700_000,
                "net_profit": 200_000,
                "vat_input": 60_000,
                "vat_output": 90_000,
            },
            {
                "tax_code": "0100000002",
                "year": 2024,
                "revenue": 950_000,
                "total_expenses": 760_000,
                "net_profit": 190_000,
                "vat_input": 62_000,
                "vat_output": 95_000,
            },
        ]
    )

    result = pipeline.predict_batch(frame)

    # Calibrated probability 0.2 and anomaly score 0.5 => risk = 26.0
    # risk = 0.8 * 0.2 * 100 + 0.2 * 0.5 * 100 = 26
    year_trend = {item["year"]: item for item in result["statistics"]["year_trend"]}
    assert year_trend["2023"]["avg_risk"] == 26.0
    assert year_trend["2024"]["avg_risk"] == 26.0

    for row in result["assessments"]:
        assert row["risk_score"] == 26.0


def test_batch_history_lookup_handles_numeric_tax_code_input():
    pipeline = TaxFraudPipeline(model_dir=".")
    pipeline._loaded = True
    pipeline.feature_engineer = _StubFeatureEngineer()
    pipeline.xgboost_model = _StubXgbModel()
    pipeline.isolation_forest = _StubIsoModel()
    pipeline._calibrate_fraud_probs = lambda probs: np.full(len(probs), 0.2, dtype=float)
    pipeline.get_serving_metadata = lambda: {
        "model_version": "fraud-test",
        "calibration_method": "test",
    }

    frame = pd.DataFrame(
        [
            {
                "tax_code": 200004979,
                "year": 2023,
                "revenue": 8_000_000,
                "total_expenses": 7_000_000,
                "net_profit": 1_000_000,
                "vat_input": 600_000,
                "vat_output": 900_000,
            },
            {
                "tax_code": 200004979,
                "year": 2024,
                "revenue": 10_000_000,
                "total_expenses": 8_500_000,
                "net_profit": 1_500_000,
                "vat_input": 700_000,
                "vat_output": 1_050_000,
            },
            {
                "tax_code": 200004980,
                "year": 2023,
                "revenue": 5_000_000,
                "total_expenses": 4_200_000,
                "net_profit": 800_000,
                "vat_input": 410_000,
                "vat_output": 520_000,
            },
            {
                "tax_code": 200004980,
                "year": 2024,
                "revenue": 5_400_000,
                "total_expenses": 4_500_000,
                "net_profit": 900_000,
                "vat_input": 430_000,
                "vat_output": 560_000,
            },
        ]
    )

    result = pipeline.predict_batch(frame)

    target_assessment = next(
        (row for row in result["assessments"] if row["tax_code"] == "200004979"),
        None,
    )
    assert target_assessment is not None
    assessment = target_assessment
    assert [item["year"] for item in assessment["yearly_history"]] == [2023, 2024]
