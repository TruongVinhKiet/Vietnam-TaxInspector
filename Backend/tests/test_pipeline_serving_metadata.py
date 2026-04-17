import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.pipeline import TaxFraudPipeline


def test_get_serving_metadata_prefers_manifest_version(tmp_path):
    pipeline = TaxFraudPipeline(model_dir=str(tmp_path))
    pipeline._loaded = True
    pipeline.model_manifest = {
        "manifest_version": "fraud-model-manifest-v1",
        "model_version": "fraud-hybrid-v99",
        "feature_contract": {"feature_set": "fraud_inference_features_v1"},
    }
    pipeline.calibration_meta = {
        "available": True,
        "method": "isotonic",
        "model_version": "fraud-hybrid-calibrator-v1",
    }

    metadata = pipeline.get_serving_metadata()

    assert metadata["model_version"] == "fraud-hybrid-v99"
    assert metadata["manifest_version"] == "fraud-model-manifest-v1"
    assert metadata["feature_set"] == "fraud_inference_features_v1"
    assert metadata["calibration_method"] == "isotonic"
    assert metadata["calibrator_available"] is True


def test_get_serving_metadata_fallback_chain(tmp_path):
    pipeline = TaxFraudPipeline(model_dir=str(tmp_path))
    pipeline._loaded = True
    pipeline.model_manifest = {}
    pipeline.calibration_meta = {
        "available": False,
        "method": "identity",
        "model_version": "fraud-hybrid-calibrator-v2",
    }

    metadata = pipeline.get_serving_metadata()
    assert metadata["model_version"] == "fraud-hybrid-calibrator-v2"

    pipeline.calibration_meta = {}
    metadata_legacy = pipeline.get_serving_metadata()
    assert metadata_legacy["model_version"] == "fraud-hybrid-legacy"
