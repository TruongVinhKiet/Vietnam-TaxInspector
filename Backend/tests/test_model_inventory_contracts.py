from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.model_inventory import REQUIRED_MODEL_KEYS, get_model_inventory_payload  # noqa: E402
from ml_engine.model_serving import ModelServingGateway  # noqa: E402
from ml_engine.tax_agent_mode_contracts import AgentModeContractRegistry  # noqa: E402
from ml_engine.tax_agent_tools import build_default_registry  # noqa: E402


def test_model_inventory_contains_required_models_and_readiness_shape() -> None:
    payload = get_model_inventory_payload()
    keys = {model["model_key"] for model in payload["models"]}

    assert payload["schema_version"] == "model-inventory-v1"
    assert set(REQUIRED_MODEL_KEYS).issubset(keys)
    assert {
        "fraud-hybrid-v2",
        "temporal-transformer-v1",
        "vae-anomaly-v1",
        "gnn-gat-v1",
        "hetero-gnn-hgt-v1",
        "tax-agent-rag-v1",
        "dpo-rlhf-v1",
    }.issubset(keys)

    hetero = next(model for model in payload["models"] if model["model_key"] == "hetero-gnn-hgt-v1")
    assert "Backend/data/models/hgt_model.pt" in hetero["artifact_paths"]
    assert "readiness_check" in hetero
    assert {"ready", "checked", "missing_count"}.issubset(hetero["readiness_check"].keys())


def test_model_serving_gateway_registers_hgt_artifact() -> None:
    ModelServingGateway.reset()
    gateway = ModelServingGateway.instance()
    status = gateway.get_status()
    hetero = status["models"]["hetero_gnn"]

    assert {"vae", "transformer", "gnn", "hetero_gnn"}.issubset(status["models"].keys())
    assert hetero["model_path"].endswith("hgt_model.pt")
    assert hetero["config_path"].endswith("hgt_config.json")


def test_model_api_server_exposes_hetero_gnn_route() -> None:
    from ml_engine.model_api_server import app

    paths = {route.path for route in app.routes}
    assert "/predict/hetero-gnn" in paths
    assert "/models/status" in paths


def test_mode_contract_allowed_tools_exist_in_tool_registry() -> None:
    registry = build_default_registry()
    tool_names = set(registry.list_tool_names())

    for mode in ("fraud", "vat", "macro", "delinquency", "legal"):
        contract = AgentModeContractRegistry.get(mode)
        assert contract.allowed_tools is not None
        assert set(contract.allowed_tools).issubset(tool_names), mode


def test_readme_mentions_all_required_model_keys() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = [key for key in REQUIRED_MODEL_KEYS if key not in readme]
    assert not missing


def test_tax_agent_infrastructure_status_includes_inventory() -> None:
    from app.routers.tax_agent import get_model_serving_status

    payload = get_model_serving_status()
    assert payload["models"]["hetero_gnn"]["model_path"].endswith("hgt_model.pt")
    assert payload["model_inventory"]["schema_version"] == "model-inventory-v1"
    assert "hetero-gnn-hgt-v1" in payload["model_inventory"]["required_model_keys"]

