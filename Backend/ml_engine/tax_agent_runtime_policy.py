"""
Runtime policy for local/offline model loading.

TaxInspector is intended to run inside an on-prem tax authority environment.
The safe default is therefore: do not download model artifacts at runtime.

Set TAX_AGENT_ALLOW_MODEL_DOWNLOAD=1 only on a development/training machine
when preparing offline artifacts for deployment.
"""

from __future__ import annotations

import os
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def model_downloads_allowed() -> bool:
    """Whether runtime code may fetch missing model files from the network."""
    return env_flag("TAX_AGENT_ALLOW_MODEL_DOWNLOAD", default=False)


def local_files_only() -> bool:
    """Transformers/SentenceTransformers local_files_only value."""
    return not model_downloads_allowed()


def apply_offline_environment() -> None:
    """Tell Hugging Face libraries to stay offline unless downloads are allowed."""
    if local_files_only():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def is_probably_local_path(model_name_or_path: str | Path | None) -> bool:
    """Return True for filesystem paths; False for remote model IDs."""
    if not model_name_or_path:
        return False
    value = str(model_name_or_path)
    return Path(value).exists() or value.startswith((".", "/", "\\"))
