from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class IntentModelArtifacts:
    model_version: str
    labels: list[str]
    vectorizer_path: str
    model_path: str


class TaxAgentIntentModel:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None
        self.labels: list[str] = []
        self.model_version: str = "tax-agent-intent-v1"

    def load(self) -> bool:
        vec_path = self.model_dir / "tax_agent_intent_vectorizer.joblib"
        model_path = self.model_dir / "tax_agent_intent_model.joblib"
        meta_path = self.model_dir / "tax_agent_intent_meta.json"
        if not vec_path.exists() or not model_path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model_version = str(meta.get("model_version") or self.model_version)
            self.labels = list(meta.get("labels") or [])
            self.vectorizer = joblib.load(vec_path)
            self.model = joblib.load(model_path)
            return True
        except Exception:
            self.vectorizer = None
            self.model = None
            self.labels = []
            return False

    def predict(self, text_value: str) -> tuple[str, float, dict[str, Any]]:
        if self.vectorizer is None or self.model is None:
            return ("general_tax_query", 0.0, {"source": "missing_artifacts"})
        X = self.vectorizer.transform([text_value or ""])
        proba = getattr(self.model, "predict_proba", None)
        if proba is None:
            pred_label = str(self.model.predict(X)[0])
            return (pred_label, 0.5, {"source": "intent_model_no_proba"})
        p = self.model.predict_proba(X)[0]
        idx = int(np.argmax(p))
        conf = float(p[idx])
        # Important: sklearn `classes_` defines the probability column order.
        classes = getattr(self.model, "classes_", None)
        if classes is not None and len(classes) > idx:
            label = str(classes[idx])
        elif self.labels and len(self.labels) > idx:
            label = str(self.labels[idx])
        else:
            label = "general_tax_query"
        return (label, conf, {"source": "intent_model", "top_prob": conf})

