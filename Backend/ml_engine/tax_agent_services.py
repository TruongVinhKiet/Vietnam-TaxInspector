"""Thin application services around the multi-agent orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import inspect

from ml_engine.tax_agent_mode_contracts import AgentModeContractRegistry


VALID_AGENT_MODES = {"fraud", "vat", "delinquency", "macro", "legal", "full"}


def normalize_agent_mode(mode: str | None) -> str:
    value = (mode or "full").lower()
    return value if value in VALID_AGENT_MODES else "full"


@dataclass
class AgentChatService:
    """Single orchestrator entrypoint for sync and streaming chat routes."""

    orchestrator: Any

    def _call_orchestrator(self, method_name: str, db, **kwargs):
        method = getattr(self.orchestrator, method_name)
        try:
            signature = inspect.signature(method)
            supports_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            supported = set(signature.parameters)
            call_kwargs = kwargs if supports_kwargs else {
                key: value for key, value in kwargs.items() if key in supported
            }
        except Exception:
            call_kwargs = dict(kwargs)
        return method(db, **call_kwargs)

    def process(
        self,
        db,
        *,
        session_id: str,
        message: str,
        user_id: int | None = None,
        top_k: int = 5,
        model_mode: str = "full",
        attachment_analysis: dict[str, Any] | None = None,
        simulation_params: dict[str, Any] | None = None,
    ) -> Any:
        resolved_mode = normalize_agent_mode(model_mode)
        return self._call_orchestrator(
            "process",
            db,
            session_id=session_id,
            message=message,
            user_id=user_id,
            top_k=top_k,
            model_mode=resolved_mode,
            attachment_analysis=attachment_analysis,
            simulation_params=simulation_params,
        )

    def stream(
        self,
        db,
        *,
        session_id: str,
        message: str,
        user_id: int | None = None,
        top_k: int = 5,
        model_mode: str = "full",
        attachment_analysis: dict[str, Any] | None = None,
        simulation_params: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        resolved_mode = normalize_agent_mode(model_mode)
        return self._call_orchestrator(
            "process_streaming",
            db,
            session_id=session_id,
            message=message,
            user_id=user_id,
            top_k=top_k,
            model_mode=resolved_mode,
            attachment_analysis=attachment_analysis,
            simulation_params=simulation_params,
        )


class AgentFileAnalysisService:
    """Shared helpers for attachment domain detection and mode validation."""

    @staticmethod
    def validate_mode_for_detection(
        *,
        model_mode: str,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_mode = normalize_agent_mode(model_mode)
        requested_domain = str(detection.get("requested_domain") or "unknown").lower()
        explicit_modes = VALID_AGENT_MODES - {"full"}
        if resolved_mode in explicit_modes and requested_domain in explicit_modes and resolved_mode != requested_domain:
            contract = AgentModeContractRegistry.get(requested_domain)
            return {
                **detection,
                "status": "mode_mismatch",
                "mode_mismatch": True,
                "model_mode": resolved_mode,
                "suggested_mode": requested_domain,
                "selected_model_bundle": contract.selected_model_bundle,
                "mode_validation": {
                    "valid": False,
                    "model_mode": resolved_mode,
                    "requested_domain": requested_domain,
                    "reason": "selected_mode_does_not_match_attachment_schema",
                },
            }
        return {**detection, "model_mode": resolved_mode}
