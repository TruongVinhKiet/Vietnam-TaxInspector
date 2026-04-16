import json
import logging
from datetime import datetime, timezone


def get_structured_logger(name: str) -> logging.Logger:
    """Create/reuse a logger that emits JSON log lines."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, action: str, **context):
    """Emit a structured JSON event for easier filtering in production logs."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **context,
    }
    message = json.dumps(payload, ensure_ascii=False, default=str)
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message)
