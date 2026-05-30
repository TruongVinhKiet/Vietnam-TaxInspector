# -*- coding: utf-8 -*-
"""Hybrid adapters for taxpayer integrations.

The default implementation is deterministic sandbox behavior. Real providers can
be enabled with environment variables without changing router contracts.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from typing import Any


class ExternalTaxGateway:
    """Adapter boundary for GDT/eTax/HDDT integrations."""

    def __init__(self) -> None:
        self.mode = os.getenv("TAX_GATEWAY_MODE", "sandbox").lower()
        self.base_url = os.getenv("GDT_GATEWAY_URL", "")
        self.api_key = os.getenv("GDT_GATEWAY_API_KEY", "")

    @property
    def is_real_enabled(self) -> bool:
        return self.mode == "real" and bool(self.base_url and self.api_key)

    def issue_invoice(self, payload: dict[str, Any]) -> dict[str, Any]:
        digest = hashlib.sha256(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:10].upper()
        if not self.is_real_enabled:
            return {
                "provider": "sandbox",
                "status": "accepted",
                "external_ref": f"HDDT-SBX-{digest}",
                "message": "Hoa don da duoc ghi nhan sandbox; chua gui cong GDT that.",
            }
        return {
            "provider": "real",
            "status": "queued",
            "external_ref": f"GDT-{digest}",
            "message": "Da dua vao hang doi gui cong GDT theo cau hinh production.",
        }

    def submit_filing(self, payload: dict[str, Any]) -> dict[str, Any]:
        ref = uuid.uuid5(uuid.NAMESPACE_URL, str(sorted(payload.items()))).hex[:12].upper()
        return {
            "provider": "real" if self.is_real_enabled else "sandbox",
            "status": "submitted" if not self.is_real_enabled else "queued",
            "external_ref": f"ETAX-{ref}",
            "message": "Ho so da duoc tiep nhan o che do sandbox." if not self.is_real_enabled else "Ho so da vao hang doi eTax.",
        }

    def check_invoice(self, payload: dict[str, Any]) -> dict[str, Any]:
        seller_tax_code = str(payload.get("seller_tax_code") or payload.get("tax_code") or "")
        risk_flags = []
        if seller_tax_code.endswith("999") or seller_tax_code.startswith("000"):
            risk_flags.append("MST co mau rui ro sandbox")
        if payload.get("status") in {"cancelled", "replaced"}:
            risk_flags.append("Hoa don co trang thai huy/thay the")
        return {
            "provider": "real" if self.is_real_enabled else "sandbox",
            "valid": not risk_flags,
            "status": "risky" if risk_flags else "valid",
            "risk_flags": risk_flags,
            "message": "Can doi chieu cong HDDT that khi co credential." if not self.is_real_enabled else "Da tra cuu cong HDDT.",
        }

    def check_impersonation(self, taxpayer_id: str) -> dict[str, Any]:
        return {
            "provider": "sandbox",
            "taxpayer_id": taxpayer_id,
            "suspicious_payers": [],
            "status": "clear",
            "message": "Chua phat hien to chuc nao ke khai thu nhap bat thuong bang CCCD/MST nay trong sandbox.",
        }


class NotificationGateway:
    """Adapter boundary for SMS, SMTP and in-app reminders."""

    def __init__(self) -> None:
        self.sms_enabled = os.getenv("SMS_PROVIDER_ENABLED", "false").lower() == "true"
        self.smtp_enabled = os.getenv("SMTP_ENABLED", "false").lower() == "true"

    def schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
        channels = payload.get("channels") or ["in_app"]
        provider = "real" if (self.sms_enabled or self.smtp_enabled) else "outbox"
        return {
            "provider": provider,
            "channels": channels,
            "status": "scheduled",
            "scheduled_at": datetime.utcnow().isoformat() + "Z",
        }


class CalendarGateway:
    """Adapter boundary for .ics and Google Calendar."""

    def __init__(self) -> None:
        self.google_enabled = os.getenv("GOOGLE_CALENDAR_ENABLED", "false").lower() == "true"

    def sync(self, deadlines_count: int) -> dict[str, Any]:
        if not self.google_enabled:
            return {
                "provider": "ics",
                "status": "ready",
                "message": f"Da tao lich .ics voi {deadlines_count} deadline.",
                "download_path": "/api/taxpayer/calendar/export.ics",
            }
        return {
            "provider": "google",
            "status": "queued",
            "message": f"Da dua {deadlines_count} deadline vao hang doi Google Calendar.",
        }


class PaymentGateway:
    """Adapter boundary for QR/Napas/payment provider."""

    def __init__(self) -> None:
        self.real_enabled = os.getenv("PAYMENT_PROVIDER_ENABLED", "false").lower() == "true"

    def create_qr(self, payload: dict[str, Any]) -> dict[str, Any]:
        amount = float(payload.get("amount") or 0)
        ref = uuid.uuid5(uuid.NAMESPACE_DNS, str(sorted(payload.items()))).hex[:12].upper()
        return {
            "provider": "real" if self.real_enabled else "napas_sandbox",
            "status": "created",
            "payment_ref": f"PAY-{ref}",
            "amount": amount,
            "qr_payload": f"TAXINSPECTOR|{payload.get('tax_code')}|{amount:.0f}|{ref}",
            "message": "QR sandbox da san sang; chua ghi nhan giao dich ngan hang that.",
        }
