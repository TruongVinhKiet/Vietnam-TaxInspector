"""
security.py – Tầng Bảo mật Trung tâm (Central Security Layer)
=================================================================
Modules:
    1. Rate Limiting (slowapi) – Chống Brute Force / DDoS
    2. Security Headers Middleware – Chống Clickjacking, XSS, MIME sniffing
"""

import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# =============================================================================
# 1. RATE LIMITER (Global)
# =============================================================================

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],         # Default: 200 req/min per IP
    storage_uri="memory://",               # In-memory store (switch to Redis in production)
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handler when rate limit is exceeded. Returns 429 with Vietnamese message."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Quá nhiều yêu cầu. Vui lòng thử lại sau.",
            "error": "rate_limit_exceeded",
        },
    )


# =============================================================================
# 2. SECURITY HEADERS MIDDLEWARE
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Inject OWASP-recommended security headers into every response.
    These headers protect against:
        - Clickjacking (X-Frame-Options)
        - MIME sniffing attacks (X-Content-Type-Options)
        - XSS reflection (X-XSS-Protection)
        - Information leakage (X-Powered-By removal, Referrer-Policy)
        - Downgrade attacks (Strict-Transport-Security in production)
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # --- Anti-Clickjacking ---
        response.headers["X-Frame-Options"] = "DENY"

        # --- Anti-MIME Sniffing ---
        response.headers["X-Content-Type-Options"] = "nosniff"

        # --- XSS Protection (legacy browsers) ---
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # --- Referrer Policy (prevent data leakage via referrer headers) ---
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # --- Permissions Policy (restrict browser features) ---
        response.headers["Permissions-Policy"] = (
            "camera=(self), microphone=(), geolocation=(), payment=()"
        )

        # --- Content Security Policy (restrict script/style sources) ---
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com https://cdn.tailwindcss.com; "
            "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' http://localhost:8000 http://127.0.0.1:8000 "
            "https://cdn.jsdelivr.net https://raw.githubusercontent.com; "
            "media-src 'self' blob:; "
            "frame-ancestors 'none';"
        )

        # --- Remove server fingerprint ---
        response.headers["X-Powered-By"] = ""
        if "server" in response.headers:
            del response.headers["server"]

        # --- HSTS (Uncomment in production behind HTTPS reverse proxy) ---
        # response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"

        return response
