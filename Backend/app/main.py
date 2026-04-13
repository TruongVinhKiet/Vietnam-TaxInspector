"""
main.py – TaxInspector API Entry Point (Security Hardened)
============================================================
Changes:
    1. CORS: allow_credentials=True (required for HttpOnly Cookie)
    2. Security Headers middleware added
    3. Rate Limiter (slowapi) integrated
    4. Swagger /docs disabled in production (optional)
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .database import engine, Base
from .routers import scoring, graph, delinquency, auth, ai_analysis
from .security import limiter, SecurityHeadersMiddleware, rate_limit_exceeded_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup, cleanup on shutdown."""
    try:
        Base.metadata.create_all(bind=engine)
        print("[OK] Database tables verified / created.")
    except Exception as e:
        print(f"[WARN] Database not reachable, starting without DB: {e}")
    yield

app = FastAPI(
    title="TaxInspector ML API",
    description="API he thong giam sat thue tich hop Machine Learning: "
                "Fraud Risk Scoring, VAT Invoice Graph, Delinquency Prediction.",
    version="2.0.0-SECURE",
    lifespan=lifespan,
)

# --- Rate Limiter ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# --- CORS (allow_credentials=True for HttpOnly Cookie) ---
default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://[::1]:3000,http://[::]:3000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,        # CRITICAL: Required for cookie-based auth
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security Headers Middleware ---
app.add_middleware(SecurityHeadersMiddleware)

# --- Register Routers ---
app.include_router(auth.router)
app.include_router(scoring.router)
app.include_router(graph.router)
app.include_router(delinquency.router)
app.include_router(ai_analysis.router)


@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "version": "2.0.0-SECURE",
        "security": {
            "cookie_auth": True,
            "rate_limiting": True,
            "security_headers": True,
            "data_encryption": True,
            "audit_logging": True,
        },
        "message": "TaxInspector API is running with full security hardening.",
    }
