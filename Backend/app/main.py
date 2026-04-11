from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .routers import scoring, graph, delinquency, auth

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
    version="1.0.0",
    lifespan=lifespan,
)

default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://[::1]:3000,http://[::]:3000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]

# CORS - cho phep Frontend truy cap API
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register Routers ---
app.include_router(auth.router)
app.include_router(scoring.router)
app.include_router(graph.router)
app.include_router(delinquency.router)


@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "message": "TaxInspector API is running. Visit /docs for Swagger UI.",
    }
