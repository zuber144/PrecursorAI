"""
app/main.py — PrecursorAI FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import alerts, cognition, dashboard, patterns, reports
from app.core.config import settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run full DB bootstrap on startup: create DB → enable pgvector → create tables."""
    await init_db()
    yield


app = FastAPI(
    title="PrecursorAI",
    description="AI-powered safety intelligence system for OIL — SIF precursor detection and pattern cognition.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
API_PREFIX = "/api/v1"
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(dashboard.router, prefix=API_PREFIX)
app.include_router(alerts.router, prefix=API_PREFIX)
app.include_router(patterns.router, prefix=API_PREFIX)
app.include_router(cognition.router, prefix=API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "PrecursorAI"}
