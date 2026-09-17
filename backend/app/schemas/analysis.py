"""
schemas/analysis.py — Tier 1 analysis Pydantic schema (full structured output)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GeminiAnalysisOutput(BaseModel):
    """
    Schema for the structured JSON output produced by Gemini in Tier 1.
    Gemini populates these fields; the deterministic risk engine computes
    risk_score and final risk_level — the LLM does NOT decide risk level.

    information_sufficiency (0-4): count of Lara-inspired factors the LLM
    considers PRESENT (context, hazard, controls, causal mechanism).
    This does NOT directly determine whether a question is asked.
    """
    sif_potential: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    hazard: str
    energy_source: str
    activity: str
    person_in_proximity: Optional[bool] = None   # Was / could a person be in the exposure zone?
    asset: Optional[str] = None
    location: Optional[str] = None
    barrier: str
    barrier_status: Literal["INTACT", "DEGRADED", "FAILED", "UNKNOWN"]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    life_saving_rules: List[str] = Field(default_factory=list)
    rationale: str
    # ── Information sufficiency (Lara et al. inspired) ────────────────────────
    information_sufficiency: int = Field(default=4, ge=0, le=4)
    information_sufficiency_reason: str = ""
    # ── Clarification (only populated when requires_followup=True) ────────────
    requires_followup: bool = False
    followup_question: Optional[str] = None
    followup_reason: Optional[str] = None   # Why this specific question is asked


class AnalysisResponse(BaseModel):
    """Full analysis as returned by the API."""
    id: uuid.UUID
    report_id: uuid.UUID
    sif_potential: Optional[bool]
    confidence: Optional[float]
    risk_score: Optional[int]
    risk_level: Optional[str]
    activity: Optional[str]
    hazard: Optional[str]
    energy_source: Optional[str]
    person_in_proximity: Optional[bool]
    barrier: Optional[str]
    barrier_status: Optional[str]
    iogp_rule: Optional[str]
    severity: Optional[str]
    rationale: Optional[str]
    information_sufficiency: Optional[int]
    information_sufficiency_reason: Optional[str]
    requires_followup: Optional[bool]
    followup_question: Optional[str]
    followup_reason: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}
