"""
models/analysis.py — ReportAnalysis ORM model
"""
import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ReportAnalysis(Base):
    __tablename__ = "report_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False, unique=True)
    sif_potential = Column(Boolean, nullable=True)
    confidence = Column(Float, nullable=True)
    risk_score = Column(Integer, nullable=True)
    risk_level = Column(String(20), nullable=True)  # ROUTINE | REVIEW | HIGH | SIF
    activity = Column(String(255), nullable=True)
    hazard = Column(String(255), nullable=True)
    energy_source = Column(String(255), nullable=True)
    person_in_proximity = Column(Boolean, nullable=True)  # Was a person in the exposure zone?
    barrier = Column(String(255), nullable=True)
    barrier_status = Column(String(50), nullable=True)    # INTACT | DEGRADED | FAILED | UNKNOWN
    iogp_rule = Column(String(255), nullable=True)
    severity = Column(String(20), nullable=True)
    rationale = Column(Text, nullable=True)
    # ── Information sufficiency (Lara et al. inspired) ────────────────────────
    information_sufficiency = Column(Integer, nullable=True)       # 0-4: number of Lara factors present
    information_sufficiency_reason = Column(Text, nullable=True)   # Why info is/isn't sufficient
    # ── Clarification ───────────────────────────────────────────────────
    requires_followup = Column(Boolean, nullable=True, default=False)
    followup_question = Column(Text, nullable=True)
    followup_reason = Column(Text, nullable=True)                  # Why this specific question
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    report = relationship("Report", back_populates="analysis")
