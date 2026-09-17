"""
models/report.py — User and Report ORM models
"""
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False, default="worker")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    reports = relationship("Report", back_populates="submitter")


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_text = Column(Text, nullable=False)
    report_type = Column(String(50), nullable=False)  # UNSAFE_ACT | UNSAFE_CONDITION | NEAR_MISS
    asset_id = Column(String(100), nullable=True)
    location = Column(String(255), nullable=True)
    submitted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), nullable=False, default="PENDING")

    submitter = relationship("User", back_populates="reports")
    analysis = relationship("ReportAnalysis", back_populates="report", uselist=False)
    embedding = relationship("ReportEmbedding", back_populates="report", uselist=False)
    pattern_links = relationship("PatternReport", back_populates="report")
    alerts = relationship("Alert", back_populates="report")
    clarifications = relationship("ReportClarification", back_populates="report", order_by="ReportClarification.created_at")
