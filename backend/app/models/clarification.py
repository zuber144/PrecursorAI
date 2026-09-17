"""
models/clarification.py - ReportClarification ORM model.

Stores the audit trail for the one-question clarification path.
When requires_followup=True, a row is created immediately with
the question and a NULL answer. The answer is filled when the
user responds via POST /reports/{id}/clarify.

This table is never modified retrospectively - it is append-only
for auditability.
"""
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ReportClarification(Base):
    __tablename__ = "report_clarifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(
        UUID(as_uuid=True),
        ForeignKey("reports.id"),
        nullable=False,
        index=True,
    )
    # The AI-generated targeted question
    question = Column(Text, nullable=False)
    # The user's answer - NULL until the user responds
    answer = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    report = relationship("Report", back_populates="clarifications")