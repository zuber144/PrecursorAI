"""
services/triage_service.py — Human-in-the-loop routing.

Rules:
  - IF confidence < settings.CONFIDENCE_THRESHOLD → route to REVIEW
  - IF requires_followup == True → route to REVIEW
  - IF risk_level in (HIGH, SIF) → create alert + route to dashboard
  - ELSE → ROUTINE storage

The LLM does NOT decide the final risk level.
Python (risk_engine) computes it deterministically.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.risk_engine import needs_human_review
from app.models.analysis import ReportAnalysis
from app.models.report import Report
from app.services.alert_service import create_sif_alert


async def route_report(analysis: ReportAnalysis, report: Report, db: AsyncSession):
    """
    Determine final status of the report based on analysis,
    and trigger alerts if necessary.
    """
    # 1. Human-in-the-loop checks (low confidence or explicit follow-up needed)
    #    OR if the risk level is higher than ROUTINE (user requested to keep non-routine in REVIEW until dismissed via Alerts)
    if needs_human_review(analysis) or analysis.risk_level != "ROUTINE":
        report.status = "REVIEW"
    else:
        report.status = "ANALYZED"

    # 2. Alert creation for severe risk levels
    if analysis.risk_level in ("HIGH", "SIF"):
        await create_sif_alert(report, analysis, db)
