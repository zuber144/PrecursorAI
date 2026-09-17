"""
api/reports.py — Report submission and retrieval endpoints (Tier 1 entry point)
"""
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.report import ReportCreate, ReportDetail, ReportListItem, ReportSubmitResponse, ClarificationSubmit
from app.services.report_service import submit_and_analyze, submit_clarification
from app.services.report_service import list_reports as list_reports_service
from app.services.report_service import get_report as get_report_service

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("", response_model=ReportSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_report(payload: ReportCreate, db: AsyncSession = Depends(get_db)):
    """
    Submit a new safety report for Tier 1 analysis.

    The report is saved, preprocessed, embedded, and run through
    RAG + Gemini + the deterministic risk engine.
    Returns a summary of the analysis result.
    """
    return await submit_and_analyze(payload, db)


@router.post("/{report_id}/clarify", response_model=ReportSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def clarify_report(report_id: UUID, payload: ClarificationSubmit, db: AsyncSession = Depends(get_db)):
    """
    Submit an answer to an AI clarification question.
    """
    try:
        return await submit_clarification(report_id, payload, db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("", response_model=List[ReportListItem])
async def list_reports(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all reports, newest first."""
    return await list_reports_service(skip, limit, db)


@router.get("/{report_id}", response_model=ReportDetail)
async def get_report(report_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get full details of a single report including its analysis."""
    report = await get_report_service(report_id, db)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report
