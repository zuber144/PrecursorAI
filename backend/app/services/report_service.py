"""
services/report_service.py — Orchestrates the full Tier 1 pipeline for a single report.

Pipeline:
  1. Save raw report
  2. Preprocessing (ai/preprocessing.py)
  3. Generate embedding (ai/embeddings.py)
  4. RAG retrieval (ai/rag.py)
  5. Ollama/Qwen structured analysis (ai/classifier.py)
  6. Validate JSON schema
  7. Deterministic risk engine (ai/risk_engine.py)
  8. Save analysis
  9. Escalation / alerting (triage_service.py)
"""

import logging
import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.classifier import classify_report
from app.ai.embeddings import embed_text
from app.ai.preprocessing import preprocess_report_text
from app.ai.rag import retrieve_relevant_chunks
from app.ai.risk_engine import compute_risk_score, determine_risk_level
from app.models.embedding import ReportEmbedding
from app.models.analysis import ReportAnalysis
from app.models.clarification import ReportClarification
from app.models.report import Report
from app.schemas.report import (
    ReportCreate,
    ReportSubmitResponse,
    AnalysisSummary,
    ReportListItem,
    ReportDetail,
    ClarificationSubmit,
)

logger = logging.getLogger(__name__)


async def submit_and_analyze(
    payload: ReportCreate,
    db: AsyncSession,
) -> ReportSubmitResponse:
    """
    End-to-end Tier 1 processing for a new report.

    Each pipeline stage is logged separately so that if something fails,
    the exact stage can be identified from the application logs.
    """

    # -------------------------------------------------------------
    # 1. CREATE AND SAVE RAW REPORT
    # -------------------------------------------------------------

    report = Report(
        report_text=payload.report_text,
        report_type=payload.report_type,
        location=payload.location,
        asset_id=payload.asset_id,
        status="PENDING",
    )

    db.add(report)
    await db.commit()
    await db.refresh(report)

    report_id = report.id

    logger.info(
        "Tier 1 started | report_id=%s",
        report_id,
    )

    try:
        # ---------------------------------------------------------
        # 2. PREPROCESSING
        # ---------------------------------------------------------

        logger.info(
            "Stage 2/9 | preprocessing | report_id=%s",
            report_id,
        )

        clean_text = preprocess_report_text(
            report.report_text
        )

        logger.info(
            "Stage 2/9 complete | cleaned_length=%s | report_id=%s",
            len(clean_text),
            report_id,
        )

        # ---------------------------------------------------------
        # 3. EMBEDDING
        # ---------------------------------------------------------

        logger.info(
            "Stage 3/9 | generating embedding | report_id=%s",
            report_id,
        )

        embedding_vec = await embed_text(
            clean_text
        )

        logger.info(
            "Stage 3/9 complete | embedding_dimensions=%s | report_id=%s",
            len(embedding_vec) if embedding_vec else 0,
            report_id,
        )

        # Save embedding
        report_emb = ReportEmbedding(
            report_id=report_id,
            embedding=embedding_vec,
            model="gemini-embedding-001",
        )

        db.add(report_emb)

        await db.flush()

        logger.info(
            "Embedding saved | report_id=%s",
            report_id,
        )

        # ---------------------------------------------------------
        # 4. RAG RETRIEVAL
        # ---------------------------------------------------------

        logger.info(
            "Stage 4/9 | RAG retrieval | report_id=%s",
            report_id,
        )

        knowledge_chunks = await retrieve_relevant_chunks(
            embedding_vec,
            db,
        )

        logger.info(
            "Stage 4/9 complete | chunks=%s | report_id=%s",
            len(knowledge_chunks) if knowledge_chunks else 0,
            report_id,
        )

        # ---------------------------------------------------------
        # 5. OLLAMA / QWEN CLASSIFICATION
        # ---------------------------------------------------------

        logger.info(
            "Stage 5/9 | Ollama/Qwen classification | report_id=%s",
            report_id,
        )

        gemini_out = await classify_report(
            clean_text,
            knowledge_chunks,
        )

        logger.info(
            "Stage 5/9 complete | report_id=%s",
            report_id,
        )

        # ---------------------------------------------------------
        # BASIC CLASSIFIER VALIDATION
        # ---------------------------------------------------------

        if gemini_out is None:
            raise ValueError(
                "Classifier returned None"
            )

        logger.info(
            "Classifier output | sif_potential=%s | confidence=%s | "
            "requires_followup=%s | report_id=%s",
            getattr(
                gemini_out,
                "sif_potential",
                None,
            ),
            getattr(
                gemini_out,
                "confidence_score",
                None,
            ),
            getattr(
                gemini_out,
                "requires_followup",
                None,
            ),
            report_id,
        )

        # ---------------------------------------------------------
        # 6. PRIMARY IOGP RULE
        # ---------------------------------------------------------

        primary_rule = (
            gemini_out.life_saving_rules[0]
            if gemini_out.life_saving_rules
            else None
        )

        # ---------------------------------------------------------
        # 6A. CLARIFICATION PATH
        # ---------------------------------------------------------

        if (
            gemini_out.requires_followup
            and gemini_out.followup_question
        ):
            logger.info(
                "Clarification required | report_id=%s",
                report_id,
            )

            # -----------------------------------------------------
            # 7. PROVISIONAL RISK SCORE
            # -----------------------------------------------------

            risk_score = compute_risk_score(
                gemini_out
            )

            risk_level = determine_risk_level(
                risk_score
            )

            logger.info(
                "Provisional risk calculated | score=%s | level=%s | "
                "report_id=%s",
                risk_score,
                risk_level,
                report_id,
            )

            # -----------------------------------------------------
            # 8. SAVE PARTIAL ANALYSIS
            # -----------------------------------------------------

            analysis = ReportAnalysis(
                report_id=report_id,
                sif_potential=gemini_out.sif_potential,
                confidence=gemini_out.confidence_score,
                risk_score=risk_score,
                risk_level=risk_level,
                activity=gemini_out.activity,
                hazard=gemini_out.hazard,
                energy_source=gemini_out.energy_source,
                person_in_proximity=gemini_out.person_in_proximity,
                barrier=gemini_out.barrier,
                barrier_status=gemini_out.barrier_status,
                iogp_rule=primary_rule,
                severity=gemini_out.severity,
                rationale=gemini_out.rationale,
                information_sufficiency=(
                    gemini_out.information_sufficiency
                ),
                information_sufficiency_reason=(
                    gemini_out.information_sufficiency_reason
                ),
                requires_followup=True,
                followup_question=(
                    gemini_out.followup_question
                ),
                followup_reason=(
                    gemini_out.followup_reason
                ),
            )

            db.add(analysis)

            # -----------------------------------------------------
            # AUDIT TRAIL
            # -----------------------------------------------------

            clarification = ReportClarification(
                report_id=report_id,
                question=gemini_out.followup_question,
            )

            db.add(clarification)

            report.status = "NEEDS_CLARIFICATION"

            await db.commit()

            logger.info(
                "Report requires clarification | report_id=%s",
                report_id,
            )

            return ReportSubmitResponse(
                report_id=report_id,
                status=report.status,
                followup_question=(
                    gemini_out.followup_question
                ),
                analysis=AnalysisSummary(
                    sif_potential=analysis.sif_potential,
                    risk_level=analysis.risk_level,
                    risk_score=analysis.risk_score,
                    iogp_rule=primary_rule,
                ),
            )

        # ---------------------------------------------------------
        # 6B. NORMAL PATH
        # ---------------------------------------------------------

        logger.info(
            "No clarification required | report_id=%s",
            report_id,
        )

        # ---------------------------------------------------------
        # 7. DETERMINISTIC RISK ENGINE
        # ---------------------------------------------------------

        risk_score = compute_risk_score(
            gemini_out
        )

        risk_level = determine_risk_level(
            risk_score
        )

        logger.info(
            "Stage 7/9 | risk calculated | score=%s | level=%s | "
            "report_id=%s",
            risk_score,
            risk_level,
            report_id,
        )

        # ---------------------------------------------------------
        # 8. SAVE ANALYSIS
        # ---------------------------------------------------------

        analysis = ReportAnalysis(
            report_id=report_id,
            sif_potential=gemini_out.sif_potential,
            confidence=gemini_out.confidence_score,
            risk_score=risk_score,
            risk_level=risk_level,
            activity=gemini_out.activity,
            hazard=gemini_out.hazard,
            energy_source=gemini_out.energy_source,
            person_in_proximity=gemini_out.person_in_proximity,
            barrier=gemini_out.barrier,
            barrier_status=gemini_out.barrier_status,
            iogp_rule=primary_rule,
            severity=gemini_out.severity,
            rationale=gemini_out.rationale,
            information_sufficiency=(
                gemini_out.information_sufficiency
            ),
            information_sufficiency_reason=(
                gemini_out.information_sufficiency_reason
            ),
            requires_followup=False,
            followup_question=None,
            followup_reason=None,
        )

        db.add(analysis)

        logger.info(
            "Stage 8/9 | analysis object created | report_id=%s",
            report_id,
        )

        # ---------------------------------------------------------
        # 9. TRIAGE / ALERTING
        # ---------------------------------------------------------

        logger.info(
            "Stage 9/9 | triage routing | report_id=%s",
            report_id,
        )

        from app.services.triage_service import route_report

        await route_report(
            analysis,
            report,
            db,
        )

        logger.info(
            "Triage routing complete | report_id=%s",
            report_id,
        )

        # ---------------------------------------------------------
        # FINAL DATABASE COMMIT
        # ---------------------------------------------------------

        report.status = "ANALYZED"

        await db.commit()

        logger.info(
            "Tier 1 pipeline COMPLETE | report_id=%s",
            report_id,
        )

        summary = AnalysisSummary(
            sif_potential=analysis.sif_potential,
            risk_level=analysis.risk_level,
            risk_score=analysis.risk_score,
            iogp_rule=analysis.iogp_rule,
        )

        return ReportSubmitResponse(
            report_id=report_id,
            status=report.status,
            analysis=summary,
        )

    except Exception as e:
        # ---------------------------------------------------------
        # ERROR HANDLING
        # ---------------------------------------------------------

        await db.rollback()

        logger.exception(
            "=================================================="
        )

        logger.exception(
            "TIER 1 PIPELINE FAILED"
        )

        logger.exception(
            "Report ID: %s",
            report_id,
        )

        logger.exception(
            "Error type: %s",
            type(e).__name__,
        )

        logger.exception(
            "Error message: %s",
            str(e),
        )

        logger.exception(
            "=================================================="
        )

        # IMPORTANT:
        #
        # Do NOT attempt another DB commit here.
        #
        # If the database transaction failed, SQLAlchemy has
        # already placed the session into a rollback-required state.
        #
        # The rollback above resets the session.

        return ReportSubmitResponse(
            report_id=report_id,
            status="ERROR",
            analysis=AnalysisSummary(
                sif_potential=False,
                risk_level="REVIEW",
                risk_score=50,
                iogp_rule=None,
            ),
        )


async def submit_clarification(
    report_id: uuid.UUID,
    payload: ClarificationSubmit,
    db: AsyncSession,
) -> ReportSubmitResponse:
    """
    Handle the user's answer to a clarification question.
    """

    stmt = (
        select(Report)
        .options(
            selectinload(Report.analysis),
            selectinload(Report.clarifications),
            selectinload(Report.embedding),
        )
        .where(
            Report.id == report_id
        )
    )

    result = await db.execute(stmt)

    report = result.scalars().first()

    if (
        not report
        or report.status != "NEEDS_CLARIFICATION"
    ):
        raise ValueError(
            "Report not found or not in NEEDS_CLARIFICATION state"
        )

    # -------------------------------------------------------------
    # FIND ACTIVE CLARIFICATION
    # -------------------------------------------------------------

    active_clarification = next(
        (
            c
            for c in report.clarifications
            if c.answer is None
        ),
        None,
    )

    if active_clarification:
        active_clarification.answer = payload.answer

    # -------------------------------------------------------------
    # COMBINE ORIGINAL REPORT + ANSWER
    # -------------------------------------------------------------

    combined_text = (
        f"ORIGINAL REPORT:\n"
        f"{report.report_text}\n\n"
        f"CLARIFICATION PROVIDED BY REPORTER:\n"
        f"Question: "
        f"{active_clarification.question} "
        f"if active_clarification else 'Unknown'\n"
        f"Answer: {payload.answer}"
    )

    try:
        # ---------------------------------------------------------
        # 1. PREPROCESS
        # ---------------------------------------------------------

        logger.info(
            "Clarification Stage 1 | preprocessing | report_id=%s",
            report_id,
        )

        clean_text = preprocess_report_text(
            combined_text
        )

        # ---------------------------------------------------------
        # 2. ORIGINAL EMBEDDING FOR RAG
        # ---------------------------------------------------------

        logger.info(
            "Clarification Stage 2 | RAG retrieval | report_id=%s",
            report_id,
        )

        if not report.embedding:
            raise ValueError(
                "Original report embedding was not found"
            )

        embedding_vec = report.embedding.embedding

        knowledge_chunks = await retrieve_relevant_chunks(
            embedding_vec,
            db,
        )

        # ---------------------------------------------------------
        # 3. SECOND-PASS OLLAMA/QWEN
        # ---------------------------------------------------------

        logger.info(
            "Clarification Stage 3 | Ollama/Qwen classification | "
            "report_id=%s",
            report_id,
        )

        gemini_out = await classify_report(
            clean_text,
            knowledge_chunks,
        )

        if gemini_out is None:
            raise ValueError(
                "Classifier returned None during clarification"
            )

        # ---------------------------------------------------------
        # 4. FINAL RISK SCORE
        # ---------------------------------------------------------

        risk_score = compute_risk_score(
            gemini_out
        )

        risk_level = determine_risk_level(
            risk_score
        )

        primary_rule = (
            gemini_out.life_saving_rules[0]
            if gemini_out.life_saving_rules
            else None
        )

        # ---------------------------------------------------------
        # 5. UPDATE EXISTING ANALYSIS
        # ---------------------------------------------------------

        analysis = report.analysis

        if not analysis:
            raise ValueError(
                "Existing analysis record was not found"
            )

        analysis.sif_potential = (
            gemini_out.sif_potential
        )

        analysis.confidence = (
            gemini_out.confidence_score
        )

        analysis.risk_score = risk_score

        analysis.risk_level = risk_level

        analysis.activity = (
            gemini_out.activity
        )

        analysis.hazard = (
            gemini_out.hazard
        )

        analysis.energy_source = (
            gemini_out.energy_source
        )

        analysis.person_in_proximity = (
            gemini_out.person_in_proximity
        )

        analysis.barrier = (
            gemini_out.barrier
        )

        analysis.barrier_status = (
            gemini_out.barrier_status
        )

        analysis.iogp_rule = primary_rule

        analysis.severity = (
            gemini_out.severity
        )

        analysis.rationale = (
            gemini_out.rationale
        )

        analysis.information_sufficiency = (
            gemini_out.information_sufficiency
        )

        analysis.information_sufficiency_reason = (
            gemini_out.information_sufficiency_reason
        )

        # Maximum one clarification question
        analysis.requires_followup = False
        analysis.followup_question = None
        analysis.followup_reason = None

        # ---------------------------------------------------------
        # 6. TRIAGE
        # ---------------------------------------------------------

        from app.services.triage_service import route_report

        await route_report(
            analysis,
            report,
            db,
        )

        # ---------------------------------------------------------
        # 7. FINAL COMMIT
        # ---------------------------------------------------------

        report.status = "ANALYZED"

        await db.commit()

        logger.info(
            "Clarification pipeline COMPLETE | report_id=%s",
            report_id,
        )

        summary = AnalysisSummary(
            sif_potential=analysis.sif_potential,
            risk_level=analysis.risk_level,
            risk_score=analysis.risk_score,
            iogp_rule=analysis.iogp_rule,
        )

        return ReportSubmitResponse(
            report_id=report_id,
            status=report.status,
            analysis=summary,
        )

    except Exception as e:
        await db.rollback()

        logger.exception(
            "=================================================="
        )

        logger.exception(
            "CLARIFICATION PIPELINE FAILED"
        )

        logger.exception(
            "Report ID: %s",
            report_id,
        )

        logger.exception(
            "Error type: %s",
            type(e).__name__,
        )

        logger.exception(
            "Error message: %s",
            str(e),
        )

        logger.exception(
            "=================================================="
        )

        return ReportSubmitResponse(
            report_id=report_id,
            status="ERROR",
        )


async def list_reports(
    skip: int,
    limit: int,
    db: AsyncSession,
) -> List[ReportListItem]:
    """Retrieve paginated list of reports."""

    stmt = (
        select(Report)
        .order_by(
            Report.created_at.desc()
        )
        .offset(skip)
        .limit(limit)
    )

    result = await db.execute(stmt)

    reports = result.scalars().all()

    return [
        ReportListItem.model_validate(r)
        for r in reports
    ]


async def get_report(
    report_id: uuid.UUID,
    db: AsyncSession,
) -> Optional[ReportDetail]:
    """Retrieve a single report."""

    stmt = (
        select(Report)
        .options(
            selectinload(Report.analysis)
        )
        .where(
            Report.id == report_id
        )
    )

    result = await db.execute(stmt)

    report = result.scalars().first()

    if not report:
        return None

    return ReportDetail.model_validate(
        report
    )