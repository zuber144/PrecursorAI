"""
tests/test_clarification_pipeline.py - Pytest suite for Tier 1 Clarification Path.

Covers the 10 scenarios requested for the Lara-inspired information sufficiency feature.
Dependencies (DB, LLM, RAG) are mocked to test the pure orchestration logic.
"""
import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.schemas.report import ReportCreate, ClarificationSubmit
from app.schemas.analysis import GeminiAnalysisOutput
from app.services.report_service import submit_and_analyze, submit_clarification
from app.models.report import Report
from app.models.analysis import ReportAnalysis


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = AsyncMock()
    def mock_add(obj):
        if hasattr(obj, 'id') and obj.id is None:
            obj.id = uuid.uuid4()
    db.add = MagicMock(side_effect=mock_add)
    
    # Mock scalars().first() to return our mock report
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    db.execute.return_value = mock_result
    return db


@pytest.fixture
def base_gemini_out():
    """Returns a valid non-SIF baseline output."""
    return GeminiAnalysisOutput(
        sif_potential=False,
        confidence_score=0.9,
        hazard="Slip trip fall",
        energy_source="Gravity",
        activity="Walking",
        person_in_proximity=False,
        barrier="Housekeeping",
        barrier_status="INTACT",
        severity="LOW",
        life_saving_rules=[],
        rationale="Standard slip hazard, no SIF potential.",
        information_sufficiency=4,
        information_sufficiency_reason="All facts known.",
        requires_followup=False,
        followup_question=None,
        followup_reason=None
    )


# ── Test Cases ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.embed_text", return_value=[0.1]*768)
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
@patch("app.services.triage_service.route_report", new_callable=AsyncMock)
async def test_1_clearly_sif_no_question(mock_route, mock_rag, mock_embed, mock_classify, mock_db, base_gemini_out):
    """Test 1: Clearly SIF report -> no question asked, goes straight to risk engine."""
    out = base_gemini_out.model_copy()
    out.sif_potential = True
    out.severity = "CRITICAL"
    out.barrier_status = "FAILED"
    out.requires_followup = False
    mock_classify.return_value = out

    payload = ReportCreate(report_type="NEAR_MISS", report_text="Fell from 10m scaffold without harness.", location="Rig 1")
    response = await submit_and_analyze(payload, mock_db)

    assert response.status == "ANALYZED"
    assert response.analysis.sif_potential is True
    assert getattr(response, 'followup_question', None) is None
    # Risk engine should have run
    assert response.analysis.risk_score > 60


@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.embed_text", return_value=[0.1]*768)
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
@patch("app.services.triage_service.route_report", new_callable=AsyncMock)
async def test_2_clearly_non_sif_no_question(mock_route, mock_rag, mock_embed, mock_classify, mock_db, base_gemini_out):
    """Test 2: Clearly non-SIF report -> no question asked."""
    mock_classify.return_value = base_gemini_out

    payload = ReportCreate(report_type="UNSAFE_CONDITION", report_text="Small puddle of water.", location="Office")
    response = await submit_and_analyze(payload, mock_db)

    assert response.status == "ANALYZED"
    assert response.analysis.sif_potential is False
    assert getattr(response, 'followup_question', None) is None


@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.embed_text", return_value=[0.1]*768)
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
@patch("app.services.triage_service.route_report", new_callable=AsyncMock)
async def test_3_ambiguous_report_asks_question(mock_route, mock_rag, mock_embed, mock_classify, mock_db, base_gemini_out):
    """Test 3 & 6: Ambiguous report with missing decision-critical info -> exactly one question."""
    out = base_gemini_out.model_copy()
    out.information_sufficiency = 2
    out.requires_followup = True
    out.followup_question = "Was the worker under the suspended load?"
    out.followup_reason = "Need to know proximity to determine SIF exposure."
    mock_classify.return_value = out

    payload = ReportCreate(report_type="NEAR_MISS", report_text="Crane lifted pipe but rigged poorly.", location="Yard")
    response = await submit_and_analyze(payload, mock_db)

    assert response.status == "NEEDS_CLARIFICATION"
    assert response.followup_question == "Was the worker under the suspended load?"
    assert response.analysis is None  # No risk scoring yet


@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
@patch("app.services.triage_service.route_report", new_callable=AsyncMock)
async def test_4_user_answers_clarification(mock_route, mock_rag, mock_classify, mock_db, base_gemini_out):
    """Test 4 & 10: User answers clarification -> reassessment runs, audit trail preserved."""
    # Setup mock DB report in NEEDS_CLARIFICATION state
    mock_report = MagicMock()
    mock_report.id = uuid.uuid4()
    mock_report.status = "NEEDS_CLARIFICATION"
    mock_report.report_text = "Crane lifted pipe but rigged poorly."
    mock_report.embedding = MagicMock()
    mock_report.embedding.embedding = [0.1]*768
    
    mock_clarification = MagicMock()
    mock_clarification.question = "Was the worker under the suspended load?"
    mock_clarification.answer = None
    mock_report.clarifications = [mock_clarification]
    
    mock_analysis = MagicMock()
    mock_report.analysis = mock_analysis

    mock_db.execute.return_value.scalars.return_value.first.return_value = mock_report

    # Mock 2nd pass LLM
    out = base_gemini_out.model_copy()
    out.sif_potential = True
    out.severity = "CRITICAL"
    mock_classify.return_value = out

    payload = ClarificationSubmit(answer="Yes, I was standing directly underneath it.")
    response = await submit_clarification(mock_report.id, payload, mock_db)

    # Asserts
    assert response.status == "ANALYZED"
    assert response.analysis.sif_potential is True
    # Verify the audit trail answer was populated
    assert mock_clarification.answer == "Yes, I was standing directly underneath it."
    # Ensure text was combined for LLM
    combined_text = mock_classify.call_args[0][0]
    assert "ORIGINAL REPORT:" in combined_text
    assert "CLARIFICATION PROVIDED" in combined_text


@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.embed_text", return_value=[0.1]*768)
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
async def test_5_missing_irrelevant_info(mock_rag, mock_embed, mock_classify, mock_db, base_gemini_out):
    """Test 5: Missing info, but irrelevant to SIF -> no question asked."""
    out = base_gemini_out.model_copy()
    out.information_sufficiency = 2  # Low score
    out.requires_followup = False    # But LLM decides it doesn't matter for SIF
    out.followup_question = None
    mock_classify.return_value = out

    payload = ReportCreate(report_type="UNSAFE_ACT", report_text="Guy wasn't wearing his safety glasses.", location="Yard")
    response = await submit_and_analyze(payload, mock_db)

    assert response.status == "ANALYZED"
    assert getattr(response, 'followup_question', None) is None


@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.embed_text", return_value=[0.1]*768)
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
async def test_7_malformed_llm_response(mock_rag, mock_embed, mock_classify, mock_db):
    """Test 7: Malformed LLM response -> gracefully handled by retry/fallback returning ERROR."""
    mock_classify.side_effect = Exception("LLM totally crashed")
    
    payload = ReportCreate(report_type="NEAR_MISS", report_text="Something happened.", location="Yard")
    response = await submit_and_analyze(payload, mock_db)

    assert response.status == "ERROR"
    assert response.analysis.risk_score == 50  # Fallback default


@pytest.mark.asyncio
@patch("app.services.report_service.classify_report")
@patch("app.services.report_service.retrieve_relevant_chunks", return_value=[])
@patch("app.services.triage_service.route_report", new_callable=AsyncMock)
async def test_8_answer_contradicts_initial_assessment(mock_route, mock_rag, mock_classify, mock_db, base_gemini_out):
    """Test 8: Clarification answer contradicts initial assessment -> reassess completely."""
    # Setup report
    mock_report = MagicMock()
    mock_report.id = uuid.uuid4()
    mock_report.status = "NEEDS_CLARIFICATION"
    mock_report.clarifications = [MagicMock(answer=None)]
    mock_report.embedding = MagicMock()
    mock_db.execute.return_value.scalars.return_value.first.return_value = mock_report

    # Initial assessment thought it might be SIF, but answer says "No"
    out = base_gemini_out.model_copy()
    out.sif_potential = False
    out.severity = "LOW"
    mock_classify.return_value = out

    payload = ClarificationSubmit(answer="No, it was a 2-inch pipe resting on the ground, just rolled slightly.")
    response = await submit_clarification(mock_report.id, payload, mock_db)

    assert response.status == "ANALYZED"
    assert response.analysis.sif_potential is False


def test_9_risk_score_is_calculated_only_by_python(base_gemini_out):
    """Test 9: Ensure risk_score is calculated only by Python and not LLM."""
    from app.ai.risk_engine import compute_risk_score
    
    # LLM schema doesn't even have a risk_score field
    assert not hasattr(base_gemini_out, 'risk_score')
    
    score = compute_risk_score(base_gemini_out)
    assert isinstance(score, int)
    assert 0 <= score <= 100