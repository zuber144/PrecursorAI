"""
services/pattern_service.py — Tier 2 pattern sweep pipeline.

Phases implemented here:
  Phase B  — Deterministic counting: find candidate assets/locations
             that cross the configured report-count threshold.
  Phase C  — Semantic clustering: group reports on the same candidate
             by cosine similarity of their stored embeddings.
  (Phases D-E — AI reasoning + rules engine — live in pattern_analyzer.py
             and are called by run_sweep() below.)
  Phase F  — Persistence: save Pattern + PatternReport rows to the DB.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.ai.embeddings import embed_text
from app.core.config import settings
from app.models.analysis import ReportAnalysis
from app.models.embedding import ReportEmbedding
from app.models.pattern import Pattern, PatternReport
from app.models.report import Report

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class CandidateGroup:
    """A candidate asset/location that crossed the report-count threshold."""
    group_key: str           # e.g. "asset:PUMP-102" or "location:Deck B"
    dimension: str           # "asset_id" or "location"
    value: str               # the actual asset_id or location string
    report_count_7d: int
    report_count_30d: int
    report_ids: List[uuid.UUID] = field(default_factory=list)


@dataclass
class ReportCluster:
    """A group of semantically similar reports inside one CandidateGroup."""
    candidate: CandidateGroup
    report_ids: List[uuid.UUID]
    similarity_scores: List[float]   # pairwise similarities for the cluster
    avg_similarity: float


# ── Phase B: Deterministic counting ──────────────────────────────────────────

async def find_candidate_groups(db: AsyncSession) -> List[CandidateGroup]:
    """
    Phase B — Steps 1-4.

    Runs COUNT+GROUP BY queries against the reports table for the last 7 and 30
    days. Returns any asset_id or location groups that cross the configured
    COGNITION_MIN_REPORTS threshold (default: 3 reports in 30 days).

    No AI involved — pure deterministic SQL math.
    """
    threshold = settings.COGNITION_MIN_REPORTS
    now = datetime.now(timezone.utc)
    cutoff_30d = now - timedelta(days=30)
    cutoff_7d = now - timedelta(days=7)

    candidates: dict[str, CandidateGroup] = {}

    # --- Count by asset_id ---------------------------------------------------
    asset_stmt = (
        select(
            Report.asset_id,
            func.count(Report.id).label("cnt_30d"),
        )
        .where(
            Report.asset_id.isnot(None),
            Report.created_at >= cutoff_30d,
            Report.status.in_(["ANALYZED", "REVIEW"]),
        )
        .group_by(Report.asset_id)
        .having(func.count(Report.id) >= threshold)
    )
    result = await db.execute(asset_stmt)
    for row in result.all():
        key = f"asset:{row.asset_id}"
        candidates[key] = CandidateGroup(
            group_key=key,
            dimension="asset_id",
            value=row.asset_id,
            report_count_7d=0,
            report_count_30d=row.cnt_30d,
        )

    # --- Count by location ---------------------------------------------------
    loc_stmt = (
        select(
            Report.location,
            func.count(Report.id).label("cnt_30d"),
        )
        .where(
            Report.location.isnot(None),
            Report.created_at >= cutoff_30d,
            Report.status.in_(["ANALYZED", "REVIEW"]),
        )
        .group_by(Report.location)
        .having(func.count(Report.id) >= threshold)
    )
    result = await db.execute(loc_stmt)
    for row in result.all():
        key = f"location:{row.location}"
        if key not in candidates:
            candidates[key] = CandidateGroup(
                group_key=key,
                dimension="location",
                value=row.location,
                report_count_7d=0,
                report_count_30d=row.cnt_30d,
            )

    if not candidates:
        logger.info("[Tier2] No candidate groups found above threshold=%d in 30d.", threshold)
        return []

    # --- Fill 7d counts ------------------------------------------------------
    for cand in candidates.values():
        col = Report.asset_id if cand.dimension == "asset_id" else Report.location
        cnt_7d_stmt = (
            select(func.count(Report.id))
            .where(
                col == cand.value,
                Report.created_at >= cutoff_7d,
                Report.status.in_(["ANALYZED", "REVIEW"]),
            )
        )
        res = await db.execute(cnt_7d_stmt)
        cand.report_count_7d = res.scalar_one_or_none() or 0

    # --- Fetch report IDs for each candidate ---------------------------------
    for cand in candidates.values():
        col = Report.asset_id if cand.dimension == "asset_id" else Report.location
        id_stmt = (
            select(Report.id)
            .where(
                col == cand.value,
                Report.created_at >= cutoff_30d,
                Report.status.in_(["ANALYZED", "REVIEW"]),
            )
            .order_by(Report.created_at.asc())
        )
        res = await db.execute(id_stmt)
        cand.report_ids = [row[0] for row in res.all()]

    logger.info(
        "[Tier2] Phase B complete — %d candidate group(s) above threshold=%d.",
        len(candidates), threshold,
    )
    return list(candidates.values())


# ── Phase C: Semantic clustering ──────────────────────────────────────────────

async def _load_or_create_embeddings(
    candidate: CandidateGroup,
    db: AsyncSession,
) -> dict[uuid.UUID, List[float]]:
    """Load existing Tier 1 embeddings and backfill historical reports that lack one."""
    if not candidate.report_ids:
        return {}

    stmt = select(ReportEmbedding).where(
        ReportEmbedding.report_id.in_(candidate.report_ids)
    )
    result = await db.execute(stmt)
    existing = result.scalars().all()

    id_to_vec: dict[uuid.UUID, List[float]] = {}
    for item in existing:
        vector = _normalise_embedding(item.embedding)
        if vector:
            id_to_vec[item.report_id] = vector

    missing_ids = [rid for rid in candidate.report_ids if rid not in id_to_vec]

    if missing_ids:
        logger.warning(
            "[Tier2] Candidate '%s' — %d/%d reports have no embedding. "
            "Backfilling historical embeddings.",
            candidate.group_key,
            len(missing_ids),
            len(candidate.report_ids),
        )

        text_stmt = select(Report.id, Report.report_text).where(
            Report.id.in_(missing_ids)
        )
        result = await db.execute(text_stmt)

        for report_id, report_text in result.all():
            try:
                vector = _normalise_embedding(await embed_text(report_text))
                if not vector:
                    logger.error(
                        "[Tier2] Empty embedding generated for report %s",
                        report_id,
                    )
                    continue

                db.add(
                    ReportEmbedding(
                        report_id=report_id,
                        embedding=vector,
                        model="gemini-embedding-001",
                    )
                )
                id_to_vec[report_id] = vector
                logger.info(
                    "[Tier2] Backfilled embedding for report %s",
                    report_id,
                )
            except Exception:
                logger.exception(
                    "[Tier2] Failed to backfill embedding for report %s",
                    report_id,
                )

        await db.flush()

    logger.info(
        "[Tier2] Candidate '%s' — usable embeddings: %d/%d",
        candidate.group_key,
        len(id_to_vec),
        len(candidate.report_ids),
    )
    return id_to_vec


def _normalise_embedding(value) -> List[float]:
    """Convert pgvector/list/numpy/string values to a plain float list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-numpy cosine similarity between two embedding vectors."""
    va = np.asarray(_normalise_embedding(a), dtype=np.float32)
    vb = np.asarray(_normalise_embedding(b), dtype=np.float32)

    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


async def cluster_candidate_reports(
    candidate: CandidateGroup,
    db: AsyncSession,
) -> List[ReportCluster]:
    """Cluster candidate reports using true greedy single-linkage semantics."""
    sim_threshold = float(settings.COGNITION_SIMILARITY_THRESHOLD)
    id_to_vec = await _load_or_create_embeddings(candidate, db)

    if len(id_to_vec) < 2:
        logger.warning(
            "[Tier2] Candidate '%s' has only %d usable embedding(s).",
            candidate.group_key,
            len(id_to_vec),
        )
        return []

    report_ids = [rid for rid in candidate.report_ids if rid in id_to_vec]
    n = len(report_ids)

    logger.info(
        "[Tier2] Candidate '%s' — comparing %d embeddings at threshold %.2f",
        candidate.group_key,
        n,
        sim_threshold,
    )

    sim_matrix: dict[tuple[int, int], float] = {}
    max_similarity = -1.0
    max_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            score = _cosine_similarity(id_to_vec[report_ids[i]], id_to_vec[report_ids[j]])
            sim_matrix[(i, j)] = score
            sim_matrix[(j, i)] = score
            if score > max_similarity:
                max_similarity = score
                max_pair = (report_ids[i], report_ids[j])

    logger.info(
        "[Tier2] Candidate '%s' — max pair similarity=%.4f, threshold=%.4f",
        candidate.group_key,
        max_similarity,
        sim_threshold,
    )

    if max_pair:
        logger.debug(
            "[Tier2] Max similarity pair: %s <-> %s",
            max_pair[0],
            max_pair[1],
        )

    visited: set[int] = set()
    clusters: List[ReportCluster] = []

    for seed in range(n):
        if seed in visited:
            continue

        members = [seed]
        join_scores = [1.0]
        visited.add(seed)

        # Expand until no new member can be connected to any current member.
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if j in visited:
                    continue

                best_score = max(
                    (sim_matrix.get((k, j), 0.0) for k in members),
                    default=0.0,
                )

                if best_score >= sim_threshold:
                    members.append(j)
                    join_scores.append(best_score)
                    visited.add(j)
                    changed = True

        # Tier 2 patterns must contain multiple reports.
        if len(members) < 2:
            continue

        cluster_ids = [report_ids[i] for i in members]
        pairwise_scores = [
            sim_matrix.get((members[a], members[b]), 0.0)
            for a in range(len(members))
            for b in range(a + 1, len(members))
        ]

        clusters.append(
            ReportCluster(
                candidate=candidate,
                report_ids=cluster_ids,
                similarity_scores=join_scores,
                avg_similarity=float(np.mean(pairwise_scores)) if pairwise_scores else 1.0,
            )
        )

    logger.info(
        "[Tier2] Candidate '%s' — %d usable report(s) -> %d multi-report cluster(s)",
        candidate.group_key,
        n,
        len(clusters),
    )

    return clusters


# ── Phase F: Persistence ──────────────────────────────────────────────────────

async def save_pattern(
    db: AsyncSession,
    *,
    pattern_type: str,
    title: str,
    description: str,
    asset_id: Optional[str],
    location: Optional[str],
    hazard: Optional[str],
    priority: str,
    confidence: float,
    evidence: list,
    report_count: int,
    cluster: ReportCluster,
) -> Pattern:
    """
    Phase F — Step 14.

    Persist a Pattern + PatternReport join rows to the database.
    Returns the saved Pattern ORM object.
    """
    # Compute first/last seen from contributing reports
    dates_stmt = (
        select(func.min(Report.created_at), func.max(Report.created_at))
        .where(Report.id.in_(cluster.report_ids))
    )
    res = await db.execute(dates_stmt)
    first_seen, last_seen = res.one()

    pattern = Pattern(
        pattern_type=pattern_type,
        title=title,
        description=description,
        asset_id=asset_id,
        location=location,
        hazard=hazard,
        priority=priority,
        confidence=confidence,
        report_count=report_count,
        first_seen=first_seen,
        last_seen=last_seen,
        evidence=evidence,
        status="ACTIVE",
    )
    db.add(pattern)
    await db.flush()  # get pattern.id without full commit

    # Link contributing reports with their similarity scores
    for i, rid in enumerate(cluster.report_ids):
        sim_score = (
            cluster.similarity_scores[i]
            if i < len(cluster.similarity_scores)
            else None
        )
        db.add(PatternReport(
            pattern_id=pattern.id,
            report_id=rid,
            similarity_score=sim_score,
        ))

    await db.commit()
    await db.refresh(pattern)
    logger.info("[Tier2] Pattern saved: %s (priority=%s)", pattern.title, pattern.priority)
    return pattern


# ── Pattern queries ───────────────────────────────────────────────────────────

async def list_patterns(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
) -> List[Pattern]:
    """Return stored patterns ordered by most recent, with report links eager-loaded."""
    stmt = (
        select(Pattern)
        .options(selectinload(Pattern.report_links))
        .order_by(Pattern.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    if status:
        stmt = stmt.where(Pattern.status == status)
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_pattern(db: AsyncSession, pattern_id: uuid.UUID) -> Optional[Pattern]:
    """Return a single pattern by ID with report links eager-loaded."""
    stmt = (
        select(Pattern)
        .options(selectinload(Pattern.report_links))
        .where(Pattern.id == pattern_id)
    )
    result = await db.execute(stmt)
    return result.scalars().first()


# ── Phase D+E orchestrator: run_sweep() ───────────────────────────────────────

async def run_sweep(db: AsyncSession) -> dict:
    """
    Full Tier 2 sweep — Phase B through E.

    Orchestrates:
      Phase B: find_candidate_groups()       — deterministic SQL counting
      Phase C: cluster_candidate_reports()   — cosine similarity clustering
      Phase D: analyze_cluster()             — Gemini multi-report reasoning
      Phase E: rules engine cross-check      — deterministic priority gate
      Phase F: save_pattern()                — persist to DB

    Returns a summary dict for SweepResult.
    """
    from app.ai.pattern_analyzer import analyze_cluster
    from app.models.report import Report
    from sqlalchemy import select as sa_select

    logger.info("[Tier2] === Sweep started ===")

    # ── Phase B: Deterministic counting ──────────────────────────────────────
    candidates = await find_candidate_groups(db)
    if not candidates:
        logger.info("[Tier2] Sweep complete — no candidates above threshold.")
        return {
            "candidates_found": 0,
            "clusters_analysed": 0,
            "patterns_saved": 0,
            "pattern_ids": [],
        }

    # ── Phase C: Semantic clustering ──────────────────────────────────────────
    all_clusters: List[ReportCluster] = []
    for cand in candidates:
        clusters = await cluster_candidate_reports(cand, db)
        all_clusters.extend(clusters)

    if not all_clusters:
        logger.warning("[Tier2] Candidates were found, but no multi-report semantic clusters met the similarity threshold.")
        return {
            "candidates_found": len(candidates),
            "clusters_analysed": 0,
            "patterns_saved": 0,
            "pattern_ids": [],
        }

    # ── Phase D + E: AI reasoning + rules engine cross-check ─────────────────
    saved_ids: List[uuid.UUID] = []

    for cluster in all_clusters:
        cand = cluster.candidate

        # Fetch the actual report texts for this cluster
        texts_stmt = (
            sa_select(Report.id, Report.report_text)
            .where(Report.id.in_(cluster.report_ids))
            .order_by(Report.created_at.asc())
        )
        text_result = await db.execute(texts_stmt)
        rows = text_result.all()
        report_texts = [r.report_text for r in rows]

        if not report_texts:
            logger.warning("[Tier2] Cluster %s has no retrievable report texts — skipping.", cand.group_key)
            continue

        # Phase D: call Gemini
        try:
            gemini_out = await analyze_cluster(cluster, report_texts)
        except ValueError as e:
            logger.error("[Tier2] Skipping cluster %s — Gemini failed: %s", cand.group_key, e)
            continue

        # ── Phase E: Rules engine cross-check ────────────────────────────────
        # Principle: only escalate to CRITICAL if BOTH:
        #   (a) AI says HIGH or CRITICAL  AND
        #   (b) Deterministic 7d count >= 2 (pattern is accelerating)
        # Otherwise cap at HIGH. This prevents AI-only escalation.
        ai_priority = gemini_out.ai_priority
        final_priority = _apply_rules_engine(
            ai_priority=ai_priority,
            report_count_30d=cand.report_count_30d,
            report_count_7d=cand.report_count_7d,
        )

        # Build a short, human-readable title
        dim_label = cand.dimension.replace("_", " ").title()
        title = f"{gemini_out.pattern_type.title()} pattern — {dim_label}: {cand.value}"

        # Phase F: persist
        pattern = await save_pattern(
            db,
            pattern_type=gemini_out.pattern_type,
            title=title,
            description=gemini_out.conclusion,
            asset_id=cand.value if cand.dimension == "asset_id" else None,
            location=cand.value if cand.dimension == "location" else None,
            hazard=gemini_out.hazard,
            priority=final_priority,
            confidence=gemini_out.confidence,
            evidence=gemini_out.evidence,
            report_count=len(cluster.report_ids),
            cluster=cluster,
        )
        saved_ids.append(pattern.id)

    logger.info(
        "[Tier2] === Sweep complete: %d candidate(s), %d cluster(s), %d pattern(s) saved ===",
        len(candidates), len(all_clusters), len(saved_ids),
    )
    return {
        "candidates_found": len(candidates),
        "clusters_analysed": len(all_clusters),
        "patterns_saved": len(saved_ids),
        "pattern_ids": saved_ids,
    }


def _apply_rules_engine(
    ai_priority: str,
    report_count_30d: int,
    report_count_7d: int,
) -> str:
    """
    Phase E — Step 13.

    Cross-checks the AI-suggested priority against deterministic thresholds.
    Returns the final, auditable priority string.

    Rules:
      - CRITICAL requires: AI says HIGH or CRITICAL  AND  7d count >= 2
      - HIGH     requires: AI says HIGH or CRITICAL  OR   30d count >= 5
      - MEDIUM   requires: AI says MEDIUM            OR   30d count >= 3
      - LOW      otherwise
    """
    is_ai_high = ai_priority in ("HIGH", "CRITICAL")
    is_accelerating = report_count_7d >= 2          # accelerating = 2+ reports in last 7d
    is_frequent_30d = report_count_30d >= 5         # frequent = 5+ in 30d

    if is_ai_high and is_accelerating:
        return "CRITICAL"
    elif is_ai_high or is_frequent_30d:
        return "HIGH"
    elif ai_priority == "MEDIUM" or report_count_30d >= settings.COGNITION_MIN_REPORTS:
        return "MEDIUM"
    else:
        return "LOW"

