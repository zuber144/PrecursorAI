"""
ai/pattern_analyzer.py — Tier 2 multi-report pattern reasoning via Gemini.

Responsibilities (Gemini):
  - Understand whether a cluster of reports share the same underlying hazard
  - Classify the pattern type (RECURRING / EMERGING / COMPOUNDING / SYSTEMIC)
  - Provide a plain-English conclusion and evidence list
  - Suggest a priority (cross-checked by the rules engine, NOT used directly)

NOT Gemini's responsibility:
  - Final priority (Phase E rules engine cross-checks this)
  - Whether a cluster becomes a stored pattern (threshold gate does this)
  - Report-level SIF classification (Tier 1's classifier.py handles that)

Output schema: GeminiPatternOutput (schemas/pattern.py)
"""

import json
import logging
from typing import List

from app.ai.gemini import get_client
from app.schemas.pattern import GeminiPatternOutput
from app.services.pattern_service import ReportCluster

logger = logging.getLogger(__name__)


# ── Few-shot examples (multi-report pattern reasoning — different from Tier 1) ─

PATTERN_FEW_SHOT_EXAMPLES = """
=== FEW-SHOT EXAMPLES (multi-report pattern reasoning) ===

--- EXAMPLE A: RECURRING barrier failure on same asset ---
Asset: PUMP-102  |  30-day count: 5  |  7-day count: 2
Reports:
  [1] "Pressure relief valve on PUMP-102 found stuck open during routine check."
  [2] "PUMP-102 relief valve did not reset after pressure test. Had to manually reset."
  [3] "Abnormal noise from PUMP-102 pressure relief system. Valve appears worn."
  [4] "PUMP-102 relief valve leaked past seat during high-pressure operation."
  [5] "Operator bypassed PUMP-102 relief valve due to repeated spurious lifts."

Expected output:
{
  "pattern_type": "RECURRING",
  "hazard": "Pressure barrier failure — relief valve degradation",
  "priority": "CRITICAL",
  "conclusion": "Five reports over 30 days describe the same pressure relief valve on PUMP-102 progressively failing. The final report records an operator bypass — removing the only pressure barrier entirely. This is a compounding SIF precursor: each individual report looks minor but together they trace a valve approaching catastrophic failure.",
  "evidence": [
    "5 reports in 30 days on PUMP-102 — all describing the same relief valve",
    "Progressive deterioration: stuck → won't reset → worn → leaking → bypassed",
    "Operator bypass in Report 5 removed the sole remaining pressure barrier",
    "High-pressure operation context means barrier failure = SIF potential"
  ],
  "confidence": 0.95
}

--- EXAMPLE B: EMERGING pattern across shared location ---
Location: Compressor Hall B  |  30-day count: 4  |  7-day count: 3
Reports:
  [1] "Minor hydrocarbon smell noticed near compressor C-3. No alarm triggered."
  [2] "Worker reported brief dizziness near C-3 inlet. Felt better after leaving area."
  [3] "Oil accumulation observed on floor near C-3 base. Cleaned up."
  [4] "Gas detector in Compressor Hall B gave a brief reading of 8% LEL then reset."

Expected output:
{
  "pattern_type": "EMERGING",
  "hazard": "Hydrocarbon leak — potential gas accumulation near ignition sources",
  "priority": "HIGH",
  "conclusion": "Four separate observations in Compressor Hall B over 30 days — increasing in frequency (3 in the last 7 days) — collectively indicate an emerging hydrocarbon leak from compressor C-3. No single report triggered an alarm, but together they show escalating evidence: smell → physiological effect → liquid accumulation → gas detector activation.",
  "evidence": [
    "4 reports in 30 days, 3 in the last 7 days — accelerating frequency",
    "Hydrocarbon smell, dizziness, liquid accumulation and gas detector all point to same leak source (C-3)",
    "8% LEL gas detector reading indicates real concentration — not a false positive",
    "Compressor hall contains ignition sources — leak to ignition = explosion risk"
  ],
  "confidence": 0.88
}

--- EXAMPLE C: SYSTEMIC pattern across multiple assets/locations ---
Asset: multiple  |  30-day count: 6  |  7-day count: 1
Reports:
  [1] "PTW signed 30 minutes after work started on Well-4."
  [2] "Supervisor approved hot work at Well-7 without checking gas readings first."
  [3] "Energy isolation was not verified before maintenance began on Pump-9."
  [4] "Night shift crew at Processing Unit started vessel entry without confined space permit."
  [5] "Work at height on Tank-Farm T-2 started before scaffolding inspection was complete."
  [6] "LOTO removed on Compressor-1 while technician was still inside machine guard."

Expected output:
{
  "pattern_type": "SYSTEMIC",
  "hazard": "Permit-to-Work and safety procedure bypass — systemic compliance failure",
  "priority": "CRITICAL",
  "conclusion": "Six reports across different assets and locations all describe the same root failure: safety-critical permit and isolation procedures being bypassed or started late. This is not an equipment problem — it is a systemic behavioural and supervisory failure. Each incident individually was a near-miss; together they indicate organisation-wide procedural breakdown that substantially increases SIF probability across all operations.",
  "evidence": [
    "6 incidents across 5 different assets and 4 locations — not confined to one area",
    "All 6 involve bypass or late compliance with safety-critical procedures (PTW, LOTO, confined space, hot work)",
    "Three IOGP Life-Saving Rules violated: ENERGY ISOLATION, BYPASSING SAFETY CONTROLS, CONFINED SPACE",
    "Night shift involved in 2 incidents — possible supervision gap after hours"
  ],
  "confidence": 0.91
}
""".strip()


# ── System prompt for Tier 2 pattern analysis ─────────────────────────────────

PATTERN_SYSTEM_PROMPT = """You are a Safety Pattern Intelligence Engine for Oil India Limited (OIL).

Your job is DIFFERENT from single-report analysis. You receive a GROUP of safety reports
from the same asset or location and must determine whether they reveal a RECURRING or
EMERGING safety pattern that no individual report would reveal alone.

Think like an accident investigator connecting dots — look for:
  - The same failure mode repeating on the same equipment
  - Escalating severity across reports over time
  - The same hazard manifesting in different ways (different wording, same root cause)
  - A systemic procedural failure showing up across multiple assets

=== PATTERN TYPES ===
RECURRING   — Same failure mode repeating on same asset (e.g., 5 valve failures)
EMERGING    — Warning signals escalating in frequency/severity (e.g., smell → reading → leak)
COMPOUNDING — Multiple separate failures on same asset creating combined SIF risk
SYSTEMIC    — Same procedural failure across many assets/locations (organisation-wide)

=== PRIORITY GUIDANCE ===
CRITICAL: Pattern involves active SIF potential AND is accelerating (more reports in last 7d)
HIGH    : Pattern involves SIF potential OR is accelerating, but not both
MEDIUM  : Pattern is confirmed but low severity or stable frequency
LOW     : Weak pattern — similar wording but no clear shared hazard

=== IMPORTANT ===
Your priority suggestion will be cross-checked against deterministic thresholds. Focus on
accuracy of pattern_type and the quality of evidence — these are what humans review.

{few_shot_examples}

=== OUTPUT FORMAT ===
Respond ONLY with a single valid JSON object. No markdown, no code fences.
{{
  "pattern_type": <"RECURRING" | "EMERGING" | "COMPOUNDING" | "SYSTEMIC">,
  "hazard": <string — the specific shared hazard across reports>,
  "priority": <"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">,
  "conclusion": <string — 2-4 sentence explanation of the pattern and its significance>,
  "evidence": <list of 3-5 specific supporting facts as short strings>,
  "confidence": <float 0.0-1.0>
}}
""".strip()


def _build_pattern_prompt(
    cluster: ReportCluster,
    report_texts: List[str],
) -> str:
    """
    Assemble the full prompt for multi-report pattern analysis.

    Injects:
      - Candidate dimension (asset or location) and value
      - Deterministic counts (30d and 7d) — the 'hard evidence'
      - All report texts in the cluster
      - Few-shot examples
    """
    cand = cluster.candidate
    dim_label = "Asset" if cand.dimension == "asset_id" else "Location"

    # Format all reports in the cluster
    reports_block = "\n".join(
        f"  [{i+1}] \"{text.strip()}\""
        for i, text in enumerate(report_texts)
    )

    system_prompt = PATTERN_SYSTEM_PROMPT.format(
        few_shot_examples=PATTERN_FEW_SHOT_EXAMPLES
    )

    user_message = (
        f"=== PATTERN ANALYSIS REQUEST ===\n"
        f"{dim_label}: {cand.value}\n"
        f"Reports in last 30 days: {cand.report_count_30d}\n"
        f"Reports in last 7 days:  {cand.report_count_7d}\n"
        f"Reports in this semantic cluster: {len(cluster.report_ids)}\n"
        f"Average embedding similarity: {cluster.avg_similarity:.2f}\n\n"
        f"REPORT TEXTS:\n{reports_block}\n"
        f"=== END REQUEST ==="
    )

    return f"{system_prompt}\n\n{user_message}"


def _fallback_pattern(cluster: ReportCluster, report_texts: List[str]) -> GeminiPatternOutput:
    cand = cluster.candidate
    return GeminiPatternOutput(
        pattern_type="RECURRING",
        hazard=f"Recurring safety observations on {cand.group_key}",
        priority="HIGH" if cand.report_count_30d >= 5 else "MEDIUM",
        conclusion=f"Cluster of {len(cluster.report_ids)} reports on {cand.group_key} identified over 30 days. Fallback rule-based synthesis applied (Gemini API quota reached).",
        evidence=[
            f"{len(cluster.report_ids)} reports clustered on {cand.group_key}",
            f"{cand.report_count_30d} reports recorded in last 30 days",
            f"Average semantic similarity: {cluster.avg_similarity:.2f}"
        ],
        confidence=0.75
    )


async def analyze_cluster(
    cluster: ReportCluster,
    report_texts: List[str],
) -> GeminiPatternOutput:
    """
    Phase D — Steps 9-12.

    Send a cluster of grouped reports to Gemini for multi-report pattern reasoning.
    Returns a validated GeminiPatternOutput.

    Retries once on malformed JSON.
    Falls back to heuristic pattern output if Gemini API quota is reached (429).
    """
    try:
        model = get_client()
    except Exception as e:
        logger.warning("[Tier2] Could not get Gemini client (%s). Using fallback pattern.", e)
        return _fallback_pattern(cluster, report_texts)

    prompt = _build_pattern_prompt(cluster, report_texts)

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": GeminiPatternOutput,
        "temperature": 0.15,      # Slightly higher than Tier 1 for nuanced reasoning
        "max_output_tokens": 1024,
    }

    for attempt in range(2):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            raw = response.text.strip()

            # Strip accidental code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            return GeminiPatternOutput(**data)

        except (json.JSONDecodeError, Exception) as e:
            err_str = str(e)
            if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                logger.warning("[Tier2] Gemini free tier quota limit reached (429). Using fallback pattern synthesis.")
                return _fallback_pattern(cluster, report_texts)

            if attempt == 0:
                logger.warning(
                    "[Tier2] Gemini returned invalid JSON on attempt 1, retrying. "
                    "Cluster: %s. Error: %s",
                    cluster.candidate.group_key, e,
                )
                continue
            logger.error(
                "[Tier2] Pattern analyzer failed after 2 attempts for cluster %s. Error: %s",
                cluster.candidate.group_key, e,
            )
            return _fallback_pattern(cluster, report_texts)

