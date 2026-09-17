"""
ai/classifier.py — Tier 1 SIF analysis via Ollama/Qwen3 (text generation).

Embeddings: Gemini (unchanged — see embeddings.py)
Generation: Ollama + Qwen3:4b (via OpenAI-compatible endpoint)

Responsibilities (Qwen):
  - Understand hazard and activity
  - Identify energy source
  - Identify barrier and barrier status
  - Determine potential SIF precursor
  - Identify relevant IOGP Life-Saving Rule
  - Explain reasoning (rationale)
  - Flag requires_followup if uncertain

NOT the LLM responsibility:
  - Final risk level (determined by risk_engine.py)
  - Risk score (computed deterministically)
  - Escalation decisions

Output schema: GeminiAnalysisOutput (schemas/analysis.py)
"""
import json
import logging
from typing import List

from app.ai.ollama_client import generate_ollama_json
from app.core.config import settings
from app.schemas.analysis import GeminiAnalysisOutput

logger = logging.getLogger(__name__)

# ── IOGP Life-Saving Rules summary (always injected — short, fits every prompt) ──

IOGP_RULES_SUMMARY = """
IOGP LIFE-SAVING RULES (9 rules — always check relevance):
1. BYPASSING SAFETY CONTROLS: Obtain authorisation before overriding or disabling safety controls.
2. CONFINED SPACE: Obtain authorisation before entering a confined space.
3. DRIVING: Do not use a phone or exceed speed limits while driving. Wear a seatbelt.
4. ENERGY ISOLATION: Verify isolation and zero energy before work begins.
5. HOT WORK: Obtain authorisation before igniting or working with sources of ignition in hazardous areas.
6. LINE OF FIRE: Keep yourself and others out of the line of fire.
7. SAFE MECHANICAL LIFTING: Conduct a risk assessment and never walk under a suspended load.
8. WORK AT HEIGHT: Obtain authorisation before working at height.
9. H2S: Never work in an area that may contain H2S without breathing equipment.
""".strip()

# ── SIF criteria (explicit rules injected every prompt) ─────────────────────

SIF_CRITERIA = """
SIF PRECURSOR CRITERIA — A report is SIF-potential if ALL THREE apply:
  1. ENERGY SOURCE present: gravitational (height/load), mechanical (moving parts),
     chemical (flammables, toxics, H2S), electrical, pressure, thermal, or kinetic.
  2. PERSON IN PROXIMITY: a person was or could be in the path of harm.
  3. BARRIER FAILED or MISSING: a safety control that should have prevented exposure
     was absent, bypassed, degraded, or failed.

If any one of the three is absent or uncertain, classify as Non-SIF-potential
but set requires_followup=true if you cannot confirm.
""".strip()

# ── Few-shot examples ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1 (SIF-potential: worker beneath suspended load, barrier failed) ---
Report: "During pipe rack installation at Duliajan rig #4, a 400 kg pipe joint slipped
from the crane hook while a rigger was directly below completing the tag-line connection.
The secondary sling was missing. The pipe fell 6 metres but fortunately missed the worker."
Output:
{
  "sif_potential": true,
  "confidence_score": 0.97,
  "hazard": "Falling object / suspended load",
  "energy_source": "Gravitational — 400 kg pipe joint at 6 m elevation",
  "activity": "Pipe rack installation using crane",
  "asset": "Rig #4 crane",
  "location": "Duliajan",
  "barrier": "Secondary sling / tag-line protocol",
  "barrier_status": "FAILED",
  "severity": "CRITICAL",
  "life_saving_rules": ["SAFE MECHANICAL LIFTING"],
  "rationale": "All three SIF criteria met: gravitational energy (400 kg at 6 m), person directly below load, secondary sling completely absent. Near-fatality.",
  "requires_followup": false,
  "followup_question": null
}

--- EXAMPLE 2 (SIF-potential: H2S exposure, PPE absent) ---
Report: "Operator entered the separator area at Nazira without checking H2S monitor
reading. Monitor was found to be reading 45 ppm. Operator had no breathing apparatus."
Output:
{
  "sif_potential": true,
  "confidence_score": 0.95,
  "hazard": "Toxic gas exposure — H2S",
  "energy_source": "Chemical — Hydrogen sulphide at 45 ppm (immediately dangerous above 50 ppm)",
  "activity": "Routine inspection / area access",
  "asset": "Separator unit",
  "location": "Nazira",
  "barrier": "H2S monitor check + breathing apparatus",
  "barrier_status": "FAILED",
  "severity": "CRITICAL",
  "life_saving_rules": ["H2S"],
  "rationale": "H2S at 45 ppm with no SCBA and no pre-entry monitor check. Lethal concentrations possible within seconds. All SIF criteria met.",
  "requires_followup": false,
  "followup_question": null
}

--- EXAMPLE 3 (Non-SIF: minor housekeeping, no energy source) ---
Report: "Oil spill of approximately 2 litres found near the mud pit area.
Area was cordoned off and cleaned within 30 minutes. No personnel were in the area."
Output:
{
  "sif_potential": false,
  "confidence_score": 0.90,
  "hazard": "Slip / environmental contamination",
  "energy_source": "None significant — small volume spill, no ignition source identified",
  "activity": "Housekeeping / spill response",
  "asset": "Mud pit area",
  "location": null,
  "barrier": "Area cordoning",
  "barrier_status": "INTACT",
  "severity": "LOW",
  "life_saving_rules": [],
  "rationale": "No person was in proximity, spill was minor and quickly contained. No energy source capable of causing SIF identified.",
  "requires_followup": false,
  "followup_question": null
}
""".strip()

# ── System prompt template ───────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a Safety Intelligence Assistant for Oil India Limited (OIL),
an upstream oil and gas company operating in Assam, India.

Your job is to analyse a single HSSE (Health, Safety, Security and Environment) observation
report -- which may be an Unsafe Act, Unsafe Condition, or Near-Miss -- and produce a
structured JSON classification of its SIF (Serious Injury and Fatality) potential.

{sif_criteria}

{iogp_rules}

=== RETRIEVED SAFETY KNOWLEDGE (use this to ground your classification) ===
{retrieved_chunks}
=== END RETRIEVED KNOWLEDGE ===

{few_shot_examples}

=== INFORMATION SUFFICIENCY ===
Score information_sufficiency (integer 0-4) based on presence of: (1) Context, (2) Specific hazard, (3) Barrier status, (4) Exposure mechanism.
Set requires_followup=true only when SIF classification is ambiguous and a specific missing fact could flip the outcome.

=== OUTPUT FORMAT ===
Respond ONLY with a single valid JSON object. No markdown, no code fences, no explanation outside the JSON.
Use exactly these fields:
{{
  "sif_potential": <boolean>,
  "confidence_score": <float 0.0-1.0>,
  "hazard": <string>,
  "energy_source": <string>,
  "activity": <string>,
  "person_in_proximity": <boolean or null>,
  "asset": <string or null>,
  "location": <string or null>,
  "barrier": <string>,
  "barrier_status": <"INTACT" | "DEGRADED" | "FAILED" | "UNKNOWN">,
  "severity": <"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">,
  "life_saving_rules": <list of applicable rule names or []>,
  "rationale": <string -- 1-3 sentences citing specific report evidence>,
  "information_sufficiency": <integer 0-4>,
  "information_sufficiency_reason": <string>,
  "requires_followup": <boolean>,
  "followup_question": <string or null>,
  "followup_reason": <string or null>
}}
""".strip()


def _build_messages(report_text: str, knowledge_chunks: List[dict]) -> List[dict]:
    """Assemble structured messages for Ollama chat endpoint."""
    if knowledge_chunks:
        chunks_text = "\n\n".join(
            f"[{i+1}] SOURCE: {c['source']} | {c['title']}\n{c['chunk_text']}"
            for i, c in enumerate(knowledge_chunks)
        )
    else:
        chunks_text = "No specific knowledge chunks retrieved. Rely on the IOGP rules and SIF criteria above."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        sif_criteria=SIF_CRITERIA,
        iogp_rules=IOGP_RULES_SUMMARY,
        retrieved_chunks=chunks_text,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )

    user_prompt = (
        f"=== REPORT TO CLASSIFY ===\n{report_text}\n=== END REPORT ===\n\n"
        f"Respond ONLY with a single valid JSON object directly. No explanations outside the JSON."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_prompt(report_text: str, knowledge_chunks: List[dict]) -> str:
    """Legacy helper assembling raw prompt."""
    msgs = _build_messages(report_text, knowledge_chunks)
    return f"{msgs[0]['content']}\n\n{msgs[1]['content']}"


def _fallback_classification(report_text: str) -> GeminiAnalysisOutput:
    """
    Rule-based heuristic fallback when Ollama is unavailable.
    Never silently returns LOW risk — marks for manual review.
    """
    text_lower = report_text.lower()

    sif_keywords = [
        "h2s", "gas leak", "explosion", "fire", "fall from", "scaffold",
        "suspended load", "crane", "electrocution", "11 kv", "high voltage",
        "confined space", "unconscious", "head injury", "loto", "lockout",
        "pressure release", "blowout", "valve", "tank", "leak"
    ]
    is_sif = any(kw in text_lower for kw in sif_keywords)

    rules = []
    if any(k in text_lower for k in ["h2s", "gas leak", "toxic"]):
        rules.append("H2S")
    if any(k in text_lower for k in ["height", "fall", "scaffold", "ladder"]):
        rules.append("WORK AT HEIGHT")
    if any(k in text_lower for k in ["crane", "load", "lifting", "rigging"]):
        rules.append("SAFE MECHANICAL LIFTING")
    if any(k in text_lower for k in ["electrical", "voltage", "loto", "isolation", "energised"]):
        rules.append("ENERGY ISOLATION")
    if any(k in text_lower for k in ["permit", "bypassed", "override"]):
        rules.append("BYPASSING SAFETY CONTROLS")
    if any(k in text_lower for k in ["confined", "vessel entry"]):
        rules.append("CONFINED SPACE")
    if any(k in text_lower for k in ["hot work", "welding", "ignition"]):
        rules.append("HOT WORK")

    hazard = "Process / Equipment Hazard" if is_sif else "Operational Safety Observation"
    severity = "HIGH" if is_sif else "MEDIUM"

    return GeminiAnalysisOutput(
        sif_potential=is_sif,
        confidence_score=0.75,
        hazard=hazard,
        energy_source="Chemical / Gravitational / Mechanical energy source" if is_sif else "General field operation",
        activity="Field observation / maintenance",
        person_in_proximity=None,
        asset=None,
        location=None,
        barrier="Safety barrier & risk control procedures",
        barrier_status="DEGRADED" if is_sif else "INTACT",
        severity=severity,
        life_saving_rules=rules,
        rationale="Automated heuristic safety classification applied (Ollama AI unavailable).",
        information_sufficiency=4,
        information_sufficiency_reason="Fallback classification assumes sufficient information to apply rules.",
        requires_followup=True,
        followup_question="Ollama AI unavailable. Please review this report manually for full safety verification.",
        followup_reason="Automated fallback triggered due to Ollama unavailability."
    )


async def classify_report(
    report_text: str,
    knowledge_chunks: List[dict],
) -> GeminiAnalysisOutput:
    """
    Send report + RAG context to Ollama/Qwen and return validated GeminiAnalysisOutput.

    Retries once on malformed / invalid JSON.
    Falls back to heuristic rule-based classifier if Ollama is unavailable.
    """
    messages = _build_messages(report_text, knowledge_chunks)

    for attempt in range(2):
        try:
            raw = await generate_ollama_json(
                messages,
                temperature=0.1,
                num_predict=800,
            )

            # Strip code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Find outermost JSON object
            s_idx = raw.find("{")
            e_idx = raw.rfind("}")
            if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                raw = raw[s_idx : e_idx + 1]

            data = json.loads(raw)
            return GeminiAnalysisOutput(**data)

        except Exception as e:
            err_str = str(e)
            if "connection" in err_str.lower() or "refused" in err_str.lower():
                logger.error("Ollama server unavailable (%s). Using fallback classifier.", e)
                return _fallback_classification(report_text)

            if attempt == 0:
                logger.warning(
                    "Ollama/Qwen returned invalid JSON on attempt 1, retrying. Error: %s", e
                )
                continue
            logger.error(
                "Ollama classifier failed after 2 attempts for report (first 100 chars): %s. Error: %s",
                report_text[:100],
                e,
            )
            return _fallback_classification(report_text)

