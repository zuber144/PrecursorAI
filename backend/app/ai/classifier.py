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

Output schema:
  GeminiAnalysisOutput (schemas/analysis.py)
"""

import json
import logging
from typing import List

from app.ai.ollama_client import generate_ollama_json
from app.schemas.analysis import GeminiAnalysisOutput

logger = logging.getLogger(__name__)


# ============================================================================
# IOGP LIFE-SAVING RULES
# ============================================================================

IOGP_RULES_SUMMARY = """
IOGP LIFE-SAVING RULES (9 rules — always check relevance):

1. BYPASSING SAFETY CONTROLS:
   Obtain authorisation before overriding or disabling safety controls.

2. CONFINED SPACE:
   Obtain authorisation before entering a confined space.

3. DRIVING:
   Do not use a phone or exceed speed limits while driving. Wear a seatbelt.

4. ENERGY ISOLATION:
   Verify isolation and zero energy before work begins.

5. HOT WORK:
   Obtain authorisation before igniting or working with sources of ignition
   in hazardous areas.

6. LINE OF FIRE:
   Keep yourself and others out of the line of fire.

7. SAFE MECHANICAL LIFTING:
   Conduct a risk assessment and never walk under a suspended load.

8. WORK AT HEIGHT:
   Obtain authorisation before working at height.

9. H2S:
   Never work in an area that may contain H2S without breathing equipment.
""".strip()


# ============================================================================
# SIF CRITERIA
# ============================================================================

SIF_CRITERIA = """
SIF PRECURSOR CRITERIA — A report is SIF-potential if ALL THREE apply:

1. ENERGY SOURCE present:
   gravitational (height/load), mechanical (moving parts),
   chemical (flammables, toxics, H2S), electrical, pressure,
   thermal, or kinetic.

2. PERSON IN PROXIMITY:
   A person was or could be in the path of harm.

3. BARRIER FAILED or MISSING:
   A safety control that should have prevented exposure was absent,
   bypassed, degraded, or failed.

If any one of the three is absent or uncertain:
- classify as Non-SIF-potential
- set requires_followup=true if a specific missing fact could change
  the classification.

Do NOT infer person_in_proximity=true merely because the report mentions
"personnel", "workers", or a work area.

Only set person_in_proximity=true when the report provides evidence that
a person was actually exposed, nearby, in the path of harm, or could
reasonably be in the immediate exposure zone.
""".strip()


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

FEW_SHOT_EXAMPLES = """
=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1 (SIF-potential: worker beneath suspended load, barrier failed) ---

Report:
"During pipe rack installation at Duliajan rig #4, a 400 kg pipe joint slipped
from the crane hook while a rigger was directly below completing the tag-line
connection. The secondary sling was missing. The pipe fell 6 metres but
fortunately missed the worker."

Output:
{
  "sif_potential": true,
  "confidence_score": 0.97,
  "hazard": "Falling object / suspended load",
  "energy_source": "Gravitational — 400 kg pipe joint at 6 m elevation",
  "activity": "Pipe rack installation using crane",
  "person_in_proximity": true,
  "asset": "Rig #4 crane",
  "location": "Duliajan",
  "barrier": "Secondary sling / tag-line protocol",
  "barrier_status": "FAILED",
  "severity": "CRITICAL",
  "life_saving_rules": ["SAFE MECHANICAL LIFTING"],
  "rationale": "All three SIF criteria met: gravitational energy (400 kg at 6 m), person directly below load, secondary sling completely absent. Near-fatality.",
  "information_sufficiency": 4,
  "information_sufficiency_reason": "Context, hazard, barrier failure, and exposure mechanism are explicitly described.",
  "requires_followup": false,
  "followup_question": null,
  "followup_reason": null
}


--- EXAMPLE 2 (SIF-potential: H2S exposure, PPE absent) ---

Report:
"Operator entered the separator area at Nazira without checking H2S monitor
reading. Monitor was found to be reading 45 ppm. Operator had no breathing
apparatus."

Output:
{
  "sif_potential": true,
  "confidence_score": 0.95,
  "hazard": "Toxic gas exposure — H2S",
  "energy_source": "Chemical — Hydrogen sulphide at 45 ppm",
  "activity": "Routine inspection / area access",
  "person_in_proximity": true,
  "asset": "Separator unit",
  "location": "Nazira",
  "barrier": "H2S monitor check + breathing apparatus",
  "barrier_status": "FAILED",
  "severity": "CRITICAL",
  "life_saving_rules": ["H2S"],
  "rationale": "The operator entered an area with detected H2S without breathing apparatus and without confirming the monitor reading, creating direct exposure potential.",
  "information_sufficiency": 4,
  "information_sufficiency_reason": "Context, hazard, failed controls, and direct personnel exposure are explicitly described.",
  "requires_followup": false,
  "followup_question": null,
  "followup_reason": null
}


--- EXAMPLE 3 (Non-SIF: minor housekeeping, no energy source) ---

Report:
"Oil spill of approximately 2 litres found near the mud pit area.
Area was cordoned off and cleaned within 30 minutes. No personnel were in
the area."

Output:
{
  "sif_potential": false,
  "confidence_score": 0.90,
  "hazard": "Slip / environmental contamination",
  "energy_source": "None significant — small volume spill, no ignition source identified",
  "activity": "Housekeeping / spill response",
  "person_in_proximity": false,
  "asset": "Mud pit area",
  "location": null,
  "barrier": "Area cordoning",
  "barrier_status": "INTACT",
  "severity": "LOW",
  "life_saving_rules": [],
  "rationale": "No person was in proximity, the spill was minor and quickly contained, and no energy source capable of causing SIF was identified.",
  "information_sufficiency": 4,
  "information_sufficiency_reason": "The report explicitly states that personnel were not in the area and describes the control response.",
  "requires_followup": false,
  "followup_question": null,
  "followup_reason": null
}
""".strip()


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """
You are a Safety Intelligence Assistant for Oil India Limited (OIL),
an upstream oil and gas company operating in Assam, India.

Your job is to analyse a single HSSE (Health, Safety, Security and Environment)
observation report — which may be an Unsafe Act, Unsafe Condition, or Near-Miss —
and produce a structured JSON classification of its SIF
(Serious Injury and Fatality) potential.

{sif_criteria}

{iogp_rules}

=== RETRIEVED SAFETY KNOWLEDGE ===
Use the retrieved knowledge to ground your classification.
Do not invent facts that are not present in the report or retrieved knowledge.

{retrieved_chunks}

=== END RETRIEVED KNOWLEDGE ===

{few_shot_examples}

=== INFORMATION SUFFICIENCY ===

Score information_sufficiency from 0 to 4:

0 = Almost no useful information.
1 = Very limited information.
2 = Some context but important safety facts are missing.
3 = Most important facts are available.
4 = Context, hazard, barrier status, and exposure mechanism are all clear.

Set requires_followup=true ONLY when:
- the SIF classification is genuinely ambiguous, AND
- one specific missing fact could change the classification.

Do not ask unnecessary clarification questions.

=== IMPORTANT CONSISTENCY RULES ===

1. person_in_proximity MUST be based on evidence in the report.
2. Do not infer person_in_proximity=true simply because "workers",
   "personnel", "staff", or "employees" are mentioned.
3. If the report explicitly says a person was nearby, exposed, underneath,
   adjacent to, inside the exposure zone, or otherwise in the path of harm,
   set person_in_proximity=true.
4. If the report explicitly says no personnel were present,
   set person_in_proximity=false.
5. If the report does not provide enough information, set person_in_proximity=null.
6. The rationale must agree with person_in_proximity.
7. The rationale must cite actual evidence from the report.
8. Do not invent measurements, distances, injuries, equipment, or exposure.
9. life_saving_rules must contain only names from the nine IOGP rules supplied above.
10. severity describes the potential consequence of the hazard. It does not
    determine the final risk level.
11. The deterministic risk engine outside the LLM will calculate the final
    risk score and risk level.

=== OUTPUT FORMAT ===

Respond ONLY with a single valid JSON object.

NO markdown.
NO code fences.
NO comments.
NO explanation before or after the JSON.

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
  "rationale": <string — 1-3 sentences citing specific report evidence>,
  "information_sufficiency": <integer 0-4>,
  "information_sufficiency_reason": <string>,
  "requires_followup": <boolean>,
  "followup_question": <string or null>,
  "followup_reason": <string or null>
}}
""".strip()


# ============================================================================
# MESSAGE BUILDERS
# ============================================================================

def _build_messages(
    report_text: str,
    knowledge_chunks: List[dict],
) -> List[dict]:
    """Assemble structured messages for Ollama chat endpoint."""

    if knowledge_chunks:
        chunks_text = "\n\n".join(
            (
                f"[{i + 1}] "
                f"SOURCE: {c.get('source', 'Unknown')} | "
                f"{c.get('title', 'Untitled')}\n"
                f"{c.get('chunk_text', '')}"
            )
            for i, c in enumerate(knowledge_chunks)
        )
    else:
        chunks_text = (
            "No specific knowledge chunks retrieved. "
            "Rely on the IOGP rules and SIF criteria above."
        )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        sif_criteria=SIF_CRITERIA,
        iogp_rules=IOGP_RULES_SUMMARY,
        retrieved_chunks=chunks_text,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )

    user_prompt = (
        "=== REPORT TO CLASSIFY ===\n"
        f"{report_text}\n"
        "=== END REPORT ===\n\n"
        "Return ONLY the JSON object."
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _build_retry_messages(
    report_text: str,
    knowledge_chunks: List[dict],
    previous_response: str,
) -> List[dict]:
    """
    Build a stricter second-attempt prompt.

    The model is explicitly told that its previous response was invalid
    and must return a pure JSON object.
    """

    messages = _build_messages(
        report_text,
        knowledge_chunks,
    )

    messages.append(
        {
            "role": "user",
            "content": (
                "Your previous response was NOT valid JSON.\n\n"
                "Do not explain anything.\n"
                "Do not use markdown.\n"
                "Do not use ``` fences.\n"
                "Do not include <think> tags.\n"
                "Return exactly ONE valid JSON object matching the schema.\n\n"
                "Previous response for correction:\n"
                f"{previous_response[:12000]}"
            ),
        }
    )

    return messages


def _build_prompt(
    report_text: str,
    knowledge_chunks: List[dict],
) -> str:
    """Legacy helper assembling raw prompt."""

    msgs = _build_messages(
        report_text,
        knowledge_chunks,
    )

    return (
        f"{msgs[0]['content']}\n\n"
        f"{msgs[1]['content']}"
    )


# ============================================================================
# QWEN RESPONSE CLEANING
# ============================================================================

def _clean_ollama_response(raw: str) -> str:
    """
    Clean common Qwen/Ollama response artifacts.

    Handles:
      - <think>...</think>
      - Markdown code fences
      - leading/trailing explanatory text
      - JSON surrounded by other text
    """

    if not raw:
        raise ValueError("Ollama returned an empty response.")

    raw = raw.strip()

    # ---------------------------------------------------------
    # Remove Qwen thinking blocks
    # ---------------------------------------------------------

    while "<think>" in raw.lower() and "</think>" in raw.lower():
        lower = raw.lower()

        start = lower.find("<think>")
        end = lower.find("</think>", start)

        if start == -1 or end == -1:
            break

        end += len("</think>")

        raw = (
            raw[:start]
            + raw[end:]
        ).strip()

    # ---------------------------------------------------------
    # Remove markdown fences
    # ---------------------------------------------------------

    if "```" in raw:
        raw = raw.replace("```json", "")
        raw = raw.replace("```JSON", "")
        raw = raw.replace("```", "")
        raw = raw.strip()

    # ---------------------------------------------------------
    # Extract JSON object
    # ---------------------------------------------------------

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No JSON object found in Ollama response."
        )

    raw = raw[start:end + 1].strip()

    return raw


# ============================================================================
# JSON VALIDATION
# ============================================================================

def _parse_and_validate(raw: str) -> GeminiAnalysisOutput:
    """
    Parse JSON and validate it against GeminiAnalysisOutput.

    Raises an exception with useful diagnostics if validation fails.
    """

    cleaned = _clean_ollama_response(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Show a small portion around the failure location.
        start = max(0, e.pos - 250)
        end = min(len(cleaned), e.pos + 250)

        context = cleaned[start:end]

        raise ValueError(
            f"Invalid JSON at character {e.pos}: {e.msg}. "
            f"Context: {context}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Ollama JSON root must be an object, got {type(data).__name__}."
        )

    try:
        return GeminiAnalysisOutput(**data)
    except Exception as e:
        raise ValueError(
            f"JSON schema validation failed: {e}. "
            f"Received keys: {list(data.keys())}"
        ) from e


# ============================================================================
# FALLBACK CLASSIFIER
# ============================================================================

def _fallback_classification(
    report_text: str,
) -> GeminiAnalysisOutput:
    """
    Rule-based heuristic fallback when Ollama is unavailable.

    Never silently returns LOW risk.
    Marks the result for manual review.
    """

    text_lower = report_text.lower()

    sif_keywords = [
        "h2s",
        "gas leak",
        "explosion",
        "fire",
        "fall from",
        "scaffold",
        "suspended load",
        "crane",
        "electrocution",
        "11 kv",
        "high voltage",
        "confined space",
        "unconscious",
        "head injury",
        "loto",
        "lockout",
        "pressure release",
        "blowout",
        "valve",
        "tank",
        "leak",
    ]

    is_sif = any(
        kw in text_lower
        for kw in sif_keywords
    )

    rules = []

    if any(
        k in text_lower
        for k in ["h2s", "gas leak", "toxic"]
    ):
        rules.append("H2S")

    if any(
        k in text_lower
        for k in ["height", "fall", "scaffold", "ladder"]
    ):
        rules.append("WORK AT HEIGHT")

    if any(
        k in text_lower
        for k in ["crane", "load", "lifting", "rigging"]
    ):
        rules.append("SAFE MECHANICAL LIFTING")

    if any(
        k in text_lower
        for k in [
            "electrical",
            "voltage",
            "loto",
            "isolation",
            "energised",
        ]
    ):
        rules.append("ENERGY ISOLATION")

    if any(
        k in text_lower
        for k in [
            "permit",
            "bypassed",
            "override",
        ]
    ):
        rules.append("BYPASSING SAFETY CONTROLS")

    if any(
        k in text_lower
        for k in [
            "confined",
            "vessel entry",
        ]
    ):
        rules.append("CONFINED SPACE")

    if any(
        k in text_lower
        for k in [
            "hot work",
            "welding",
            "ignition",
        ]
    ):
        rules.append("HOT WORK")

    hazard = (
        "Process / Equipment Hazard"
        if is_sif
        else "Operational Safety Observation"
    )

    severity = (
        "HIGH"
        if is_sif
        else "MEDIUM"
    )

    return GeminiAnalysisOutput(
        sif_potential=is_sif,
        confidence_score=0.75,
        hazard=hazard,
        energy_source=(
            "Chemical / Gravitational / Mechanical energy source"
            if is_sif
            else "General field operation"
        ),
        activity="Field observation / maintenance",
        person_in_proximity=None,
        asset=None,
        location=None,
        barrier="Safety barrier & risk control procedures",
        barrier_status=(
            "DEGRADED"
            if is_sif
            else "INTACT"
        ),
        severity=severity,
        life_saving_rules=rules,
        rationale=(
            "Automated heuristic safety classification applied "
            "(Ollama AI unavailable)."
        ),
        information_sufficiency=4,
        information_sufficiency_reason=(
            "Fallback classification assumes sufficient information "
            "to apply rules."
        ),
        requires_followup=True,
        followup_question=(
            "Ollama AI unavailable. Please review this report "
            "manually for full safety verification."
        ),
        followup_reason=(
            "Automated fallback triggered due to Ollama unavailability."
        ),
    )


# ============================================================================
# MAIN CLASSIFIER
# ============================================================================

async def classify_report(
    report_text: str,
    knowledge_chunks: List[dict],
) -> GeminiAnalysisOutput:
    """
    Send report + RAG context to Ollama/Qwen and return validated output.

    Behavior:

      Attempt 1
          ↓
      Qwen response
          ↓
      Clean response
          ↓
      Parse JSON
          ↓
      Validate schema

      If invalid:
          ↓
      Log exact response + error
          ↓
      Attempt 2 with correction prompt

      If Ollama unavailable:
          ↓
      Rule-based fallback

      If both attempts fail:
          ↓
      Rule-based fallback
    """

    messages = _build_messages(
        report_text,
        knowledge_chunks,
    )

    previous_raw = ""

    for attempt in range(2):

        try:
            logger.info(
                "Sending Tier 1 classification request to "
                "Ollama/Qwen (attempt %s/2).",
                attempt + 1,
            )

            raw = await generate_ollama_json(
                messages,
                temperature=0.1,
                num_predict=1200,
            )

            previous_raw = raw or ""

            logger.debug(
                "Raw Ollama/Qwen response on attempt %s:\n%s",
                attempt + 1,
                previous_raw[:12000],
            )

            result = _parse_and_validate(
                previous_raw
            )

            logger.info(
                "Ollama/Qwen classification successful "
                "on attempt %s.",
                attempt + 1,
            )

            return result

        except Exception as e:

            err_str = str(e)

            # -----------------------------------------------------
            # Ollama unavailable
            # -----------------------------------------------------

            if any(
                phrase in err_str.lower()
                for phrase in [
                    "connection refused",
                    "connection error",
                    "connecterror",
                    "failed to connect",
                    "cannot connect",
                ]
            ):
                logger.error(
                    "Ollama server unavailable: %s. "
                    "Using fallback classifier.",
                    e,
                )

                return _fallback_classification(
                    report_text
                )

            # -----------------------------------------------------
            # First attempt failed
            # -----------------------------------------------------

            if attempt == 0:

                logger.warning(
                    "Ollama/Qwen returned invalid output "
                    "on attempt 1.\n"
                    "Error: %s\n"
                    "Raw response:\n%s",
                    e,
                    previous_raw[:12000],
                )

                messages = _build_retry_messages(
                    report_text,
                    knowledge_chunks,
                    previous_raw,
                )

                continue

            # -----------------------------------------------------
            # Second attempt failed
            # -----------------------------------------------------

            logger.error(
                "Ollama/Qwen classification failed after "
                "2 attempts.\n"
                "Report: %s\n"
                "Error: %s\n"
                "Last raw response:\n%s",
                report_text[:500],
                e,
                previous_raw[:12000],
            )

            return _fallback_classification(
                report_text
            )

    # Defensive fallback
    return _fallback_classification(
        report_text
    )