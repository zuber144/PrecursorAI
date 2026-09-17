"""
ai/classifier.py — Gemini-powered Tier 1 SIF analysis.

Responsibilities (Gemini):
  - Understand hazard and activity
  - Identify energy source
  - Identify barrier and barrier status
  - Determine potential SIF precursor
  - Identify relevant IOGP Life-Saving Rule
  - Explain reasoning (rationale)
  - Flag requires_followup if uncertain

NOT Gemini's responsibility:
  - Final risk level (determined by risk_engine.py)
  - Risk score (computed deterministically)
  - Escalation decisions

Output schema: GeminiAnalysisOutput (schemas/analysis.py)
"""
import json
import logging
from typing import List

from app.ai.gemini import get_client
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

--- EXAMPLE 3 (SIF-potential: energy isolation not verified) ---
Report: "Maintenance technician began work on the HV transformer at Jorhat substation 
before the LOTO permit was signed. The equipment was still energised at 11 kV."
Output:
{
  "sif_potential": true,
  "confidence_score": 0.98,
  "hazard": "Electrical shock / electrocution",
  "energy_source": "Electrical — 11 kV high voltage",
  "activity": "Electrical maintenance on HV transformer",
  "asset": "HV transformer",
  "location": "Jorhat substation",
  "barrier": "LOTO (Lockout-Tagout) permit",
  "barrier_status": "FAILED",
  "severity": "CRITICAL",
  "life_saving_rules": ["ENERGY ISOLATION", "BYPASSING SAFETY CONTROLS"],
  "rationale": "Worker in direct contact with live 11 kV equipment. LOTO procedure completely bypassed. Electrocution was imminent.",
  "requires_followup": false,
  "followup_question": null
}

--- EXAMPLE 4 (Non-SIF: minor housekeeping, no energy source) ---
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
  "rationale": "No person was in proximity, spill was minor and quickly contained. No energy source capable of causing SIF identified. Routine housekeeping issue.",
  "requires_followup": false,
  "followup_question": null
}

--- EXAMPLE 5 (Non-SIF: driver seatbelt, barrier intact) ---
Report: "Driver observed not wearing seatbelt while driving OIL vehicle on the 
Duliajan field road. He was stopped and counselled. Seatbelt was worn for remainder of journey."
Output:
{
  "sif_potential": false,
  "confidence_score": 0.82,
  "hazard": "Road traffic / vehicle accident",
  "energy_source": "Kinetic — moving vehicle",
  "activity": "Vehicle driving",
  "asset": "OIL field vehicle",
  "location": "Duliajan field road",
  "barrier": "Seatbelt",
  "barrier_status": "FAILED",
  "severity": "MEDIUM",
  "life_saving_rules": ["DRIVING"],
  "rationale": "Barrier (seatbelt) was missing but corrective action was immediate. No near-miss event occurred. Risk is real but does not meet SIF threshold without an actual incident trigger.",
  "requires_followup": false,
  "followup_question": null
}

--- EXAMPLE 6 (Non-SIF: administrative / documentation) ---
Report: "Permit to work was issued 20 minutes after work had already started 
at well #12. Supervisor signed off retroactively. No injuries."
Output:
{
  "sif_potential": false,
  "confidence_score": 0.78,
  "hazard": "Procedural / permit to work violation",
  "energy_source": "Unknown — work type not specified",
  "activity": "Unspecified well work",
  "asset": "Well #12",
  "location": null,
  "barrier": "Permit to Work system",
  "barrier_status": "DEGRADED",
  "severity": "MEDIUM",
  "life_saving_rules": ["BYPASSING SAFETY CONTROLS"],
  "rationale": "PTW issued retroactively is a serious procedural failure. However, the specific work type and energy source are unspecified — cannot confirm SIF potential without knowing the hazard.",
  "requires_followup": true,
  "followup_question": "What specific work was being performed at Well #12 and what energy sources were present?"
}

--- EXAMPLE 7 (Ambiguous — requires followup) ---
Report: "Worker reported feeling dizzy near the gas compression station. 
He rested for 10 minutes and felt better. No formal check was done."
Output:
{
  "sif_potential": true,
  "confidence_score": 0.61,
  "hazard": "Possible gas / toxic exposure",
  "energy_source": "Chemical — possible gas leak near compression station",
  "activity": "Work near gas compression station",
  "asset": "Gas compression station",
  "location": null,
  "barrier": "Gas detection / area monitoring",
  "barrier_status": "UNKNOWN",
  "severity": "HIGH",
  "life_saving_rules": ["H2S"],
  "rationale": "Dizziness near a gas compression station is a red flag for toxic gas or oxygen-deficient atmosphere exposure. Barrier status is unknown — no monitor check was performed. Requires immediate investigation.",
  "requires_followup": true,
  "followup_question": "Was there any gas leak or H2S alarm at the compression station at the time? What was the worker's exact task and location?"
}

--- EXAMPLE 8 (Ambiguous — work at height, incomplete info) ---
Report: "Scaffolding was found with missing planks on level 3 of the 
Jorhat processing unit. Reported by a passerby."
Output:
{
  "sif_potential": true,
  "confidence_score": 0.72,
  "hazard": "Fall from height / falling object",
  "energy_source": "Gravitational — level 3 height (estimated 8-10 m)",
  "activity": "Scaffolding use / access at height",
  "asset": "Scaffolding at Jorhat processing unit",
  "location": "Jorhat processing unit",
  "barrier": "Scaffold planking / working platform",
  "barrier_status": "FAILED",
  "severity": "HIGH",
  "life_saving_rules": ["WORK AT HEIGHT", "SAFE MECHANICAL LIFTING"],
  "rationale": "Missing scaffold planks at approximately 10 m height is a direct fall hazard. If any worker accesses level 3, SIF is likely. Barrier is clearly failed.",
  "requires_followup": true,
  "followup_question": "Is anyone currently working on level 3 of this scaffold? Has work been stopped and the area barricaded?"
}
""".strip()

# \u2500\u2500 System prompt template \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

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

=== INFORMATION SUFFICIENCY ASSESSMENT (Lara et al. 2024 inspired) ===

Before classifying, internally assess the report against these FOUR factors:

  Factor 1 -- CONTEXT/ENVIRONMENT: Is the operational setting, work location, or
              environmental condition described or clearly implied?
  Factor 2 -- HAZARD: Is the specific hazard or threat to life clearly identifiable?
  Factor 3 -- CONTROLS/BARRIERS: Is the status of relevant safety barriers known
              (present, absent, bypassed, degraded)?
  Factor 4 -- CAUSAL/EXPOSURE MECHANISM: Is there enough information to understand
              HOW a person could be harmed (the exposure pathway)?

For each factor, mark it PRESENT, ABSENT, or UNKNOWN.
Count the number of PRESENT factors -> this is information_sufficiency (0-4).

CRITICAL RULE: A low information_sufficiency score does NOT automatically mean
you should ask a question. The ONLY reason to ask is:

  "Is there one specific missing fact whose answer could change the SIF/non-SIF
   determination?"

=== CLARIFICATION DECISION RULES ===

Set requires_followup=true ONLY when BOTH of the following are true:
  1. The SIF classification is genuinely ambiguous -- you cannot make a defensible
     determination from the available information.
  2. There is ONE specific missing piece of information that could materially
     flip the outcome between SIF-potential and non-SIF-potential.

Do NOT ask when:
  * The report clearly meets SIF criteria -> classify and proceed.
  * The report clearly does NOT meet SIF criteria -> classify and proceed.
  * Information is missing but would not change the SIF/non-SIF determination.
  * The classification can be made defensibly at HIGH or CRITICAL with what is given.

Maximum one clarification question per report submission.

=== QUESTION DESIGN RULES (only relevant when requires_followup=true) ===

The followup_question MUST:
  * Ask only for the single highest-value missing fact.
  * Be directly related to SIF determination.
  * Be understandable to a field worker or safety officer.
  * NOT ask for information already present in the report.
  * NOT be a compound question (no "and" joining two separate facts).
  * NOT be a generic prompt like "Can you provide more details?"
  * NOT lead the user toward a SIF answer.

BAD examples (do NOT produce these):
  X  "Can you provide more information about the incident?"
  X  "Was the worker exposed, was the barrier failed, and what was the energy source?"
  X  "Were there any safety controls in place?"

GOOD examples (produce questions like these):
  +  "Was the worker inside the potential drop zone when the pipe was suspended?"
  +  "Was the equipment still pressurised when the leak occurred?"
  +  "Was the isolation barrier in place and confirmed to be functioning at the time?"
  +  "At what height above ground was the worker when they lost footing?"

=== OUTPUT FORMAT ===
Respond ONLY with a single valid JSON object. No markdown, no code fences, no explanation outside the JSON.
Use exactly these fields:
{{
  "sif_potential": <boolean>,
  "confidence_score": <float 0.0-1.0>,
  "hazard": <string -- specific hazard type>,
  "energy_source": <string -- energy source and magnitude if known>,
  "activity": <string -- what was being done>,
  "person_in_proximity": <boolean or null -- was / could a person be in the exposure zone>,
  "asset": <string or null -- equipment/asset involved>,
  "location": <string or null -- site/area if mentioned>,
  "barrier": <string -- safety control that should prevent harm>,
  "barrier_status": <"INTACT" | "DEGRADED" | "FAILED" | "UNKNOWN">,
  "severity": <"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">,
  "life_saving_rules": <list of applicable rule names from the 9 IOGP rules, or []>,
  "rationale": <string -- 1-3 sentence explanation citing specific report evidence>,
  "information_sufficiency": <integer 0-4 -- number of Lara factors marked PRESENT>,
  "information_sufficiency_reason": <string -- one sentence explaining which factors are present/absent/unknown>,
  "requires_followup": <boolean>,
  "followup_question": <string or null -- ONE targeted question, null if requires_followup=false>,
  "followup_reason": <string or null -- why this specific fact is decision-critical, null if requires_followup=false>
}}
""".strip()



def _build_prompt(report_text: str, knowledge_chunks: List[dict]) -> str:
    """Assemble the full prompt with RAG context injected."""
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

    return f"{system_prompt}\n\n=== REPORT TO CLASSIFY ===\n{report_text}\n=== END REPORT ==="


def _fallback_classification(report_text: str) -> GeminiAnalysisOutput:
    """
    Rule-based heuristic fallback classification when Gemini API is unavailable 
    or daily free tier quota is reached (HTTP 429).
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
        rationale="Automated heuristic safety classification applied (Gemini AI API daily quota limit reached).",
        information_sufficiency=4,
        information_sufficiency_reason="Fallback classification assumes sufficient information to apply rules.",
        requires_followup=True,
        followup_question="Gemini API quota exceeded. Please review this report manually for full safety verification.",
        followup_reason="Automated fallback triggered due to quota limit."
    )


async def classify_report(
    report_text: str,
    knowledge_chunks: List[dict],
) -> GeminiAnalysisOutput:
    """
    Send report + RAG context to Gemini and return validated GeminiAnalysisOutput.

    Retries once on malformed / invalid JSON.
    Falls back to heuristic rule-based classifier if Gemini free tier quota is reached (429).
    """
    try:
        model = get_client()
    except Exception as e:
        logger.warning("Could not initialize Gemini client (%s). Using fallback classifier.", e)
        return _fallback_classification(report_text)

    prompt = _build_prompt(report_text, knowledge_chunks)

    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.1,      # Low temperature for consistent, reliable JSON
        "max_output_tokens": 1024,
    }

    for attempt in range(2):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            raw = response.text.strip()

            # Strip accidental code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            return GeminiAnalysisOutput(**data)

        except (json.JSONDecodeError, Exception) as e:
            err_str = str(e)
            if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                logger.warning("Gemini free tier daily quota limit reached (429). Using rule-based fallback classification.")
                return _fallback_classification(report_text)

            if attempt == 0:
                logger.warning(
                    "Gemini returned invalid JSON on attempt 1, retrying. Error: %s", e
                )
                continue
            logger.error(
                "Gemini classifier failed after 2 attempts for report (first 100 chars): %s. Error: %s",
                report_text[:100],
                e,
            )
            return _fallback_classification(report_text)


