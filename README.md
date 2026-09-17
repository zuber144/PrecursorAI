# PrecursorAI

> **AI-powered safety intelligence system for Oil India Limited (OIL)**
> Real-time SIF precursor detection (Tier 1) + historical cross-report pattern analysis (Tier 2)

---

## What It Does

Oil field workers submit safety observations — unsafe acts, unsafe conditions, near-misses.
PrecursorAI analyses every report in real time, scores it for **Serious Injury and Fatality (SIF)** potential, and sweeps historical reports to find recurring patterns no single report reveals alone.

**The core design principle:** AI reasons and explains. Deterministic code decides.
Gemini classifies hazards. Python computes the final risk score, risk level, and priority. The LLM never makes the final escalation call — every decision is auditable.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite, TailwindCSS, Recharts, Lucide Icons |
| **Backend** | FastAPI (Python 3.11), SQLAlchemy (async), Pydantic v2 |
| **Database** | PostgreSQL 17 + pgvector extension |
| **AI / LLM** | Google Gemini 3.6 Flash (`gemini-3.6-flash`) |
| **Embeddings** | Gemini Embedding 001 (`gemini-embedding-001`) — 3072-dimensional vectors |
| **Vector Search** | pgvector `<=>` cosine distance operator (native PostgreSQL) |

---

## Architecture Overview

```
Frontend (React/Vite :5173)
    |
    v
FastAPI Backend (:8000)
    |
    |-- Tier 1: Real-time single-report pipeline  (on every POST /reports)
    |       |
    |       |-- Preprocessing -> Embedding -> RAG Retrieval
    |       |-- Gemini Classification (hazard, SIF, barrier, severity)
    |       |-- Deterministic Risk Engine (score + level)
    |       `-- Triage & Alerting
    |
    `-- Tier 2: Historical pattern sweep  (on POST /patterns/sweep)
            |
            |-- SQL COUNT queries (Phase B)
            |-- Cosine similarity clustering (Phase C)
            |-- Gemini multi-report reasoning (Phase D)
            |-- Deterministic rules engine (Phase E)
            `-- Pattern persistence (Phase F)
    |
    v
PostgreSQL + pgvector
    |-- reports
    |-- report_analysis
    |-- report_embeddings       <- 3072-dim vectors per report
    |-- knowledge_chunks        <- 438 IOGP/OIL rule chunks
    |-- knowledge_embeddings    <- 3072-dim vectors per chunk
    |-- patterns
    |-- pattern_reports         <- join table (traceability)
    `-- alerts
```

---

## Tier 1 — Real-Time Single-Report Analysis

**Triggered:** Every time a report is submitted via `POST /api/v1/reports`
**Goal:** Within seconds, classify every incoming report for SIF potential and route it appropriately.

### Pipeline (9 steps)

```
Report Text
    |
    v  Step 1: Preprocessing (preprocessing.py)
Strip control chars, collapse whitespace, truncate to 4000 chars
    |
    v  Step 2: Embedding (embeddings.py)
gemini-embedding-001 -> 3072-dimensional float vector
    |
    v  Step 3: Save ReportEmbedding to DB
Stored in report_embeddings table for Tier 2 reuse
    |
    v  Step 4: RAG Retrieval (rag.py)
pgvector cosine similarity: report vector <=> knowledge_embeddings
Top-5 IOGP/OIL safety rule chunks retrieved
    |
    v  Step 5: Gemini Classification (classifier.py)
Structured JSON prompt -> gemini-3.6-flash
8 few-shot examples, injected RAG chunks, SIF criteria, IOGP rules
    |
    v  Step 6: Validate GeminiAnalysisOutput schema
Pydantic validation; retries once on malformed JSON
    |
    v  Step 7: Deterministic Risk Engine (risk_engine.py)
compute_risk_score() + determine_risk_level()  <- AI never does this
    |
    v  Step 8: Save ReportAnalysis to DB
    |
    v  Step 9: Triage & Alerting (triage_service.py)
Route to ANALYZED | REVIEW, create Alert if HIGH or SIF
```

### RAG Knowledge Base

- **438 chunks** split from two IOGP/OIL safety rulebooks
- Ingested via `scripts/ingest_knowledge.py`
- Chunk size: 300 tokens, 50-token overlap
- Stored in `knowledge_chunks` + `knowledge_embeddings` (pgvector)
- At query time: top-5 most-similar chunks injected verbatim into the Gemini prompt

### SIF Criteria (Three-Factor Test)

A report is classified as **SIF-potential** only if **all three** are present:

| Factor | Description |
|---|---|
| **Energy source** | Gravitational, mechanical, chemical (H2S/flammables), electrical, pressure, thermal, or kinetic |
| **Person in proximity** | A person was or could be in the path of harm |
| **Barrier failed/missing** | A safety control that should have prevented exposure was absent, bypassed, degraded, or failed |

If any one factor is absent or uncertain, the AI evaluates information sufficiency (Lara et al. 2024 model). If the missing information is critical to the SIF determination, the system halts scoring, sets `requires_followup = true`, and asks the user **one targeted question**. Once answered, the evaluation resumes.

### Risk Score Formula

The risk score is only computed by the Python engine *after* all necessary clarifications (if any) are resolved:

```
risk_score = barrier_weight + severity_weight + sif_bonus - confidence_penalty
```

**Barrier status weights:**

| Barrier Status | Weight |
|---|---|
| FAILED | +40 |
| DEGRADED | +20 |
| UNKNOWN | +10 |
| INTACT | +0 |

**Severity weights:**

| Severity | Weight |
|---|---|
| CRITICAL | +30 |
| HIGH | +20 |
| MEDIUM | +10 |
| LOW | +5 |

**SIF bonus:** +30 if `sif_potential = true`

**Confidence penalty:** If `confidence_score < 0.6`: subtract `(0.6 - confidence) x 20` points (max -12 pts)

**Score capped at [0, 100]**

### Risk Level Thresholds

| Score Range | Risk Level | Action |
|---|---|---|
| 80-100 | SIF | Alert created, immediate escalation |
| 60-79 | HIGH | Alert created, dashboard flagged |
| 40-59 | REVIEW | Routed to Safety Officer queue |
| 0-39 | ROUTINE | Stored, no immediate escalation |

### Human Review Routing

A report is sent to REVIEW status (requiring Safety Officer action) if either:
- `confidence_score < 0.75` (configured via `CONFIDENCE_THRESHOLD`)
- `requires_followup = true` (AI flagged missing context)

### Gemini Output Fields (Tier 1)

Gemini classifies these fields — **it does NOT set risk level:**

| Field | Description |
|---|---|
| `sif_potential` | Boolean — does this meet the three-factor SIF test? |
| `confidence_score` | 0.0-1.0 — how certain is the classification? |
| `hazard` | Specific hazard type |
| `energy_source` | Energy type and magnitude |
| `activity` | What the worker was doing |
| `barrier` | The safety control that should have prevented harm |
| `barrier_status` | INTACT / DEGRADED / FAILED / UNKNOWN |
| `severity` | LOW / MEDIUM / HIGH / CRITICAL |
| `life_saving_rules` | Which of the 9 IOGP Life-Saving Rules apply |
| `rationale` | 1-3 sentence explanation citing specific report evidence |
| `requires_followup` | True if information is insufficient to classify confidently |
| `followup_question` | The specific question a Safety Officer should ask |

---

## Tier 2 — Historical Pattern Sweep

**Triggered:** `POST /api/v1/patterns/sweep` (on-demand button in dashboard)
**Goal:** Find recurring safety patterns across many stored reports — things no single report reveals alone.

### Pipeline (Phases B-F)

```
Phase B: Deterministic Counting (SQL)
    SELECT asset_id, COUNT(*) FROM reports
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY asset_id HAVING COUNT(*) >= 3
    -> Candidate asset/location shortlist

Phase C: Semantic Clustering (NumPy cosine similarity)
    For each candidate:
    1. Pull stored embeddings from report_embeddings
    2. Compute pairwise cosine similarity matrix
    3. Greedy single-linkage clustering at similarity >= 0.80
    -> Clusters of semantically related reports

Phase D: Gemini Multi-Report Reasoning
    For each cluster: build prompt with ALL report texts + counts
    + 3 few-shot examples (RECURRING / EMERGING / SYSTEMIC)
    -> GeminiPatternOutput: pattern_type, hazard, ai_priority,
                            conclusion, evidence[], confidence

Phase E: Rules Engine Cross-Check (deterministic gate)
    final_priority = _apply_rules_engine(ai_priority, count_30d, count_7d)
    -> Final priority NEVER set by AI alone

Phase F: Persistence
    Save Pattern + PatternReport join rows to DB
    (traceability: every pattern links to contributing report IDs)
```

### Cosine Similarity Formula

```
similarity(a, b) = (a . b) / (||a|| x ||b||)
```

Computed in pure NumPy over 3072-dimensional Gemini embedding vectors.

**Single-linkage clustering:** A report joins a cluster if its cosine similarity to **any existing cluster member** >= `COGNITION_SIMILARITY_THRESHOLD` (default: 0.80).

This catches "oil dripping" and "hydrocarbon accumulation" as the same issue — even with completely different wording.

### Tier 2 Rules Engine (Phase E)

The AI's `ai_priority` is **never used directly**. It is cross-checked:

| Condition | Final Priority |
|---|---|
| AI says HIGH/CRITICAL AND 7-day count >= 2 (accelerating) | CRITICAL |
| AI says HIGH/CRITICAL OR 30-day count >= 5 (frequent) | HIGH |
| AI says MEDIUM OR 30-day count >= 3 (threshold) | MEDIUM |
| Otherwise | LOW |

This double-lock prevents the AI from independently triggering critical alerts.

### Pattern Types

| Type | Meaning |
|---|---|
| RECURRING | Same failure mode repeating on same asset |
| EMERGING | Warning signals escalating in frequency/severity |
| COMPOUNDING | Multiple separate failures on same asset creating combined SIF risk |
| SYSTEMIC | Same procedural failure across many assets/locations |

### Gemini Output Fields (Tier 2)

| Field | Description |
|---|---|
| `pattern_type` | RECURRING / EMERGING / COMPOUNDING / SYSTEMIC |
| `hazard` | The shared hazard across the cluster of reports |
| `ai_priority` | Suggested priority — cross-checked before use |
| `conclusion` | 2-4 sentence explanation of the pattern |
| `evidence` | List of 3-5 specific supporting facts from the reports |
| `confidence` | 0.0-1.0 |

---

## Configuration (backend/.env)

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/precursorai
GEMINI_API_KEY=your_key_here

# Tier 1 thresholds
CONFIDENCE_THRESHOLD=0.75         # Below this -> route to REVIEW
RAG_TOP_K=5                       # Knowledge chunks injected per prompt

# Tier 2 thresholds
COGNITION_MIN_REPORTS=3           # Min reports on same asset to be a candidate
COGNITION_SIMILARITY_THRESHOLD=0.80  # Cosine similarity threshold for clustering
```

---

## API Endpoints

### Reports (Tier 1)
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/reports | Submit report -> triggers full Tier 1 pipeline |
| GET | /api/v1/reports | List all reports (paginated) |
| GET | /api/v1/reports/{id} | Full report detail including AI risk assessment |

### Patterns (Tier 2)
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/patterns/sweep | Trigger full Tier 2 sweep (Phases B-F) |
| GET | /api/v1/patterns | List stored patterns |
| GET | /api/v1/patterns/{id} | Pattern detail with contributing report IDs |

### Dashboard & Alerts
| Method | Endpoint | Description |
|---|---|---|
| GET | /api/v1/dashboard/summary | Live metrics (polled every 7s) |
| GET | /api/v1/alerts | List alerts |
| PATCH | /api/v1/alerts/{id}/read | Mark alert as read |

---

## Running Locally

```powershell
# Backend (Terminal 1)
cd backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --port 8000

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

The database, pgvector extension, all tables, and the 438 RAG knowledge chunks are permanently persisted. No re-setup needed between sessions.

---

## First-Time Setup

```powershell
# 1. Install pgvector
.\install_pgvector.ps1

# 2. Install Python dependencies
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Ingest IOGP knowledge base
python scripts\ingest_knowledge.py

# 4. (Optional) Seed sample reports
python scripts\seed_reports.py
```

---

## Design Principles

1. **AI reasons, Python decides.** Gemini classifies hazards and explains patterns. Risk score, risk level, and final priority are always computed by deterministic Python code.

2. **Double-lock escalation.** A report only becomes CRITICAL or SIF if both the AI assessment AND the deterministic threshold agree.

3. **Full traceability.** Every pattern links back to the exact contributing report IDs and their similarity scores. Every report links to its full AI rationale.

4. **Reuse over rebuild.** Tier 2 reuses the same embedding function and pgvector infrastructure built for Tier 1. No separate vector database.
