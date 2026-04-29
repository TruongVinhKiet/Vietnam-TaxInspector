# TaxInspector AI Agent Guide (Internal / Offline)

Generated: 2026-04-28

This document is the **single onboarding reference** for all code related to the internal Tax AI Agent.  
Goal: a future model/agent can read this file, scan the referenced files, and continue development safely.

## 1) What the “AI Agent” is in this repo

The internal agent is a **governed, offline** assistant that:
- infers **intent** from a user message
- retrieves **grounding evidence** (knowledge chunks)
- applies **policy gates** (abstain/escalate vs answer)
- composes a **grounded response** + citations
- logs **every decision** for auditability

Key design principle: **prefer abstain/escalate over low-evidence answers** (tax/legal safety).

## 2) Key entrypoints (read these first)

- **API router**: `Backend/app/routers/tax_agent.py`
  - Endpoint: `POST /api/tax-agent/chat`
  - Responsibilities: session/turn creation, intent routing, retrieval, policy gate, answer composition, trace logging.

- **App auto-migrations**: `Backend/app/main.py`
  - Creates / ensures agent-related tables on startup (idempotent `CREATE TABLE IF NOT EXISTS ...`).

- **DB schema source-of-truth**: `Database/init_db.sql`
  - Contains the DDL for agent tables, retrieval, index governance, and red-team scaffolding.

## 3) Data model (tables) — what gets logged and why

### 3.1 Knowledge corpus (RAG)
- `knowledge_documents`: document metadata (key/title/type/authority/status)
- `knowledge_document_versions`: versioned raw text + parsed json + content hash
- `knowledge_chunks`: chunked text for retrieval (`chunk_key` is unique)
- `knowledge_chunk_embeddings`: embedding vectors stored in JSONB (current: hash-tfidf v1)
- `knowledge_citations`: extracted legal references per chunk (basic extraction)
- `retrieval_logs`: query → retrieved chunk keys + scores + latency

Ingestion script:
- `Backend/app/scripts/ingest_tax_knowledge.py`

Seed/minimal KB (for smoke/eval):
- `Backend/app/scripts/seed_minimal_tax_knowledge.py`

### 3.2 Agent sessions & traces
- `agent_sessions`: session-level metadata
- `agent_turns`: user/assistant turns, including `normalized_intent` + `citations_json`
- `agent_tool_calls`: tool calls used by the agent (e.g., retrieval)
- `agent_decision_traces`: final decision summary: intent, track, abstain/escalate, evidence_json, answer_text

### 3.3 Policy guardrails
- `policy_rules`: current rule definitions (high-level)
- `policy_rule_versions`: version history of rule configs (governance-grade)
- `policy_execution_logs`: per-turn per-rule decision logs

### 3.4 Governance for agent-like systems

- **Prompt registry/versioning** (for future grounded synthesis upgrades):
  - `prompt_registry`, `prompt_versions`, `prompt_rollouts`
  - Seed: `Backend/app/scripts/seed_agent_governance.py`

- **Tool outcomes** (normalize success/failure/side-effects):
  - `tool_execution_outcomes`

- **Adjudication / human-in-the-loop** (disputes + authoritative label):
  - `adjudication_cases`

- **Red-team safety regression**:
  - `redteam_scenarios`, `redteam_run_results`

## 4) Models currently implemented for the agent

### 4.1 Intent model (learned-first, offline)
- Loader: `Backend/ml_engine/tax_agent_intent_model.py`
  - Important implementation detail: **use `model.classes_`** to map `predict_proba` outputs to labels.

- Training: `Backend/ml_engine/train_tax_agent_intent.py`
  - Mode A (bootstrap): generates thousands of Vietnamese paraphrases across intents.
  - Mode B (retrain): if `agent_turns` exists, it pulls supervised examples and blends with bootstrap.
  - Artifacts written to: `Backend/data/models/`
    - `tax_agent_intent_vectorizer.joblib`
    - `tax_agent_intent_model.joblib`
    - `tax_agent_intent_meta.json`

### 4.2 Retrieval (BM25 + dense + lexical)
- Core scoring utilities: `Backend/ml_engine/tax_agent_retrieval.py`
- Used in router: `Backend/app/routers/tax_agent.py`
  - Stage 1: candidate pool from DB (currently limited by SQL `LIMIT 400`)
  - Stage 2: **BM25** + **dense dot** (hash-tfidf embedding) + **lexical overlap**
  - Logging: `retrieval_logs`

### 4.3 Reranker (top-N -> top-k)
- Reranker: `Backend/ml_engine/tax_agent_reranker.py`
  - Currently weights-based (learned-lite placeholder) to keep system fully offline.
  - Future upgrade: replace with a trained reranker model and keep the same API.

## 5) Evaluation + readiness gates

### 5.1 Offline+online eval harness
- Runner: `Backend/app/scripts/run_tax_agent_eval.py`
- Cases: `Backend/app/scripts/tax_agent_eval_cases.jsonl`
- Outputs:
  - DB: `agent_eval_runs` (metrics_json)
  - File: `Backend/reports/tax_agent_eval_latest.json`

Key metrics currently logged:
- `offline.intent_accuracy`
- `offline.retrieval_hit_rate`
- `offline.citation_rate`
- `offline.abstain_rate`
- `online.retrieval_hit_rate_ge2`
- latency p95 metrics

### 5.2 Readiness gates (30/60/90)
- `Backend/reports/tax_agent_readiness_gates.md`

## 6) Index governance (pre-ANN; measurable)

Even before enabling pgvector/FAISS, we track index lifecycle:
- `vector_index_registry`: index record (embedding model/dim/corpus hash)
- `vector_index_quality_runs`: periodic quality logs (hit-rate/latency proxies)

Scripts:
- `Backend/app/scripts/register_vector_index.py`
- `Backend/app/scripts/evaluate_vector_index_quality.py`

## 7) Development workflow (safe iterative)

### Step 0: Ensure schemas exist
- Run the API once (startup auto-migrations), or run scripts that create tables (eval/seed scripts).

### Step 1: Ingest knowledge
- Use `ingest_tax_knowledge.py` for real docs.
- For smoke testing: `seed_minimal_tax_knowledge.py`.

### Step 2: Train intent model
- Run `Backend/ml_engine/train_tax_agent_intent.py`.
- If real traffic exists, ensure `agent_turns` is populated to enable retraining.

### Step 3: Evaluate
- Run `Backend/app/scripts/run_tax_agent_eval.py`.
- Track `agent_eval_runs` and compare run-to-run.

### Step 4: Rollout safely
- Use prompt/policy versioning + red-team regression before canary.

## 8) Known constraints / current limitations
- Knowledge retrieval currently scans a limited recent window (`LIMIT 400`) and ranks in Python.
- Dense embedding is hash-based TFIDF (fast/offline but not semantic-strong).
- Reranker is weights-based (placeholder).
- To reach “enterprise-grade”, next upgrades are:
  - ANN index + true dense embeddings
  - trained reranker
  - grounded synthesis model (offline) with citation faithfulness checks
  - closed-loop learning from feedback/adjudication

