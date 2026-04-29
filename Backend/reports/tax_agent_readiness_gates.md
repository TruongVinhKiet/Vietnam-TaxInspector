# Tax Agent Production Readiness Gates (30/60/90)

Generated: 2026-04-28

This document defines **deployment gates** and **rollback criteria** for the internal tax agent stack (no third-party LLM).

## Scope
- **Intent routing**: `Backend/app/routers/tax_agent.py`
- **Knowledge ingestion**: `Backend/app/scripts/ingest_tax_knowledge.py`
- **Retrieval telemetry**: `retrieval_logs`, `agent_eval_runs`
- **Governance**: `prompt_registry/*`, `policy_rule_versions`, `tool_execution_outcomes`, `adjudication_cases`, `redteam_*`

## Gate Philosophy
- Prefer **abstain/escalate** over low-evidence answers (tax/legal safety).
- Every rollout must be **measurable**: offline eval + online telemetry.
- Any regression beyond threshold triggers **automatic rollback to last good prompt/policy/model version**.

---

## 30 Days — “Pilot-Ready”

### Required Deliverables
- **Eval harness exists and runs**: `agent_eval_runs` has weekly runs for `suite_key=tax_agent_core_v1`.
- **Policy & prompt versioning seeded**: `prompt_registry`, `prompt_versions`, `policy_rule_versions` populated.
- **Red-team seed suite exists**: `redteam_scenarios` has baseline taxonomy coverage.
- **Retrieval index registry exists**: `vector_index_registry` has at least 1 ready index record.

### Quality Gates (minimum)
- **Intent accuracy**: `offline.intent_accuracy >= 0.70` on curated eval suite.
- **Abstain safety**: `offline.abstain_rate` is acceptable and explained (target range depends on KB completeness).
- **Grounding**: `offline.citation_rate >= 0.85` (answers produced only when citations exist).
- **Latency**: `offline.retrieval_latency_p95_ms <= 250ms` on pilot DB size.

### Rollback Triggers
- **Any** increase of unsafe answers (non-abstain with <2 citations) above 1%.
- Intent accuracy drop \(>\) 10% relative.
- Retrieval p95 latency \(>\) 2× baseline for 2 consecutive runs.

---

## 60 Days — “Shadow + Canary”

### Required Deliverables
- **Shadow comparisons**: run prompt/policy variants via `prompt_rollouts` (staging) and log outcomes.
- **Index quality runs**: `vector_index_quality_runs` produced weekly; includes hit-rate and latency.
- **Adjudication loop**: `adjudication_cases` used for disputed/uncertain decisions (human-in-the-loop).
- **Tool outcome logging**: `tool_execution_outcomes` produced for agent tool calls.

### Quality Gates (minimum)
- **Retrieval hit-rate** (online): `online.retrieval_hit_rate_ge2 >= 0.70` over last 14 days.
- **Red-team pass rate**: `>= 0.95` on “high severity” scenarios (must abstain/guard correctly).
- **Stability**: no drift alerts unresolved beyond SLA window (if drift alerting enabled).

### Rollback Triggers
- Any “high severity” red-team failure.
- Online hit-rate drops \(>\) 15% for 3 consecutive days.
- Escalation rate spikes without corresponding policy change record.

---

## 90 Days — “Production-Grade”

### Required Deliverables
- **Champion/Challenger for agent policies**: store/compare best prompt/policy versions with gates.
- **Closed-loop learning queue**: feedback → curated labels → retrain intent/reranker (if added).
- **SLOs defined**: p95 latency, grounding, abstain precision, and safety pass rates.
- **Incident process**: every rollback creates an audit event and links to the failing eval runs.

### Quality Gates (minimum)
- **Safety**: 0 tolerance for privacy exfiltration / instruction override scenarios.
- **Grounding**: citation faithfulness \(>=\) target (measured by adjudicated grounding labels when available).
- **Reliability**: service uptime and latency SLOs met for 30 consecutive days.

### Rollback Triggers
- Any confirmed unsafe disclosure.
- Any regression in safety suite or adjudication-confirmed incorrect legal answer beyond threshold.

