---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - taxinspector
  - tax-agent
  - tool-calling
  - lora
  - sft
  - offline-runtime
---

# TaxInspector Tax Agent LoRA V5

## Model Details

This directory contains the local LoRA adapter used by the TaxInspector
multi-agent tax assistant.  The adapter is trained to classify Vietnamese tax
requests, emit constrained tool calls, and support grounded legal/RAG workflows.

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter type: PEFT LoRA
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- LoRA rank/alpha/dropout: `r=16`, `alpha=32`, `dropout=0.05`
- Primary runtime role: planning/tool selection only
- Production runtime default: offline/local artifacts only
- Last model-card update: 2026-05-27

## Intended Use

Use this adapter inside the TaxInspector orchestrator, not as a standalone legal
authority.  The model may suggest one canonical backend tool call; the backend
must still enforce `ToolRegistry`, `AgentModeContractRegistry`, schema
validation, GraphRAG citation checks, compliance gates, and audit logging.

Suitable uses:

- Route officer questions to tax-risk, VAT, debt, macro, OCR, or legal tools.
- Produce structured `<tool_call>` JSON for local backend tools.
- Help synthesize tool-backed answers after the orchestrator has retrieved
evidence.

Out-of-scope uses:

- Final legal decisions without citation/effective-date verification.
- Automatic enforcement actions without officer review.
- Runtime calls to external LLM APIs or remote model repositories.
- General-purpose chatbot use outside the TaxInspector workflow.

## Offline Deployment Policy

Runtime deployments must set no external model-download dependency.  The code
defaults to offline behavior unless `TAX_AGENT_ALLOW_MODEL_DOWNLOAD=1` is set on
a development/training machine.

Expected runtime behavior:

- Load adapter/tokenizer from this directory.
- Load the base model only from a pre-populated local Hugging Face cache or an
  exported local artifact.
- Fall back to deterministic planner/templates if local model files are missing.
- Prefer GGUF/llama.cpp or an OpenAI-compatible local endpoint for CPU serving
  on 12GB RAM machines.

Optional local GGUF server profile:

```powershell
$env:TAX_AGENT_LOCAL_LLM_ENDPOINT = "http://127.0.0.1:8080"
$env:TAX_AGENT_LOCAL_LLM_MODEL = "tax-agent-qwen2.5-1.5b-q4"
```

## Training Data

The current production generator is:

- `Backend/scripts/generate_mega_agent_dataset_v4.py`
- Default output: `Backend/data/agent_ultimate_dataset_v4.jsonl`
- Legacy alias: `Backend/data/agent_ultimate_dataset.jsonl`

The generator now validates every emitted tool call against the canonical
backend tool contract and adds stable `train/dev/test` split metadata.  Deprecated
training names such as `gnn_vat_fraud`, `run_hetero_gnn`, and
`escalate_to_debate` must not appear in new model outputs.

Recommended next release gate:

- At least 1,000 intent/mode cases.
- At least 300 legal-grounding cases.
- At least 200 tool-call cases.
- At least 100 adversarial/prompt-injection cases.
- At least 100 CSV/Excel/OCR workflow cases.

## Evaluation Requirements

A model release is acceptable only if it improves or preserves:

- Tool-call exact match and F1.
- Intent and mode routing accuracy.
- Legal groundedness and citation coverage.
- Hallucination/unsupported-claim rate.
- CPU latency and peak memory on Core i7 8th gen, 12GB RAM.
- No network access during runtime smoke tests.

DPO/RLHF may be run only after enough approved feedback/correction pairs exist
and the post-DPO model passes the same groundedness and safety gates.

## Known Limitations

- A 1.5B model is too small to be the source of all tax knowledge.
- Legal answers must come from GraphRAG/RAG evidence and citation verification.
- The adapter may still emit malformed JSON; runtime schema validation is
  mandatory.
- CPU Transformer inference can be slow; GGUF quantized serving is preferred for
  office workstations.
- Synthetic data can overfit tool names unless evaluated against real anonymized
  officer workflows.

## Artifact Checksums

SHA256:

- `adapter_config.json`: `6B7D8C6F0CDFF1660679E70F7805F1952E721BE95362E6FA2504E4B55B095B26`
- `adapter_model.safetensors`: `3843966C3F2916128BAD45534E03FF024C7FAB77E4F4BE2EB4414C35A513D3BA`
- `tokenizer.json`: `3FD169731D2CBDE95E10BF356D66D5997FD885DD8DBB6FB4684DA3F23B2585D8`
- `tax_agent_lora_v5.zip`: `996DC7AEE9C65ACA2BB39DD11F68A4201C4B2F1E463FE6EF2583F9259369F83A`

## 2026-05-27 Retrain/Export Smoke Artifacts

These artifacts prove the local PEFT/llama.cpp pipeline on the target machine,
but they are not promoted as the production adapter because they were trained
for one CPU step only.

- 1.5B LoRA smoke run:
  `Backend/data/models/tax_agent_lora_v5_retrain_1p5b_onestep`
  - Base: `Qwen/Qwen2.5-1.5B-Instruct`
  - Training data: `Backend/data/agent_ultimate_dataset_v4.jsonl`
  - Examples/epochs: `1` example, `1` epoch
  - Target modules: `q_proj`
  - LoRA rank/alpha: `r=1`, `alpha=2`
  - Trainable/total params: `86,016 / 1,543,800,320`
  - CPU train time: `524.7s`
  - Adapter SHA256: `266162C15DAC4757E922F496C4BD9B88509D69C12F2AA19E07B555556B54817A`

- LoRA GGUF:
  `Backend/data/models/tax_agent_lora_v5_retrain_1p5b_onestep.gguf`
  - Size: `176,480` bytes
  - SHA256: `0AD0458F9B84A8E59E2F40D3499AF97831E7800B272F3896F5EE0340F5644B91`

- Base GGUF for local runtime smoke:
  `Backend/data/models/qwen2_5_0_5b_instruct_q4_k_m.gguf`
  - Source: local cache for `Qwen/Qwen2.5-0.5B-Instruct`
  - Quantization: `Q4_K_M` from F16 using `llama-quantize`
  - Size: `397,807,520` bytes
  - SHA256: `0E1AFF558BE982D8755DD403DE67EECBD65F5616FC5E9ABEEA1EA97B964EE27F`

Telemetry screenshots from the same run:

- `Backend/data/screenshots/telemetry_desktop.png`
- `Backend/data/screenshots/telemetry_mobile.png`
- `Backend/data/screenshots/telemetry_capture_diagnostics.json`

## Reproducibility Notes

Training and export should happen on a development machine or Colab, then the
resulting artifacts should be copied into the on-prem deployment.  Runtime nodes
should not need Hugging Face network access.

For development downloads only:

```powershell
$env:TAX_AGENT_ALLOW_MODEL_DOWNLOAD = "1"
```

For production/offline runtime, leave that variable unset.
