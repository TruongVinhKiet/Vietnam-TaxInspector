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

# TaxInspector Tax Agent LoRA V5 — Production

## Model Details

This directory contains the **production LoRA adapter** used by the TaxInspector
multi-agent tax assistant. The adapter was trained on the full canonical dataset
(101,465 training records + 16,718 evaluation records) using QLoRA on Google
Colab T4 GPU.

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter type: PEFT LoRA (QLoRA 4-bit NF4)
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- LoRA rank/alpha/dropout: `r=16`, `alpha=32`, `dropout=0.05`
- Primary runtime role: planning/tool selection only
- Production runtime default: offline/local artifacts only
- Last model-card update: 2026-06-11

## Training Results (Production V5)

| Metric | Value |
|---|---|
| Total training records | 101,465 |
| Total evaluation records | 16,718 |
| Training steps completed | 6,342 |
| Epochs | 1 |
| **Final Training Loss** | **0.029538** |
| **Final Validation Loss** | **0.135369** |
| Trainable parameters | 4,358,144 / 1,548,072,448 (0.28%) |
| Batch size (effective) | 16 (per_device=2 × grad_accum=8) |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Precision | FP16 (mixed) |
| Quantization (training) | QLoRA NF4 double-quant |
| Max sequence length | 1,024 tokens |
| Hardware | Google Colab T4 GPU (15GB VRAM) |
| Training time (approx.) | ~18 hours across multiple sessions |

### Training Loss Progression

| Checkpoint | Training Loss | Notes |
|---|---|---|
| Step 1,000 | ~0.12 | Early convergence |
| Step 2,000 | ~0.07 | Strong tool-call learning |
| Step 3,000 | ~0.05 | Legal grounding improving |
| Step 4,000 | ~0.04 | Near convergence |
| Step 5,000 | ~0.03 | Stabilized |
| Step 6,000 | ~0.03 | Final plateau |
| **Step 6,342** | **0.0295** | **Final (Validation: 0.1354)** |

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

Dataset composition (130,000 total records):

- Training split: 101,465 records (78%)
- Dev/Eval split: 16,718 records (13%)
- Test split: 11,817 records (9%)

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

## Artifact Checksums (Production V5 Final)

SHA256:

- `adapter_config.json`: `20AA6FF2E87483540449D3E273ABED03B0C2F1AEBB09200B24D12DAC522CD2D5`
- `adapter_model.safetensors`: `BE03521F76C03812BF79FF5DEC58A0AB89D89CB9ED4172FCC6589A17160B8C55`
- `tokenizer.json`: `2F55E63353D3D978B390D346BAE531BE8B83BC9532C0BE500D62B7253AA4C595`

## Checkpoint History

Training was performed across multiple Colab sessions with resume capability:

| Checkpoint | Date | Status |
|---|---|---|
| checkpoint-1000 | 2026-05-28 | ✅ Archived |
| checkpoint-2000 | 2026-05-30 | ✅ Archived |
| checkpoint-3000 | 2026-06-04 | ✅ Archived |
| checkpoint-4000 | 2026-06-05 | ✅ Archived |
| checkpoint-5000 | 2026-06-06 | ✅ Archived |
| checkpoint-6000 | 2026-06-10 | ✅ Archived |
| **final_adapter** | **2026-06-11** | **✅ Production** |

## Reproducibility Notes

Training and export should happen on a development machine or Colab, then the
resulting artifacts should be copied into the on-prem deployment.  Runtime nodes
should not need Hugging Face network access.

For development downloads only:

```powershell
$env:TAX_AGENT_ALLOW_MODEL_DOWNLOAD = "1"
```

For production/offline runtime, leave that variable unset.

### Framework versions

- PEFT 0.19.1
- Transformers (latest as of 2026-06)
- BitsAndBytes (QLoRA NF4)
- PyTorch 2.x (CUDA, FP16)
