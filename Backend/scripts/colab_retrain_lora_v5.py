"""
Google Colab LoRA V5 Production Training Script
================================================
Copy file này vào Colab notebook, chạy từng cell.
GPU: T4 (free tier) hoặc tốt hơn.

Hướng dẫn:
1. Upload agent_ultimate_dataset_v5.jsonl lên Google Drive
2. Mount Drive trong Colab
3. Chạy từng section bên dưới

Trước khi upload lên Colab, chạy trên máy local:
  cd e:\\TaxInspector\\Backend
  python scripts/generate_mega_agent_dataset_v4.py --total-simple 90000 --total-legal 30000
  => Output: data/agent_ultimate_dataset_v4.jsonl (~482MB)
  => Copy file này lên Google Drive
"""

# ============================================================
# CELL 1: Cài đặt dependencies
# ============================================================
# !pip install -q torch transformers peft datasets accelerate bitsandbytes sentencepiece

# ============================================================
# CELL 2: Mount Google Drive & kiểm tra GPU
# ============================================================
COLAB_SCRIPT = '''
import torch, os, json, time, hashlib
from pathlib import Path

# --- Kiểm tra GPU ---
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("VRAM:", f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A")

# --- Mount Drive ---
from google.colab import drive
drive.mount("/content/drive")

# --- Paths ---
DRIVE_DIR = Path("/content/drive/MyDrive/TaxInspector")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = DRIVE_DIR / "agent_ultimate_dataset_v4.jsonl"
OUTPUT_DIR = Path("/content/tax_agent_lora_v5_production")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}. Upload it first!"
print(f"Dataset: {DATASET_PATH} ({DATASET_PATH.stat().st_size / 1e6:.1f}MB)")
'''

# ============================================================
# CELL 3: Load & validate dataset
# ============================================================
LOAD_DATASET_SCRIPT = '''
import json, re
from collections import Counter

# --- Canonical tools (must match tax_agent_tool_contracts.py) ---
CANONICAL_TOOL_NAMES = frozenset({
    "knowledge_search", "company_risk_lookup", "delinquency_check",
    "invoice_risk_scan", "vat_refund_risk", "gnn_analysis",
    "motif_detection", "ring_scoring", "ownership_analysis",
    "temporal_delinquency_deep", "hetero_gnn_risk", "vae_anomaly_scan",
    "causal_uplift_recommend", "top_n_risky_companies", "company_name_search",
    "nlp_red_flag_scan", "revenue_forecast", "entity_resolution_check",
    "ocr_document_process", "macro_forecast",
})

DEPRECATED_ALIASES = {
    "gnn_vat_fraud": "gnn_analysis",
    "run_hetero_gnn": "hetero_gnn_risk",
    "run_vae_anomaly": "vae_anomaly_scan",
    "predict_delinquency": "temporal_delinquency_deep",
    "causal_uplift_action": "causal_uplift_recommend",
    "query_legal_graphrag": "knowledge_search",
    "run_macro_simulation": "macro_forecast",
}

records = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Total records: {len(records)}")

# Validate tool calls
invalid_count = 0
deprecated_count = 0
for r in records:
    for msg in r.get("messages", r.get("conversations", [])):
        content = str(msg.get("content", msg.get("value", "")))
        m = re.search(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", content, re.DOTALL)
        if not m:
            continue
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name", "")
            if name in DEPRECATED_ALIASES:
                deprecated_count += 1
            elif name not in CANONICAL_TOOL_NAMES:
                invalid_count += 1
        except json.JSONDecodeError:
            invalid_count += 1

splits = Counter(r.get("metadata", {}).get("split", "unknown") for r in records)
kinds = Counter(r.get("metadata", {}).get("kind", "unknown") for r in records)

print(f"Splits: {dict(splits)}")
print(f"Kinds: {dict(kinds)}")
print(f"Deprecated tool names: {deprecated_count}")
print(f"Invalid tool calls: {invalid_count}")

if deprecated_count > 0 or invalid_count > 0:
    print("\\n⚠️ DATASET CẦN REGENERATE! Chạy generate_mega_agent_dataset_v4.py trên máy local trước.")
else:
    print("\\n✅ Dataset hợp lệ, sẵn sàng train.")
'''

# ============================================================
# CELL 4: Prepare training data
# ============================================================
PREPARE_SCRIPT = '''
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024  # Tăng lên 2048 nếu VRAM cho phép

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def record_to_text(record):
    """Convert JSONL record to training text."""
    messages = record.get("messages")
    if messages:
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[SYSTEM]\\n{content}\\n[/SYSTEM]")
            elif role == "user":
                parts.append(f"[USER]\\n{content}\\n[/USER]")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\\n{content}")
            elif role == "tool":
                parts.append(f"[TOOL_RESULT]\\n{content}\\n[/TOOL_RESULT]")
        return "\\n".join(parts)

    # Legacy conversations format
    convos = record.get("conversations", [])
    if len(convos) >= 3:
        return (
            f"[SYSTEM]\\n{convos[0].get('value','')}\\n[/SYSTEM]\\n"
            f"[USER]\\n{convos[1].get('value','')}\\n[/USER]\\n"
            f"[ASSISTANT]\\n{convos[2].get('value','')}"
        )
    return None

# Filter by split
train_records = [r for r in records if r.get("metadata", {}).get("split", "train") == "train"]
eval_records = [r for r in records if r.get("metadata", {}).get("split") in ("dev", "eval", "validation")]

# Fallback split if no metadata
if not eval_records and len(train_records) == len(records):
    split_idx = int(len(records) * 0.9)
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

print(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

# Convert to text
train_texts = [t for t in (record_to_text(r) for r in train_records) if t]
eval_texts = [t for t in (record_to_text(r) for r in eval_records) if t]
print(f"Train texts: {len(train_texts)}, Eval texts: {len(eval_texts)}")

# Tokenize
MAX_TRAIN = 50000  # Giới hạn để không OOM, tăng nếu VRAM đủ
train_texts = train_texts[:MAX_TRAIN]

print("Tokenizing...")
train_enc = tokenizer(
    train_texts, truncation=True, padding=True,
    max_length=MAX_SEQ_LENGTH, return_tensors="pt",
)
train_enc["labels"] = train_enc["input_ids"].clone()

eval_enc = tokenizer(
    eval_texts[:2000], truncation=True, padding=True,
    max_length=MAX_SEQ_LENGTH, return_tensors="pt",
)
eval_enc["labels"] = eval_enc["input_ids"].clone()

class TokenizedDataset(TorchDataset):
    def __init__(self, enc):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = enc["labels"]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

train_dataset = TokenizedDataset(train_enc)
eval_dataset = TokenizedDataset(eval_enc)
print(f"Tokenized train: {len(train_dataset)}, eval: {len(eval_dataset)}")
'''

# ============================================================
# CELL 5: Train LoRA
# ============================================================
TRAIN_SCRIPT = '''
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# --- Load base model ---
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# --- LoRA config ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

# --- Training args ---
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,              # 3-5 epochs cho production
    per_device_train_batch_size=4,   # T4: 4, A100: 16
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch = 16
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=50,
    gradient_checkpointing=True,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# --- Train ---
print("\\n🚀 Starting LoRA training...")
t0 = time.time()
trainer.train()
duration = time.time() - t0
print(f"\\n✅ Training complete in {duration/60:.1f} minutes")

# --- Save adapter ---
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Adapter saved to {OUTPUT_DIR}")
'''

# ============================================================
# CELL 6: Evaluate
# ============================================================
EVAL_SCRIPT = '''
import re, json
from collections import Counter

# Quick eval on test split
test_records = [r for r in records if r.get("metadata", {}).get("split") == "test"]
if not test_records:
    test_records = eval_records[:200]
print(f"Evaluating on {len(test_records)} test records...")

model.eval()
correct_tool = 0
total_tool = 0
correct_intent = 0

for i, rec in enumerate(test_records[:200]):
    expected_tool = rec.get("metadata", {}).get("expected_tool")
    if not expected_tool:
        continue

    user_msg = ""
    for msg in rec.get("messages", []):
        if msg.get("role") == "user":
            user_msg = msg["content"]
            break
    if not user_msg:
        continue

    # Generate
    messages = [
        {"role": "system", "content": rec["messages"][0]["content"]},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256,
            temperature=0.1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse tool call
    m = re.search(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", response, re.DOTALL)
    if m:
        try:
            payload = json.loads(m.group(1))
            predicted_tool = payload.get("name", "")
            # Canonicalize
            predicted_tool = DEPRECATED_ALIASES.get(predicted_tool, predicted_tool)
            expected_canonical = DEPRECATED_ALIASES.get(expected_tool, expected_tool)

            total_tool += 1
            if predicted_tool == expected_canonical:
                correct_tool += 1
        except json.JSONDecodeError:
            total_tool += 1

    if (i + 1) % 50 == 0:
        print(f"  Evaluated {i+1}/{min(200, len(test_records))}...")

tool_accuracy = correct_tool / max(1, total_tool)
print(f"\\n📊 Evaluation Results:")
print(f"   Tool-call accuracy: {correct_tool}/{total_tool} = {tool_accuracy:.1%}")
print(f"   (Target: >= 85%)")

# Save eval report
eval_report = {
    "total_test": len(test_records),
    "evaluated": min(200, len(test_records)),
    "tool_call_correct": correct_tool,
    "tool_call_total": total_tool,
    "tool_call_accuracy": round(tool_accuracy, 4),
    "base_model": BASE_MODEL,
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": 3,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
eval_path = OUTPUT_DIR / "eval_report.json"
eval_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False))
print(f"   Report saved: {eval_path}")
'''

# ============================================================
# CELL 7: Export & copy to Drive
# ============================================================
EXPORT_SCRIPT = '''
import shutil

# --- Copy adapter to Drive ---
DRIVE_OUTPUT = DRIVE_DIR / "tax_agent_lora_v5_production"
if DRIVE_OUTPUT.exists():
    shutil.rmtree(DRIVE_OUTPUT)
shutil.copytree(OUTPUT_DIR, DRIVE_OUTPUT)
print(f"✅ Adapter copied to Google Drive: {DRIVE_OUTPUT}")

# --- Compute checksums ---
for f in sorted(DRIVE_OUTPUT.glob("*")):
    if f.is_file():
        h = hashlib.sha256(f.read_bytes()).hexdigest().upper()
        print(f"  {f.name}: SHA256={h[:16]}... ({f.stat().st_size:,} bytes)")

print(f"""
\\n{'='*60}
  HOÀN TẤT! Các bước tiếp theo:
{'='*60}

1. Tải thư mục từ Drive về máy:
   Google Drive > TaxInspector > tax_agent_lora_v5_production

2. Copy vào project:
   Copy tất cả file vào: e:\\\\TaxInspector\\\\Backend\\\\tax_agent\\\\

3. (Tùy chọn) Export GGUF cho llama.cpp:
   Cần merge adapter + base rồi convert.
   Xem cell tiếp theo.

4. Verify trên máy local:
   cd e:\\\\TaxInspector\\\\Backend
   python test_agent_v5_local.py
""")
'''

# ============================================================
# CELL 8 (Optional): Merge & export GGUF
# ============================================================
GGUF_EXPORT_SCRIPT = '''
# --- Merge LoRA into base model ---
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Merging LoRA adapter into base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float16,
)
merged_model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))
merged_model = merged_model.merge_and_unload()

MERGED_DIR = Path("/content/tax_agent_merged")
MERGED_DIR.mkdir(exist_ok=True)
merged_model.save_pretrained(str(MERGED_DIR))
tokenizer.save_pretrained(str(MERGED_DIR))
print(f"Merged model saved to {MERGED_DIR}")

# --- Convert to GGUF ---
# Cần cài llama.cpp
# !pip install -q llama-cpp-python
# !git clone https://github.com/ggerganov/llama.cpp.git /content/llama_cpp
# !cd /content/llama_cpp && pip install -r requirements.txt

# !python /content/llama_cpp/convert_hf_to_gguf.py \\
#     /content/tax_agent_merged \\
#     --outfile /content/drive/MyDrive/TaxInspector/tax_agent_v5_q4_k_m.gguf \\
#     --outtype q4_k_m

print("Xem hướng dẫn GGUF export trong comment ở trên.")
'''

# ============================================================
# Full notebook content for copy-paste
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TaxInspector Colab Training Script")
    print("  Copy các CELL vào Google Colab notebook")
    print("=" * 60)
    print()
    print("CELL 1: !pip install -q torch transformers peft datasets accelerate bitsandbytes sentencepiece")
    print()
    print("CELL 2 (Mount & Check GPU):")
    print(COLAB_SCRIPT)
    print()
    print("CELL 3 (Load & Validate Dataset):")
    print(LOAD_DATASET_SCRIPT)
    print()
    print("CELL 4 (Prepare Training Data):")
    print(PREPARE_SCRIPT)
    print()
    print("CELL 5 (Train LoRA):")
    print(TRAIN_SCRIPT)
    print()
    print("CELL 6 (Evaluate):")
    print(EVAL_SCRIPT)
    print()
    print("CELL 7 (Export to Drive):")
    print(EXPORT_SCRIPT)
    print()
    print("CELL 8 (Optional - GGUF Export):")
    print(GGUF_EXPORT_SCRIPT)
