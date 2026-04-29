"""
tax_agent_llm_model.py - Custom LLM for Tax Intelligence (Phase 6)
====================================================================
Local LLM inference engine with LoRA fine-tuning & quantization.

Tier 1: Fine-tuned local LLM (LoRA adapter)
Tier 2: Base model + in-context learning
Tier 3: Template synthesis fallback

Recommended: Qwen2.5-1.5B for 12GB RAM (i7-8th gen CPU)
"""
from __future__ import annotations

import json, logging, os, time, threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


class LLMTier(str, Enum):
    FINETUNED = "finetuned"
    BASE_FEW_SHOT = "base_few_shot"
    TEMPLATE = "template"


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str | None = None
    quantization: str = "4bit"
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.85
    repetition_penalty: float = 1.15
    device: str = "cpu"
    max_memory_mb: int = 6000
    context_window: int = 2048


@dataclass
class LLMResponse:
    text: str
    tier: LLMTier
    model_name: str
    tokens_generated: int
    latency_ms: float
    cached: bool = False


@dataclass
class LoRATrainingConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "data/models/tax_llm_lora"
    training_data: str = "data/llm_training.jsonl"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    num_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True


FEW_SHOT_EXAMPLES = {
    "vat_refund_risk": {
        "query": "Dieu kien hoan thue VAT cho DN xuat khau?",
        "answer": "Theo Dieu 13 Luat Thue GTGT va TT 219/2013: 1) Co hang hoa XK 2) Thue dau vao >= 300tr 3) Co hop dong XK 4) Thanh toan qua NH",
    },
    "invoice_risk": {
        "query": "Dau hieu hoa don gia?",
        "answer": "Theo ND 123/2020: 1) Ben ban khong KD thuc 2) Gia tri bat thuong 3) MST ngung HD 4) Khong co chung tu van chuyen",
    },
    "delinquency": {
        "query": "Bien phap xu ly no dong thue?",
        "answer": "Theo Luat QLT 38/2019: 1) Nhac no (D59) 2) Phat 0.03%/ngay 3) Cuong che trich TK (D62) 4) Phong toa TS 5) Thu hoi GP",
    },
}


class TaxAgentLLM:
    """Custom LLM: Fine-tuned -> Base+FewShot -> Template fallback."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._model = None
        self._tokenizer = None
        self._tier = LLMTier.TEMPLATE
        self._loaded = False

    @property
    def tier(self) -> LLMTier:
        return self._tier

    def load(self) -> LLMTier:
        for loader in (self._try_load_finetuned, self._try_load_base):
            tier = loader()
            if tier:
                return tier
        self._tier = LLMTier.TEMPLATE
        self._loaded = True
        logger.info("[TaxLLM] Using template fallback")
        return LLMTier.TEMPLATE

    def _try_load_finetuned(self) -> Optional[LLMTier]:
        adapter = self.config.adapter_path or str(Path(__file__).parent.parent / "data/models/tax_llm_lora")
        if not Path(adapter).exists():
            return None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(self.config.model_name, trust_remote_code=True, low_cpu_mem_usage=True)
            self._model = PeftModel.from_pretrained(base, adapter)
            self._model.eval()
            self._tier = LLMTier.FINETUNED
            self._loaded = True
            logger.info("[TaxLLM] LoRA model loaded from %s", adapter)
            return LLMTier.FINETUNED
        except Exception as exc:
            logger.warning("[TaxLLM] LoRA load fail: %s", exc)
            return None

    def _try_load_base(self) -> Optional[LLMTier]:
        try:
            import psutil
            if psutil.virtual_memory().available / (1024**2) < 4000:
                return None
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name, trust_remote_code=True, low_cpu_mem_usage=True)
            self._model.eval()
            self._tier = LLMTier.BASE_FEW_SHOT
            self._loaded = True
            return LLMTier.BASE_FEW_SHOT
        except Exception as exc:
            logger.warning("[TaxLLM] Base load fail: %s", exc)
            return None

    def generate(self, query: str, context: str = "", intent: str = "general_tax_query",
                 *, evidence: list[dict] | None = None, max_new_tokens: int | None = None) -> LLMResponse:
        if not self._loaded:
            self.load()
        t0 = time.perf_counter()
        if self._tier in (LLMTier.FINETUNED, LLMTier.BASE_FEW_SHOT):
            resp = self._model_generate(query, context, intent, evidence, max_new_tokens)
        else:
            resp = self._template_generate(query, context, intent, evidence)
        resp.latency_ms = (time.perf_counter() - t0) * 1000.0
        return resp

    def generate_stream(self, query: str, context: str = "", intent: str = "general_tax_query") -> Generator[str, None, None]:
        if self._tier == LLMTier.TEMPLATE:
            resp = self._template_generate(query, context, intent, None)
            for w in resp.text.split():
                yield w + " "
                time.sleep(0.02)
            return
        if not self._model:
            yield "He thong chua san sang."
            return
        try:
            from transformers import TextIteratorStreamer
            prompt = self._build_prompt(query, context, intent, None)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            kw = {**inputs, "streamer": streamer, "max_new_tokens": self.config.max_new_tokens,
                  "temperature": self.config.temperature, "do_sample": True}
            t = threading.Thread(target=self._model.generate, kwargs=kw)
            t.start()
            for chunk in streamer:
                yield chunk
            t.join()
        except ImportError:
            yield self.generate(query, context, intent).text

    def _model_generate(self, query, context, intent, evidence, max_tokens):
        import torch
        prompt = self._build_prompt(query, context, intent, evidence)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.context_window)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens or self.config.max_new_tokens,
                temperature=self.config.temperature, top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty, do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return LLMResponse(text=self._tokenizer.decode(gen, skip_special_tokens=True).strip(),
                           tier=self._tier, model_name=self.config.model_name,
                           tokens_generated=len(gen), latency_ms=0)

    def _template_generate(self, query, context, intent, evidence):
        prefixes = {"vat_refund_risk": "Ve hoan thue GTGT", "invoice_risk": "Ve rui ro hoa don",
                     "delinquency": "Ve no dong thue", "osint_ownership": "Ve cau truc so huu",
                     "transfer_pricing": "Ve chuyen gia", "audit_selection": "Ve lua chon thanh tra"}
        p = [f"{prefixes.get(intent, 'Ve van de thue')}:\n"]
        if context:
            p.append(f"Du lieu:\n{context[:500]}\n")
        if evidence:
            for ev in evidence[:3]:
                p.append(f"- {ev.get('title','')}: {ev.get('content','')[:150]}")
        p.append("\nKhuyen nghi: doi chieu them nghiep vu truoc khi quyet dinh.")
        text = "\n".join(p)
        return LLMResponse(text=text, tier=LLMTier.TEMPLATE, model_name="template",
                           tokens_generated=len(text.split()), latency_ms=0)

    def _build_prompt(self, query, context, intent, evidence):
        system = "Ban la tro ly AI thue Viet Nam. Tra loi chinh xac, co trich dan phap luat."
        fs = FEW_SHOT_EXAMPLES.get(intent)
        fs_text = ""
        if fs and self._tier == LLMTier.BASE_FEW_SHOT:
            fs_text = "\nVi du:\nQ: " + fs["query"] + "\nA: " + fs["answer"] + "\n"
        ctx = "\nNgu canh:\n" + context[:800] + "\n" if context else ""
        ev_text = ""
        if evidence:
            items = ["- " + e.get("title", "") + ": " + e.get("content", "")[:200] for e in evidence[:3]]
            ev_text = "\nBang chung:\n" + "\n".join(items) + "\n"
        parts = ["[SYSTEM]\n", system, fs_text, "\n[/SYSTEM]\n"]
        parts.append(ctx)
        parts.append(ev_text)
        parts.append("\n[USER]\n" + query + "\n[/USER]\n[ASSISTANT]\n")
        return "".join(parts)

    @property
    def is_available(self) -> bool:
        return self._loaded


class LoRATrainer:
    """
    LoRA fine-tuning pipeline for custom tax LLM.

    Usage:
        trainer = LoRATrainer(LoRATrainingConfig())
        trainer.train()  # Fine-tune with LoRA
        trainer.export()  # Export merged model
    """

    def __init__(self, config: LoRATrainingConfig | None = None):
        self.config = config or LoRATrainingConfig()

    def prepare_dataset(self) -> dict:
        """Load and prepare the training dataset from JSONL."""
        data_path = Path(self.config.training_data)
        if not data_path.exists():
            logger.warning("[LoRATrainer] No training data at %s", data_path)
            return {"train": [], "eval": []}

        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                convos = record.get("conversations", [])
                if len(convos) >= 3:
                    examples.append({
                        "system": convos[0].get("value", ""),
                        "user": convos[1].get("value", ""),
                        "assistant": convos[2].get("value", ""),
                    })

        # 90/10 train/eval split
        split_idx = int(len(examples) * 0.9)
        return {"train": examples[:split_idx], "eval": examples[split_idx:]}

    def train(self) -> dict:
        """
        Run LoRA fine-tuning using HF Trainer (no TRL dependency).

        Returns training summary dict.
        Requires: transformers, peft, datasets
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            logger.error("[LoRATrainer] Missing dependency: %s", exc)
            return {"status": "error", "message": f"Missing: {exc}"}

        dataset = self.prepare_dataset()
        if not dataset["train"]:
            return {"status": "error", "message": "No training data"}

        logger.info("[LoRATrainer] Starting training with %d examples", len(dataset["train"]))

        # Load tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, trust_remote_code=True, low_cpu_mem_usage=True,
        )

        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )

        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("[LoRATrainer] Trainable: %d / %d (%.2f%%)", trainable, total, trainable / total * 100)

        # Format and tokenize dataset
        def format_example(ex):
            return "[SYSTEM]\n" + ex["system"] + "\n[/SYSTEM]\n[USER]\n" + ex["user"] + "\n[/USER]\n[ASSISTANT]\n" + ex["assistant"]

        train_texts = [format_example(ex) for ex in dataset["train"]]

        # Tokenize
        encodings = tokenizer(
            train_texts, truncation=True, padding=True,
            max_length=self.config.context_window if hasattr(self.config, 'context_window') else 2048,
            return_tensors="pt",
        )
        # For causal LM, labels = input_ids
        encodings["labels"] = encodings["input_ids"].clone()

        from torch.utils.data import Dataset as TorchDataset

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

        train_dataset = TokenizedDataset(encodings)

        # Training arguments
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            save_strategy="epoch",
            logging_steps=1,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="none",
            use_cpu=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        t0 = time.perf_counter()
        trainer.train()
        duration = time.perf_counter() - t0

        # Save adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        summary = {
            "status": "success",
            "output_dir": str(output_dir),
            "training_examples": len(train_texts),
            "trainable_params": trainable,
            "total_params": total,
            "duration_seconds": round(duration, 1),
            "epochs": self.config.num_epochs,
        }

        logger.info("[LoRATrainer] Training complete: %s", summary)
        return summary

    def export_merged(self, output_dir: str | None = None) -> dict:
        """Export merged model (base + LoRA) for standalone deployment."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            adapter_path = self.config.output_dir
            if not Path(adapter_path).exists():
                return {"status": "error", "message": "No adapter found"}

            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True, low_cpu_mem_usage=True)
            model = PeftModel.from_pretrained(base, adapter_path)
            merged = model.merge_and_unload()

            out = output_dir or str(Path(adapter_path) / "merged")
            Path(out).mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(out)
            tokenizer.save_pretrained(out)

            return {"status": "success", "output_dir": out}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


# Singleton
_llm_instance: TaxAgentLLM | None = None

def get_tax_llm() -> TaxAgentLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = TaxAgentLLM()
    return _llm_instance
