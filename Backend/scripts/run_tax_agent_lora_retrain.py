"""
Controlled LoRA retraining entrypoint for the tax multi-agent planner.

This script is intentionally small: it wraps ``LoRATrainer`` with CLI
arguments, writes a JSON summary, and keeps artifacts out of the runtime
adapter directory unless the caller explicitly chooses that output path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.tax_agent_llm_model import LoRATrainer, LoRATrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain Tax Agent LoRA adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--training-data", default=str(BACKEND_DIR / "data" / "agent_ultimate_dataset_v4.jsonl"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="q_proj,v_proj,k_proj,o_proj")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.allow_download:
        os.environ["TAX_AGENT_ALLOW_MODEL_DOWNLOAD"] = "1"

    config = LoRATrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        training_data=args.training_data,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        max_records=args.max_records,
        max_seq_length=args.max_seq_length,
    )
    trainer = LoRATrainer(config)
    prepared = trainer.prepare_dataset()
    print(
        json.dumps(
            {
                "event": "dataset_prepared",
                "train": len(prepared["train"]),
                "eval": len(prepared["eval"]),
                "base_model": args.base_model,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    started = time.perf_counter()
    result = trainer.train()
    result["wall_seconds"] = round(time.perf_counter() - started, 1)

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "training_finished", "summary_json": str(summary_path), **result}, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
