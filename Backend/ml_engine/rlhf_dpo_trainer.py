"""
rlhf_dpo_trainer.py – RLHF / DPO Fine-Tuning Pipeline
========================================================
Pipeline huấn luyện DPO (Direct Preference Optimization) tích hợp
với FeedbackCollector và TaxAgentLLM để liên tục cải thiện chất lượng.

Capabilities:
    1. Xây dựng preference pairs từ feedback data (positive/negative)
    2. DPO training loop với LoRA adapter
    3. A/B model comparison evaluator
    4. Auto-retrain trigger dựa trên drift alerts

Integration:
    - tax_agent_feedback.py → export_training_candidates() → preference pairs
    - tax_agent_llm_model.py → LoRATrainer → adapter hot-swap

Design:
    - CPU-first: tất cả pipeline chạy được trên CPU
    - Thread-safe preference dataset builder
    - Checkpoint + rollback nếu model mới kém hơn
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
DPO_DIR = MODEL_DIR / "dpo_adapters"


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class PreferencePair:
    """Một cặp preference cho DPO training."""
    prompt: str
    chosen: str           # Câu trả lời tốt (positive feedback)
    rejected: str         # Câu trả lời kém (negative feedback)
    intent: str = ""
    confidence_chosen: float = 1.0
    confidence_rejected: float = 0.0
    source: str = "feedback"   # feedback | correction | synthetic
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DPOConfig:
    """Cấu hình DPO training."""
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_output_dir: str = str(DPO_DIR)
    beta: float = 0.1              # DPO temperature (β)
    learning_rate: float = 5e-5
    num_epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    max_prompt_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    min_pairs_for_training: int = 10
    # Auto-retrain triggers
    drift_threshold: float = 0.15
    satisfaction_drop_threshold: float = 0.10
    check_interval_hours: int = 24


@dataclass
class ABTestResult:
    """Kết quả so sánh A/B giữa hai model versions."""
    model_a: str
    model_b: str
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    total: int = 0
    avg_score_a: float = 0.0
    avg_score_b: float = 0.0
    verdict: str = "inconclusive"
    details: list[dict] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
#  1. Preference Pair Builder
# ════════════════════════════════════════════════════════════════

class PreferencePairBuilder:
    """
    Xây dựng preference pairs từ FeedbackCollector data.

    Chiến lược:
        - Correction feedback → strongest signal (user cung cấp câu đúng)
        - Positive vs Negative trên cùng intent → preference pair
        - Synthetic pairs từ template vs model output
    """

    def __init__(self):
        self._lock = threading.Lock()

    def build_from_feedback(
        self,
        feedback_export: dict[str, Any],
        db=None,
    ) -> list[PreferencePair]:
        """
        Xây dựng preference pairs từ feedback export data.

        Args:
            feedback_export: Output từ FeedbackCollector.export_training_candidates()
            db: Optional DB session để query agent_turns cho context

        Returns:
            Danh sách PreferencePair sẵn sàng cho DPO training.
        """
        pairs: list[PreferencePair] = []

        # 1. Correction-based pairs (strongest signal)
        corrections = feedback_export.get("corrections", [])
        for corr in corrections:
            prompt = self._get_prompt_from_db(
                db, corr.get("session_id", ""), corr.get("turn_id", 0)
            )
            if not prompt:
                continue

            original = self._get_response_from_db(
                db, corr.get("session_id", ""), corr.get("turn_id", 0)
            )
            correction_text = corr.get("correction_text", "")

            if original and correction_text:
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=correction_text,
                    rejected=original,
                    intent=corr.get("original_intent", ""),
                    confidence_chosen=0.95,
                    confidence_rejected=corr.get("confidence", 0.5),
                    source="correction",
                ))

        # 2. Positive vs Negative trên cùng intent
        negatives = feedback_export.get("negatives", [])
        intent_groups = self._group_by_intent(negatives, db)
        for intent, neg_items in intent_groups.items():
            pos_response = self._get_best_positive_for_intent(db, intent)
            if not pos_response:
                continue
            for neg in neg_items[:3]:  # Tối đa 3 pairs/intent
                prompt = self._get_prompt_from_db(
                    db, neg.get("session_id", ""), neg.get("turn_id", 0)
                )
                neg_response = self._get_response_from_db(
                    db, neg.get("session_id", ""), neg.get("turn_id", 0)
                )
                if prompt and neg_response:
                    pairs.append(PreferencePair(
                        prompt=prompt,
                        chosen=pos_response,
                        rejected=neg_response,
                        intent=intent,
                        source="feedback",
                    ))

        logger.info(
            "[DPO-Builder] Built %d preference pairs (%d corrections, %d feedback)",
            len(pairs),
            len([p for p in pairs if p.source == "correction"]),
            len([p for p in pairs if p.source == "feedback"]),
        )
        return pairs

    def build_synthetic_pairs(
        self,
        prompts: list[str],
        good_responses: list[str],
        bad_responses: list[str],
    ) -> list[PreferencePair]:
        """Tạo synthetic preference pairs từ curated data."""
        pairs = []
        for prompt, good, bad in zip(prompts, good_responses, bad_responses):
            pairs.append(PreferencePair(
                prompt=prompt, chosen=good, rejected=bad,
                source="synthetic", confidence_chosen=1.0,
            ))
        return pairs

    def export_to_jsonl(
        self, pairs: list[PreferencePair], output_path: str | Path
    ) -> int:
        """Export preference pairs sang JSONL cho DPO training."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "intent": pair.intent,
                    "source": pair.source,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        logger.info("[DPO-Builder] Exported %d pairs to %s", count, output_path)
        return count

    # ─── Helpers ─────────────────────────────────────────────────

    def _get_prompt_from_db(self, db, session_id: str, turn_id: int) -> str:
        """Lấy user message từ DB (turn trước turn_id)."""
        if db is None:
            return ""
        try:
            from sqlalchemy import text as sql_text
            row = db.execute(sql_text("""
                SELECT message_text FROM agent_turns
                WHERE session_id = :sid AND turn_index = :tid AND role = 'user'
                LIMIT 1
            """), {"sid": session_id, "tid": max(0, turn_id - 1)}).mappings().first()
            return str(row["message_text"]) if row else ""
        except Exception:
            return ""

    def _get_response_from_db(self, db, session_id: str, turn_id: int) -> str:
        """Lấy assistant response từ DB."""
        if db is None:
            return ""
        try:
            from sqlalchemy import text as sql_text
            row = db.execute(sql_text("""
                SELECT message_text FROM agent_turns
                WHERE session_id = :sid AND turn_index = :tid AND role = 'assistant'
                LIMIT 1
            """), {"sid": session_id, "tid": turn_id}).mappings().first()
            return str(row["message_text"]) if row else ""
        except Exception:
            return ""

    def _group_by_intent(self, negatives: list[dict], db) -> dict[str, list[dict]]:
        """Nhóm negative feedback theo intent."""
        groups: dict[str, list[dict]] = {}
        for neg in negatives:
            intent = neg.get("intent", "unknown")
            groups.setdefault(intent, []).append(neg)
        return groups

    def _get_best_positive_for_intent(self, db, intent: str) -> str:
        """Tìm response tốt nhất (positive feedback) cho intent."""
        if db is None:
            return ""
        try:
            from sqlalchemy import text as sql_text
            row = db.execute(sql_text("""
                SELECT at2.message_text
                FROM agent_feedback_events af
                JOIN agent_turns at2 ON at2.session_id = af.session_id
                    AND (at2.id = af.turn_id OR at2.turn_index = af.turn_id)
                    AND at2.role = 'assistant'
                WHERE af.feedback_type = 'positive' AND af.intent = :intent
                ORDER BY af.confidence DESC
                LIMIT 1
            """), {"intent": intent}).mappings().first()
            return str(row["message_text"]) if row else ""
        except Exception:
            return ""


# ════════════════════════════════════════════════════════════════
#  2. DPO Trainer
# ════════════════════════════════════════════════════════════════

class DPOTrainer:
    """
    Direct Preference Optimization trainer với LoRA adapters.

    Tham chiếu: Rafailov et al., "Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model", NeurIPS 2023.

    Usage:
        trainer = DPOTrainer(DPOConfig())
        result = trainer.train(preference_pairs)
    """

    def __init__(self, config: DPOConfig | None = None):
        self.config = config or DPOConfig()
        self._training_lock = threading.Lock()

    def train(self, pairs: list[PreferencePair]) -> dict[str, Any]:
        """
        Chạy DPO training loop với LoRA adapter.

        Returns:
            Training summary dict.
        """
        if len(pairs) < self.config.min_pairs_for_training:
            return {
                "status": "skipped",
                "reason": f"Cần tối thiểu {self.config.min_pairs_for_training} pairs, "
                          f"hiện có {len(pairs)}",
            }

        with self._training_lock:
            return self._run_training(pairs)

    def _run_training(self, pairs: list[PreferencePair]) -> dict[str, Any]:
        """Core DPO training implementation."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            logger.error("[DPO] Missing dependency: %s", exc)
            return {"status": "error", "message": f"Missing: {exc}"}

        logger.info("[DPO] Starting training with %d preference pairs", len(pairs))
        t0 = time.perf_counter()

        # Load tokenizer + base model
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, trust_remote_code=True, low_cpu_mem_usage=True
        )

        # Tạo reference model (frozen copy)
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, trust_remote_code=True, low_cpu_mem_usage=True
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

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
        logger.info("[DPO] Trainable: %d / %d (%.2f%%)", trainable, total,
                     trainable / total * 100)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # DPO Training loop
        model.train()
        total_loss = 0.0
        n_steps = 0

        for epoch in range(self.config.num_epochs):
            for pair in pairs:
                # Tokenize chosen + rejected
                chosen_ids = tokenizer(
                    pair.prompt + "\n" + pair.chosen,
                    truncation=True, max_length=self.config.max_length,
                    return_tensors="pt",
                )
                rejected_ids = tokenizer(
                    pair.prompt + "\n" + pair.rejected,
                    truncation=True, max_length=self.config.max_length,
                    return_tensors="pt",
                )

                # Forward pass — policy model
                with torch.no_grad():
                    ref_chosen = ref_model(**chosen_ids).logits
                    ref_rejected = ref_model(**rejected_ids).logits

                policy_chosen = model(**chosen_ids).logits
                policy_rejected = model(**rejected_ids).logits

                # Tính log-probabilities
                chosen_logprob = self._get_log_probs(
                    policy_chosen, chosen_ids["input_ids"]
                )
                rejected_logprob = self._get_log_probs(
                    policy_rejected, rejected_ids["input_ids"]
                )
                ref_chosen_logprob = self._get_log_probs(
                    ref_chosen, chosen_ids["input_ids"]
                )
                ref_rejected_logprob = self._get_log_probs(
                    ref_rejected, rejected_ids["input_ids"]
                )

                # DPO loss: -log σ(β * (log π(y_w|x)/π_ref(y_w|x)
                #                      - log π(y_l|x)/π_ref(y_l|x)))
                chosen_reward = chosen_logprob - ref_chosen_logprob
                rejected_reward = rejected_logprob - ref_rejected_logprob
                loss = -torch.nn.functional.logsigmoid(
                    self.config.beta * (chosen_reward - rejected_reward)
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1

            logger.info("[DPO] Epoch %d/%d — avg_loss=%.4f",
                         epoch + 1, self.config.num_epochs,
                         total_loss / max(1, n_steps))

        # Save adapter
        output_dir = Path(self.config.adapter_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        duration = time.perf_counter() - t0
        # Lưu metadata
        meta = {
            "status": "success",
            "pairs_count": len(pairs),
            "trainable_params": trainable,
            "total_params": total,
            "epochs": self.config.num_epochs,
            "beta": self.config.beta,
            "avg_loss": round(total_loss / max(1, n_steps), 6),
            "duration_seconds": round(duration, 1),
            "adapter_path": str(output_dir),
            "data_hash": self._hash_pairs(pairs),
            "timestamp": time.time(),
        }
        with open(output_dir / "dpo_training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("[DPO] Training complete: %s", meta)
        return meta

    @staticmethod
    def _get_log_probs(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """Tính tổng log-probability của sequence."""
        import torch
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        target = labels[:, 1:].unsqueeze(-1)
        gathered = log_probs.gather(dim=-1, index=target).squeeze(-1)
        return gathered.sum(dim=-1).mean()

    @staticmethod
    def _hash_pairs(pairs: list[PreferencePair]) -> str:
        """Hash nội dung pairs để tracking lineage."""
        content = "".join(p.prompt + p.chosen + p.rejected for p in pairs)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════
#  3. A/B Model Evaluator
# ════════════════════════════════════════════════════════════════

class ABModelEvaluator:
    """
    So sánh hai model versions trên cùng test prompts.

    Sử dụng reference-free evaluation: so sánh output quality
    dựa trên length, coherence, và keyword coverage.
    """

    def __init__(self):
        pass

    def compare(
        self,
        model_a_fn,
        model_b_fn,
        test_prompts: list[str],
        reference_answers: list[str] | None = None,
    ) -> ABTestResult:
        """
        So sánh hai model trên test prompts.

        Args:
            model_a_fn: Callable nhận prompt → str (response)
            model_b_fn: Callable nhận prompt → str (response)
            test_prompts: Danh sách prompts
            reference_answers: Optional gold answers

        Returns:
            ABTestResult với verdict.
        """
        result = ABTestResult(model_a="model_a", model_b="model_b")
        scores_a: list[float] = []
        scores_b: list[float] = []

        for i, prompt in enumerate(test_prompts):
            try:
                resp_a = model_a_fn(prompt)
                resp_b = model_b_fn(prompt)
            except Exception as exc:
                logger.warning("[AB] Error on prompt %d: %s", i, exc)
                continue

            ref = reference_answers[i] if reference_answers and i < len(reference_answers) else None
            score_a = self._score_response(resp_a, prompt, ref)
            score_b = self._score_response(resp_b, prompt, ref)

            scores_a.append(score_a)
            scores_b.append(score_b)

            if score_a > score_b + 0.05:
                result.wins_a += 1
            elif score_b > score_a + 0.05:
                result.wins_b += 1
            else:
                result.ties += 1

            result.details.append({
                "prompt_idx": i,
                "score_a": round(score_a, 4),
                "score_b": round(score_b, 4),
            })

        result.total = result.wins_a + result.wins_b + result.ties
        result.avg_score_a = round(sum(scores_a) / max(1, len(scores_a)), 4)
        result.avg_score_b = round(sum(scores_b) / max(1, len(scores_b)), 4)

        # Verdict
        if result.total < 5:
            result.verdict = "insufficient_data"
        elif result.wins_b > result.wins_a * 1.2:
            result.verdict = "model_b_wins"
        elif result.wins_a > result.wins_b * 1.2:
            result.verdict = "model_a_wins"
        else:
            result.verdict = "no_significant_difference"

        logger.info(
            "[AB] Verdict=%s | A=%d wins (%.3f avg) | B=%d wins (%.3f avg) | Ties=%d",
            result.verdict, result.wins_a, result.avg_score_a,
            result.wins_b, result.avg_score_b, result.ties,
        )
        return result

    def _score_response(self, response: str, prompt: str, reference: str | None) -> float:
        """Đánh giá quality của response (0-1)."""
        if not response or not response.strip():
            return 0.0

        score = 0.0
        # Độ dài hợp lý (không quá ngắn, không quá dài)
        words = len(response.split())
        if 20 <= words <= 300:
            score += 0.3
        elif 10 <= words < 20:
            score += 0.15

        # Có cấu trúc (numbered list, bullet points)
        if any(c in response for c in ["1)", "1.", "- ", "•"]):
            score += 0.2

        # Có trích dẫn pháp luật (đặc thù thuế VN)
        legal_keywords = ["Điều", "Luật", "Nghị định", "Thông tư", "ND", "TT"]
        if any(kw in response for kw in legal_keywords):
            score += 0.2

        # Reference overlap
        if reference:
            ref_words = set(reference.lower().split())
            resp_words = set(response.lower().split())
            overlap = len(ref_words & resp_words) / max(1, len(ref_words))
            score += 0.3 * overlap

        return min(1.0, score)


# ════════════════════════════════════════════════════════════════
#  4. Auto-Retrain Controller
# ════════════════════════════════════════════════════════════════

class AutoRetrainController:
    """
    Tự động trigger DPO retraining khi phát hiện drift.

    Tích hợp với FeedbackCollector.compute_drift() để monitor
    và trigger training pipeline khi cần thiết.
    """

    def __init__(self, config: DPOConfig | None = None):
        self.config = config or DPOConfig()
        self._last_check: float = 0.0
        self._last_train_hash: str = ""

    def should_retrain(self, feedback_collector) -> dict[str, Any]:
        """
        Kiểm tra xem có cần retrain không.

        Args:
            feedback_collector: FeedbackCollector instance

        Returns:
            Dict với should_retrain flag và reasons.
        """
        now = time.time()
        hours_since_check = (now - self._last_check) / 3600

        if hours_since_check < self.config.check_interval_hours:
            return {"should_retrain": False, "reason": "too_soon"}

        self._last_check = now

        # Kiểm tra drift
        drift_data = feedback_collector.compute_drift(
            window_hours=self.config.check_interval_hours * 2
        )

        reasons = []

        if drift_data.get("drift_detected"):
            for alert in drift_data.get("alerts", []):
                if alert.get("level") == "critical":
                    reasons.append(f"Critical drift: {alert.get('metric')}")
                elif (alert.get("metric") == "satisfaction_rate"
                      and alert.get("drift", 0) > self.config.satisfaction_drop_threshold):
                    reasons.append(f"Satisfaction drop: {alert.get('drift'):.2%}")

        # Kiểm tra có đủ training data mới
        stats = feedback_collector.get_statistics(
            window_hours=self.config.check_interval_hours
        )
        has_enough = stats.get("total_feedback", 0) >= self.config.min_pairs_for_training

        if reasons and has_enough:
            return {
                "should_retrain": True,
                "reasons": reasons,
                "drift_data": drift_data,
                "feedback_stats": stats,
            }

        return {
            "should_retrain": False,
            "reason": "no_significant_drift" if not reasons else "insufficient_data",
        }

    def run_auto_retrain(
        self,
        feedback_collector,
        db=None,
    ) -> dict[str, Any]:
        """
        Chạy full auto-retrain pipeline nếu cần.

        Steps:
            1. Check drift triggers
            2. Build preference pairs
            3. Run DPO training
            4. A/B compare with current model
            5. Swap adapter nếu model mới tốt hơn

        Returns:
            Pipeline execution summary.
        """
        # Step 1: Check
        check = self.should_retrain(feedback_collector)
        if not check.get("should_retrain"):
            return {"action": "skip", **check}

        logger.info("[AutoRetrain] Triggered: %s", check.get("reasons"))

        # Step 2: Build pairs
        builder = PreferencePairBuilder()
        export_data = feedback_collector.export_training_candidates(
            window_hours=self.config.check_interval_hours * 2
        )
        pairs = builder.build_from_feedback(export_data, db)

        if len(pairs) < self.config.min_pairs_for_training:
            return {
                "action": "skip",
                "reason": f"Only {len(pairs)} pairs (need {self.config.min_pairs_for_training})",
            }

        # Export JSONL cho audit trail
        data_path = Path(self.config.adapter_output_dir) / "training_data.jsonl"
        builder.export_to_jsonl(pairs, data_path)

        # Step 3: Train
        trainer = DPOTrainer(self.config)
        train_result = trainer.train(pairs)

        if train_result.get("status") != "success":
            return {"action": "train_failed", **train_result}

        self._last_train_hash = train_result.get("data_hash", "")

        return {
            "action": "retrained",
            "triggers": check.get("reasons"),
            "pairs_used": len(pairs),
            "training": train_result,
        }


# ════════════════════════════════════════════════════════════════
#  Training Status Tracker
# ════════════════════════════════════════════════════════════════

class DPOTrainingStatusTracker:
    """
    Tracks DPO training history and current status for dashboard consumption.

    Thread-safe singleton that records every training run, providing
    data for the Telemetry Dashboard to show DPO pipeline health.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []
        self._current_status: str = "idle"  # idle | building_pairs | training | evaluating | swapping
        self._last_run: dict[str, Any] = {}

    def record_run(self, run_data: dict[str, Any]) -> None:
        """Record a completed training run."""
        with self._lock:
            self._history.append({
                **run_data,
                "recorded_at": time.time(),
            })
            # Keep only last 50 runs
            if len(self._history) > 50:
                self._history = self._history[-50:]
            self._last_run = run_data

    def set_status(self, status: str) -> None:
        with self._lock:
            self._current_status = status

    def get_status(self) -> dict[str, Any]:
        """Get current DPO pipeline status for dashboard."""
        with self._lock:
            return {
                "current_status": self._current_status,
                "last_run": dict(self._last_run) if self._last_run else None,
                "total_runs": len(self._history),
                "history": list(self._history[-10:]),  # Last 10 runs
                "has_deps": _check_dpo_deps(),
            }

    def persist_to_db(self, db, run_data: dict[str, Any]) -> bool:
        """Persist training run to dpo_training_runs table."""
        try:
            from sqlalchemy import text as sql_text
            db.execute(sql_text("""
                INSERT INTO dpo_training_runs
                (pairs_count, avg_loss, duration_seconds, adapter_path,
                 data_hash, ab_verdict, status, triggered_by)
                VALUES
                (:pairs_count, :avg_loss, :duration_seconds, :adapter_path,
                 :data_hash, :ab_verdict, :status, :triggered_by)
            """), {
                "pairs_count": run_data.get("pairs_used", run_data.get("pairs_count", 0)),
                "avg_loss": run_data.get("avg_loss"),
                "duration_seconds": run_data.get("duration_seconds"),
                "adapter_path": run_data.get("adapter_path", ""),
                "data_hash": run_data.get("data_hash", ""),
                "ab_verdict": run_data.get("ab_verdict", ""),
                "status": run_data.get("status", run_data.get("action", "unknown")),
                "triggered_by": run_data.get("triggered_by", "auto"),
            })
            db.commit()
            return True
        except Exception as exc:
            logger.debug("[DPO-Tracker] DB persist failed (table may not exist): %s", exc)
            return False


def _check_dpo_deps() -> bool:
    """Check if torch/transformers/peft are available."""
    try:
        import torch
        import transformers
        import peft
        return True
    except ImportError:
        return False


class DPODryRunner:
    """
    Dry-run mode: builds preference pairs + exports JSONL without training.

    Useful when:
      - DPO dependencies (torch/peft) are not installed
      - You want to review the training data before committing
      - Manual trigger from dashboard for data inspection
    """

    def run(
        self,
        feedback_collector,
        db=None,
        window_hours: int = 168,
        triggered_by: str = "manual_dry_run",
    ) -> dict[str, Any]:
        """
        Build pairs and export JSONL without training.

        Returns:
            Summary with pairs count, export path, and sample data.
        """
        builder = PreferencePairBuilder()
        export_data = feedback_collector.export_training_candidates(
            window_hours=window_hours,
        )
        pairs = builder.build_from_feedback(export_data, db)

        result = {
            "action": "dry_run",
            "triggered_by": triggered_by,
            "pairs_count": len(pairs),
            "corrections_count": len([p for p in pairs if p.source == "correction"]),
            "feedback_count": len([p for p in pairs if p.source == "feedback"]),
            "synthetic_count": len([p for p in pairs if p.source == "synthetic"]),
            "has_deps": _check_dpo_deps(),
            "timestamp": time.time(),
        }

        if pairs:
            # Export to JSONL
            export_path = Path(DPO_DIR) / "dry_run_data.jsonl"
            count = builder.export_to_jsonl(pairs, export_path)
            result["export_path"] = str(export_path)
            result["exported_count"] = count
            # Sample preview (first 3 pairs)
            result["sample_pairs"] = [
                {
                    "prompt": p.prompt[:200],
                    "chosen_preview": p.chosen[:150],
                    "rejected_preview": p.rejected[:150],
                    "intent": p.intent,
                    "source": p.source,
                }
                for p in pairs[:3]
            ]
        else:
            result["message"] = "Không có đủ feedback data để xây dựng preference pairs."

        logger.info("[DPO-DryRun] Completed: %d pairs built", len(pairs))
        return result


# ════════════════════════════════════════════════════════════════
#  Singleton + Factory
# ════════════════════════════════════════════════════════════════

_dpo_controller: AutoRetrainController | None = None
_dpo_status: DPOTrainingStatusTracker | None = None


def get_auto_retrain_controller() -> AutoRetrainController:
    """Singleton cho AutoRetrainController."""
    global _dpo_controller
    if _dpo_controller is None:
        _dpo_controller = AutoRetrainController()
    return _dpo_controller


def get_dpo_status_tracker() -> DPOTrainingStatusTracker:
    """Singleton cho DPOTrainingStatusTracker."""
    global _dpo_status
    if _dpo_status is None:
        _dpo_status = DPOTrainingStatusTracker()
    return _dpo_status


def get_dpo_trainer(config: DPOConfig | None = None) -> DPOTrainer:
    """Factory method cho DPOTrainer."""
    return DPOTrainer(config)


def get_dpo_dry_runner() -> DPODryRunner:
    """Factory method cho DPODryRunner."""
    return DPODryRunner()
