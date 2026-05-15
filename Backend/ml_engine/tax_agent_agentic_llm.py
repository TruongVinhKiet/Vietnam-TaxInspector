"""
tax_agent_agentic_llm.py — Agentic LLM Engine (LoRA V5)
========================================================
Load LoRA V5 adapter và chạy inference để tự động chọn Tool.

Output format:
    <thought>Suy luận nghiệp vụ...</thought>
    <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

Fallback: Nếu model không load được hoặc output không hợp lệ,
trả về None để Orchestrator dùng pipeline heuristic cũ.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Đường dẫn mặc định tới adapter V4
_DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "tax_agent"

# System Prompt đồng bộ 100% với dataset huấn luyện
AGENTIC_SYSTEM_PROMPT = """Bạn là TaxInspector AI - Trợ lý Thanh tra Thuế và tư vấn pháp luật thuế.
Nhiệm vụ: hiểu yêu cầu tiếng Việt tự nhiên, kể cả không dấu/viết tắt/sai ký tự nhẹ; chọn đúng công cụ/model; không tự bịa kết quả.

Quy tắc trả lời khi chưa có tool result:
- Nếu là tác vụ nghiệp vụ, suy nghĩ ngắn trong <thought>, sau đó gọi đúng một tool trong <tool_call> JSON rồi dừng.
- Nếu thiếu MST/tệp/thông tin bắt buộc, hỏi lại rõ ràng thay vì gọi tool sai.
- Nếu chỉ là chào hỏi/cảm ơn/hỏi khả năng, trả lời trực tiếp, không gọi tool.

Quy tắc sau khi nhận tool result pháp luật:
- Tổng hợp bằng tiếng Việt rõ ràng, có căn cứ, điều kiện áp dụng, bước xử lý, cảnh báo rủi ro.
- Ưu tiên GraphRAG/knowledge graph, authority path và tình trạng hiệu lực; không kết luận vượt quá chứng cứ.

Công cụ khả dụng:
1. top_n_risky_companies(n): Danh sách top N doanh nghiệp rủi ro cao nhất.
2. company_risk_lookup(tax_code): Tra cứu hồ sơ rủi ro tổng thể của một MST.
3. gnn_vat_fraud(tax_code): Pipeline gian lận VAT kết hợp graph/GNN.
4. gnn_analysis(tax_code): Phân tích mạng lưới giao dịch VAT.
5. invoice_risk_scan(tax_code, period): Rà soát rủi ro hóa đơn đầu vào/đầu ra.
6. vat_refund_risk(tax_code, period): Đánh giá rủi ro hồ sơ hoàn thuế VAT.
7. vae_anomaly_scan(tax_code): Quét bất thường hóa đơn bằng VAE.
8. motif_detection(tax_code, max_hops): Phát hiện motif/vòng lặp giao dịch.
9. ring_scoring(tax_code): Chấm điểm vòng giao dịch VAT.
10. ownership_analysis(tax_code): Phân tích sở hữu chéo, UBO, common controller.
11. hetero_gnn_risk(tax_code, node_type): Đánh giá rủi ro bằng HeteroGNN/HGT.
12. entity_resolution_check(query, tax_code): So khớp thực thể, alias, MST/tên công ty.
13. company_name_search(query, limit): Tìm doanh nghiệp theo tên.
14. nlp_red_flag_scan(tax_code, text): Quét red flag NLP trên mô tả/hồ sơ.
15. delinquency_check(tax_code, horizon_days): Dự báo rủi ro nợ đọng/chậm nộp.
16. temporal_delinquency_deep(tax_code, horizon_days): Dự báo nợ đọng bằng Temporal Transformer.
17. causal_uplift_recommend(tax_code, objective): Đề xuất biện pháp can thiệp/thu hồi nợ tối ưu.
18. revenue_forecast(tax_code, periods): Dự báo doanh thu/nghĩa vụ thuế tương lai.
19. macro_forecast(scenario): Mô phỏng vĩ mô và kịch bản chính sách.
20. ocr_document_process(document_type, language): OCR hóa đơn/chứng từ.
21. knowledge_search(query, top_k): Tra cứu pháp luật thuế qua RAG/GraphRAG.
22. escalate_to_debate(tax_code): Mở phiên tranh biện đa đặc vụ AI."""

# Tập hợp tên tool hợp lệ (đồng bộ với ModeContracts)
VALID_TOOL_NAMES = {
    "top_n_risky_companies", "company_risk_lookup", "gnn_vat_fraud", "gnn_analysis",
    "invoice_risk_scan", "vat_refund_risk", "vae_anomaly_scan", "motif_detection",
    "ring_scoring", "ownership_analysis", "hetero_gnn_risk", "entity_resolution_check",
    "company_name_search", "nlp_red_flag_scan", "delinquency_check",
    "temporal_delinquency_deep", "causal_uplift_recommend", "revenue_forecast",
    "macro_forecast", "ocr_document_process", "knowledge_search", "escalate_to_debate",
}

# Ánh xạ từ tool_name (agent output) sang intent (orchestrator input)
TOOL_TO_INTENT_MAP = {
    "top_n_risky_companies": "top_n_query",
    "company_risk_lookup": "general_tax_query",
    "gnn_vat_fraud": "vat_network_analysis",
    "gnn_analysis": "vat_network_analysis",
    "invoice_risk_scan": "invoice_risk",
    "vat_refund_risk": "vat_network_analysis",
    "vae_anomaly_scan": "invoice_risk",
    "motif_detection": "vat_network_analysis",
    "ring_scoring": "vat_network_analysis",
    "ownership_analysis": "osint_ownership",
    "hetero_gnn_risk": "general_tax_query",
    "entity_resolution_check": "general_tax_query",
    "company_name_search": "general_tax_query",
    "nlp_red_flag_scan": "invoice_risk",
    "delinquency_check": "delinquency",
    "temporal_delinquency_deep": "delinquency",
    "causal_uplift_recommend": "delinquency",
    "revenue_forecast": "delinquency",
    "macro_forecast": "macro_forecast",
    "ocr_document_process": "general_tax_query",
    "knowledge_search": "general_tax_query",
    "escalate_to_debate": "general_tax_query",
}


@dataclass
class AgenticDecision:
    """Kết quả inference từ LLM V4."""
    thought: str            # Nội dung thẻ <thought>
    tool_name: str          # Tên tool được chọn
    tool_args: dict         # Arguments của tool
    raw_output: str         # Output thô từ model
    confidence: float       # 1.0 nếu parse thành công
    mapped_intent: str      # Intent tương ứng để Orchestrator hiểu


class AgenticLLM:
    """
    Agentic LLM Engine — Load LoRA V4 adapter, chạy inference,
    parse output thành AgenticDecision.

    Thiết kế singleton, lazy-load để tối ưu RAM.
    """

    def __init__(self, adapter_dir: str | Path | None = None):
        self._adapter_dir = Path(adapter_dir) if adapter_dir else _DEFAULT_ADAPTER_DIR
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def load(self) -> bool:
        """Load LoRA V4 adapter. Trả về True nếu thành công."""
        if self._loaded:
            return self._available

        adapter_config = self._adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            logger.warning(
                "[AgenticLLM] Adapter không tìm thấy tại %s — sử dụng pipeline heuristic",
                self._adapter_dir,
            )
            self._loaded = True
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            # Đọc base model name từ adapter config
            with open(adapter_config, "r", encoding="utf-8") as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")

            logger.info("[AgenticLLM] Đang tải base model: %s", base_model_name)

            # Load tokenizer từ adapter (đã lưu cùng khi train)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._adapter_dir), trust_remote_code=True,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            # Kiểm tra GPU availability
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # CPU fallback — không quantize
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            self._model = PeftModel.from_pretrained(base_model, str(self._adapter_dir))
            self._model.eval()
            self._available = True
            self._loaded = True
            logger.info("[AgenticLLM] ✓ LoRA V5 loaded thành công từ %s", self._adapter_dir)
            return True

        except Exception as exc:
            logger.warning("[AgenticLLM] ✗ Không thể load model: %s — fallback heuristic", exc)
            self._loaded = True
            self._available = False
            return False

    def infer(self, query: str) -> Optional[AgenticDecision]:
        """
        Chạy inference: nhận câu hỏi, trả về AgenticDecision.
        Trả về None nếu model không khả dụng hoặc output không hợp lệ.
        """
        if not self._loaded:
            self.load()
        if not self._available:
            return None

        try:
            import torch

            t0 = time.perf_counter()

            messages = [
                {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(text, return_tensors="pt")

            # Di chuyển input tới đúng device của model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    repetition_penalty=1.05,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            latency = (time.perf_counter() - t0) * 1000.0
            logger.info("[AgenticLLM] Inference hoàn tất trong %.0fms", latency)

            return self._parse_output(response)

        except Exception as exc:
            logger.warning("[AgenticLLM] Inference thất bại: %s", exc)
            return None

    def _parse_output(self, raw: str) -> Optional[AgenticDecision]:
        """Parse output thô thành AgenticDecision có cấu trúc."""
        # Trích xuất <thought>
        thought_match = re.search(r"<thought>(.*?)</thought>", raw, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Trích xuất <tool_call> JSON
        tool_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.DOTALL)
        if not tool_match:
            logger.debug("[AgenticLLM] Không tìm thấy <tool_call> trong output: %s", raw[:200])
            return None

        try:
            tool_json = json.loads(tool_match.group(1))
        except json.JSONDecodeError as exc:
            logger.debug("[AgenticLLM] JSON parse thất bại: %s — raw: %s", exc, tool_match.group(1)[:200])
            return None

        tool_name = tool_json.get("name", "")
        tool_args = tool_json.get("arguments", {})

        # Validate tên tool
        if tool_name not in VALID_TOOL_NAMES:
            logger.debug("[AgenticLLM] Tool không hợp lệ: '%s'", tool_name)
            return None

        mapped_intent = TOOL_TO_INTENT_MAP.get(tool_name, "general_tax_query")

        return AgenticDecision(
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_output=raw,
            confidence=1.0,
            mapped_intent=mapped_intent,
        )


# ═══════════════════════════════════════════
#  Singleton — dùng chung toàn hệ thống
# ═══════════════════════════════════════════
_agentic_llm_instance: AgenticLLM | None = None


def get_agentic_llm() -> AgenticLLM:
    """Lấy singleton AgenticLLM."""
    global _agentic_llm_instance
    if _agentic_llm_instance is None:
        _agentic_llm_instance = AgenticLLM()
    return _agentic_llm_instance
