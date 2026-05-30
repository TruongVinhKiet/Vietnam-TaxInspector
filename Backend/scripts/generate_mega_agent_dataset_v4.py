"""
generate_mega_agent_dataset_v4.py - Production-grade Tax Agent SFT dataset.

This generator is intentionally aligned with the real backend tool registry and
mode contracts.  It creates a large ChatML-style JSONL dataset for Qwen/LoRA
training with:

- all registered TaxInspector agent tools, not only the early V3 subset;
- legal GraphRAG multi-turn examples with tool result -> grounded answer;
- Vietnamese no-accent, abbreviation, and typo variants;
- clarification and smalltalk records, including short noisy greetings.

Default output is >100k rows and also writes the legacy alias used by the
Colab notebook: Backend/data/agent_ultimate_dataset.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "data" / "agent_ultimate_dataset_v4.jsonl"
LATEST_OUTPUT = BASE_DIR / "data" / "agent_ultimate_dataset.jsonl"
DEFAULT_SPLIT_RATIOS = {"train": 80, "dev": 10, "test": 10}

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_engine.tax_agent_tool_contracts import (  # noqa: E402
    CANONICAL_TOOL_NAMES,
    tool_prompt_lines,
    validate_tool_call,
)


def strip_accents(value: str) -> str:
    text = unicodedata.normalize("NFD", value or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def repair_mojibake(value: Any) -> Any:
    """Repair common UTF-8-as-Latin-1 mojibake if a data file ever contains it."""
    if isinstance(value, str):
        if any(marker in value for marker in ("Ã", "Â", "á»", "áº", "Ä", "Æ")):
            try:
                repaired = value.encode("latin1").decode("utf-8")
                if sum(ch in repaired for ch in "ăâđêôơưÁÀẢÃẠấầẩẫậếềểễệ") > 0:
                    return repaired
            except Exception:
                return value
        return value
    if isinstance(value, list):
        return [repair_mojibake(v) for v in value]
    if isinstance(value, tuple):
        return tuple(repair_mojibake(v) for v in value)
    if isinstance(value, dict):
        return {k: repair_mojibake(v) for k, v in value.items()}
    return value


def normalize_query(value: str) -> str:
    text = strip_accents(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return clean_spaces(text)


def split_for_group(split_group: str, ratios: dict[str, int] | None = None) -> str:
    """Assign a stable train/dev/test split for a semantic group."""
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    total = sum(ratios.values()) or 100
    digest = hashlib.sha1(normalize_query(split_group).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % total
    running = 0
    for name in ("train", "dev", "test"):
        running += int(ratios.get(name, 0))
        if bucket < running:
            return name
    return "train"


def first_user_message(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def assign_split(record: dict[str, Any], split_group: str | None = None) -> None:
    metadata = record.setdefault("metadata", {})
    group = split_group or metadata.get("split_group") or first_user_message(record)
    metadata["split_group"] = normalize_query(str(group or "ungrouped"))
    metadata["split"] = split_for_group(metadata["split_group"])


def validate_dataset_tool_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return invalid tool-call diagnostics; empty list means the dataset is safe."""
    invalid: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        for message in record.get("messages", []):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                invalid.append({"row": idx, "reason": f"invalid_json:{exc}"})
                continue
            ok, canonical_name, _args, reason = validate_tool_call(payload)
            if not ok:
                invalid.append({
                    "row": idx,
                    "tool": payload.get("name"),
                    "canonical_tool": canonical_name,
                    "reason": reason,
                })
    return invalid


def apply_abbreviations(value: str) -> str:
    replacements = [
        (r"\bmã số thuế\b", "MST"),
        (r"\bdoanh nghiệp\b", "DN"),
        (r"\bcông ty\b", "cty"),
        (r"\bhóa đơn điện tử\b", "HĐĐT"),
        (r"\bgiá trị gia tăng\b", "GTGT"),
        (r"\bthu nhập cá nhân\b", "TNCN"),
        (r"\bthu nhập doanh nghiệp\b", "TNDN"),
        (r"\bchuyển khoản\b", "ck"),
        (r"\bpháp lý\b", "pl"),
        (r"\bquy định\b", "qđ"),
    ]
    out = value
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def drop_one_character(value: str, rng: random.Random) -> str:
    tokens = value.split()
    candidates = [i for i, token in enumerate(tokens) if len(token) >= 4 and token.isalpha()]
    if not candidates:
        return value
    idx = rng.choice(candidates)
    token = tokens[idx]
    drop_idx = rng.randrange(1, len(token))
    tokens[idx] = token[:drop_idx] + token[drop_idx + 1:]
    return " ".join(tokens)


def make_noisy_variants(value: str, rng: random.Random, *, max_variants: int = 4) -> list[str]:
    """Generate realistic Vietnamese user variants: no accents, abbreviations, typos."""
    base = clean_spaces(value)
    variants: list[str] = []
    normalized = strip_accents(base)
    abbreviated = apply_abbreviations(base)
    no_accent_abbrev = strip_accents(abbreviated)
    for item in (base, normalized, abbreviated, no_accent_abbrev):
        item = clean_spaces(item)
        if item and item not in variants:
            variants.append(item)

    lower_norm = normalize_query(base)
    if lower_norm.startswith("xin chao") or lower_norm.startswith("chao"):
        variants.extend(["xin chao", "xn chào", "xi chà", "chao b", "alo ban oi"])

    typo_source = no_accent_abbrev if rng.random() < 0.75 else normalized
    typo = drop_one_character(typo_source, rng)
    if typo and typo not in variants:
        variants.append(typo)

    rng.shuffle(variants)
    return variants[:max_variants]


def make_tax_code(rng: random.Random) -> str:
    return f"{rng.randint(100000000, 9999999999):010d}"


def make_company_name(rng: random.Random) -> str:
    prefixes = ["Công ty TNHH", "CTCP", "Doanh nghiệp tư nhân", "Công ty cổ phần"]
    stems = ["Minh Phát", "An Bình", "Hải Nam", "Tân Thịnh", "Hoàng Gia", "Việt Á", "Đông Dương"]
    suffixes = ["Trading", "Logistics", "Tech", "Foods", "Services", "Holdings"]
    return f"{rng.choice(prefixes)} {rng.choice(stems)} {rng.choice(suffixes)}"


ToolArgsBuilder = Callable[[random.Random], dict[str, Any]]


TOOL_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "name": "top_n_risky_companies",
        "mode": "fraud",
        "description": "Danh sách top N doanh nghiệp rủi ro cao nhất.",
        "thought": "Yêu cầu cần bảng xếp hạng doanh nghiệp rủi ro. Gọi top_n_risky_companies.",
        "args": lambda rng: {"n": rng.choice([5, 10, 20, 50])},
        "templates": [
            "Cho tôi top {n} doanh nghiệp rủi ro cao nhất",
            "Liệt kê {n} DN cần thanh tra gấp",
            "Xếp hạng {n} công ty có điểm gian lận cao nhất",
        ],
    },
    {
        "name": "company_risk_lookup",
        "mode": "fraud",
        "description": "Tra cứu hồ sơ rủi ro tổng thể của một MST.",
        "thought": "Cần chấm điểm rủi ro cho một doanh nghiệp cụ thể. Gọi company_risk_lookup.",
        "args": lambda rng: {"tax_code": make_tax_code(rng)},
        "templates": [
            "Phân tích rủi ro tổng thể MST {tax_code}",
            "Chấm điểm gian lận cho doanh nghiệp {tax_code}",
            "Doanh nghiệp {tax_code} có nên đưa vào diện thanh tra không",
        ],
    },
    {
        "name": "gnn_analysis",
        "mode": "fraud",
        "description": "Pipeline gian lận VAT kết hợp graph/GNN.",
        "thought": "Yêu cầu đánh giá gian lận VAT bằng tín hiệu đồ thị. Gọi gnn_analysis.",
        "args": lambda rng: {"tax_code": make_tax_code(rng)},
        "templates": [
            "Đánh giá gian lận VAT bằng GNN cho MST {tax_code}",
            "Chạy pipeline graph fraud cho doanh nghiệp {tax_code}",
        ],
    },
    {
        "name": "gnn_analysis",
        "mode": "vat",
        "description": "Phân tích mạng lưới giao dịch VAT.",
        "thought": "Người dùng muốn truy vết mạng lưới hóa đơn. Gọi gnn_analysis.",
        "args": lambda rng: {"tax_code": make_tax_code(rng)},
        "templates": [
            "Truy vết mạng lưới VAT của công ty {tax_code}",
            "Vẽ sơ đồ giao dịch hóa đơn quanh MST {tax_code}",
            "Phân tích GNN mạng lưới đầu vào đầu ra của {tax_code}",
        ],
    },
    {
        "name": "invoice_risk_scan",
        "mode": "vat",
        "description": "Rà soát rủi ro hóa đơn đầu vào/đầu ra.",
        "thought": "Yêu cầu kiểm tra rủi ro hóa đơn. Gọi invoice_risk_scan.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "period": rng.choice(["2024Q4", "2025Q1", "2025Q2"])},
        "templates": [
            "Kiểm tra hóa đơn đầu vào của MST {tax_code}",
            "Rà soát hóa đơn bất hợp pháp kỳ {period} cho {tax_code}",
        ],
    },
    {
        "name": "vat_refund_risk",
        "mode": "vat",
        "description": "Đánh giá rủi ro hồ sơ hoàn thuế VAT.",
        "thought": "Yêu cầu liên quan hoàn thuế GTGT cần pipeline VAT refund. Gọi vat_refund_risk.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "period": rng.choice(["2024Q4", "2025Q1"])},
        "templates": [
            "Đánh giá rủi ro hoàn thuế GTGT của {tax_code}",
            "Hồ sơ xin hoàn thuế VAT kỳ {period} của {tax_code} có đáng ngờ không",
        ],
    },
    {
        "name": "vae_anomaly_scan",
        "mode": "vat",
        "description": "Quét bất thường hóa đơn bằng VAE.",
        "thought": "Cần phát hiện anomaly hóa đơn bằng deep learning. Gọi vae_anomaly_scan.",
        "args": lambda rng: {"tax_code": make_tax_code(rng)},
        "templates": [
            "Quét bất thường hóa đơn bằng VAE cho {tax_code}",
            "Tìm outlier giao dịch của doanh nghiệp {tax_code}",
        ],
    },
    {
        "name": "motif_detection",
        "mode": "vat",
        "description": "Phát hiện motif/vòng lặp giao dịch.",
        "thought": "Cần tìm vòng giao dịch khép kín và motif đáng ngờ. Gọi motif_detection.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "max_hops": rng.choice([2, 3, 4])},
        "templates": [
            "Phát hiện vòng lặp hóa đơn quanh MST {tax_code}",
            "Tìm motif giao dịch đáng ngờ của {tax_code}",
        ],
    },
    {
        "name": "ring_scoring",
        "mode": "vat",
        "description": "Chấm điểm vòng giao dịch VAT.",
        "thought": "Yêu cầu chấm điểm ring/cycle VAT. Gọi ring_scoring.",
        "args": lambda rng: {"tax_code": make_tax_code(rng)},
        "templates": [
            "Chấm điểm vòng giao dịch VAT cho {tax_code}",
            "Vòng hóa đơn của {tax_code} rủi ro mức nào",
        ],
    },
    {
        "name": "ownership_analysis",
        "mode": "fraud",
        "description": "Phân tích sở hữu chéo, UBO, common controller.",
        "thought": "Cần truy vết chủ sở hữu và cấu trúc liên kết. Gọi ownership_analysis.",
        "args": lambda rng: {"tax_code": make_tax_code(rng)},
        "templates": [
            "Ai là chủ sở hữu thực sự của doanh nghiệp {tax_code}",
            "Phân tích sở hữu chéo và UBO của {tax_code}",
        ],
    },
    {
        "name": "hetero_gnn_risk",
        "mode": "fraud",
        "description": "Đánh giá rủi ro bằng HeteroGNN/HGT.",
        "thought": "Cần model đồ thị dị thể cho hồ sơ có nhiều loại thực thể. Gọi hetero_gnn_risk.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "node_type": "company"},
        "templates": [
            "Chạy HeteroGNN đánh giá rủi ro cho công ty {tax_code}",
            "Phân tích HGT trên mạng lưới doanh nghiệp {tax_code}",
        ],
    },
    {
        "name": "entity_resolution_check",
        "mode": "vat",
        "description": "So khớp thực thể, alias, MST/tên công ty.",
        "thought": "Yêu cầu nghi ngờ trùng/thay tên thực thể. Gọi entity_resolution_check.",
        "args": lambda rng: {"query": make_company_name(rng), "tax_code": make_tax_code(rng)},
        "templates": [
            "Kiểm tra {company_name} có alias hay trùng thực thể với MST {tax_code} không",
            "So khớp entity resolution cho công ty {company_name}",
        ],
    },
    {
        "name": "company_name_search",
        "mode": "fraud",
        "description": "Tìm doanh nghiệp theo tên.",
        "thought": "Người dùng đưa tên doanh nghiệp thay vì MST. Gọi company_name_search.",
        "args": lambda rng: {"name": make_company_name(rng), "limit": rng.choice([5, 10])},
        "templates": [
            "Tìm thông tin về {company_name}",
            "Tra cứu công ty tên {company_name}",
        ],
    },
    {
        "name": "nlp_red_flag_scan",
        "mode": "fraud",
        "description": "Quét red flag NLP trên mô tả/hồ sơ.",
        "thought": "Cần phân tích dấu hiệu đỏ trong văn bản/hồ sơ. Gọi nlp_red_flag_scan.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "text": rng.choice(["lỗ liên tục nhưng doanh thu tăng mạnh", "mua bán hóa đơn lòng vòng"])},
        "templates": [
            "Quét red flag NLP cho hồ sơ {tax_code}",
            "Đọc mô tả hồ sơ {tax_code} xem có dấu hiệu gian lận không",
        ],
    },
    {
        "name": "delinquency_check",
        "mode": "delinquency",
        "description": "Dự báo rủi ro nợ đọng/chậm nộp.",
        "thought": "Yêu cầu dự báo nợ đọng thuế. Gọi delinquency_check.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "horizon_days": rng.choice([30, 60, 90])},
        "templates": [
            "Dự báo khả năng nợ đọng thuế 90 ngày của {tax_code}",
            "Công ty {tax_code} có nguy cơ chậm nộp kỳ tới không",
        ],
    },
    {
        "name": "temporal_delinquency_deep",
        "mode": "delinquency",
        "description": "Dự báo nợ đọng bằng Temporal Transformer.",
        "thought": "Cần mô hình chuỗi thời gian chuyên sâu cho nợ đọng. Gọi temporal_delinquency_deep.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "horizon_days": rng.choice([90, 180])},
        "templates": [
            "Chạy Temporal Transformer dự báo nợ đọng cho {tax_code}",
            "Phân tích chuỗi thời gian thanh toán thuế của {tax_code}",
        ],
    },
    {
        "name": "causal_uplift_recommend",
        "mode": "delinquency",
        "description": "Đề xuất biện pháp can thiệp/thu hồi nợ tối ưu.",
        "thought": "Người dùng cần chọn hành động cưỡng chế tối ưu. Gọi causal_uplift_recommend.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "objective": "maximize_recovery"},
        "templates": [
            "Nên áp dụng biện pháp cưỡng chế nào hiệu quả nhất cho {tax_code}",
            "So sánh phong tỏa tài khoản và ngừng hóa đơn để thu nợ {tax_code}",
        ],
    },
    {
        "name": "revenue_forecast",
        "mode": "delinquency",
        "description": "Dự báo doanh thu/nghĩa vụ thuế tương lai.",
        "thought": "Cần dự báo doanh thu hoặc nghĩa vụ thuế. Gọi revenue_forecast.",
        "args": lambda rng: {"tax_code": make_tax_code(rng), "periods": rng.choice([2, 4, 8])},
        "templates": [
            "Dự báo doanh thu quý tới của {tax_code}",
            "Ước lượng nghĩa vụ thuế 4 kỳ tiếp theo cho {tax_code}",
        ],
    },
    {
        "name": "macro_forecast",
        "mode": "macro",
        "description": "Mô phỏng vĩ mô và kịch bản chính sách.",
        "thought": "Yêu cầu mô phỏng tham số vĩ mô. Gọi macro_forecast.",
        "args": lambda rng: {"scenario": {"gdp_growth": round(rng.uniform(2.0, 8.0), 1), "vat_rate": rng.choice([8, 10, 12]), "cpi": round(rng.uniform(2.0, 5.5), 1)}},
        "templates": [
            "Mô phỏng nếu GDP tăng {gdp}% và VAT là {vat}%",
            "Chạy kịch bản vĩ mô với CPI {cpi}% và thuế GTGT {vat}%",
        ],
    },
    {
        "name": "ocr_document_process",
        "mode": "vat",
        "description": "OCR hóa đơn/chứng từ.",
        "thought": "Yêu cầu đọc chứng từ/hóa đơn bằng OCR. Gọi ocr_document_process.",
        "args": lambda rng: {
            "file_path": "<uploaded_file>",
            "document_type": rng.choice(["invoice", "receipt", "tax_notice"]),
            "language": "vi",
        },
        "templates": [
            "Đọc hóa đơn tôi đính kèm và trích xuất thông tin thuế",
            "OCR chứng từ thuế này rồi kiểm tra trường dữ liệu",
        ],
    },
    {
        "name": "knowledge_search",
        "mode": "legal",
        "description": "Tra cứu pháp luật thuế qua RAG/GraphRAG.",
        "thought": "Đây là câu hỏi pháp luật thuế, cần tra cứu căn cứ. Gọi knowledge_search.",
        "args": lambda rng: {"query": rng.choice(LEGAL_TOOL_QUERIES), "top_k": 5},
        "templates": [
            "Căn cứ pháp lý về thời hạn nộp tờ khai GTGT quý là gì",
            "Mua hàng trên 20 triệu cần chuyển khoản để khấu trừ thuế đúng không",
            "Hộ kinh doanh bán hàng online phải nộp thuế gì",
        ],
    },
]

_UNKNOWN_BLUEPRINT_TOOLS = [spec["name"] for spec in TOOL_BLUEPRINTS if spec["name"] not in CANONICAL_TOOL_NAMES]
if _UNKNOWN_BLUEPRINT_TOOLS:
    TOOL_BLUEPRINTS = [spec for spec in TOOL_BLUEPRINTS if spec["name"] in CANONICAL_TOOL_NAMES]
_DEDUPED_BLUEPRINTS: dict[str, dict[str, Any]] = {}
for _spec in TOOL_BLUEPRINTS:
    _DEDUPED_BLUEPRINTS.setdefault(_spec["name"], _spec)
TOOL_BLUEPRINTS = list(_DEDUPED_BLUEPRINTS.values())


LEGAL_TOOL_QUERIES = [
    "Luật Quản lý thuế 38/2019/QH14 thời hạn nộp hồ sơ khai thuế",
    "Nghị định 125/2020/NĐ-CP xử phạt chậm nộp tờ khai thuế",
    "Thông tư 40/2021/TT-BTC thuế hộ kinh doanh cá nhân kinh doanh",
    "Thông tư 96/2015/TT-BTC thanh toán không dùng tiền mặt 20 triệu",
    "Nghị định 123/2020/NĐ-CP hóa đơn điện tử sai sót",
]


PREFIXES = ["", "Hệ thống ơi, ", "Hãy ", "Vui lòng ", "Tôi cần ", "Giúp tôi ", "Làm ơn "]
SUFFIXES = ["", " nhé.", " giúp tôi.", " ngay.", " cho tôi.", " trong hôm nay.", " để báo cáo lãnh đạo."]

SMALLTALK_QUERIES = [
    "xin chào",
    "xin chao",
    "xn chào",
    "xi chà",
    "chào bạn",
    "alo bạn ơi",
    "bạn làm được gì",
    "agent có thể giúp gì",
    "cảm ơn",
    "cam on ban",
]

CLARIFICATION_QUERIES = [
    "Phân tích doanh nghiệp này giúp tôi",
    "Kiểm tra rủi ro công ty đó",
    "Hồ sơ này có vấn đề gì không",
    "Cho tôi biết có nên thanh tra không",
    "Tôi muốn hỏi về thuế nhưng chưa biết bắt đầu từ đâu",
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legal_deep_data() -> list[dict[str, Any]]:
    """Load all curated legal datasets and citizen FAQ snippets."""
    records: list[dict[str, Any]] = []
    sources = [
        ("generate_mega_agent_dataset.py", "LEGAL_DEEP_DATA"),
        ("legal_deep_data.py", "LEGAL_DEEP_DATA_NEW"),
        ("legal_deep_data_2.py", "LEGAL_DEEP_DATA_NEW_2"),
        ("legal_deep_data_3.py", "LEGAL_DEEP_DATA_NEW_3"),
        ("legal_deep_data_4.py", "LEGAL_DEEP_DATA_NEW_4"),
    ]
    for filename, attr in sources:
        try:
            module = _load_module(SCRIPT_DIR / filename)
            values = getattr(module, attr, []) if module else []
            for item in values:
                fixed = repair_mojibake(item)
                if all(fixed.get(k) for k in ("q", "tool_query", "doc_title", "doc_snippet", "answer")):
                    records.append(fixed)
        except Exception as exc:
            print(f"[WARN] Cannot load {filename}:{attr}: {exc}")

    try:
        from ml_engine.tax_agent_citizen_legal import SNIPPETS

        for snippet in SNIPPETS:
            q = f"{snippet.title} quy định như thế nào?"
            records.append({
                "q": q,
                "tool_query": f"{snippet.title} {' '.join(snippet.keywords)} {snippet.legal_reference}",
                "doc_title": snippet.legal_reference,
                "doc_snippet": snippet.text,
                "answer": (
                    f"Dựa trên nhóm căn cứ **{snippet.legal_reference}**, có thể xử lý như sau:\n\n"
                    f"{snippet.text}\n\n"
                    "**Bước xử lý đề xuất:**\n"
                    + "\n".join(f"- {step}" for step in snippet.next_steps)
                    + "\n\nLưu ý: cần đối chiếu văn bản pháp luật chính thức và tình trạng hiệu lực trước khi ra quyết định ràng buộc."
                ),
            })
    except Exception as exc:
        print(f"[WARN] Cannot load citizen legal snippets: {exc}")

    unique: dict[str, dict[str, Any]] = {}
    for item in records:
        key = normalize_query(item["q"])
        unique[key] = item
    return list(unique.values())


def build_system_prompt() -> str:
    return (
        "Bạn là TaxInspector AI - Trợ lý Thanh tra Thuế và tư vấn pháp luật thuế.\n"
        "Nhiệm vụ: hiểu yêu cầu tiếng Việt tự nhiên, kể cả không dấu/viết tắt/sai ký tự nhẹ; "
        "chọn đúng công cụ/model; không tự bịa kết quả.\n\n"
        "Quy tắc trả lời khi chưa có tool result:\n"
        "- Nếu là tác vụ nghiệp vụ, suy nghĩ ngắn trong <thought>, sau đó gọi đúng một tool trong <tool_call> JSON rồi dừng.\n"
        "- Nếu thiếu MST/tệp/thông tin bắt buộc, hỏi lại rõ ràng thay vì gọi tool sai.\n"
        "- Nếu chỉ là chào hỏi/cảm ơn/hỏi khả năng, trả lời trực tiếp, không gọi tool.\n\n"
        "Quy tắc sau khi nhận tool result pháp luật:\n"
        "- Tổng hợp bằng tiếng Việt rõ ràng, có căn cứ, điều kiện áp dụng, bước xử lý, cảnh báo rủi ro.\n"
        "- Ưu tiên GraphRAG/knowledge graph, authority path và tình trạng hiệu lực; không kết luận vượt quá chứng cứ.\n\n"
        "Công cụ khả dụng:\n" + "\n".join(tool_prompt_lines())
    )


SYSTEM_PROMPT = build_system_prompt()


def tool_call_content(tool_name: str, args: dict[str, Any]) -> str:
    payload = {"name": tool_name, "arguments": args}
    ok, canonical_name, canonical_args, reason = validate_tool_call(payload)
    if not ok:
        raise ValueError(f"Invalid tool call for training data: {tool_name} ({reason})")
    payload = {"name": canonical_name, "arguments": canonical_args}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def render_template(template: str, args: dict[str, Any], rng: random.Random) -> str:
    tax_code = args.get("tax_code") or args.get("entity_id") or make_tax_code(rng)
    values = {
        "tax_code": tax_code,
        "n": args.get("n", rng.choice([5, 10, 20])),
        "period": args.get("period") or args.get("refund_period") or "2025Q1",
        "gdp": round((args.get("scenario") or {}).get("gdp_growth", 5.5), 1),
        "vat": (args.get("scenario") or {}).get("vat_rate", 10),
        "cpi": round((args.get("scenario") or {}).get("cpi", 3.5), 1),
        "company_name": args.get("name") or args.get("query") or make_company_name(rng),
    }
    return template.format(**values)


def make_task_record(query: str, tool_name: str, args: dict[str, Any], thought: str) -> dict[str, Any]:
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"<thought>{thought}</thought>\n<tool_call>{tool_call_content(tool_name, args)}</tool_call>"},
        ],
        "metadata": {
            "kind": "tool_routing",
            "expected_tool": tool_name,
        },
    }
    assign_split(record, query)
    return record


def make_legal_record(data_item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    base_query = data_item["q"]
    query = rng.choice(make_noisy_variants(
        f"{rng.choice(PREFIXES)}{base_query}{rng.choice(SUFFIXES)}",
        rng,
        max_variants=5,
    ))
    args = {"query": data_item["tool_query"], "top_k": 5}
    thought = "Đây là câu hỏi pháp luật thuế. Cần tra cứu Knowledge Graph/GraphRAG trước khi tư vấn."
    tool_result = {
        "status": "success",
        "query": data_item["tool_query"],
        "results": [
            {
                "title": data_item["doc_title"],
                "content": data_item["doc_snippet"],
                "score": 0.94,
                "source_type": "legal",
                "effective_status": "active_or_verify",
            }
        ],
        "knowledge_graph": {
            "anchors": [data_item["doc_title"]],
            "authority_path": [
                {"display_name": data_item["doc_title"], "authority_rank": 1, "effective_status": "active_or_verify"}
            ],
            "relations": [
                {"source": data_item["doc_title"], "relation": "guides_or_amends", "target": "tax_obligation"}
            ],
        },
        "retrieval_quality": {
            "grounding": "graph_rag_or_citizen_fallback",
            "requires_effective_date_check": True,
        },
    }
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"<thought>{thought}</thought>\n<tool_call>{tool_call_content('knowledge_search', args)}</tool_call>"},
            {"role": "tool", "name": "knowledge_search", "content": json.dumps(tool_result, ensure_ascii=False)},
            {"role": "assistant", "content": data_item["answer"]},
        ],
        "metadata": {
            "kind": "legal_graphrag_answer",
            "expected_tool": "knowledge_search",
        },
    }
    assign_split(record, base_query)
    return record


def make_smalltalk_record(query: str) -> dict[str, Any]:
    normalized = normalize_query(query)
    if "cam on" in normalized or "thank" in normalized:
        answer = "Rất vui được hỗ trợ. Khi cần phân tích thêm, bạn cứ gửi MST, tên doanh nghiệp, câu hỏi pháp lý hoặc tệp dữ liệu."
    elif "lam duoc gi" in normalized or "giup gi" in normalized:
        answer = (
            "Tôi có thể tra cứu pháp luật thuế, phân tích rủi ro gian lận, truy vết mạng VAT, "
            "dự báo nợ đọng, mô phỏng vĩ mô và xử lý CSV/Excel/hóa đơn."
        )
    else:
        answer = "Xin chào! Bạn có thể gửi câu hỏi thuế, MST/tên doanh nghiệp, hoặc upload tệp cần phân tích."
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"kind": "smalltalk", "expected_tool": None},
    }
    assign_split(record, query)
    return record


def make_clarification_record(query: str) -> dict[str, Any]:
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": (
                    "Tôi cần thêm thông tin để chọn đúng model. Bạn vui lòng cung cấp MST/tên doanh nghiệp, "
                    "loại hồ sơ hoặc tệp dữ liệu cần phân tích."
                ),
            },
        ],
        "metadata": {"kind": "clarification", "expected_tool": None},
    }
    assign_split(record, query)
    return record


def generate(
    total_simple: int = 90000,
    total_legal: int = 30000,
    total_smalltalk: int = 5000,
    total_clarification: int = 5000,
    *,
    seed: int = 42,
    output_path: Path | str = OUTPUT,
    write_latest_alias: bool = True,
) -> dict[str, Any]:
    """Generate the dataset and return a quality summary."""
    rng = random.Random(seed)
    output_path = Path(output_path)
    legal_data = load_legal_deep_data()
    if len(legal_data) < 50:
        raise RuntimeError(f"Legal deep corpus too small: {len(legal_data)} records")

    dataset: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    per_tool = total_simple // len(TOOL_BLUEPRINTS)
    remainder = total_simple % len(TOOL_BLUEPRINTS)
    for idx, spec in enumerate(TOOL_BLUEPRINTS):
        target = per_tool + (1 if idx < remainder else 0)
        for _ in range(target):
            args = spec["args"](rng)
            template = rng.choice(spec["templates"])
            base_query = render_template(template, args, rng)
            wrapped = f"{rng.choice(PREFIXES)}{base_query}{rng.choice(SUFFIXES)}"
            query = rng.choice(make_noisy_variants(wrapped, rng, max_variants=4))
            record = make_task_record(query, spec["name"], args, spec["thought"])
            record["metadata"]["mode"] = spec["mode"]
            assign_split(record, base_query)
            dataset.append(record)
            counts[spec["name"]] += 1

    for _ in range(total_legal):
        record = make_legal_record(rng.choice(legal_data), rng)
        dataset.append(record)
        counts["knowledge_search_deep_answer"] += 1

    for _ in range(total_smalltalk):
        base = rng.choice(SMALLTALK_QUERIES)
        query = rng.choice(make_noisy_variants(base, rng, max_variants=6))
        dataset.append(make_smalltalk_record(query))
        counts["smalltalk"] += 1

    for _ in range(total_clarification):
        query = rng.choice(make_noisy_variants(rng.choice(CLARIFICATION_QUERIES), rng, max_variants=4))
        dataset.append(make_clarification_record(query))
        counts["clarification"] += 1

    invalid_tool_calls = validate_dataset_tool_calls(dataset)
    if invalid_tool_calls:
        preview = invalid_tool_calls[:5]
        raise RuntimeError(f"Generated dataset contains invalid tool calls: {preview}")

    rng.shuffle(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if write_latest_alias and output_path.resolve() != LATEST_OUTPUT.resolve():
        shutil.copyfile(output_path, LATEST_OUTPUT)

    split_distribution = Counter(
        str(record.get("metadata", {}).get("split") or "unknown")
        for record in dataset
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    summary = {
        "output": str(output_path),
        "latest_alias": str(LATEST_OUTPUT) if write_latest_alias else None,
        "total_records": len(dataset),
        "legal_seed_records": len(legal_data),
        "tool_count": len(TOOL_BLUEPRINTS),
        "distribution": dict(sorted(counts.items())),
        "split_distribution": dict(sorted(split_distribution.items())),
        "canonical_tools": sorted(CANONICAL_TOOL_NAMES),
        "deprecated_tools_dropped": list(_UNKNOWN_BLUEPRINT_TOOLS),
        "seed": seed,
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TaxInspector agent SFT JSONL dataset.")
    parser.add_argument("--total-simple", type=int, default=90000)
    parser.add_argument("--total-legal", type=int, default=30000)
    parser.add_argument("--total-smalltalk", type=int, default=5000)
    parser.add_argument("--total-clarification", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-latest-alias", action="store_true")
    args = parser.parse_args()
    generate(
        total_simple=args.total_simple,
        total_legal=args.total_legal,
        total_smalltalk=args.total_smalltalk,
        total_clarification=args.total_clarification,
        seed=args.seed,
        output_path=args.output,
        write_latest_alias=not args.no_latest_alias,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
