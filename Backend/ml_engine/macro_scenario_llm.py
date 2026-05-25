"""
Natural-language macro scenario interpreter with HITL memory.

The interpreter first searches approved human-reviewed memories. If no close
match exists, it can call an LLM provider via environment variables, then falls
back to deterministic rules. LLM output is never treated as training memory
until the user approves/rates it.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"
MEMORY_PATH = DATA_DIR / "macro_text_scenario_memory.jsonl"


PROVIDER_PRIORITY = ("gemini", "openrouter", "github", "groq", "cohere")


def try_llm_interpretation(prompt: str) -> Optional[Dict[str, Any]]:
    for provider in PROVIDER_PRIORITY:
        try:
            payload = _call_provider(provider, prompt)
            if payload:
                payload["llm_provider"] = provider
                return payload
        except Exception as e:
            print(f"[LLM Waterfall] Provider {provider} failed: {e}")
            continue
    return None


def _call_provider(provider: str, prompt: str) -> Optional[Dict[str, Any]]:
    if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
        return _post_gemini(os.environ["GEMINI_API_KEY"], prompt)
    if provider == "openrouter" and os.environ.get("OPENROUTER_API_KEY"):
        return _post_openai_compatible(
            "https://openrouter.ai/api/v1/chat/completions",
            os.environ["OPENROUTER_API_KEY"],
            "openrouter/auto",
            prompt,
        )
    if provider == "groq" and os.environ.get("GROQ_API_KEY"):
        return _post_openai_compatible(
            "https://api.groq.com/openai/v1/chat/completions",
            os.environ["GROQ_API_KEY"],
            "llama-3.1-8b-instant",
            prompt,
        )
    if provider == "github" and (os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_PAT")):
        token = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        return _post_openai_compatible(
            "https://models.inference.ai.azure.com/chat/completions",
            token,
            "gpt-4o-mini",
            prompt,
        )
    if provider == "cohere" and os.environ.get("COHERE_API_KEY"):
        return _post_cohere(os.environ["COHERE_API_KEY"], prompt)
    return None


def try_trained_model_interpretation(text: str) -> Optional[Dict[str, Any]]:
    """Use the reviewed local model before paying an LLM call."""
    try:
        from ml_engine.macro_retrain_pipeline import predict_scenario_from_trained_model
        return predict_scenario_from_trained_model(text)
    except Exception:
        return None


@dataclass
class ScenarioMemoryHit:
    payload: Dict[str, Any]
    similarity: float


def interpret_text_scenario(
    text: str,
    *,
    province_code: Optional[str] = None,
    horizon_years: int = 5,
    force_llm: bool = False,
) -> Dict[str, Any]:
    clean = text.strip()
    if not clean:
        raise ValueError("Scenario text is empty.")

    memory_hit = None if force_llm else find_approved_memory(clean)
    if memory_hit:
        payload = dict(memory_hit.payload)
        payload["source"] = "memory"
        payload["memory_similarity"] = round(memory_hit.similarity, 4)
        payload["llm_provider"] = payload.get("llm_provider") or "approved_memory"
        payload["scenario_id"] = f"memory-{uuid.uuid4().hex[:12]}"
        payload["province_code"] = province_code
        payload["horizon_years"] = horizon_years
        return payload

    model_payload = None if force_llm else try_trained_model_interpretation(clean)
    if model_payload:
        model_payload["source"] = "trained_memory_model"
        payload = normalize_interpretation_payload(model_payload, clean)
        payload["scenario_id"] = f"model-{uuid.uuid4().hex[:12]}"
        payload["province_code"] = province_code
        payload["horizon_years"] = horizon_years
        payload["requires_human_review"] = True
        return payload

    prompt = build_interpretation_prompt(clean, province_code=province_code, horizon_years=horizon_years)
    llm_payload = try_llm_interpretation(prompt)
    if not llm_payload:
        llm_payload = rule_based_interpretation(clean)
        llm_payload["source"] = "rule_fallback"
    else:
        llm_payload["source"] = "llm"

    payload = normalize_interpretation_payload(llm_payload, clean)
    payload["scenario_id"] = f"scenario-{uuid.uuid4().hex[:12]}"
    payload["province_code"] = province_code
    payload["horizon_years"] = horizon_years
    payload["requires_human_review"] = True
    return payload


def find_approved_memory(text: str, *, min_similarity: float = 0.86) -> Optional[ScenarioMemoryHit]:
    normalized = normalize_text(text)
    best: Optional[ScenarioMemoryHit] = None
    for row in load_memory_rows():
        if row.get("review_status") != "approved":
            continue
        if float(row.get("rating") or 0.0) < 4.0:
            continue
        score = SequenceMatcher(None, normalized, str(row.get("normalized_text") or "")).ratio()
        if score >= min_similarity and (not best or score > best.similarity):
            best = ScenarioMemoryHit(payload=dict(row.get("payload") or {}), similarity=score)
    return best


def remember_scenario_feedback(
    *,
    text: str,
    payload: Dict[str, Any],
    rating: float,
    approved: bool,
    notes: str = "",
    reviewer: str = "user",
) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "memory_id": f"macro-memory-{uuid.uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_text": text,
        "normalized_text": normalize_text(text),
        "payload": payload,
        "rating": max(0.0, min(5.0, float(rating))),
        "review_status": "approved" if approved and rating >= 4 else "rejected",
        "reviewer": reviewer,
        "notes": notes,
    }
    with MEMORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_memory_rows() -> List[Dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def memory_status() -> Dict[str, Any]:
    rows = load_memory_rows()
    approved = [r for r in rows if r.get("review_status") == "approved"]
    return {
        "memory_path": str(MEMORY_PATH),
        "total": len(rows),
        "approved": len(approved),
        "rejected": len([r for r in rows if r.get("review_status") == "rejected"]),
        "avg_rating": round(sum(float(r.get("rating") or 0) for r in approved) / max(1, len(approved)), 3),
    }


def build_interpretation_prompt(text: str, *, province_code: Optional[str], horizon_years: int) -> str:
    return f"""Bạn là chuyên gia kinh tế vĩ mô Việt Nam và mô hình hóa chính sách thuế.

Hãy đọc kịch bản giả định sau và trả về JSON thuần, không markdown.

Kịch bản người dùng:
{text}

Tỉnh trọng tâm nếu có: {province_code or "toàn quốc"}
Kỳ hạn mô phỏng: {horizon_years} năm

Schema JSON:
{{
  "scenario_title": "tiêu đề ngắn",
  "event_type": "policy|trade_war|natural_disaster|pandemic|financial_crisis|trade_agreement|growth|geopolitics|infrastructure_shock|sanction|war|unknown",
  "severity": "low|medium|high|extreme",
  "affected_provinces": [],
  "affected_sectors": [],
  "macro_parameters": {{
    "gdp_delta_pct": -2.0,
    "tax_rate_delta": 0.0,
    "compliance_delta": -0.02,
    "unemployment_delta": 0.5,
    "fdi_delta_pct": -5.0
  }},
  "candidate_events": [
    {{
      "headline": "tiêu đề bài báo giả lập",
      "summary": "tóm tắt 2 câu",
      "probability": 0.55,
      "impact_level": "low|medium|high|extreme"
    }}
  ],
  "reasoning_brief": "giải thích ngắn vì sao ra hệ số này",
  "confidence": 0.72
}}

Quy tắc:
- Nếu tác động toàn quốc, affected_provinces = [].
- Không bịa số quá cực đoan trừ chiến tranh/cấm vận/đại dịch.
- Thuế suất tăng 30-40% với hàng Việt Nam thường là trade_war hoặc sanction, tác động GDP/FDI âm, thất nghiệp dương.
- Đại dịch X tác động tiêu cực tới dịch vụ, du lịch, logistics, y tế và chuỗi cung ứng.
"""


# Forward to providers defined above
# The actual functions try_llm_interpretation and _call_provider are declared at the top of the file.


def _post_openai_compatible(url: str, api_key: str, model: str, prompt: str) -> Optional[Dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 1200,
    }
    data = _http_json(url, body, {"Authorization": f"Bearer {api_key}"})
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return extract_json(text)


def _post_gemini(api_key: str, prompt: str) -> Optional[Dict[str, Any]]:
    # Try production v1 endpoint first
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.15}}
        data = _http_json(url, body, {})
        parts = ((((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or [])
        text = "\n".join(str(p.get("text") or "") for p in parts).strip()
        result = extract_json(text)
        if result:
            return result
    except Exception as e:
        print(f"[Gemini v1 REST Failed] {e}, trying v1beta...")
        
    # Fallback to v1beta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.15}}
    data = _http_json(url, body, {})
    parts = ((((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or [])
    text = "\n".join(str(p.get("text") or "") for p in parts).strip()
    return extract_json(text)


def _post_cohere(api_key: str, prompt: str) -> Optional[Dict[str, Any]]:
    body = {
        "model": "command-r7b-12-2024",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
    }
    data = _http_json("https://api.cohere.com/v2/chat", body, {"Authorization": f"Bearer {api_key}"})
    content = data.get("message", {}).get("content", [])
    text = "\n".join(str(part.get("text") or "") for part in content).strip()
    return extract_json(text)


def _http_json(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=18) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def rule_based_interpretation(text: str) -> Dict[str, Any]:
    normalized = normalize_text(text)
    percent_values = [float(x) for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", normalized.replace(",", "."))]
    pct = max(percent_values) if percent_values else 0.0
    params = {"gdp_delta_pct": 0.0, "tax_rate_delta": 0.0, "compliance_delta": 0.0, "unemployment_delta": 0.0, "fdi_delta_pct": 0.0}
    event_type = "unknown"
    severity = "medium"
    sectors: List[str] = []

    if any(k in normalized for k in ["thue", "tariff", "danh thue", "ap thue", "my danh thue"]):
        event_type = "trade_war"
        severity = "high" if pct >= 25 else "medium"
        params.update({
            "gdp_delta_pct": -min(8.0, max(1.0, pct * 0.12)),
            "tax_rate_delta": 0.0,
            "compliance_delta": -min(0.06, pct / 1000.0),
            "unemployment_delta": min(2.5, pct * 0.035),
            "fdi_delta_pct": -min(18.0, pct * 0.35),
        })
        sectors = ["Xuất khẩu", "Sản xuất", "Logistics", "FDI"]
    elif any(k in normalized for k in ["cam van", "sanction", "chien tranh", "world war", "the chien"]):
        event_type = "sanction" if "cam van" in normalized or "sanction" in normalized else "war"
        severity = "extreme"
        params.update({"gdp_delta_pct": -9.0, "compliance_delta": -0.05, "unemployment_delta": 3.0, "fdi_delta_pct": -25.0})
        sectors = ["Năng lượng", "Xuất nhập khẩu", "Tài chính", "Logistics"]
    elif any(k in normalized for k in ["dai dich", "pandemic", "dich x", "covid"]):
        event_type = "pandemic"
        severity = "extreme"
        params.update({"gdp_delta_pct": -6.0, "compliance_delta": -0.04, "unemployment_delta": 2.2, "fdi_delta_pct": -14.0})
        sectors = ["Du lịch", "Dịch vụ", "Logistics", "Y tế"]
    elif any(k in normalized for k in ["bao", "lu", "han han", "thien tai"]):
        event_type = "natural_disaster"
        severity = "high"
        params.update({"gdp_delta_pct": -2.5, "compliance_delta": -0.015, "unemployment_delta": 0.6, "fdi_delta_pct": -3.0})
        sectors = ["Nông nghiệp", "Hạ tầng", "Logistics"]

    title = text.strip()[:96] or "Kịch bản vĩ mô giả định"
    return {
        "scenario_title": title,
        "event_type": event_type,
        "severity": severity,
        "affected_provinces": [],
        "affected_sectors": sectors,
        "macro_parameters": params,
        "candidate_events": build_candidate_event_cards(title, event_type, severity, params),
        "reasoning_brief": "Suy luận bằng bộ quy tắc vĩ mô dựa trên từ khóa, phần trăm cú sốc và dữ liệu sự kiện lịch sử.",
        "confidence": 0.58 if event_type == "unknown" else 0.68,
    }


def build_candidate_event_cards(title: str, event_type: str, severity: str, params: Dict[str, float]) -> List[Dict[str, Any]]:
    direction = "suy giảm" if params.get("gdp_delta_pct", 0) < 0 else "cải thiện"
    return [
        {
            "headline": f"{title}: thị trường bước vào pha điều chỉnh",
            "summary": f"Tác động {event_type} có thể khiến GDP {direction}, thất nghiệp thay đổi khoảng {params.get('unemployment_delta', 0):+.1f} điểm phần trăm.",
            "probability": 0.62,
            "impact_level": severity,
        },
        {
            "headline": "Doanh nghiệp điều chỉnh chuỗi cung ứng và kế hoạch thuế",
            "summary": "Các ngành nhạy với xuất nhập khẩu, logistics và vốn FDI sẽ phản ứng trước; thu ngân sách biến động theo độ trễ 1-3 quý.",
            "probability": 0.54,
            "impact_level": "medium" if severity in ("low", "medium") else "high",
        },
    ]


def normalize_interpretation_payload(payload: Dict[str, Any], text: str) -> Dict[str, Any]:
    params = dict(payload.get("macro_parameters") or {})
    normalized_params = {
        "gdp_delta_pct": _clamp_float(params.get("gdp_delta_pct"), -30.0, 20.0, 0.0),
        "tax_rate_delta": _clamp_float(params.get("tax_rate_delta"), -0.10, 0.10, 0.0),
        "compliance_delta": _clamp_float(params.get("compliance_delta"), -0.30, 0.30, 0.0),
        "unemployment_delta": _clamp_float(params.get("unemployment_delta"), -5.0, 10.0, 0.0),
        "fdi_delta_pct": _clamp_float(params.get("fdi_delta_pct"), -60.0, 60.0, 0.0),
    }
    return {
        "scenario_title": str(payload.get("scenario_title") or text[:96] or "Kịch bản vĩ mô").strip(),
        "event_type": str(payload.get("event_type") or "unknown"),
        "severity": str(payload.get("severity") or "medium"),
        "affected_provinces": [str(x) for x in payload.get("affected_provinces", [])],
        "affected_sectors": [str(x) for x in payload.get("affected_sectors", [])],
        "macro_parameters": normalized_params,
        "candidate_events": payload.get("candidate_events") or build_candidate_event_cards(text[:96], str(payload.get("event_type") or "unknown"), str(payload.get("severity") or "medium"), normalized_params),
        "reasoning_brief": str(payload.get("reasoning_brief") or ""),
        "confidence": _clamp_float(payload.get("confidence"), 0.0, 1.0, 0.55),
        "llm_provider": payload.get("llm_provider"),
        "source": payload.get("source"),
        "scenario_text": text,
    }


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[^\w\s%.,-]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    return round(max(low, min(high, number)), 4)
