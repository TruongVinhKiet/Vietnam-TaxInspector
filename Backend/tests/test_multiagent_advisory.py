# -*- coding: utf-8 -*-
"""
Multi-Agent Tax Advisory System — Comprehensive Test Suite
==========================================================
Sends diverse real-world tax consultation queries to the V2 streaming endpoint
and captures the full orchestration output for quality review.
"""
from __future__ import annotations

import json
import sys
import time
import requests
from dataclasses import dataclass, field
from typing import Any

API_BASE = "http://localhost:8000/api/tax-agent"
STREAM_URL = f"{API_BASE}/chat/v2/stream"

# ═══════════════════════════════════════════════════════════════════════
#  Test Cases — diverse real-world scenarios
# ═══════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": "TC-01",
        "label": "VAT 8% — Nghị định 72/2024",
        "message": "Công ty tôi kinh doanh dịch vụ ăn uống, nhà hàng. Xin hỏi theo Nghị định 72/2024/NĐ-CP, tôi có được áp dụng thuế GTGT 8% không? Thời hạn áp dụng đến khi nào?",
        "model_mode": "legal",
        "expected_intents": ["vat_refund_risk", "general_tax_query"],
    },
    {
        "id": "TC-02",
        "label": "Ưu đãi TNDN — Đầu tư mở rộng",
        "message": "Doanh nghiệp FDI của chúng tôi đang mở rộng nhà máy ở KCN Long Hậu, Long An. Vốn đầu tư thêm 50 tỷ. Xin hỏi chúng tôi có được hưởng ưu đãi thuế TNDN cho phần thu nhập tăng thêm không?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query", "transfer_pricing"],
    },
    {
        "id": "TC-03",
        "label": "Hoàn thuế GTGT — Dự án đầu tư",
        "message": "Công ty chúng tôi có dự án đầu tư xây dựng nhà xưởng, đang trong giai đoạn đầu tư, chưa phát sinh doanh thu. Số thuế GTGT đầu vào chưa khấu trừ hết là 500 triệu. Xin hỏi có đủ điều kiện để xin hoàn thuế GTGT không? Cần lưu ý gì khi nộp hồ sơ?",
        "model_mode": "legal",
        "expected_intents": ["vat_refund_risk"],
    },
    {
        "id": "TC-04",
        "label": "Hóa đơn điện tử — Bán lẻ",
        "message": "Tôi kinh doanh chuỗi quán cà phê. Cơ quan thuế yêu cầu phải chuyển sang hóa đơn điện tử từ máy tính tiền. Xin hỏi quy định cụ thể như thế nào? Có bắt buộc không? Và hóa đơn từ máy tính tiền có được tính chi phí hợp lý khi tính thuế TNDN không?",
        "model_mode": "legal",
        "expected_intents": ["invoice_risk", "general_tax_query"],
    },
    {
        "id": "TC-05",
        "label": "Rủi ro gian lận — Doanh nghiệp MST 123456",
        "message": "Phân tích rủi ro gian lận cho doanh nghiệp có mã số thuế 0316123456. Doanh nghiệp này có dấu hiệu bất thường: doanh thu tăng vọt 300% nhưng lợi nhuận giảm, đồng thời có nhiều giao dịch liên kết với công ty ở Singapore.",
        "model_mode": "fraud",
        "expected_intents": ["invoice_risk", "osint_ownership", "transfer_pricing"],
    },
    {
        "id": "TC-06",
        "label": "Dự báo nợ đọng — Doanh nghiệp chậm nộp",
        "message": "Doanh nghiệp MST 0312345678 đã chậm nộp thuế 3 kỳ liên tiếp. Tổng nợ hiện tại khoảng 2.1 tỷ đồng. Xin phân tích xu hướng và dự báo khả năng thu hồi nợ trong 6 tháng tới. Đề xuất biện pháp xử lý phù hợp.",
        "model_mode": "delinquency",
        "expected_intents": ["delinquency"],
    },
    {
        "id": "TC-07",
        "label": "Giao dịch liên kết — Chi phí lãi vay",
        "message": "Doanh nghiệp MST 0311223344 có vốn điều lệ 100 tỷ, vay của công ty mẹ 50 tỷ. Tổng chi phí lãi vay trong kỳ là 10 tỷ, EBITDA là 20 tỷ. Xin hỏi chi phí lãi vay có bị khống chế theo Nghị định 132 không? Chi tiết cách tính và rủi ro?",
        "model_mode": "legal",
        "expected_intents": ["transfer_pricing"],
    },
    {
        "id": "TC-08",
        "label": "Bất đồng/Gian lận — Hoàn thuế VAT gỗ",
        "message": "Đề nghị phân tích hồ sơ hoàn thuế GTGT 20 tỷ của Công ty TNHH Xuất khẩu Gỗ A (MST: 0319988776). Công ty có mua hàng từ các F1, F2 nhưng có thông tin một số công ty F2 đã bỏ địa chỉ kinh doanh. Hồ sơ thanh toán chủ yếu bằng hình thức bù trừ công nợ.",
        "model_mode": "full",
        "expected_intents": ["vat_refund_risk", "invoice_risk", "osint_ownership"],
    },
    {
        "id": "TC-09",
        "label": "Kiểm tra hiệu lực — Cảnh báo Luật Cũ",
        "message": "Vui lòng cho tôi biết về thời hạn áp dụng giảm thuế VAT 8% theo Nghị định 72 đối với doanh nghiệp MST 0315566778. Nếu năm 2025 công ty xuất hóa đơn thì có được giảm không?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-10",
        "label": "Hỏi tự nhiên — Bán hàng Shopee/TikTok",
        "message": "Dạ chào anh/chị, em bán quần áo trên Shopee và TikTok Shop, tháng kiếm được cỡ 15 triệu. Tiền chuyển về tài khoản cá nhân. Cho em hỏi em có phải đóng thuế không ạ? Nếu có thì tính thuế làm sao?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-11",
        "label": "Hỏi tự nhiên — Mở tiệm tạp hóa",
        "message": "Chú định mở một cái tiệm tạp hóa nhỏ ở quê, không đăng ký công ty gì cả, mở dạng hộ gia đình thôi. Doanh thu một năm chắc chưa tới 100 triệu đâu. Vậy chú có phải đóng thuế GTGT hay thuế thu nhập cá nhân gì không?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-12",
        "label": "Hỏi tự nhiên — Phạt nộp trễ tiền mạng",
        "message": "Em có mở một tiệm net nhỏ xíu MST 0317778889, tháng này quên đóng tiền thuế môn bài mất rồi. Không biết có bị khóa tài khoản ngân hàng hay cấm xuất cảnh không vậy mọi người?",
        "model_mode": "legal",
        "expected_intents": ["delinquency", "general_tax_query"],
    },
    {
        "id": "TC-13",
        "label": "Hỏi tự nhiên — Lương 25 triệu đóng thuế bao nhiêu",
        "message": "Anh ơi cho em hỏi, em đi làm công ty lương gross 25 triệu/tháng, chưa lập gia đình, không có người phụ thuộc. Vậy em bị trừ thuế thu nhập cá nhân bao nhiêu tiền một tháng ạ? Tính kiểu gì vậy ạ?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-14",
        "label": "Hỏi tự nhiên — Chi phí tiếp khách có được trừ thuế",
        "message": "Công ty em thường xuyên mời khách hàng đi ăn uống, chi phí khoảng 50-70 triệu/tháng. Kế toán bảo một số khoản không được trừ khi tính thuế TNDN. Vậy cụ thể chi phí tiếp khách thì cần chứng từ gì để được trừ thuế? Có giới hạn mức nào không?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-15",
        "label": "Hỏi tự nhiên — Quên xuất hóa đơn bị phạt sao",
        "message": "Dạ em bán hàng cho khách mà quên không xuất hóa đơn điện tử, giờ khách gọi lại đòi. Cho em hỏi nếu cơ quan thuế biết thì em bị phạt bao nhiêu? Có cách nào xuất hóa đơn bổ sung không ạ?",
        "model_mode": "legal",
        "expected_intents": ["invoice_risk", "general_tax_query"],
    },
    {
        "id": "TC-16",
        "label": "Hỏi tự nhiên — Làm freelance có đóng thuế",
        "message": "Em là freelancer thiết kế đồ họa, nhận việc từ nhiều khách hàng cả trong và ngoài nước. Thu nhập khoảng 30-40 triệu/tháng nhưng không có hợp đồng lao động cố định. Vậy em có phải đóng thuế không? Thuế gì? Kê khai ở đâu?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-17",
        "label": "Hỏi tự nhiên — Cho thuê nhà có phải đóng thuế",
        "message": "Nhà tôi có một căn nhà cho thuê 15 triệu/tháng, cho thuê đã 2 năm rồi mà chưa đóng thuế gì hết. Giờ nghe nói thuế đang rà soát người cho thuê nhà. Xin hỏi tôi phải đóng những loại thuế gì? Bị phạt không? Tính từ lúc nào?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-18",
        "label": "Hỏi tự nhiên — DN mới thanh toán tiền mặt 30 triệu",
        "message": "Công ty mới thành lập MST 0318889900, em vừa mua một lô hàng trị giá 35 triệu đồng. Bên bán xuất hóa đơn đầy đủ nhưng em trả bằng tiền mặt. Vậy khoản chi này có được trừ khi tính thuế TNDN không? Kế toán nói không được, đúng không ạ?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
    {
        "id": "TC-19",
        "label": "Hỏi tự nhiên — Nộp tờ khai trễ 15 ngày",
        "message": "Chị ơi cho em hỏi, công ty em MST 0314455667 quên nộp tờ khai thuế GTGT tháng 3 rồi, giờ là tháng 5 luôn rồi, tức là trễ khoảng 45 ngày. Bị phạt bao nhiêu vậy chị? Có cần làm đơn xin giảm nhẹ gì không?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query", "delinquency"],
    },
    {
        "id": "TC-20",
        "label": "Hỏi tự nhiên — Công ty khấu trừ thuế TNCN sai",
        "message": "Em thấy công ty khấu trừ thuế TNCN của em sai rồi. Lương em 20 triệu, có 1 con nhỏ là người phụ thuộc, mà tháng nào cũng bị trừ 1.5 triệu tiền thuế. Em tính lại thấy bị trừ nhiều quá. Vậy em phải làm gì để được hoàn lại tiền thuế đóng thừa?",
        "model_mode": "legal",
        "expected_intents": ["general_tax_query"],
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  SSE Stream Parser
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    test_id: str
    label: str
    status: str = "pending"
    answer: str = ""
    intent: str = ""
    confidence: float = 0.0
    complexity: str = ""
    tools_used: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    legal_workspace: dict = field(default_factory=dict)
    events_log: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""
    escalation_required: bool = False
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_stream_test(case: dict) -> TestResult:
    result = TestResult(test_id=case["id"], label=case["label"])
    payload = {
        "message": case["message"],
        "model_mode": case["model_mode"],
        "top_k": 8,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            STREAM_URL,
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        result.status = "error"
        result.error = str(e)
        return result

    text_chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            raw = line[6:]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if current_event == "thinking":
                step = data.get("step", "")
                detail = data.get("detail", "")
                result.events_log.append(f"[thinking:{step}] {detail}")

            elif current_event == "tool_start":
                tool = data.get("tool", "")
                result.events_log.append(f"[tool_start] {tool}")

            elif current_event == "tool_done":
                tool = data.get("tool", "")
                status = data.get("status", "")
                lat = data.get("latency_ms", 0)
                result.events_log.append(f"[tool_done] {tool} — {status} ({lat:.0f}ms)")

            elif current_event == "sub_agent":
                agent = data.get("agent", "")
                status = data.get("status", "")
                detail = data.get("detail", "")
                result.events_log.append(f"[sub_agent] {agent}: {status} — {detail}")

            elif current_event == "debate":
                result.events_log.append(f"[debate] consensus={data.get('consensus_score', 'N/A')}")

            elif current_event == "text_chunk":
                text_chunks.append(data.get("chunk", ""))

            elif current_event == "done":
                result.answer = data.get("answer", "".join(text_chunks))
                result.intent = data.get("intent", "")
                result.confidence = data.get("intent_confidence", 0.0)
                result.complexity = data.get("complexity", "")
                result.tools_used = data.get("tools_used", [])
                result.citations = data.get("citations", [])
                result.latency_ms = data.get("latency_ms", 0.0)
                result.escalation_required = data.get("escalation_required", False)
                result.recommendations = data.get("recommendations", [])
                result.warnings = data.get("compliance_warnings", [])
                result.legal_workspace = data.get("legal_workspace", {})
                result.status = "success"

    if not result.answer:
        result.answer = "".join(text_chunks)
    if result.status == "pending":
        result.status = "success" if result.answer else "no_answer"

    result.latency_ms = result.latency_ms or (time.perf_counter() - t0) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Report Generator
# ═══════════════════════════════════════════════════════════════════════

def print_report(results: list[TestResult]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  MULTI-AGENT TAX ADVISORY SYSTEM — TEST REPORT")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    for r in results:
        print(f"\n{'─' * 90}")
        print(f"  [{r.test_id}] {r.label}")
        print(f"  Status: {r.status.upper()} | Intent: {r.intent} | Confidence: {r.confidence:.2%}")
        print(f"  Complexity: {r.complexity} | Latency: {r.latency_ms:.0f}ms | Tools: {len(r.tools_used)}")
        if r.escalation_required:
            print(f"  !! ESCALATION REQUIRED !!")
        print(f"{'─' * 90}")

        # Event pipeline
        print("\n  Pipeline Events:")
        for evt in r.events_log:
            print(f"    {evt}")

        # Tools
        if r.tools_used:
            print(f"\n  Tools Used: {', '.join(r.tools_used)}")

        # Citations
        if r.citations:
            print(f"\n  Citations ({len(r.citations)}):")
            for c in r.citations[:5]:
                print(f"    - [{c.get('citation_key', c.get('chunk_key', ''))}] "
                      f"{c.get('title', '')} (score={c.get('score', 0):.3f})")

        # Legal Workspace
        lw = r.legal_workspace
        if lw:
            print(f"\n  Legal Workspace:")
            if lw.get("facts"):
                print(f"    Facts: {lw['facts']}")
            if lw.get("assumptions"):
                print(f"    Assumptions: {lw['assumptions']}")
            if lw.get("open_questions"):
                print(f"    Open Questions: {lw['open_questions'][:3]}")
            if lw.get("verifications"):
                v_ok = sum(1 for v in lw["verifications"] if v.get("is_verified"))
                v_fail = sum(1 for v in lw["verifications"] if not v.get("is_verified"))
                print(f"    Verifications: {v_ok} supported, {v_fail} unsupported")
            if lw.get("escalations"):
                print(f"    Escalations: {lw['escalations']}")

        # Recommendations
        if r.recommendations:
            print(f"\n  Recommendations:")
            for rec in r.recommendations[:5]:
                print(f"    * {rec[:150]}")

        # Warnings
        if r.warnings:
            print(f"\n  Compliance Warnings:")
            for w in r.warnings:
                print(f"    ! {w}")

        # Answer (truncated for readability)
        print(f"\n  Answer Preview (first 1200 chars):")
        answer_lines = r.answer[:1200].split("\n")
        for al in answer_lines:
            print(f"    | {al}")
        if len(r.answer) > 1200:
            print(f"    | ... [{len(r.answer)} total chars]")

        if r.error:
            print(f"\n  ERROR: {r.error}")

    # Summary
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    success = sum(1 for r in results if r.status == "success")
    errors = sum(1 for r in results if r.status == "error")
    no_answer = sum(1 for r in results if r.status == "no_answer")
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    avg_confidence = sum(r.confidence for r in results if r.confidence > 0) / max(1, sum(1 for r in results if r.confidence > 0))
    answer_lengths = [len(r.answer) for r in results if r.answer]
    avg_answer_len = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

    print(f"  Total tests:     {len(results)}")
    print(f"  Success:         {success}")
    print(f"  Errors:          {errors}")
    print(f"  No answer:       {no_answer}")
    print(f"  Avg latency:     {avg_latency:.0f}ms")
    print(f"  Avg confidence:  {avg_confidence:.2%}")
    print(f"  Avg answer len:  {avg_answer_len:.0f} chars")
    print(f"  Escalations:     {sum(1 for r in results if r.escalation_required)}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Starting Multi-Agent Tax Advisory Test Suite...")
    print(f"Target: {STREAM_URL}")
    print(f"Test cases: {len(TEST_CASES)}\n")

    results: list[TestResult] = []
    for i, case in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] Running: {case['id']} — {case['label']}...")
        result = run_stream_test(case)
        results.append(result)
        print(f"  -> {result.status.upper()} ({result.latency_ms:.0f}ms) intent={result.intent}")
        # Small delay between tests
        if i < len(TEST_CASES) - 1:
            time.sleep(1)

    print_report(results)

    # Save full JSON results
    output_path = "test_results_multiagent.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [{
                "test_id": r.test_id,
                "label": r.label,
                "status": r.status,
                "intent": r.intent,
                "confidence": r.confidence,
                "complexity": r.complexity,
                "tools_used": r.tools_used,
                "citations": r.citations,
                "legal_workspace": r.legal_workspace,
                "answer": r.answer,
                "latency_ms": r.latency_ms,
                "escalation_required": r.escalation_required,
                "recommendations": r.recommendations,
                "warnings": r.warnings,
                "events_log": r.events_log,
                "error": r.error,
            } for r in results],
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
