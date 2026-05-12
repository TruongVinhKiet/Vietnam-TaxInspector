import sys
from dataclasses import dataclass

sys.path.append("e:/TaxInspector/Backend")

from ml_engine.tax_agent_planner import TaxAgentPlanner
from ml_engine.tax_agent_task_router import TaskRouter


@dataclass(frozen=True)
class ModeEvalCase:
    query: str
    intent: str
    mode: str
    expected_domain: str
    attachment: dict | None = None
    expect_mismatch: bool = False


import random

def _build_cases() -> list[ModeEvalCase]:
    random.seed(42)
    cases: list[ModeEvalCase] = []
    
    fraud_templates = [
        "Xin danh sách top {n} doanh nghiệp có rủi ro cao nhất tại khu vực {loc}.",
        "Chấm điểm rủi ro cho công ty có MST {mst}.",
        "Đánh giá hồ sơ rủi ro và các dấu hiệu gian lận của doanh nghiệp {mst}.",
        "Lọc ra {n} công ty cần thanh tra gấp trong tháng này.",
        "Phân tích bất thường trong hồ sơ kê khai của MST {mst} giúp tôi.",
        "Kiểm tra xem công ty {mst} có nằm trong diện rủi ro cao không?",
        "Chạy mô hình phân tích gian lận cho lô dữ liệu này.",
        "Xác định rủi ro tổng thể cho danh sách doanh nghiệp vừa upload.",
    ]
    
    vat_templates = [
        "Truy vết mạng lưới VAT và vòng lặp hóa đơn của MST {mst}.",
        "Phân tích rủi ro hoàn thuế GTGT cho doanh nghiệp {mst}.",
        "Kiểm tra hóa đơn đầu vào đầu ra của công ty {mst} có dấu hiệu mua bán hóa đơn không.",
        "Vẽ sơ đồ mạng lưới giao dịch VAT cho MST {mst}.",
        "Chấm điểm rủi ro mạng lưới GNN cho hệ sinh thái của công ty {mst}.",
        "Rà soát chuỗi cung ứng và rủi ro hoàn thuế của {mst}.",
    ]
    
    delinquency_templates = [
        "Dự báo khả năng nợ đọng thuế trong 30, 60, 90 ngày của doanh nghiệp {mst}.",
        "Công ty {mst} có nguy cơ chậm nộp thuế trong kỳ tới không?",
        "Chạy mô hình chuỗi thời gian dự báo nợ cho MST {mst}.",
        "Đánh giá rủi ro trễ hạn thanh toán thuế của doanh nghiệp {mst}.",
        "Phân tích khả năng thu hồi nợ đọng của công ty có MST {mst}.",
    ]
    
    macro_templates = [
        "Chạy mô phỏng vĩ mô với kịch bản VAT {vat}%, tăng trưởng GDP {gdp}%, CPI {cpi}%.",
        "Dự báo tác động nguồn thu nếu điều chỉnh tỷ lệ thất nghiệp lên {unemp}% và lãi suất {rate}%.",
        "Khởi chạy bảng điều khiển mô phỏng với thông số hiện tại.",
        "Mô phỏng kịch bản kinh tế: thuế TNDN {cit}%, biến động tỷ giá {fx}%.",
        "Phân tích độ nhạy của các biến số vĩ mô đối với chính sách giảm thuế.",
        "Điều chỉnh tham số mô phỏng: diện thanh tra {audit}% và chạy lại kịch bản 5 năm.",
    ]
    
    legal_templates = [
        "Tra cứu căn cứ pháp lý về điều kiện hoàn thuế GTGT hàng xuất khẩu.",
        "Theo Nghị định {nd}, mức phạt chậm nộp tờ khai thuế môn bài là bao nhiêu?",
        "Hướng dẫn thủ tục đăng ký người phụ thuộc giảm trừ gia cảnh thuế TNCN.",
        "Luật quản lý thuế quy định thế nào về các chi phí được trừ khi tính thuế TNDN?",
        "Tìm kiếm quy định pháp luật liên quan đến giao dịch liên kết và chuyển giá.",
        "Hộ kinh doanh bán hàng online trên Shopee phải nộp những loại thuế nào?",
    ]
    
    smalltalk_templates = [
        "Chào bạn, chúc một ngày tốt lành.",
        "Cảm ơn hệ thống đã hỗ trợ nhiệt tình.",
        "Agent có thể làm được những việc gì?",
        "Xin chào, tôi cần giúp đỡ phân tích dữ liệu thuế.",
        "Tuyệt vời, cảm ơn agent nhé.",
    ]
    
    # Generate 150 Fraud (Top N)
    for i in range(150):
        t = random.choice(fraud_templates[:4])
        q = t.format(n=random.randint(5, 50), loc=f"tỉnh {i}", mst=f"010{i:07d}")
        cases.append(ModeEvalCase(query=q, intent="top_n_query", mode="full", expected_domain="fraud"))
        
    # Generate 150 Fraud (Single/Audit)
    for i in range(150):
        t = random.choice(fraud_templates[1:6])
        q = t.format(n=10, loc="HN", mst=f"010{i:07d}")
        cases.append(ModeEvalCase(query=q, intent="audit_selection", mode="full", expected_domain="fraud"))

    # Generate 200 VAT
    for i in range(200):
        t = random.choice(vat_templates)
        q = t.format(mst=f"020{i:07d}")
        cases.append(ModeEvalCase(query=q, intent="vat_network_analysis", mode="full", expected_domain="vat"))
        
    # Generate 150 Delinquency
    for i in range(150):
        t = random.choice(delinquency_templates)
        q = t.format(mst=f"030{i:07d}")
        cases.append(ModeEvalCase(query=q, intent="delinquency", mode="full", expected_domain="delinquency"))
        
    # Generate 150 Macro
    for i in range(150):
        t = random.choice(macro_templates)
        q = t.format(vat=random.randint(8,12), gdp=round(random.uniform(5,7), 1), cpi=round(random.uniform(2,4), 1), unemp=round(random.uniform(2,3), 1), rate=random.randint(5,8), cit=random.randint(15,22), fx=0, audit=random.randint(2,8))
        cases.append(ModeEvalCase(query=q, intent="macro_forecast", mode="full", expected_domain="macro"))
        
    # Generate 100 Legal
    for i in range(100):
        t = random.choice(legal_templates)
        q = t.format(nd=f"12{i}/2024")
        cases.append(ModeEvalCase(query=q, intent="general_tax_query", mode="full", expected_domain="legal"))
        
    # Generate 50 Smalltalk
    for i in range(50):
        cases.append(ModeEvalCase(query=random.choice(smalltalk_templates), intent="smalltalk", mode="full", expected_domain="general"))
        
    # Generate 50 Mismatch (Batch upload in legal mode)
    for i in range(50):
        q = random.choice(fraud_templates[6:])
        cases.append(ModeEvalCase(
            query=q,
            intent="batch_analysis",
            mode="legal",
            expected_domain="fraud",
            attachment={
                "status": "detected",
                "analysis_type": "risk_csv",
                "requested_domain": "fraud",
                "filename": f"danh_sach_dn_{i}.csv",
            },
            expect_mismatch=True,
        ))
        
    return cases


def test_agent_mode_router_1000_case_quality_gate():
    router = TaskRouter()
    planner = TaxAgentPlanner()
    cases = _build_cases()
    assert len(cases) == 1000

    correct_routes = 0
    focus_violations = 0
    mode_mismatch_correct = 0
    tool_precision_ok = 0

    for case in cases:
        decision = router.route(
            query=case.query,
            intent=case.intent,
            model_mode=case.mode,
            has_attachment=bool(case.attachment),
            attachment_analysis=case.attachment,
        )

        if case.expect_mismatch:
            if decision.mode_mismatch and decision.suggested_mode == case.expected_domain:
                mode_mismatch_correct += 1
                correct_routes += 1
            continue

        if decision.requested_domain == case.expected_domain:
            correct_routes += 1

        selected_tools = decision.allowed_tools or set()
        if not decision.allow_legal and "knowledge_search" not in selected_tools:
            tool_precision_ok += 1
        elif decision.allow_legal:
            tool_precision_ok += 1

        if decision.route_violation:
            focus_violations += 1

        if case.expected_domain == "macro":
            plan = planner.plan(
                query=case.query,
                intent=case.intent,
                intent_confidence=0.9,
                model_mode=case.mode,
            )
            assert "macro_forecast" in [step.tool_name for step in plan.steps]

    route_accuracy = correct_routes / len(cases)
    tool_precision = tool_precision_ok / max(1, len(cases) - 100)
    focus_violation_rate = focus_violations / len(cases)

    assert route_accuracy >= 0.95
    assert tool_precision >= 0.98
    assert focus_violation_rate <= 0.05
    assert mode_mismatch_correct == 50
