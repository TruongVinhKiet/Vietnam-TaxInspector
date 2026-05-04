import sys
sys.path.append('e:/TaxInspector/Backend')
from ml_engine.tax_agent_orchestrator import TaxAgentOrchestrator

orch = TaxAgentOrchestrator()
msg1 = "Công ty tôi kinh doanh dịch vụ ăn uống, nhà hàng. Xin hỏi theo Nghị định 72/2024/NĐ-CP, tôi có được áp dụng thuế GTGT 8% không? Thời hạn áp dụng đến khi nào?"
msg2 = "Doanh nghiệp FDI của chúng tôi đang mở rộng nhà máy ở KCN Long Hậu, Long An. Vốn đầu tư thêm 50 tỷ. Xin hỏi chúng tôi có được hưởng ưu đãi thuế TNDN cho phần thu nhập tăng thêm không?"

print("msg1:", orch._rule_based_intent(msg1))
print("msg2:", orch._rule_based_intent(msg2))
