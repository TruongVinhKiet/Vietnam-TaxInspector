import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import json
import logging
from ml_engine.tax_agent_agentic_llm import get_agentic_llm
from ml_engine.tax_agent_llm_model import get_tax_llm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_tests():
    print("="*80)
    print("  🚀 BÀI KIỂM TRA TÍCH HỢP AGENT ULTIMATE V5 (LOCAL)")
    print("="*80)
    
    agentic_llm = get_agentic_llm()
    if not agentic_llm.load():
        print("❌ Lỗi: Không thể tải LoRA V5 adapter.")
        return
        
    tax_llm = get_tax_llm()
    tax_llm.load()

    test_queries = [
        "Tôi cần truy vết mạng lưới VAT của doanh nghiệp 888777666.",
        "Phân tích cấu trúc sở hữu chéo của công ty 000111000.",
        "Sử dụng hóa đơn mua ngoài chợ trời không hợp pháp bị phạt bao nhiêu tiền?",
        "Thời hạn nộp tờ khai thuế GTGT quý là bao giờ? Tôi sợ bị trễ hạn.",
        "Mua hàng giá trị bao nhiêu thì bắt buộc phải chuyển khoản để được khấu trừ thuế?"
    ]
    
    for idx, q in enumerate(test_queries, 1):
        print(f"\n[TEST {idx}] 👤 User: {q}")
        print("-" * 80)
        
        t0 = time.time()
        
        # Bước 1: Gọi V5 LLM để lấy thought và tool_call
        decision = agentic_llm.infer(q)
        
        if not decision:
            print("🤖 Lỗi inference hoặc output không đúng định dạng.")
            continue
            
        print(f"⚡ Bước 1 (Agentic LLM - {time.time() - t0:.2f}s):")
        print(f"  Thought: {decision.thought}")
        print(f"  Tool: {decision.tool_name}")
        print(f"  Args: {decision.tool_args}")
        
        # Bước 2: Nếu là knowledge_search, giả lập kết quả trả về từ RAG
        if decision.tool_name == "knowledge_search":
            print(f"\n⚖️ Bước 2 (Tổng hợp câu trả lời dựa trên Tool):")
            
            # Giả lập kết quả RAG
            mock_context = ""
            if "ngoài chợ" in q or "bất hợp pháp" in q:
                mock_context = "Nghị định 125/2020/NĐ-CP: Phạt tiền từ 20.000.000 đồng đến 50.000.000 đồng đối với hành vi sử dụng hóa đơn, chứng từ không hợp pháp."
            elif "thời hạn" in q:
                mock_context = "Điều 44 Luật Quản lý thuế 38/2019/QH14: Chậm nhất là ngày cuối cùng của tháng đầu của quý tiếp theo quý phát sinh nghĩa vụ thuế đối với trường hợp khai và nộp theo quý."
            elif "20 triệu" in q or "chuyển khoản" in q:
                mock_context = "Thông tư 96/2015/TT-BTC: Hóa đơn mua hàng hóa, dịch vụ từng lần từ 20 triệu đồng trở lên (đã gồm VAT) phải có chứng từ thanh toán không dùng tiền mặt."
            else:
                mock_context = "Không tìm thấy thông tin pháp luật liên quan."
                
            prompt_q = f"Dựa vào văn bản sau, hãy tư vấn cho người dùng: {q}\nTuyệt đối không bịa thêm mức phạt."
            t1 = time.time()
            final_resp = tax_llm.generate(query=prompt_q, context=mock_context)
            
            print(final_resp.text)
            print(f"  (Sinh trong {time.time() - t1:.2f}s)")
            
        print("\n" + "="*80)

if __name__ == "__main__":
    run_tests()
