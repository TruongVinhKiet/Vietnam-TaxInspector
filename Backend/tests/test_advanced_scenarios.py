import requests
import time
import uuid

URL = "http://localhost:8000/api/tax-agent/chat/v2"

# Khởi tạo một Session ID mới để không bị dính context cũ
SESSION_ID = f"test_session_{uuid.uuid4().hex[:8]}"

scenarios = [
    # 1. Nhóm Giao tiếp tự nhiên & Khởi tạo
    {
        "name": "1. Smalltalk / Greeting",
        "query": "Xin chào, bạn có thể giúp tôi làm gì?",
        "mode": "fast"
    },
    {
        "name": "2. Ambiguous Clarification",
        "query": "Kiểm tra giúp tôi công ty này với.",
        "mode": "fast"
    },
    
    # 2. Nhóm Pháp lý chung & GraphRAG
    {
        "name": "3. Legal - Căn cứ pháp lý",
        "query": "Căn cứ pháp lý nào quy định về mức phạt chậm nộp thuế GTGT? Trích dẫn rõ điều luật.",
        "mode": "fast"
    },
    {
        "name": "4. Legal - Hoàn thuế",
        "query": "Điều kiện để doanh nghiệp được hoàn thuế xuất khẩu là gì?",
        "mode": "fast"
    },

    # 3. Nhóm Phân tích Rủi ro & Bất thường
    {
        "name": "5. Risk - Hoàn thuế GTGT",
        "query": "Hãy đánh giá rủi ro hoàn thuế GTGT và bất thường hóa đơn của doanh nghiệp có MST 0100109106.",
        "mode": "full"
    },
    {
        "name": "6. Risk - NLP Red Flag",
        "query": "Kiểm tra hồ sơ rủi ro và quét NLP các hóa đơn đầu vào của công ty FPT (MST 0101248141).",
        "mode": "full"
    },

    # 4. Nhóm Khai thác Đồ thị Giao dịch & Sở hữu
    {
        "name": "7. Graph - Điều tra mạng lưới",
        "query": "Điều tra mạng lưới giao dịch và bóc tách cấu trúc sở hữu của MST 0300588569. Có dấu hiệu thao túng không?",
        "mode": "full"
    },
    {
        "name": "8. OSINT - Công ty bình phong",
        "query": "Kiểm tra xem MST 9900000001 có phải là công ty bình phong (Shell Company) không?",
        "mode": "full"
    },

    # 5. Nhóm Dự báo & Nợ đọng
    {
        "name": "9. Forecasting - Dự báo doanh thu & nợ",
        "query": "Dự báo doanh thu quý tới và khả năng nợ thuế của doanh nghiệp 0100109106.",
        "mode": "full"
    },

    # 6. Nhóm Truy vấn Ngôn ngữ Tự nhiên & Danh sách
    {
        "name": "10. NL Query - Top N Risky",
        "query": "Hãy liệt kê cho tôi danh sách 5 doanh nghiệp có điểm rủi ro gian lận thuế cao nhất hiện nay.",
        "mode": "fast"
    },
    {
        "name": "11. NL Query - Company Name Lookup",
        "query": "Tra cứu thông tin cơ bản của doanh nghiệp mang tên Hòa Phát.",
        "mode": "fast"
    }
]

print(f"Bắt đầu chạy Test Advanced Scenarios (Session: {SESSION_ID})")
print("Đảm bảo bạn đã khởi động Uvicorn server: uvicorn app.main:app --reload")

for i, s in enumerate(scenarios, 1):
    print(f"\n{'='*60}\nKịch bản {i}: {s['name']}")
    print(f"Câu hỏi: {s['query']}")
    print(f"Chế độ: {s['mode']}")
    
    payload = {
        "message": s["query"],
        "session_id": SESSION_ID,
        "model_mode": s["mode"]
    }
    
    start_time = time.time()
    try:
        resp = requests.post(URL, json=payload, timeout=60)
        
        if resp.status_code == 200:
            data = resp.json()
            latency = time.time() - start_time
            
            print(f"\n[Kết quả - {latency:.2f}s]")
            print(f"✅ Status: 200 OK")
            print(f"🔍 Intent phân loại: {data.get('intent')}")
            print(f"⚖️ Answer Contract: {data.get('answer_contract')}")
            print(f"🛠 Tools đã gọi: {data.get('tools_used', [])}")
            
            if data.get('route_violation'):
                print(f"⚠️ Route Violation: True (Focus Score: {data.get('focus_score')})")
            
            print(f"🤖 Trả lời (Preview): {data.get('answer', '').replace(chr(10), ' ')[:200]}...")
        else:
            print(f"\n❌ Lỗi Server: Status {resp.status_code}")
            print(resp.text[:200])
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Lỗi Kết nối: Server Uvicorn chưa chạy. Vui lòng chạy `uvicorn app.main:app --reload`")
        break
    except Exception as e:
        print(f"\n❌ Lỗi Exception: {e}")
        
    time.sleep(1) # Nghỉ 1 giây giữa các request để server xử lý
