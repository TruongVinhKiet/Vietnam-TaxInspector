import requests
import json
import time

URL = "http://localhost:8000/api/tax-agent/chat/v2"

scenarios = [
    {
        "name": "Scenario 1: Smalltalk / Greeting",
        "query": "Xin chào, bạn có thể giúp tôi làm gì?",
        "mode": "full"
    },
    {
        "name": "Scenario 2: Data Lookup / Top N",
        "query": "Hãy liệt kê cho tôi danh sách 5 doanh nghiệp có điểm rủi ro gian lận thuế cao nhất hiện nay.",
        "mode": "full"
    },
    {
        "name": "Scenario 3: Legal Consultation",
        "query": "Căn cứ pháp lý nào quy định về mức phạt chậm nộp thuế GTGT? Trích dẫn rõ điều luật.",
        "mode": "full"
    },
    {
        "name": "Scenario 4: VAT Graph Analysis",
        "query": "Phân tích mạng lưới hóa đơn và các rủi ro của công ty FPT.",
        "mode": "full"
    }
]

for s in scenarios:
    print(f"\n{'='*50}\n{s['name']}\nQuery: {s['query']}")
    payload = {
        "message": s["query"],
        "session_id": "test_session_123",
        "model_mode": s["mode"]
    }
    start = time.time()
    try:
        resp = requests.post(URL, json=payload)
        data = resp.json()
        latency = time.time() - start
        
        print(f"Latency: {latency:.2f}s")
        print(f"Intent: {data.get('intent')}")
        print(f"Answer Contract: {data.get('answer_contract')}")
        print(f"Tools Used: {data.get('tools_used', [])}")
        print(f"Route Violation: {data.get('route_violation')}")
        print(f"Focus Score: {data.get('focus_score')}")
        print(f"Response Preview: {data.get('answer', '')[:200]}...")
    except Exception as e:
        print("Error:", e)
