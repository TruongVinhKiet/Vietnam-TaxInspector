import json

with open("e:/TaxInspector/Backend/test_results_multiagent.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    print(f"{d['test_id']} | {d['label']} | intent={d['intent']} | confidence={d['confidence']:.1%}")
    # Check language quality
    answer = d.get("answer", "")
    has_old = any(s in answer for s in ["Tu van phap ly", "Ket luan ngan", "Can doi chieu", "Dieu kien ap dung", "Buoc xu ly tiep theo", "Ngoai le / rui ro", "Can cu va chuoi quan he"])
    has_new = any(s in answer for s in ["Tư vấn pháp lý", "Kết luận ngắn", "Cần đối chiếu", "Điều kiện áp dụng", "Bước xử lý tiếp theo", "Ngoại lệ / Rủi ro", "Căn cứ và chuỗi quan hệ"])
    print(f"  Language: OLD={has_old} NEW={has_new}")
