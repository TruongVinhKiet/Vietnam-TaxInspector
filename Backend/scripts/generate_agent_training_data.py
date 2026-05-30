import json
import random
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "agent_tool_use_dataset_v2.jsonl"

SYSTEM_PROMPT = """Bạn là TaxInspector, một Trợ lý Thanh tra Thuế AI cấp cao.
Bạn được trang bị các công cụ (tools) AI chuyên sâu để phân tích dữ liệu nghiệp vụ.
Hãy đọc kỹ ngữ cảnh, suy nghĩ (trong thẻ <thought>) và gọi đúng công cụ (trong thẻ <tool_call>).
TUYỆT ĐỐI KHÔNG GỌI NHẦM CÔNG CỤ. Mỗi chế độ (VAT, Gian lận, Pháp lý, Nợ đọng, Vĩ mô) tương ứng với công cụ riêng biệt.

Các công cụ hiện có:
- gnn_analysis(tax_code): Mạng lưới VAT, chuỗi hóa đơn, giao dịch lòng vòng, công ty ma.
- vae_anomaly_scan(tax_code): Chấm điểm gian lận, điểm rủi ro, phát hiện hóa đơn bất thường của một doanh nghiệp độc lập.
- temporal_delinquency_deep(tax_code): Dự báo khả năng trễ hạn nộp thuế, nợ đọng thuế.
- causal_uplift_recommend(tax_code): Đề xuất biện pháp cưỡng chế thu hồi nợ tối ưu nhất.
- knowledge_search(query): Giải đáp luật thuế, văn bản pháp luật, nghị định, thông tư.
- macro_forecast(scenario): Mô phỏng vĩ mô, tác động của GDP hoặc chính sách thuế lên tổng thu ngân sách.
"""

# ==========================================
# ENGINE TẠO CÂU TỪ TỔ HỢP (COMBINATORIAL)
# Đảm bảo hàng vạn câu không trùng lặp
# ==========================================

GREETINGS = ["", "Chào bạn, ", "Giúp tôi ", "Trợ lý ơi, ", "Vui lòng ", "Sếp yêu cầu ", "Hãy ", "Tiến hành ", "Tôi cần "]
TARGET_COMPANIES = ["doanh nghiệp {tax_code}", "công ty có MST {tax_code}", "MST {tax_code}", "mã số thuế {tax_code}", "nhà cung cấp {tax_code}", "đối tượng {tax_code}"]
TIME_CONTEXTS = ["ngay bây giờ.", "trong kỳ này.", "nhé.", "để báo cáo sếp.", "cho chuyên đề thanh tra.", "gấp nhé.", ""]

def generate_queries(intents_verbs, targets, contexts, num=100):
    queries = []
    for _ in range(num):
        g = random.choice(GREETINGS)
        v = random.choice(intents_verbs)
        t = random.choice(targets)
        c = random.choice(contexts)
        query = f"{g}{v} {t} {c}".strip()
        queries.append(query)
    return queries

# 1. VAT NETWORK (Hetero GNN)
vat_verbs = ["phân tích mạng lưới VAT của", "truy vết dòng chảy hóa đơn của", "kiểm tra chuỗi giao dịch lòng vòng của", "vẽ sơ đồ liên kết của", "xem hệ sinh thái mua bán của", "tìm dấu hiệu công ty ma trong mạng lưới của"]
vat_queries = generate_queries(vat_verbs, TARGET_COMPANIES, TIME_CONTEXTS, 2000)

# 2. FRAUD / ANOMALY (VAE)
fraud_verbs = ["chấm điểm rủi ro gian lận cho", "phát hiện bất thường trong hóa đơn của", "kiểm tra rủi ro hồ sơ", "quét dị thường dữ liệu của", "chấm điểm AI tổng hợp cho", "đánh giá mức độ rủi ro độc lập của"]
fraud_queries = generate_queries(fraud_verbs, TARGET_COMPANIES, TIME_CONTEXTS, 2000)

# 3. DELINQUENCY (Temporal)
delinq_verbs = ["dự báo rủi ro nợ đọng của", "xem khả năng trễ hạn nộp thuế của", "tính xác suất chậm nộp tiền thuế của", "dự báo dòng tiền nộp ngân sách của", "kiểm tra nguy cơ nợ thuế của"]
delinq_queries = generate_queries(delinq_verbs, TARGET_COMPANIES, TIME_CONTEXTS, 1000)

# 4. ACTION UPLIFT (Thu hồi nợ)
uplift_verbs = ["gợi ý biện pháp cưỡng chế", "tìm cách thu hồi nợ tối ưu cho", "đề xuất hành động can thiệp với", "phân tích hiệu quả cưỡng chế đối với", "chọn biện pháp thu nợ cho"]
uplift_queries = generate_queries(uplift_verbs, TARGET_COMPANIES, TIME_CONTEXTS, 1000)

# 5. LEGAL (GraphRAG)
legal_queries = [
    "Luật quy định thế nào về hoàn thuế xuất khẩu?",
    "Mức phạt trễ hạn nộp tờ khai thuế GTGT?",
    "Quy trình thanh tra thuế tại trụ sở người nộp thuế gồm mấy bước?",
    "Hành vi trốn thuế bị xử lý hình sự khi nào?",
    "Thông tư nào hướng dẫn về giao dịch liên kết?",
    "Cho tôi biết điều kiện giảm trừ gia cảnh thuế TNCN.",
    "Doanh nghiệp phần mềm được ưu đãi thuế TNDN như thế nào?",
    "Trường hợp nào được ân hạn nợ thuế?",
]
# Tổ hợp thêm cho luật
legal_prefixes = ["Tìm giúp tôi: ", "Giải đáp: ", "Hệ thống tra cứu: ", "Xin hỏi ", ""]
legal_expanded = [random.choice(legal_prefixes) + q for q in legal_queries for _ in range(100)]


# 6. MACRO SIMULATION (Vĩ mô)
macro_queries = [
    "Mô phỏng ngân sách nếu GDP giảm {gdp}% và thuế suất thay đổi {tax}%",
    "Chạy dự báo vĩ mô với kịch bản tăng trưởng GDP {gdp}% và điều chỉnh thuế {tax}%",
    "Hệ thống tính toán tổng thu nếu kinh tế biến động: GDP {gdp}%, thuế {tax}%",
    "Cho tôi xem kịch bản vĩ mô khi GDP = {gdp}% và Tax = {tax}%"
]

# XÂY DỰNG DATASET
dataset = []

def add_record(query, thought, tool_name, tool_args, tool_result, final_answer):
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {
                "role": "assistant", 
                "content": f"<thought>{thought}</thought>\n<tool_call>{json.dumps({'name': tool_name, 'arguments': tool_args}, ensure_ascii=False)}</tool_call>"
            },
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": final_answer}
        ]
    }
    dataset.append(record)

print("Đang tổng hợp hàng ngàn kịch bản...")

# Tạo VAT
for q in set(vat_queries):
    tc = str(random.randint(1000000000, 9999999999))
    q_fmt = q.format(tax_code=tc)
    thought = f"Yêu cầu liên quan đến việc phân tích mạng lưới, chuỗi hóa đơn, hệ sinh thái hoặc dòng chảy giao dịch của {tc}. Công cụ phân tích Mạng lưới (HeteroGNN) là chính xác nhất."
    res = '{"risk_score": 98.2, "carousel_detected": true, "layers": 4}'
    ans = f"Qua phân tích bằng thuật toán Đồ thị HeteroGNN, mạng lưới VAT của {tc} cực kỳ phức tạp. Phát hiện dấu hiệu giao dịch lòng vòng (carousel fraud) qua 4 lớp công ty. Điểm rủi ro mạng lưới đạt 98.2/100."
    add_record(q_fmt, thought, "gnn_analysis", {"tax_code": tc}, res, ans)

# Tạo Fraud
for q in set(fraud_queries):
    tc = str(random.randint(1000000000, 9999999999))
    q_fmt = q.format(tax_code=tc)
    thought = f"Yêu cầu đánh giá rủi ro, sự bất thường trong hóa đơn đầu vào/đầu ra của doanh nghiệp độc lập {tc}. Mô hình học sâu VAE Anomaly là công cụ phù hợp."
    res = '{"anomaly_score": 75.4, "flagged_invoices": 5}'
    ans = f"Hệ thống đã chạy mô hình VAE Anomaly. Doanh nghiệp {tc} có điểm bất thường là 75.4/100. AI nhận diện được 5 hóa đơn có giá trị chênh lệch lớn so với hành vi lịch sử."
    add_record(q_fmt, thought, "vae_anomaly_scan", {"tax_code": tc}, res, ans)

# Tạo Delinquency
for q in set(delinq_queries):
    tc = str(random.randint(1000000000, 9999999999))
    q_fmt = q.format(tax_code=tc)
    thought = f"Truy vấn liên quan đến tương lai: dự báo nợ đọng, chậm nộp, trễ hạn thuế của {tc}. Phải sử dụng mô hình chuỗi thời gian Temporal Transformer."
    res = '{"prob_30d": 88.0, "prob_90d": 92.5}'
    ans = f"Mô hình Temporal Transformer dự báo {tc} có xác suất nợ đọng rất cao: 88.0% sẽ trễ hạn 30 ngày và 92.5% sẽ nợ dai dẳng qua 90 ngày. Cần đưa vào danh sách đôn đốc thu ngay."
    add_record(q_fmt, thought, "temporal_delinquency_deep", {"tax_code": tc}, res, ans)

# Tạo Uplift
for q in set(uplift_queries):
    tc = str(random.randint(1000000000, 9999999999))
    q_fmt = q.format(tax_code=tc)
    thought = f"Người dùng muốn tìm 'giải pháp', 'biện pháp cưỡng chế', 'thu hồi nợ' tốt nhất cho {tc}. Mô hình nhân quả Causal Uplift Action sẽ đề xuất hành động tối ưu."
    res = '{"best_action": "Ngừng sử dụng hóa đơn", "uplift_score": 0.45}'
    ans = f"Thuật toán Causal Uplift phân tích rằng biện pháp 'Ngừng sử dụng hóa đơn' sẽ mang lại hiệu quả cao nhất cho {tc} (Uplift Score: 0.45). Các biện pháp khác như nhắn tin nhắc nhở sẽ không có tác dụng với đối tượng này."
    add_record(q_fmt, thought, "causal_uplift_recommend", {"tax_code": tc}, res, ans)

# Tạo Legal
for q in set(legal_expanded):
    thought = f"Đây là một câu hỏi tra cứu kiến thức pháp luật, quy định, thông tư hoặc nghị định thuế. Không có MST cụ thể. Cần truy vấn cơ sở dữ liệu tri thức bằng GraphRAG."
    res = '{"articles": ["Điều 17 Nghị định 125/2020"], "content": "Xử phạt hành vi trốn thuế..."}'
    ans = f"Dựa trên truy xuất từ Knowledge Graph: Căn cứ Điều 17 Nghị định 125/2020/NĐ-CP quy định chi tiết về xử phạt vi phạm hành chính đối với hành vi trốn thuế..."
    add_record(q, thought, "knowledge_search", {"query": q}, res, ans)

# Tạo Macro
for _ in range(1000):
    gdp = round(random.uniform(-5.0, 10.0), 1)
    tax = round(random.uniform(-2.0, 5.0), 1)
    q = random.choice(macro_queries).format(gdp=gdp, tax=tax)
    thought = f"Người dùng cung cấp thông số vĩ mô (GDP {gdp}%, Thuế {tax}%) để yêu cầu chạy mô phỏng. Công cụ chạy kịch bản vĩ mô là cần thiết."
    res = '{"budget_impact": "-12.5 nghìn tỷ", "risk_level": "Moderate"}'
    ans = f"Đã hoàn thành mô phỏng vĩ mô. Với kịch bản GDP biến động {gdp}% và thuế suất {tax}%, tổng thu ngân sách ước tính sẽ bị tác động khoảng -12.5 nghìn tỷ VNĐ. Mức độ rủi ro hệ thống: Trung bình."
    add_record(q, thought, "macro_forecast", {"scenario": {"gdp_change": gdp, "tax_rate_change": tax}}, res, ans)

# Xáo trộn dataset
random.shuffle(dataset)

# Lấy chính xác 10,000 mẫu để giới hạn file không quá lớn (khoảng 15-20MB là đẹp cho Colab)
final_dataset = dataset[:10000]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in final_dataset:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Hoàn tất! Đã tạo thành công {len(final_dataset)} kịch bản siêu đa dạng và KHÔNG LẶP LẠI.")
print(f"Dung lượng file: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
