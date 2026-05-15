"""
generate_mega_agent_dataset.py - Ultimate Dataset Generator V3
==============================================================
Sinh 10,000 kịch bản huấn luyện Agent đồng bộ 100% với kiến trúc thực:
- Tool names lấy từ tax_agent_mode_contracts.py & tax_agent_planner.py
- Intent names lấy từ test_agent_mode_1000_eval.py
- Câu hỏi lấy cảm hứng từ test_agent_comprehensive_eval.py
- Thought phản ánh đúng logic routing của TaskRouter
"""
import json, random, itertools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "data" / "agent_ultimate_dataset.jsonl"

SYSTEM_PROMPT = """Bạn là TaxInspector AI - Trợ lý Thanh tra Thuế thông minh.
Khi nhận câu hỏi, hãy suy nghĩ trong thẻ <thought>, sau đó gọi công cụ phù hợp trong thẻ <tool_call>.
KHÔNG được tự bịa kết quả. Chỉ gọi tool rồi DỪNG LẠI.

Công cụ:
1. top_n_risky_companies(n): Danh sách top N doanh nghiệp rủi ro cao nhất.
2. company_risk_lookup(tax_code): Tra cứu hồ sơ rủi ro của một doanh nghiệp.
3. gnn_analysis(tax_code): Phân tích đồ thị GNN mạng lưới giao dịch VAT.
4. vae_anomaly_scan(tax_code): Quét bất thường hóa đơn bằng VAE.
5. invoice_risk_scan(tax_code): Rà soát rủi ro hóa đơn đầu vào/đầu ra.
6. motif_detection(tax_code): Phát hiện mẫu giao dịch vòng lặp đáng ngờ.
7. ring_scoring(tax_code): Chấm điểm vòng giao dịch VAT.
8. ownership_analysis(tax_code): Phân tích cấu trúc sở hữu chéo.
9. delinquency_check(tax_code): Kiểm tra rủi ro nợ đọng thuế.
10. temporal_delinquency_deep(tax_code): Dự báo nợ đọng bằng Temporal Transformer.
11. causal_uplift_recommend(tax_code): Đề xuất biện pháp cưỡng chế tối ưu.
12. revenue_forecast(tax_code): Dự báo doanh thu quý tới.
13. macro_forecast(scenario): Mô phỏng vĩ mô tác động kinh tế.
14. knowledge_search(query): Tra cứu luật thuế, văn bản pháp luật.
15. escalate_to_debate(tax_code): Mở phiên tranh biện đa đặc vụ AI."""

# ══════════════════════════════════════════
# BUILDING BLOCKS - Xây dựng câu hỏi tổ hợp
# ══════════════════════════════════════════
P = ["", "Hệ thống ơi, ", "Hãy ", "Vui lòng ", "Tôi cần ", "Giúp tôi ", "Bạn hãy ", "Sếp yêu cầu ", "Cần gấp: ", "Làm ơn "]
S = ["", " nhé.", " gấp.", " giúp tôi.", " ngay.", " cho tôi.", " đi.", " để báo cáo sếp.", " trong hôm nay.", " càng sớm càng tốt."]

# ── 1. FRAUD: top_n_risky_companies ──
TOP_N_TEMPLATES = [
    "cho tôi danh sách top {n} doanh nghiệp rủi ro cao nhất",
    "lọc ra {n} công ty có điểm gian lận cao nhất hệ thống",
    "xuất báo cáo top {n} đối tượng cần thanh tra gấp",
    "xếp hạng {n} doanh nghiệp nguy hiểm nhất hiện nay",
    "tìm {n} mã số thuế có bất thường lớn nhất",
    "liệt kê {n} hồ sơ rủi ro ưu tiên thanh tra",
    "hiển thị bảng xếp hạng {n} công ty rủi ro hàng đầu",
    "ai là {n} con nợ hoặc gian lận lớn nhất",
    "trích xuất danh sách {n} đối tượng giám sát đặc biệt",
    "cho xem top {n} doanh nghiệp đang trong diện cảnh báo",
]

# ── 2. FRAUD: company_risk_lookup ──
RISK_TEMPLATES = [
    "chấm điểm rủi ro cho doanh nghiệp MST {tc}",
    "đánh giá hồ sơ rủi ro của công ty {tc}",
    "kiểm tra xem {tc} có nằm trong diện rủi ro cao không",
    "phân tích bất thường trong hồ sơ kê khai của {tc}",
    "tra cứu mức độ nguy hiểm của mã số thuế {tc}",
    "xếp hạng rủi ro tổng thể cho doanh nghiệp {tc}",
    "hồ sơ rủi ro hiện tại của công ty mã {tc} thế nào",
    "đánh giá xem MST {tc} có cần đưa vào diện thanh tra không",
    "kiểm tra điểm rủi ro gian lận của {tc}",
    "quét toàn diện hồ sơ rủi ro doanh nghiệp {tc}",
]

# ── 3. VAT: gnn_analysis ──
GNN_TEMPLATES = [
    "truy vết mạng lưới VAT của công ty {tc}",
    "phân tích đồ thị giao dịch hóa đơn của {tc}",
    "vẽ sơ đồ mạng lưới giao dịch VAT cho MST {tc}",
    "kiểm tra hóa đơn đầu vào đầu ra của {tc} có dấu hiệu mua bán lòng vòng không",
    "chạy thuật toán GNN phân tích liên kết giao dịch {tc}",
    "tìm kiếm chuỗi giao dịch bất thường quanh doanh nghiệp {tc}",
    "rà soát hệ sinh thái hóa đơn VAT xung quanh {tc}",
    "phân tích rủi ro mạng lưới GNN cho MST {tc}",
    "kiểm tra dòng chảy hóa đơn của công ty {tc}",
    "truy vết chuỗi cung ứng và mạng lưới của {tc}",
]

# ── 4. VAT: vae_anomaly_scan ──
VAE_TEMPLATES = [
    "quét bất thường hóa đơn của doanh nghiệp {tc} bằng mô hình VAE",
    "phát hiện dị thường trong giao dịch của MST {tc}",
    "chạy mô hình VAE để tìm hóa đơn bất thường của {tc}",
    "dùng học sâu quét anomaly cho hồ sơ {tc}",
    "phân tích bất thường thống kê trong dữ liệu hóa đơn {tc}",
    "kiểm tra outlier trong giao dịch của doanh nghiệp {tc}",
    "chạy anomaly detection cho toàn bộ hóa đơn MST {tc}",
    "quét sâu tìm gian lận ẩn trong hồ sơ {tc}",
]

# ── 5. VAT: invoice_risk_scan ──
INVOICE_TEMPLATES = [
    "rà soát rủi ro hóa đơn đầu vào đầu ra của {tc}",
    "kiểm tra hóa đơn có dấu hiệu bất hợp pháp của MST {tc}",
    "quét rủi ro toàn bộ hóa đơn của doanh nghiệp {tc}",
    "phân tích chất lượng hóa đơn VAT của công ty {tc}",
    "tìm hóa đơn đáng ngờ trong hồ sơ kê khai {tc}",
    "kiểm tra tính hợp lệ của hóa đơn đầu vào MST {tc}",
]

# ── 6. VAT: motif_detection ──
MOTIF_TEMPLATES = [
    "phát hiện mẫu giao dịch vòng lặp của {tc}",
    "tìm kiếm vòng lặp hóa đơn VAT quanh MST {tc}",
    "kiểm tra xem {tc} có tham gia đường dây mua bán hóa đơn không",
    "phát hiện motif giao dịch bất thường của doanh nghiệp {tc}",
    "chạy thuật toán phát hiện vòng giao dịch cho {tc}",
    "tìm chuỗi hóa đơn khép kín liên quan đến {tc}",
]

# ── 7. DELINQUENCY: delinquency_check ──
DELI_TEMPLATES = [
    "dự báo khả năng nợ đọng thuế trong 30 60 90 ngày của {tc}",
    "công ty {tc} có nguy cơ chậm nộp thuế trong kỳ tới không",
    "kiểm tra rủi ro nợ đọng thuế của doanh nghiệp {tc}",
    "đánh giá rủi ro trễ hạn thanh toán thuế của {tc}",
    "cảnh báo sớm nợ đọng cho MST {tc}",
    "phân tích khả năng thu hồi nợ đọng của công ty {tc}",
    "{tc} có khả năng trở thành con nợ khó đòi không",
    "tính xác suất nợ thuế dai dẳng của {tc}",
]

# ── 8. DELINQUENCY: causal_uplift_recommend ──
UPLIFT_TEMPLATES = [
    "đề xuất biện pháp cưỡng chế thu hồi nợ tối ưu cho {tc}",
    "nên áp dụng hành động can thiệp nào để đòi nợ từ {tc}",
    "so sánh hiệu quả phong tỏa tài khoản và ngừng hóa đơn cho {tc}",
    "gợi ý giải pháp thu nợ phù hợp nhất cho doanh nghiệp {tc}",
    "chạy Causal Uplift để tìm cách xử lý nợ của MST {tc}",
    "đâu là hành động mang lại hiệu suất thu nợ cao nhất cho {tc}",
    "tính toán hiệu quả các biện pháp cưỡng chế đối với {tc}",
]

# ── 9. MACRO: macro_forecast ──
MACRO_TEMPLATES = [
    "chạy mô phỏng vĩ mô với kịch bản VAT {vat}% và GDP {gdp}%",
    "dự báo tác động nguồn thu nếu GDP {gdp}% và thuế suất GTGT {vat}%",
    "mô phỏng kịch bản kinh tế: thuế TNDN {cit}% và GDP {gdp}%",
    "phân tích độ nhạy ngân sách với GDP {gdp}% và thuế {vat}%",
    "nếu GDP là {gdp}% và thuế điều chỉnh {vat}% thì tổng thu sẽ thế nào",
    "đánh giá tác động hệ thống khi kinh tế tăng {gdp}% và thuế {vat}%",
    "khởi động bảng điều khiển mô phỏng vĩ mô với GDP {gdp}%",
    "chạy kịch bản 5 năm với tham số GDP {gdp}% và VAT {vat}%",
]

# ── 10. LEGAL: knowledge_search ──
LEGAL_TEMPLATES = [
    "tra cứu căn cứ pháp lý về điều kiện hoàn thuế GTGT hàng xuất khẩu",
    "mức phạt chậm nộp tờ khai thuế hiện nay là bao nhiêu",
    "hướng dẫn thủ tục đăng ký người phụ thuộc giảm trừ gia cảnh TNCN",
    "luật quản lý thuế quy định thế nào về chi phí được trừ khi tính thuế TNDN",
    "quy định pháp luật liên quan đến giao dịch liên kết và chuyển giá",
    "hộ kinh doanh bán hàng online trên Shopee phải nộp thuế gì",
    "khi nào bắt buộc sử dụng hóa đơn điện tử",
    "truy cứu trách nhiệm hình sự hành vi trốn thuế từ số tiền bao nhiêu",
    "quy trình thủ tục hoàn thuế GTGT gồm những bước nào",
    "thông tư hướng dẫn xử phạt chậm nộp tờ khai",
    "doanh nghiệp có giao dịch liên kết cần nộp phụ lục nào",
    "xử lý thế nào khi hóa đơn điện tử viết sai",
    "khung hình phạt đối với tội trốn thuế",
    "điều kiện để được hoàn thuế GTGT là gì",
    "thuế suất TNDN ưu đãi cho doanh nghiệp mới thành lập",
]

# ── 11. DEBATE: escalate_to_debate ──
DEBATE_TEMPLATES = [
    "mở phiên tòa tranh biện AI cho hồ sơ phức tạp của {tc}",
    "tổ chức Multi-Agent Debate để phân tích đa chiều MST {tc}",
    "kích hoạt hội chẩn đa đặc vụ AI cho doanh nghiệp {tc}",
    "hồ sơ {tc} rất mâu thuẫn, đưa ra tòa án AI để xét xử lại",
    "cần ý kiến phản biện từ nhiều AI về doanh nghiệp {tc}",
    "chạy phiên tranh biện đa đặc vụ cho MST {tc}",
    "khởi động Debate Engine đánh giá lại điểm rủi ro của {tc}",
    "triệu tập tòa án AI hội chẩn về trường hợp {tc}",
]

# ══════════════════════════════════════════
# THOUGHT TEMPLATES - Đa dạng hóa suy luận
# ══════════════════════════════════════════
def thought_variants(base_thoughts):
    """Trả về 1 thought ngẫu nhiên từ danh sách."""
    return random.choice(base_thoughts)

THOUGHTS = {
    "top_n_risky_companies": [
        "Người dùng cần danh sách tổng hợp các doanh nghiệp rủi ro cao nhất. Gọi top_n_risky_companies.",
        "Yêu cầu xếp hạng và lọc top đối tượng rủi ro. Công cụ phù hợp: top_n_risky_companies.",
        "Đây là truy vấn danh sách top N, không cần MST cụ thể. Dùng top_n_risky_companies.",
    ],
    "company_risk_lookup": [
        "Cần đánh giá rủi ro tổng thể cho một doanh nghiệp cụ thể. Gọi company_risk_lookup.",
        "Tra cứu hồ sơ rủi ro độc lập của MST được yêu cầu. Dùng company_risk_lookup.",
        "Yêu cầu kiểm tra điểm rủi ro của một công ty. Công cụ: company_risk_lookup.",
    ],
    "gnn_analysis": [
        "Người dùng muốn truy vết mạng lưới giao dịch VAT, cần phân tích đồ thị. Gọi gnn_analysis.",
        "Yêu cầu phân tích liên kết hóa đơn và mạng lưới. Công cụ GNN phù hợp: gnn_analysis.",
        "Truy vết chuỗi giao dịch, sơ đồ mạng lưới VAT. Dùng gnn_analysis.",
    ],
    "vae_anomaly_scan": [
        "Cần phát hiện bất thường trong hóa đơn bằng deep learning. Gọi vae_anomaly_scan.",
        "Yêu cầu quét anomaly, tìm giao dịch dị thường. Công cụ: vae_anomaly_scan.",
        "Phân tích bất thường thống kê trong dữ liệu. Dùng vae_anomaly_scan.",
    ],
    "invoice_risk_scan": [
        "Rà soát rủi ro hóa đơn đầu vào đầu ra. Gọi invoice_risk_scan.",
        "Kiểm tra chất lượng và tính hợp lệ hóa đơn. Dùng invoice_risk_scan.",
        "Quét rủi ro toàn bộ hóa đơn VAT. Công cụ: invoice_risk_scan.",
    ],
    "motif_detection": [
        "Tìm kiếm vòng lặp giao dịch, mẫu mua bán hóa đơn khép kín. Gọi motif_detection.",
        "Phát hiện motif giao dịch bất thường, đường dây hóa đơn. Dùng motif_detection.",
        "Cần kiểm tra chuỗi hóa đơn vòng lặp. Công cụ: motif_detection.",
    ],
    "delinquency_check": [
        "Dự báo rủi ro nợ đọng, chậm nộp thuế. Gọi delinquency_check.",
        "Đánh giá nguy cơ trễ hạn nộp thuế trong tương lai. Dùng delinquency_check.",
        "Kiểm tra khả năng nợ đọng thuế 30/60/90 ngày. Công cụ: delinquency_check.",
    ],
    "causal_uplift_recommend": [
        "Tìm biện pháp cưỡng chế thu hồi nợ hiệu quả nhất. Gọi causal_uplift_recommend.",
        "Đề xuất hành động can thiệp tối ưu để đòi nợ. Dùng causal_uplift_recommend.",
        "So sánh hiệu quả các biện pháp cưỡng chế. Công cụ: causal_uplift_recommend.",
    ],
    "macro_forecast": [
        "Mô phỏng tác động kinh tế vĩ mô lên thu ngân sách. Gọi macro_forecast.",
        "Chạy kịch bản kinh tế với các tham số GDP và thuế suất. Dùng macro_forecast.",
        "Đánh giá độ nhạy ngân sách với biến số vĩ mô. Công cụ: macro_forecast.",
    ],
    "knowledge_search": [
        "Đây là câu hỏi pháp luật thuế, không liên quan đến MST cụ thể. Gọi knowledge_search.",
        "Tra cứu quy định, văn bản pháp luật thuế. Dùng knowledge_search.",
        "Câu hỏi về luật, thông tư, nghị định thuế. Công cụ: knowledge_search.",
    ],
    "escalate_to_debate": [
        "Hồ sơ phức tạp, cần nhiều góc nhìn AI phản biện. Gọi escalate_to_debate.",
        "Kích hoạt phiên tranh biện đa đặc vụ cho hồ sơ mâu thuẫn. Dùng escalate_to_debate.",
        "Cần hội chẩn AI đa chiều. Công cụ: escalate_to_debate.",
    ],
}

def make_tc():
    return str(random.randint(1000000000, 9999999999))

LEGAL_DEEP_DATA = [
    {
        "q": "Thuế suất thuế GTGT năm 2024 được giảm xuống bao nhiêu và áp dụng cho mặt hàng nào?",
        "tool_query": "Nghị định 72/2024/NĐ-CP giảm thuế GTGT",
        "doc_title": "Nghị định 72/2024/NĐ-CP",
        "doc_snippet": "Giảm thuế giá trị gia tăng đối với các nhóm hàng hóa, dịch vụ đang áp dụng mức thuế suất 10%... Mức thuế suất áp dụng 8%",
        "answer": "Dựa trên kết quả tra cứu **Nghị định 72/2024/NĐ-CP**, hệ thống xin tư vấn như sau:\n\n**1. Mức giảm thuế suất:**\n- Thuế suất thuế Giá trị gia tăng (GTGT) được giảm từ **10% xuống 8%**.\n- Đối với cơ sở kinh doanh tính thuế theo tỷ lệ % trên doanh thu, được giảm 20% mức tỷ lệ % để tính thuế GTGT.\n\n**2. Đối tượng áp dụng:**\nÁp dụng đối với các nhóm hàng hóa, dịch vụ đang chịu mức thuế suất 10%, **TRỪ** các nhóm hàng hóa, dịch vụ sau:\n- Viễn thông, hoạt động tài chính, ngân hàng, chứng khoán, bảo hiểm.\n- Kinh doanh bất động sản, kim loại và sản phẩm từ kim loại đúc sẵn, sản phẩm khai khoáng (không kể khai thác than), than cốc, dầu mỏ tinh chế, sản phẩm hóa chất.\n- Sản phẩm hàng hóa và dịch vụ chịu thuế tiêu thụ đặc biệt.\n- Công nghệ thông tin theo pháp luật về công nghệ thông tin.\n\n**3. Thời gian hiệu lực:**\nChính sách giảm thuế này áp dụng từ ngày **01/07/2024 đến hết ngày 31/12/2024**."
    },
    {
        "q": "Doanh nghiệp có giao dịch liên kết thì chi phí lãi vay bị khống chế như thế nào?",
        "tool_query": "Nghị định 132/2020/NĐ-CP khống chế chi phí lãi vay giao dịch liên kết",
        "doc_title": "Nghị định 132/2020/NĐ-CP",
        "doc_snippet": "Tổng chi phí lãi vay (sau khi trừ lãi tiền gửi và lãi cho vay) được trừ khi xác định thu nhập chịu thuế TNDN không vượt quá 30% của tổng lợi nhuận thuần cộng chi phí lãi vay cộng chi phí khấu hao (EBITDA).",
        "answer": "Theo quy định tại **Điều 16, Nghị định 132/2020/NĐ-CP** quy định về quản lý thuế đối với doanh nghiệp có giao dịch liên kết, việc khống chế chi phí lãi vay được thực hiện như sau:\n\n**1. Mức khống chế:**\nTổng chi phí lãi vay phát sinh trong kỳ của người nộp thuế (sau khi trừ lãi tiền gửi và lãi cho vay) được trừ khi xác định thu nhập chịu thuế Thu nhập doanh nghiệp (TNDN) **không vượt quá 30%** của tổng lợi nhuận thuần từ hoạt động kinh doanh trong kỳ cộng chi phí lãi vay sau khi trừ lãi tiền gửi và lãi cho vay cộng chi phí khấu hao phát sinh trong kỳ (EBITDA).\n\n**2. Xử lý phần chi phí vượt mức:**\nPhần chi phí lãi vay không được trừ (phần vượt quá 30%) được chuyển sang kỳ tính thuế tiếp theo khi xác định tổng chi phí lãi vay được trừ trong trường hợp tổng chi phí lãi vay phát sinh được trừ của kỳ tính thuế tiếp theo thấp hơn mức quy định. Thời gian chuyển chi phí lãi vay tính liên tục không quá 05 năm kể từ năm tiếp sau năm phát sinh chi phí lãi vay không được trừ.\n\n**Khuyến nghị:** Doanh nghiệp cần rà soát lại cơ cấu vốn, hạn chế vay nợ quá cao giữa các công ty liên kết để tránh việc chi phí lãi vay bị loại khỏi chi phí hợp lý khi tính thuế TNDN."
    },
    {
        "q": "Mức phạt chậm nộp hồ sơ khai thuế là bao nhiêu?",
        "tool_query": "Nghị định 125/2020/NĐ-CP phạt chậm nộp hồ sơ khai thuế",
        "doc_title": "Nghị định 125/2020/NĐ-CP",
        "doc_snippet": "Xử phạt hành vi chậm nộp hồ sơ khai thuế: Phạt cảnh cáo từ 01-05 ngày... Phạt 15-25 triệu đồng nếu nộp chậm trên 90 ngày",
        "answer": "Căn cứ **Điều 13 Nghị định 125/2020/NĐ-CP**, mức phạt đối với hành vi chậm nộp hồ sơ khai thuế được quy định tùy thuộc vào thời gian chậm nộp như sau:\n\n1. **Phạt cảnh cáo:** Nếu chậm nộp từ 01 ngày đến 05 ngày và có tình tiết giảm nhẹ.\n2. **Phạt tiền từ 2.000.000 đồng đến 5.000.000 đồng:** Nếu chậm nộp từ 01 ngày đến 30 ngày (trừ trường hợp bị phạt cảnh cáo).\n3. **Phạt tiền từ 5.000.000 đồng đến 8.000.000 đồng:** Nếu chậm nộp từ 31 ngày đến 60 ngày.\n4. **Phạt tiền từ 8.000.000 đồng đến 15.000.000 đồng:** Nếu chậm nộp từ 61 ngày đến 90 ngày hoặc chậm nộp từ 91 ngày trở lên nhưng không phát sinh số thuế phải nộp.\n5. **Phạt tiền từ 15.000.000 đồng đến 25.000.000 đồng:** Nếu nộp hồ sơ khai thuế quá thời hạn trên 90 ngày, thuộc trường hợp phát sinh số thuế phải nộp và người nộp thuế đã nộp đủ số tiền thuế, tiền chậm nộp trước khi cơ quan thuế công bố quyết định thanh tra, kiểm tra.\n\n**Lưu ý:** Ngoài tiền phạt hành chính, nếu phát sinh số thuế phải nộp, người nộp thuế còn phải nộp thêm tiền chậm nộp tiền thuế tính theo mức **0,03%/ngày** tính trên số tiền thuế chậm nộp (theo Điều 59 Luật Quản lý thuế số 38/2019/QH14)."
    },
    {
        "q": "Doanh thu bao nhiêu thì hộ kinh doanh phải nộp thuế?",
        "tool_query": "Thông tư 40/2021/TT-BTC doanh thu tính thuế hộ kinh doanh",
        "doc_title": "Thông tư 40/2021/TT-BTC",
        "doc_snippet": "Hộ kinh doanh, cá nhân kinh doanh có doanh thu từ hoạt động sản xuất, kinh doanh trong năm dương lịch từ 100 triệu đồng trở xuống thì KHÔNG phải nộp thuế GTGT và KHÔNG phải nộp thuế TNCN.",
        "answer": "Theo quy định tại **Điều 4 Thông tư 40/2021/TT-BTC** của Bộ Tài chính hướng dẫn thuế GTGT, thuế TNCN đối với hộ kinh doanh, cá nhân kinh doanh:\n\n- Hộ kinh doanh, cá nhân kinh doanh có doanh thu từ hoạt động sản xuất, kinh doanh trong năm dương lịch **từ 100 triệu đồng trở xuống** thì **KHÔNG** phải nộp thuế Giá trị gia tăng (GTGT) và **KHÔNG** phải nộp thuế Thu nhập cá nhân (TNCN).\n- Trường hợp doanh thu trong năm dương lịch **trên 100 triệu đồng**, hộ kinh doanh phải nộp thuế GTGT và thuế TNCN theo tỷ lệ quy định đối với từng ngành nghề kinh doanh.\n\n**Ví dụ về mức thuế khoán (theo Điều 7 TT 40):**\n- Phân phối, cung cấp hàng hóa (bán buôn, bán lẻ): Thuế suất GTGT 1% + TNCN 0.5% = Tổng 1.5% trên doanh thu.\n- Dịch vụ, xây dựng (không bao thầu nguyên vật liệu): Thuế suất GTGT 5% + TNCN 2% = Tổng 7% trên doanh thu.\n\n**Lưu ý:** Hộ kinh doanh vẫn phải thực hiện nghĩa vụ đăng ký thuế, nộp lệ phí môn bài theo quy định (trừ trường hợp doanh thu dưới 100 triệu/năm thì được miễn lệ phí môn bài)."
    },
    {
        "q": "Mua hàng giá trị bao nhiêu thì bắt buộc phải chuyển khoản để được khấu trừ thuế?",
        "tool_query": "Thông tư 96/2015/TT-BTC điều kiện thanh toán không dùng tiền mặt",
        "doc_title": "Thông tư 96/2015/TT-BTC",
        "doc_snippet": "Hóa đơn mua hàng hóa, dịch vụ từng lần từ 20 triệu đồng trở lên (đã gồm VAT) phải có chứng từ thanh toán không dùng tiền mặt.",
        "answer": "Theo quy định tại **Thông tư 96/2015/TT-BTC** (sửa đổi, bổ sung Điều 6 Thông tư 78/2014/TT-BTC) về chi phí được trừ khi tính thuế TNDN và quy định về khấu trừ thuế GTGT đầu vào:\n\nĐối với hóa đơn mua hàng hóa, dịch vụ từng lần có giá trị **từ 20 triệu đồng trở lên** (giá này đã bao gồm thuế GTGT), doanh nghiệp **bắt buộc phải có chứng từ thanh toán không dùng tiền mặt** (ví dụ: ủy nhiệm chi, chuyển khoản qua ngân hàng từ tài khoản của công ty mua sang tài khoản của công ty bán) để được:\n\n1. **Khấu trừ thuế GTGT đầu vào:** Nếu thanh toán bằng tiền mặt, số thuế GTGT đầu vào của hóa đơn đó sẽ không được khấu trừ.\n2. **Tính vào chi phí hợp lý (chi phí được trừ) khi tính thuế TNDN:** Nếu thanh toán bằng tiền mặt, toàn bộ giá trị hóa đơn trước thuế cũng sẽ bị loại khỏi chi phí hợp lý, làm tăng số thuế TNDN phải nộp.\n\n**Lưu ý quan trọng:**\n- Việc mua hàng hóa, dịch vụ của cùng một nhà cung cấp trong cùng một ngày, mặc dù mỗi hóa đơn có giá trị dưới 20 triệu đồng nhưng tổng giá trị các hóa đơn trong ngày từ 20 triệu đồng trở lên thì cũng bắt buộc phải thanh toán qua ngân hàng.\n- Chứng từ nộp tiền mặt trực tiếp vào tài khoản của bên bán **không** được coi là chứng từ thanh toán không dùng tiền mặt."
    }
]

try:
    from legal_deep_data import LEGAL_DEEP_DATA_NEW
    from legal_deep_data_2 import LEGAL_DEEP_DATA_NEW_2
    from legal_deep_data_3 import LEGAL_DEEP_DATA_NEW_3
    from legal_deep_data_4 import LEGAL_DEEP_DATA_NEW_4
    
    LEGAL_DEEP_DATA.extend(LEGAL_DEEP_DATA_NEW)
    LEGAL_DEEP_DATA.extend(LEGAL_DEEP_DATA_NEW_2)
    LEGAL_DEEP_DATA.extend(LEGAL_DEEP_DATA_NEW_3)
    LEGAL_DEEP_DATA.extend(LEGAL_DEEP_DATA_NEW_4)
except ImportError as e:
    print(f"Warning: Could not import extended legal deep data - {e}")

def make_legal_record(data_item):
    query = f"{random.choice(P)}{data_item['q']}{random.choice(S)}".strip()
    query = query[0].upper() + query[1:]
    
    thought = "Đây là câu hỏi pháp luật thuế. Tôi cần tra cứu cơ sở tri thức bằng knowledge_search, sau đó tổng hợp câu trả lời chi tiết cho người dùng."
    tool_args = {"query": data_item["tool_query"]}
    tc_json = json.dumps({"name": "knowledge_search", "arguments": tool_args}, ensure_ascii=False)
    
    tool_result = json.dumps({
        "status": "success",
        "results": [
            {
                "title": data_item["doc_title"],
                "content": data_item["doc_snippet"],
                "score": 0.95
            }
        ]
    }, ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"<thought>{thought}</thought>\n<tool_call>{tc_json}</tool_call>"},
            {"role": "tool", "name": "knowledge_search", "content": tool_result},
            {"role": "assistant", "content": data_item["answer"]}
        ]
    }

def make_record(query, tool_name, tool_args):
    t = thought_variants(THOUGHTS[tool_name])
    tc_json = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"<thought>{t}</thought>\n<tool_call>{tc_json}</tool_call>"}
        ]
    }

def generate(total=10000):
    dataset = []
    per = total // 12  # 12 categories (11 tools + 1 legal deep)

    # 1. top_n
    for _ in range(per):
        n = random.choice([5, 10, 15, 20, 30, 50])
        q = f"{random.choice(P)}{random.choice(TOP_N_TEMPLATES).format(n=n)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "top_n_risky_companies", {"n": n}))

    # 2. company_risk_lookup
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(RISK_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "company_risk_lookup", {"tax_code": tc}))

    # 3. gnn_analysis
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(GNN_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "gnn_analysis", {"tax_code": tc}))

    # 4. vae_anomaly_scan
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(VAE_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "vae_anomaly_scan", {"tax_code": tc}))

    # 5. invoice_risk_scan
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(INVOICE_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "invoice_risk_scan", {"tax_code": tc}))

    # 6. motif_detection
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(MOTIF_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "motif_detection", {"tax_code": tc}))

    # 7. delinquency_check
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(DELI_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "delinquency_check", {"tax_code": tc}))

    # 8. causal_uplift_recommend
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(UPLIFT_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "causal_uplift_recommend", {"tax_code": tc}))

    # 9. macro_forecast
    for _ in range(per):
        gdp = round(random.uniform(-5.0, 10.0), 1)
        vat = random.choice([8, 10, 12, 15])
        cit = random.choice([15, 17, 18, 20, 22])
        q = f"{random.choice(P)}{random.choice(MACRO_TEMPLATES).format(gdp=gdp, vat=vat, cit=cit)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "macro_forecast", {"scenario": {"gdp": gdp, "vat_rate": vat}}))

    # 10. knowledge_search
    for _ in range(per):
        base_q = random.choice(LEGAL_TEMPLATES)
        q = f"{random.choice(P)}{base_q}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "knowledge_search", {"query": base_q}))

    # 11. escalate_to_debate
    for _ in range(per):
        tc = make_tc()
        q = f"{random.choice(P)}{random.choice(DEBATE_TEMPLATES).format(tc=tc)}{random.choice(S)}".strip()
        q = q[0].upper() + q[1:]
        dataset.append(make_record(q, "escalate_to_debate", {"tax_code": tc}))

    # 12. legal_deep_answer
    for _ in range(per):
        data_item = random.choice(LEGAL_DEEP_DATA)
        dataset.append(make_legal_record(data_item))

    random.shuffle(dataset)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Đã tạo {len(dataset)} kịch bản đẳng cấp tại: {OUTPUT}")

if __name__ == "__main__":
    # Keep this legacy file import-compatible, but route CLI usage to the
    # production generator so the Colab notebook receives the full tool/GraphRAG
    # dataset instead of the old 10k subset.
    from generate_mega_agent_dataset_v4 import generate as generate_v4

    generate_v4()
