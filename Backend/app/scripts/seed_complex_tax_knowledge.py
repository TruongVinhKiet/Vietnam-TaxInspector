# -*- coding: utf-8 -*-
"""
Seed Complex Tax Knowledge — Real Vietnamese Tax Documents
===========================================================
Sources: chinhphu.vn, mof.gov.vn, gdt.gov.vn, luatvietnam.vn (researched 2026-05-04)
"""
from __future__ import annotations
import json
from app.scripts.ingest_tax_knowledge import ingest_document


DOCUMENTS = [
    # ── 1. Nghị định 132/2020/NĐ-CP — Chuyển giá ──────────────────────
    {
        "document_key": "nd_132_2020_ndcp",
        "title": "Nghị định 132/2020/NĐ-CP — Quản lý thuế đối với giao dịch liên kết",
        "doc_type": "decree", "authority": "Chính phủ",
        "source_uri": "chinhphu.vn", "version_tag": "v1",
        "content": (
            "Điều 1. Phạm vi điều chỉnh\n"
            "Nghị định này quy định nguyên tắc, phương pháp, trình tự xác định giá giao dịch liên kết; "
            "quyền và nghĩa vụ của người nộp thuế có phát sinh giao dịch liên kết.\n\n"
            "Điều 5. Các bên có quan hệ liên kết\n"
            "Một doanh nghiệp nắm giữ trực tiếp hoặc gián tiếp ít nhất 25% vốn góp của chủ sở hữu "
            "của doanh nghiệp kia.\n\n"
            "Điều 16. Khống chế chi phí lãi vay\n"
            "Tổng chi phí lãi vay (sau khi trừ lãi tiền gửi và lãi cho vay) được trừ khi xác định "
            "thu nhập chịu thuế TNDN không vượt quá 30% của tổng lợi nhuận thuần cộng chi phí lãi vay "
            "cộng chi phí khấu hao (EBITDA).\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
    # ── 2. Nghị định 72/2024/NĐ-CP — Giảm VAT 8% ─────────────────────
    {
        "document_key": "nd_72_2024_ndcp",
        "title": "Nghị định 72/2024/NĐ-CP — Giảm thuế GTGT từ 10% xuống 8%",
        "doc_type": "decree", "authority": "Chính phủ",
        "source_uri": "chinhphu.vn", "version_tag": "v1",
        "content": (
            "Điều 1. Giảm thuế giá trị gia tăng\n"
            "Giảm thuế GTGT đối với hàng hóa, dịch vụ đang áp dụng mức thuế suất 10%, TRỪ:\n"
            "a) Viễn thông, tài chính, ngân hàng, chứng khoán, bảo hiểm, bất động sản, kim loại đúc sẵn, "
            "khai khoáng, than cốc, dầu mỏ tinh chế, sản phẩm hóa chất.\n"
            "b) Sản phẩm chịu thuế tiêu thụ đặc biệt.\n"
            "c) Công nghệ thông tin.\n\n"
            "Mức thuế suất áp dụng: 8% (phương pháp khấu trừ).\n\n"
            "Điều 2. Hiệu lực: từ 01/07/2024 đến hết 31/12/2024.\n"
            "Trạng thái: Hết hiệu lực từ 01/01/2025.\n"
        ),
    },
    # ── 3. Luật Quản lý thuế 38/2019/QH14 ─────────────────────────────
    {
        "document_key": "luat_38_2019_qh14",
        "title": "Luật Quản lý thuế số 38/2019/QH14",
        "doc_type": "law", "authority": "Quốc hội",
        "source_uri": "quochoi.vn", "version_tag": "v1",
        "content": (
            "Điều 17. Trách nhiệm của người nộp thuế\n"
            "Khai thuế chính xác, trung thực, đầy đủ và nộp hồ sơ thuế đúng thời hạn.\n\n"
            "Điều 59. Xử lý đối với việc chậm nộp tiền thuế\n"
            "Người nộp thuế chậm nộp tiền thuế so với thời hạn quy định thì ngoài việc nộp đủ tiền thuế, "
            "phải nộp tiền chậm nộp theo mức 0,03%/ngày tính trên số tiền thuế chậm nộp.\n\n"
            "Điều 125. Các biện pháp cưỡng chế thi hành quyết định hành chính về quản lý thuế\n"
            "1. Trích tiền từ tài khoản ngân hàng, phong tỏa tài khoản.\n"
            "2. Khấu trừ một phần tiền lương hoặc thu nhập.\n"
            "3. Dừng thủ tục hải quan.\n"
            "4. Ngừng sử dụng hóa đơn.\n"
            "5. Kê biên tài sản, bán đấu giá.\n"
            "6. Thu tiền, tài sản khác do bên thứ ba nắm giữ.\n"
            "7. Thu hồi giấy chứng nhận đăng ký doanh nghiệp.\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
    # ── 4. Nghị định 123/2020/NĐ-CP — Hóa đơn chứng từ ──────────────
    {
        "document_key": "nd_123_2020_ndcp",
        "title": "Nghị định 123/2020/NĐ-CP — Quy định về hóa đơn, chứng từ",
        "doc_type": "decree", "authority": "Chính phủ",
        "source_uri": "chinhphu.vn", "version_tag": "v1",
        "content": (
            "Điều 3. Hình thức hóa đơn\n"
            "1. Hóa đơn điện tử có mã của cơ quan thuế.\n"
            "2. Hóa đơn điện tử không có mã của cơ quan thuế.\n\n"
            "Điều 4. Đối tượng áp dụng hóa đơn điện tử\n"
            "Doanh nghiệp, hợp tác xã, hộ kinh doanh, cá nhân kinh doanh đều phải sử dụng hóa đơn "
            "điện tử khi bán hàng hóa, cung cấp dịch vụ.\n\n"
            "Điều 9. Thời điểm lập hóa đơn\n"
            "Thời điểm lập hóa đơn đối với bán hàng hóa là thời điểm chuyển giao quyền sở hữu hoặc "
            "quyền sử dụng hàng hóa cho người mua. Đối với cung cấp dịch vụ là thời điểm hoàn thành "
            "việc cung cấp dịch vụ hoặc thời điểm lập hóa đơn thanh toán.\n\n"
            "Điều 15. Hóa đơn điện tử khởi tạo từ máy tính tiền\n"
            "Cơ sở kinh doanh bán lẻ (siêu thị, nhà hàng, cửa hàng tiện lợi) sử dụng máy tính tiền "
            "có kết nối chuyển dữ liệu điện tử với cơ quan thuế.\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
    # ── 5. Nghị định 125/2020/NĐ-CP — Xử phạt thuế, hóa đơn ─────────
    {
        "document_key": "nd_125_2020_ndcp",
        "title": "Nghị định 125/2020/NĐ-CP — Xử phạt vi phạm hành chính về thuế, hóa đơn",
        "doc_type": "decree", "authority": "Chính phủ",
        "source_uri": "chinhphu.vn", "version_tag": "v1",
        "content": (
            "Điều 13. Xử phạt hành vi chậm nộp hồ sơ khai thuế\n"
            "- Phạt cảnh cáo: nộp chậm từ 01 đến 05 ngày (có tình tiết giảm nhẹ).\n"
            "- Phạt 2–5 triệu đồng: nộp chậm từ 01 đến 30 ngày.\n"
            "- Phạt 5–8 triệu đồng: nộp chậm từ 31 đến 60 ngày.\n"
            "- Phạt 8–15 triệu đồng: nộp chậm từ 61 đến 90 ngày.\n"
            "- Phạt 15–25 triệu đồng: nộp chậm trên 90 ngày nhưng không phát sinh số thuế phải nộp.\n\n"
            "Điều 17. Xử phạt hành vi trốn thuế\n"
            "Phạt tiền 1 lần số thuế trốn khi có tình tiết giảm nhẹ, "
            "phạt tới 3 lần số thuế trốn khi có tình tiết tăng nặng.\n"
            "Các hành vi: không nộp hồ sơ đăng ký thuế; sử dụng hóa đơn không hợp pháp; "
            "ghi sai giá trị trên hóa đơn; khai man để giảm thuế phải nộp.\n\n"
            "Điều 24. Xử phạt vi phạm về hóa đơn điện tử\n"
            "- Phạt 4–8 triệu: không lập hóa đơn khi bán hàng.\n"
            "- Phạt 10–20 triệu: sử dụng hóa đơn không hợp pháp.\n"
            "Mức phạt tối đa: 100 triệu (tổ chức), 50 triệu (cá nhân).\n"
            "Trạng thái: Còn hiệu lực (sửa đổi bởi NĐ 310/2025 từ 16/01/2026).\n"
        ),
    },
    # ── 6. Thông tư 40/2021/TT-BTC — Hộ kinh doanh ──────────────────
    {
        "document_key": "tt_40_2021_tt_btc",
        "title": "Thông tư 40/2021/TT-BTC — Thuế GTGT, TNCN đối với hộ kinh doanh, cá nhân kinh doanh",
        "doc_type": "circular", "authority": "Bộ Tài chính",
        "source_uri": "mof.gov.vn", "version_tag": "v1",
        "content": (
            "Điều 4. Nguyên tắc tính thuế\n"
            "Hộ kinh doanh, cá nhân kinh doanh có doanh thu từ hoạt động sản xuất, kinh doanh "
            "trong năm dương lịch từ 100 triệu đồng trở xuống thì KHÔNG phải nộp thuế GTGT "
            "và KHÔNG phải nộp thuế TNCN.\n\n"
            "Điều 7. Tỷ lệ thuế tính trên doanh thu\n"
            "- Phân phối, cung cấp hàng hóa: GTGT 1% + TNCN 0.5% = 1.5%\n"
            "- Dịch vụ, xây dựng: GTGT 5% + TNCN 2% = 7%\n"
            "- Sản xuất, vận tải, dịch vụ gắn liền hàng hóa: GTGT 3% + TNCN 1.5% = 4.5%\n"
            "- Hoạt động kinh doanh khác: GTGT 2% + TNCN 1% = 3%\n\n"
            "Điều 8. Cá nhân bán hàng online qua mạng, không có địa điểm kinh doanh cố định "
            "nộp thuế theo từng lần phát sinh.\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
    # ── 7. Thông tư 96/2015/TT-BTC — Chi phí được trừ TNDN ──────────
    {
        "document_key": "tt_96_2015_tt_btc",
        "title": "Thông tư 96/2015/TT-BTC — Chi phí được trừ và không được trừ khi tính thuế TNDN",
        "doc_type": "circular", "authority": "Bộ Tài chính",
        "source_uri": "mof.gov.vn", "version_tag": "v1",
        "content": (
            "Điều 4 (sửa đổi Điều 6 TT 78/2014). Chi phí được trừ khi xác định thu nhập chịu thuế:\n\n"
            "Doanh nghiệp được trừ mọi khoản chi nếu đáp ứng ĐỦ 3 điều kiện:\n"
            "1. Khoản chi thực tế phát sinh liên quan đến hoạt động sản xuất, kinh doanh.\n"
            "2. Khoản chi có đủ hóa đơn, chứng từ hợp pháp.\n"
            "3. Hóa đơn mua hàng hóa, dịch vụ từng lần từ 20 triệu đồng trở lên (đã gồm VAT) "
            "phải có chứng từ thanh toán không dùng tiền mặt.\n\n"
            "Các khoản chi KHÔNG được trừ (trích):\n"
            "- Tiền lương đã hạch toán nhưng thực tế không chi trả.\n"
            "- Tiền thưởng cho người lao động không ghi trong hợp đồng lao động, thỏa ước, quy chế.\n"
            "- Chi phí khấu hao tài sản cố định không sử dụng cho SXKD.\n"
            "- Phần chi phí nguyên vật liệu vượt định mức tiêu hao.\n"
            "- Chi phí thuê tài sản cá nhân không có hợp đồng thuê và chứng từ trả tiền.\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
    # ── 8. Luật Thuế TNCN (hợp nhất) — Biểu thuế lũy tiến ───────────
    {
        "document_key": "luat_thue_tncn_hop_nhat",
        "title": "Luật Thuế TNCN số 04/2007/QH12 (sửa đổi 2012, 2014) — Biểu thuế lũy tiến từng phần",
        "doc_type": "law", "authority": "Quốc hội",
        "source_uri": "quochoi.vn", "version_tag": "v1",
        "content": (
            "Điều 22. Biểu thuế lũy tiến từng phần áp dụng cho thu nhập từ tiền lương, tiền công:\n"
            "Bậc 1: Đến 5 triệu — 5%\n"
            "Bậc 2: Trên 5 đến 10 triệu — 10%\n"
            "Bậc 3: Trên 10 đến 18 triệu — 15%\n"
            "Bậc 4: Trên 18 đến 32 triệu — 20%\n"
            "Bậc 5: Trên 32 đến 52 triệu — 25%\n"
            "Bậc 6: Trên 52 đến 80 triệu — 30%\n"
            "Bậc 7: Trên 80 triệu — 35%\n\n"
            "Giảm trừ gia cảnh (theo NQ 954/2020/UBTVQH14):\n"
            "- Bản thân: 11 triệu/tháng (132 triệu/năm).\n"
            "- Mỗi người phụ thuộc: 4,4 triệu/tháng.\n\n"
            "Thu nhập tính thuế = Thu nhập chịu thuế – Các khoản giảm trừ.\n"
            "Trạng thái: Còn hiệu lực (sẽ thay bằng Luật TNCN 2025 từ 01/07/2026).\n"
        ),
    },
    # ── 9. Thông tư 80/2021/TT-BTC — Hồ sơ hoàn thuế ────────────────
    {
        "document_key": "tt_80_2021_tt_btc",
        "title": "Thông tư 80/2021/TT-BTC — Hướng dẫn Luật Quản lý thuế về hồ sơ, thủ tục hoàn thuế",
        "doc_type": "circular", "authority": "Bộ Tài chính",
        "source_uri": "mof.gov.vn", "version_tag": "v1",
        "content": (
            "Điều 28. Hồ sơ hoàn thuế GTGT đối với dự án đầu tư\n"
            "1. Giấy đề nghị hoàn trả khoản thu NSNN (Mẫu 01/HT).\n"
            "2. Bảng kê hóa đơn, chứng từ hàng hóa dịch vụ mua vào (Mẫu 01-1/HT) "
            "nếu chưa gửi hóa đơn điện tử đến CQT.\n\n"
            "Điều 29. Phân loại hồ sơ hoàn thuế\n"
            "- Hoàn thuế TRƯỚC, kiểm tra SAU: Doanh nghiệp không thuộc diện rủi ro cao.\n"
            "- Kiểm tra TRƯỚC, hoàn thuế SAU: Doanh nghiệp mới thành lập dưới 12 tháng; "
            "DN có lịch sử vi phạm; DN hoàn thuế lần đầu.\n\n"
            "Thời hạn giải quyết: 06 ngày (hoàn trước), 40 ngày (kiểm tra trước).\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
    # ── 10. Nghị định 52/2013/NĐ-CP — Thương mại điện tử ────────────
    {
        "document_key": "nd_52_2013_ndcp",
        "title": "Nghị định 52/2013/NĐ-CP — Thương mại điện tử (sửa đổi bổ sung)",
        "doc_type": "decree", "authority": "Chính phủ",
        "source_uri": "chinhphu.vn", "version_tag": "v1",
        "content": (
            "Người bán hàng trên sàn TMĐT (Shopee, Lazada, TikTok Shop) hoặc qua mạng xã hội "
            "(Facebook, Zalo) có nghĩa vụ tự kê khai và nộp thuế nếu doanh thu vượt ngưỡng "
            "100 triệu đồng/năm theo Thông tư 40/2021/TT-BTC.\n\n"
            "Sàn TMĐT có trách nhiệm cung cấp thông tin người bán cho cơ quan thuế "
            "và thực hiện khấu trừ thuế thay cho người bán theo yêu cầu cơ quan quản lý.\n"
            "Trạng thái: Còn hiệu lực.\n"
        ),
    },
]


def seed_complex_docs():
    for d in DOCUMENTS:
        print(f"Ingesting {d['document_key']}...")
        result = ingest_document(
            document_key=d["document_key"],
            title=d["title"],
            doc_type=d["doc_type"],
            authority=d["authority"],
            source_uri=d["source_uri"],
            version_tag=d["version_tag"],
            content=d["content"],
        )
        print(f"  -> {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    seed_complex_docs()
