import sys
sys.stdout.reconfigure(encoding='utf-8')
from ingest_tax_knowledge import TaxKnowledgeIngestor

DOCS = [
    # --- QUẢN LÝ THUẾ (Tax Administration) ---
    {
        'key': 'LUAT_38_2019',
        'title': 'Luật Quản lý thuế số 38/2019/QH14',
        'type': 'law',
        'authority_rank': 90,
        'effective_from': '2020-07-01',
        'content': '''Luật Quản lý thuế số 38/2019/QH14
Điều 1. Phạm vi điều chỉnh
Luật này quy định việc quản lý các loại thuế, các khoản thu khác thuộc ngân sách nhà nước.
Điều 2. Đối tượng áp dụng
1. Người nộp thuế bao gồm: tổ chức, hộ gia đình, hộ kinh doanh, cá nhân nộp thuế.
Điều 17. Trách nhiệm của người nộp thuế
1. Đăng ký thuế, sử dụng mã số thuế theo quy định của pháp luật.
2. Khai thuế chính xác, trung thực, đầy đủ.
3. Nộp tiền thuế, tiền chậm nộp, tiền phạt đầy đủ, đúng thời hạn.
Điều 42. Nguyên tắc khai thuế, tính thuế
1. Người nộp thuế phải khai chính xác, trung thực, đầy đủ các nội dung trong tờ khai thuế theo mẫu do Bộ trưởng Bộ Tài chính quy định.
''',
        'relations': []
    },
    {
        'key': 'ND_126_2020',
        'title': 'Nghị định 126/2020/NĐ-CP Quy định chi tiết một số điều của Luật Quản lý thuế',
        'type': 'decree',
        'authority_rank': 80,
        'effective_from': '2020-12-05',
        'content': '''Nghị định 126/2020/NĐ-CP
Điều 7. Hồ sơ khai thuế
1. Hồ sơ khai thuế đối với loại thuế khai và nộp theo tháng là tờ khai thuế tháng.
2. Hồ sơ khai thuế đối với loại thuế khai và nộp theo quý là tờ khai thuế quý.
Điều 8. Các loại thuế khai theo tháng, khai theo quý, khai theo năm, khai theo từng lần phát sinh nghĩa vụ thuế và khai quyết toán thuế
1. Các loại thuế, khoản thu khác thuộc ngân sách nhà nước do cơ quan quản lý thuế thu khai theo tháng.
''',
        'relations': [
            {'target': 'LUAT_38_2019', 'type': 'implements', 'weight': 1.0}
        ]
    },
    {
        'key': 'TT_80_2021',
        'title': 'Thông tư 80/2021/TT-BTC Hướng dẫn thi hành một số điều của Luật Quản lý thuế và NĐ 126/2020',
        'type': 'circular',
        'authority_rank': 70,
        'effective_from': '2022-01-01',
        'content': '''Thông tư 80/2021/TT-BTC
Điều 1. Phạm vi điều chỉnh
Thông tư này hướng dẫn thi hành một số điều của Luật Quản lý thuế số 38/2019/QH14 và Nghị định số 126/2020/NĐ-CP.
Điều 12. Phân bổ số thuế phải nộp đối với người nộp thuế có đơn vị phụ thuộc, địa điểm kinh doanh khác tỉnh
1. Người nộp thuế có hoạt động, kinh doanh trên nhiều địa bàn cấp tỉnh khác nơi có trụ sở chính thực hiện hạch toán tập trung tại trụ sở chính.
''',
        'relations': [
            {'target': 'LUAT_38_2019', 'type': 'implements', 'weight': 1.0},
            {'target': 'ND_126_2020', 'type': 'interprets', 'weight': 0.95}
        ]
    },

    # --- THUẾ GIÁ TRỊ GIA TĂNG (VAT) ---
    {
        'key': 'LUAT_GTGT_13_2008',
        'title': 'Luật Thuế giá trị gia tăng số 13/2008/QH12',
        'type': 'law',
        'authority_rank': 90,
        'effective_from': '2009-01-01',
        'content': '''Luật Thuế giá trị gia tăng số 13/2008/QH12
Điều 2. Thuế giá trị gia tăng
Thuế giá trị gia tăng là thuế tính trên giá trị tăng thêm của hàng hóa, dịch vụ phát sinh trong quá trình từ sản xuất, lưu thông đến tiêu dùng.
Điều 5. Đối tượng không chịu thuế
Sản phẩm trồng trọt, chăn nuôi, thủy sản nuôi trồng, đánh bắt chưa chế biến thành các sản phẩm khác hoặc chỉ qua sơ chế thông thường.
Điều 13. Các trường hợp hoàn thuế
1. Cơ sở kinh doanh nộp thuế giá trị gia tăng theo phương pháp khấu trừ thuế nếu có số thuế giá trị gia tăng đầu vào chưa được khấu trừ hết.
''',
        'relations': []
    },
    {
        'key': 'LUAT_GTGT_31_2013_SDBS',
        'title': 'Luật sửa đổi, bổ sung một số điều của Luật Thuế giá trị gia tăng số 31/2013/QH13',
        'type': 'law',
        'authority_rank': 90,
        'effective_from': '2014-01-01',
        'content': '''Luật số 31/2013/QH13 Sửa đổi, bổ sung Luật Thuế GTGT
Điều 1. Sửa đổi, bổ sung một số điều của Luật thuế giá trị gia tăng
3. Bổ sung khoản 2 Điều 8 về mức thuế suất 5% đối với bán, cho thuê, cho thuê mua nhà ở xã hội.
7. Sửa đổi Điều 13 về các trường hợp hoàn thuế đối với hàng hóa, dịch vụ xuất khẩu.
''',
        'relations': [
            {'target': 'LUAT_GTGT_13_2008', 'type': 'amends', 'weight': 0.95}
        ]
    },
    {
        'key': 'ND_209_2013',
        'title': 'Nghị định 209/2013/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều Luật Thuế GTGT',
        'type': 'decree',
        'authority_rank': 80,
        'effective_from': '2014-01-01',
        'content': '''Nghị định 209/2013/NĐ-CP
Điều 9. Khấu trừ thuế giá trị gia tăng đầu vào
1. Thuế giá trị gia tăng đầu vào của hàng hóa, dịch vụ dùng cho sản xuất, kinh doanh hàng hóa, dịch vụ chịu thuế giá trị gia tăng được khấu trừ toàn bộ.
2. Điều kiện khấu trừ thuế GTGT đầu vào là phải có hóa đơn GTGT hợp pháp, có chứng từ thanh toán không dùng tiền mặt đối với hàng hóa từ 20 triệu đồng trở lên.
''',
        'relations': [
            {'target': 'LUAT_GTGT_13_2008', 'type': 'implements', 'weight': 1.0},
            {'target': 'LUAT_GTGT_31_2013_SDBS', 'type': 'related_to', 'weight': 0.8}
        ]
    },
    {
        'key': 'TT_219_2013',
        'title': 'Thông tư 219/2013/TT-BTC Hướng dẫn thi hành Luật Thuế GTGT và NĐ 209/2013',
        'type': 'circular',
        'authority_rank': 70,
        'effective_from': '2014-01-01',
        'content': '''Thông tư 219/2013/TT-BTC
Điều 15. Điều kiện khấu trừ thuế giá trị gia tăng đầu vào
1. Có hóa đơn giá trị gia tăng hợp pháp của hàng hóa, dịch vụ mua vào.
2. Có chứng từ thanh toán không dùng tiền mặt đối với hàng hóa, dịch vụ mua vào từ hai mươi triệu đồng trở lên.
3. Chứng từ thanh toán qua ngân hàng được hiểu là có chứng từ chứng minh việc chuyển tiền từ tài khoản của bên mua sang tài khoản của bên bán.
''',
        'relations': [
            {'target': 'ND_209_2013', 'type': 'interprets', 'weight': 0.95},
            {'target': 'LUAT_GTGT_13_2008', 'type': 'implements', 'weight': 1.0}
        ]
    },

    # --- HÓA ĐƠN ĐIỆN TỬ (E-Invoices) ---
    {
        'key': 'ND_123_2020',
        'title': 'Nghị định 123/2020/NĐ-CP Quy định về hóa đơn, chứng từ',
        'type': 'decree',
        'authority_rank': 80,
        'effective_from': '2022-07-01',
        'content': '''Nghị định 123/2020/NĐ-CP
Điều 4. Nguyên tắc lập, quản lý, sử dụng hóa đơn, chứng từ
1. Khi bán hàng hóa, cung cấp dịch vụ, người bán phải lập hóa đơn để giao cho người mua. Hóa đơn phải được lập theo định dạng chuẩn dữ liệu của cơ quan thuế.
Điều 9. Thời điểm lập hóa đơn
1. Thời điểm lập hóa đơn đối với bán hàng hóa là thời điểm chuyển giao quyền sở hữu hoặc quyền sử dụng hàng hóa cho người mua, không phân biệt đã thu được tiền hay chưa thu được tiền.
''',
        'relations': [
            {'target': 'LUAT_38_2019', 'type': 'implements', 'weight': 1.0}
        ]
    },
    {
        'key': 'TT_78_2021',
        'title': 'Thông tư 78/2021/TT-BTC Hướng dẫn thực hiện một số điều của Luật Quản lý thuế và NĐ 123 về hóa đơn',
        'type': 'circular',
        'authority_rank': 70,
        'effective_from': '2022-07-01',
        'content': '''Thông tư 78/2021/TT-BTC
Điều 5. Sử dụng hóa đơn điện tử có mã của cơ quan thuế được khởi tạo từ máy tính tiền có kết nối chuyển dữ liệu điện tử với cơ quan thuế.
1. Đối tượng áp dụng: Doanh nghiệp, hộ kinh doanh nộp thuế theo phương pháp kê khai hoạt động kinh doanh bán lẻ trực tiếp đến người tiêu dùng.
Điều 7. Xử lý hóa đơn điện tử, bảng tổng hợp dữ liệu hóa đơn điện tử đã gửi cơ quan thuế có sai sót.
''',
        'relations': [
            {'target': 'ND_123_2020', 'type': 'interprets', 'weight': 0.95},
            {'target': 'LUAT_38_2019', 'type': 'implements', 'weight': 0.8}
        ]
    },

    # --- THUẾ THU NHẬP DOANH NGHIỆP (CIT) ---
    {
        'key': 'LUAT_TNDN_14_2008',
        'title': 'Luật Thuế thu nhập doanh nghiệp số 14/2008/QH12',
        'type': 'law',
        'authority_rank': 90,
        'effective_from': '2009-01-01',
        'content': '''Luật Thuế thu nhập doanh nghiệp số 14/2008/QH12
Điều 2. Người nộp thuế
1. Người nộp thuế thu nhập doanh nghiệp là tổ chức hoạt động sản xuất, kinh doanh hàng hóa, dịch vụ có thu nhập chịu thuế theo quy định của Luật này.
Điều 9. Các khoản chi được trừ và không được trừ khi xác định thu nhập chịu thuế
1. Doanh nghiệp được trừ mọi khoản chi nếu đáp ứng đủ các điều kiện sau đây:
a) Khoản chi thực tế phát sinh liên quan đến hoạt động sản xuất, kinh doanh của doanh nghiệp;
b) Khoản chi có đủ hóa đơn, chứng từ hợp pháp theo quy định của pháp luật.
''',
        'relations': []
    },
    {
        'key': 'ND_218_2013',
        'title': 'Nghị định 218/2013/NĐ-CP Quy định chi tiết và hướng dẫn thi hành Luật Thuế TNDN',
        'type': 'decree',
        'authority_rank': 80,
        'effective_from': '2014-02-15',
        'content': '''Nghị định 218/2013/NĐ-CP
Điều 9. Các khoản chi được trừ và không được trừ khi xác định thu nhập chịu thuế
1. Trừ các khoản chi quy định tại Khoản 2 Điều này, doanh nghiệp được trừ mọi khoản chi nếu đáp ứng đủ các điều kiện.
2. Các khoản chi không được trừ bao gồm: Chi phí khấu hao tài sản cố định không đúng quy định; Chi phí tiền lương, tiền công không có chứng từ thanh toán.
''',
        'relations': [
            {'target': 'LUAT_TNDN_14_2008', 'type': 'implements', 'weight': 1.0}
        ]
    },
    {
        'key': 'TT_78_2014',
        'title': 'Thông tư 78/2014/TT-BTC Hướng dẫn thi hành Nghị định số 218/2013/NĐ-CP',
        'type': 'circular',
        'authority_rank': 70,
        'effective_from': '2014-08-02',
        'content': '''Thông tư 78/2014/TT-BTC
Điều 6. Các khoản chi được trừ và không được trừ khi xác định thu nhập chịu thuế
1. Điều kiện khoản chi được trừ:
a) Khoản chi thực tế phát sinh liên quan đến hoạt động sản xuất, kinh doanh của doanh nghiệp.
b) Khoản chi có đủ hoá đơn, chứng từ hợp pháp theo quy định của pháp luật.
c) Khoản chi nếu có hoá đơn mua hàng hoá, dịch vụ từng lần có giá trị từ 20 triệu đồng trở lên (giá đã bao gồm thuế GTGT) khi thanh toán phải có chứng từ thanh toán không dùng tiền mặt.
2. Các khoản chi không được trừ: Chi tiền lương, tiền công, tiền thưởng cho người lao động đã hạch toán vào chi phí sản xuất kinh doanh trong kỳ nhưng thực tế không chi trả hoặc không có chứng từ thanh toán.
''',
        'relations': [
            {'target': 'ND_218_2013', 'type': 'interprets', 'weight': 0.95},
            {'target': 'LUAT_TNDN_14_2008', 'type': 'implements', 'weight': 1.0},
            {'target': 'TT_219_2013', 'type': 'related_to', 'weight': 0.6} # Liên quan quy định 20 triệu đồng
        ]
    },

    # --- CROSS-REFERENCES & PENALTIES (Xử phạt VPHC) ---
    {
        'key': 'ND_125_2020',
        'title': 'Nghị định 125/2020/NĐ-CP Xử phạt vi phạm hành chính về thuế, hóa đơn',
        'type': 'decree',
        'authority_rank': 80,
        'effective_from': '2020-12-05',
        'content': '''Nghị định 125/2020/NĐ-CP
Điều 16. Xử phạt hành vi khai sai dẫn đến thiếu số tiền thuế phải nộp hoặc tăng số tiền thuế được miễn, giảm, hoàn
1. Phạt 20% số tiền thuế khai thiếu hoặc số tiền thuế đã được miễn, giảm, hoàn cao hơn so với quy định đối với các hành vi: Khai sai căn cứ tính thuế, số tiền thuế được khấu trừ.
Điều 17. Xử phạt hành vi trốn thuế
1. Phạt tiền 1 lần số thuế trốn đối với người nộp thuế có từ một tình tiết giảm nhẹ trở lên khi thực hiện một trong các hành vi vi phạm: Không nộp hồ sơ đăng ký thuế; Sử dụng hóa đơn không hợp pháp để hạch toán giá trị hàng hóa, dịch vụ mua vào làm giảm số tiền thuế phải nộp.
''',
        'relations': [
            {'target': 'LUAT_38_2019', 'type': 'implements', 'weight': 1.0},
            {'target': 'ND_123_2020', 'type': 'related_to', 'weight': 0.8} # Liên quan hóa đơn không hợp pháp
        ]
    },
    
    # --- RỦI RO THUẾ (Tax Risks & Fraud) ---
    {
        'key': 'QD_78_2023',
        'title': 'Quyết định 78/QĐ-TCT 2023 Ban hành Bộ chỉ số tiêu chí đánh giá rủi ro người nộp thuế',
        'type': 'decision',
        'authority_rank': 60,
        'effective_from': '2023-02-02',
        'content': '''Quyết định 78/QĐ-TCT năm 2023
Điều 1. Ban hành kèm theo Quyết định này Bộ chỉ số tiêu chí đánh giá rủi ro người nộp thuế là doanh nghiệp.
Mục II. Bộ tiêu chí đánh giá rủi ro:
1. Nhóm tiêu chí về sự tuân thủ pháp luật thuế: Tần suất nộp chậm tờ khai; Tần suất bị xử phạt vi phạm hành chính về thuế.
2. Nhóm tiêu chí về dấu hiệu rủi ro hóa đơn: Tỷ lệ Hóa đơn hủy/tổng số hóa đơn sử dụng lớn bất thường; Doanh thu tăng đột biến nhưng thuế GTGT phát sinh thấp.
''',
        'relations': [
            {'target': 'LUAT_38_2019', 'type': 'implements', 'weight': 0.8},
            {'target': 'ND_125_2020', 'type': 'related_to', 'weight': 0.7}
        ]
    }
]

def main():
    print(f"Bắt đầu nạp {len(DOCS)} văn bản pháp luật vào Knowledge Graph...")
    ingestor = TaxKnowledgeIngestor()
    for doc in DOCS:
        ingestor.ingest(doc)
    
    print("\n--- Finalizing graph relations ---")
    ingestor.finalize_relations()
    ingestor.close()
    print("Done! Đã xây dựng xong Knowledge Graph đa tầng đẳng cấp.")

if __name__ == "__main__":
    main()
