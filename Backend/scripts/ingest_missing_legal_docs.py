import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ingest_tax_knowledge import TaxKnowledgeIngestor

docs_to_ingest = [
    {
        'key': 'NQ_954_2020_UBTVQH14',
        'title': 'Nghị quyết 954/2020/UBTVQH14 — Điều chỉnh mức giảm trừ gia cảnh thuế TNCN',
        'type': 'resolution',
        'authority_rank': 75,
        'effective_from': '2020-07-01',
        'content': '''Nghị quyết số 954/2020/UBTVQH14 do Ủy ban Thường vụ Quốc hội ban hành ngày 02 tháng 06 năm 2020 về việc điều chỉnh mức giảm trừ gia cảnh của thuế thu nhập cá nhân.
        
Điều 1. Điều chỉnh mức giảm trừ gia cảnh
Điều chỉnh mức giảm trừ gia cảnh quy định tại khoản 1 Điều 19 của Luật Thuế thu nhập cá nhân số 04/2007/QH12 đã được sửa đổi, bổ sung một số điều theo Luật số 26/2012/QH13 như sau:
1. Mức giảm trừ đối với đối tượng nộp thuế là 11 triệu đồng/tháng (132 triệu đồng/năm);
2. Mức giảm trừ đối với mỗi người phụ thuộc là 4,4 triệu đồng/tháng.

Điều 2. Hiệu lực thi hành
1. Nghị quyết này có hiệu lực thi hành từ ngày 01 tháng 07 năm 2020 và áp dụng từ kỳ tính thuế năm 2020.
2. Các trường hợp đã tạm nộp thuế theo mức giảm trừ gia cảnh quy định tại khoản 1 Điều 19 của Luật Thuế thu nhập cá nhân số 04/2007/QH12 đã được sửa đổi, bổ sung một số điều theo Luật số 26/2012/QH13 được xác định lại số thuế thu nhập cá nhân phải nộp theo mức giảm trừ gia cảnh quy định tại Nghị quyết này khi quyết toán thuế thu nhập cá nhân năm 2020.''',
        'relations': [
            {'target': 'luat_thue_tncn_hop_nhat', 'type': 'supplements'}
        ]
    },
    {
        'key': 'LUAT_GTGT_48_2024_QH15',
        'title': 'Luật Thuế giá trị gia tăng số 48/2024/QH15',
        'type': 'law',
        'authority_rank': 90,
        'effective_from': '2025-07-01',
        'content': '''Luật Thuế giá trị gia tăng số 48/2024/QH15 được Quốc hội thông qua ngày 26/11/2024, có hiệu lực từ ngày 01/07/2025.
        
Điều 4. Đối tượng không chịu thuế
1. Doanh thu bán hàng hóa, dịch vụ của hộ kinh doanh, cá nhân kinh doanh có mức doanh thu hàng năm từ 200 triệu đồng trở xuống thì không phải nộp thuế GTGT. (Tăng từ mức 100 triệu đồng trước đây).
2. Hàng hóa nhập khẩu để ủng hộ, tài trợ cho phòng chống thiên tai, thảm họa, dịch bệnh, chiến tranh.
3. Phân bón, máy móc thiết bị chuyên dùng phục vụ sản xuất nông nghiệp, tàu đánh bắt xa bờ (chuyển sang nhóm chịu thuế suất 5%).

Điều 7. Giá tính thuế
Giá tính thuế đối với hàng hóa, dịch vụ dùng để khuyến mại theo pháp luật thương mại được xác định bằng 0. Đối với hàng hóa nhập khẩu, trị giá tính thuế là trị giá hải quan cộng với thuế nhập khẩu, thuế tiêu thụ đặc biệt, thuế bảo vệ môi trường (nếu có).

Điều 9. Thuế suất 0%
Áp dụng đối với hàng hóa, dịch vụ xuất khẩu (cung cấp cho tổ chức, cá nhân ở nước ngoài và tiêu dùng ngoài Việt Nam; cung cấp vào khu phi thuế quan để phục vụ sản xuất xuất khẩu).

Điều 14. Tổ chức quản lý sàn giao dịch thương mại điện tử
Tổ chức quản lý sàn giao dịch thương mại điện tử, nền tảng số có chức năng thanh toán có trách nhiệm khấu trừ, kê khai và nộp thuế thay cho hộ kinh doanh, cá nhân kinh doanh trên sàn theo quy định pháp luật quản lý thuế.''',
        'relations': [
            {'target': 'LUAT_GTGT_13_2008', 'type': 'replaces'}
        ]
    },
    {
        'key': 'ND_64_2024_NDCP',
        'title': 'Nghị định 64/2024/NĐ-CP: Gia hạn nộp thuế GTGT, TNDN, TNCN năm 2024',
        'type': 'decree',
        'authority_rank': 60,
        'effective_from': '2024-06-17',
        'content': '''Nghị định 64/2024/NĐ-CP ban hành ngày 17/06/2024 quy định về việc gia hạn thời hạn nộp thuế giá trị gia tăng (GTGT), thuế thu nhập doanh nghiệp (TNDN), thuế thu nhập cá nhân (TNCN) và tiền thuê đất trong năm 2024.

Điều 2. Đối tượng áp dụng
Áp dụng cho doanh nghiệp, tổ chức, hộ gia đình, cá nhân hoạt động sản xuất trong các ngành nông nghiệp, lâm nghiệp, thủy sản, xây dựng, sản xuất thực phẩm, dệt may, cơ khí, xuất bản, điện ảnh và doanh nghiệp nhỏ/siêu nhỏ.

Điều 3. Gia hạn thời hạn nộp thuế
1. Thuế giá trị gia tăng (GTGT):
Gia hạn thời hạn nộp thuế đối với số thuế GTGT phát sinh phải nộp (bao gồm cả số thuế phân bổ cho các địa phương cấp tỉnh khác nơi người nộp thuế có trụ sở chính, số thuế nộp theo từng lần phát sinh) của kỳ tính thuế từ tháng 5 đến tháng 9 năm 2024 và quý II, quý III năm 2024. Thời gian gia hạn là 05 tháng đối với số thuế GTGT của tháng 5, tháng 6 và quý II/2024.

2. Thuế thu nhập doanh nghiệp (TNDN):
Gia hạn thời hạn nộp thuế đối với số thuế TNDN tạm nộp của quý II kỳ tính thuế năm 2024. Thời gian gia hạn là 03 tháng, kể từ ngày kết thúc thời hạn nộp thuế TNDN theo quy định.

3. Thuế GTGT và thuế TNCN của hộ kinh doanh, cá nhân kinh doanh:
Gia hạn thời hạn nộp thuế đối với số tiền thuế phải nộp phát sinh năm 2024 của hộ kinh doanh, cá nhân kinh doanh. Hộ kinh doanh, cá nhân kinh doanh thực hiện nộp số tiền thuế được gia hạn chậm nhất là ngày 30 tháng 12 năm 2024.''',
        'relations': [
            {'target': 'LUAT_38_2019', 'type': 'implements'}
        ]
    },
    {
        'key': 'ND_102_2021_NDCP',
        'title': 'Nghị định 102/2021/NĐ-CP: Sửa đổi quy định xử phạt hành chính về thuế, hóa đơn',
        'type': 'decree',
        'authority_rank': 60,
        'effective_from': '2021-11-16',
        'content': '''Nghị định 102/2021/NĐ-CP ban hành ngày 16/11/2021 sửa đổi, bổ sung một số điều của các nghị định về xử phạt vi phạm hành chính trong lĩnh vực thuế, hải quan; hải quan; kế toán, kiểm toán độc lập.

Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số 125/2020/NĐ-CP ngày 19/10/2020.
Sửa đổi quy định về thẩm quyền xử phạt của các chức danh: Phạt tiền đến 50.000.000 đồng đối với Chi cục trưởng Chi cục Thuế, Trưởng ban Cục Thuế. 
Sửa đổi quy định về hành vi sử dụng hóa đơn không hợp pháp: Phạt tiền từ 20.000.000 đồng đến 50.000.000 đồng đối với hành vi sử dụng hóa đơn không hợp pháp, sử dụng không hợp pháp hóa đơn để hạch toán giá trị hàng hóa, dịch vụ mua vào làm giảm số tiền thuế phải nộp hoặc làm tăng số tiền thuế được hoàn, số tiền thuế được miễn, giảm nhưng khi cơ quan thuế thanh tra, kiểm tra phát hiện, người mua chứng minh được lỗi vi phạm thuộc về bên bán hàng và người mua đã hạch toán kế toán đầy đủ theo quy định.

Biện pháp khắc phục hậu quả: Buộc nộp đủ số tiền thuế trốn, thiếu; buộc nộp tiền chậm nộp tiền thuế vào ngân sách nhà nước.''',
        'relations': [
            {'target': 'nd_125_2020_ndcp', 'type': 'modifies'}
        ]
    },
    {
        'key': 'TT_111_2013_TT_BTC',
        'title': 'Thông tư 111/2013/TT-BTC — Hướng dẫn thực hiện Luật Thuế thu nhập cá nhân',
        'type': 'circular',
        'authority_rank': 50,
        'effective_from': '2013-10-01',
        'content': '''Thông tư 111/2013/TT-BTC do Bộ Tài chính ban hành ngày 15/08/2013 hướng dẫn thực hiện Luật Thuế thu nhập cá nhân.

Điều 7. Căn cứ tính thuế đối với thu nhập từ tiền lương, tiền công
Căn cứ tính thuế đối với thu nhập từ tiền lương, tiền công là thu nhập tính thuế và thuế suất.
Thu nhập tính thuế được xác định bằng thu nhập chịu thuế trừ các khoản giảm trừ: giảm trừ gia cảnh, bảo hiểm, quỹ hưu trí tự nguyện, đóng góp từ thiện.

Thuế suất thuế thu nhập cá nhân đối với thu nhập từ tiền lương, tiền công được áp dụng theo Biểu thuế lũy tiến từng phần:
Bậc 1: Đến 5 triệu đồng/tháng (Thuế suất 5%)
Bậc 2: Trên 5 triệu đến 10 triệu đồng/tháng (Thuế suất 10%)
Bậc 3: Trên 10 triệu đến 18 triệu đồng/tháng (Thuế suất 15%)
Bậc 4: Trên 18 triệu đến 32 triệu đồng/tháng (Thuế suất 20%)
Bậc 5: Trên 32 triệu đến 52 triệu đồng/tháng (Thuế suất 25%)
Bậc 6: Trên 52 triệu đến 80 triệu đồng/tháng (Thuế suất 30%)
Bậc 7: Trên 80 triệu đồng/tháng (Thuế suất 35%)

Điều 25. Khấu trừ thuế và chứng từ khấu trừ thuế
Khấu trừ thuế là việc tổ chức, cá nhân trả thu nhập thực hiện tính trừ số thuế phải nộp vào thu nhập của người nộp thuế trước khi trả thu nhập. 
Khấu trừ 10%: Tổ chức, cá nhân trả tiền công, tiền thù lao, tiền chi khác cho cá nhân cư trú không ký hợp đồng lao động hoặc ký hợp đồng lao động dưới ba (03) tháng có tổng mức trả thu nhập từ hai triệu (2.000.000) đồng/lần trở lên thì phải khấu trừ thuế theo mức 10% trên thu nhập trước khi trả cho cá nhân.''',
        'relations': [
            {'target': 'luat_thue_tncn_hop_nhat', 'type': 'implements'}
        ]
    },
    {
        'key': 'TT_92_2015_TT_BTC',
        'title': 'Thông tư 92/2015/TT-BTC — Sửa đổi, bổ sung quy định về thuế GTGT và TNCN',
        'type': 'circular',
        'authority_rank': 50,
        'effective_from': '2015-07-30',
        'content': '''Thông tư 92/2015/TT-BTC do Bộ Tài chính ban hành ngày 15/06/2015 hướng dẫn thực hiện thuế giá trị gia tăng và thuế thu nhập cá nhân đối với cá nhân cư trú có hoạt động kinh doanh; hướng dẫn thực hiện một số nội dung sửa đổi, bổ sung về thuế thu nhập cá nhân.

Điều 21. Sửa đổi, bổ sung Điều 16 Thông tư 111/2013/TT-BTC
Cá nhân có thu nhập từ tiền lương, tiền công ủy quyền cho tổ chức, cá nhân trả thu nhập quyết toán thuế thay trong các trường hợp: 
a) Cá nhân chỉ có thu nhập từ tiền lương, tiền công ký hợp đồng lao động từ 03 (ba) tháng trở lên tại một tổ chức, cá nhân trả thu nhập và thực tế đang làm việc tại đó vào thời điểm ủy quyền quyết toán thuế.
b) Cá nhân có thu nhập từ tiền lương, tiền công ký hợp đồng lao động từ 03 tháng trở lên tại một nơi, đồng thời có thu nhập vãng lai ở các nơi khác bình quân tháng trong năm không quá 10 triệu đồng đã được đơn vị trả thu nhập khấu trừ thuế 10% nếu không có yêu cầu quyết toán thuế đối với phần thu nhập này.

Điều 11. Căn cứ tính thuế đối với hoạt động cho thuê tài sản
Đối với cá nhân cho thuê tài sản, doanh thu tính thuế GTGT và doanh thu tính thuế TNCN là doanh thu bao gồm thuế. Nếu tổng doanh thu trong năm dương lịch từ 100 triệu đồng trở xuống thì không phải nộp thuế GTGT và thuế TNCN.''',
        'relations': [
            {'target': 'TT_111_2013_TT_BTC', 'type': 'modifies'}
        ]
    },
    {
        'key': 'TT_130_2016_TT_BTC',
        'title': 'Thông tư 130/2016/TT-BTC — Sửa đổi Thông tư 219 về thuế GTGT và Thông tư 78 về thuế TNDN',
        'type': 'circular',
        'authority_rank': 50,
        'effective_from': '2016-10-01',
        'content': '''Thông tư 130/2016/TT-BTC ngày 12/08/2016 hướng dẫn Nghị định số 100/2016/NĐ-CP quy định chi tiết và hướng dẫn thi hành một số điều của Luật sửa đổi, bổ sung một số điều của Luật Thuế giá trị gia tăng, Luật Thuế tiêu thụ đặc biệt và Luật Quản lý thuế.

Điều 1. Sửa đổi, bổ sung Thông tư số 219/2013/TT-BTC về thuế GTGT.
Cơ sở kinh doanh nộp thuế GTGT theo phương pháp khấu trừ thuế nếu có dự án đầu tư mới đang trong giai đoạn đầu tư có số thuế GTGT của hàng hóa, dịch vụ mua vào sử dụng cho đầu tư mà chưa được khấu trừ và có số thuế còn lại từ 300 triệu đồng trở lên thì được hoàn thuế GTGT.
Đối với dự án đầu tư của cơ sở kinh doanh không góp đủ số vốn điều lệ như đã đăng ký hoặc kinh doanh ngành nghề đầu tư kinh doanh có điều kiện nhưng chưa đủ các điều kiện kinh doanh thì không được hoàn thuế GTGT mà được kết chuyển sang kỳ tiếp theo.

Điều 4. Sửa đổi, bổ sung Thông tư số 78/2014/TT-BTC về thuế TNDN.
Doanh nghiệp được hưởng ưu đãi thuế TNDN đối với phần thu nhập phát sinh từ dự án đầu tư mới. Không áp dụng ưu đãi thuế thu nhập doanh nghiệp đối với thu nhập từ chuyển nhượng vốn, chuyển nhượng quyền góp vốn; thu nhập từ chuyển nhượng bất động sản, dự án đầu tư; thu nhập từ hoạt động tìm kiếm, thăm dò, khai thác dầu khí, tài nguyên quý hiếm khác.''',
        'relations': [
            {'target': 'TT_219_2013', 'type': 'modifies'},
            {'target': 'TT_78_2014', 'type': 'modifies'}
        ]
    },
    {
        'key': 'TT_100_2021_TT_BTC',
        'title': 'Thông tư 100/2021/TT-BTC — Sửa đổi, bổ sung Thông tư 40/2021 về thuế hộ kinh doanh',
        'type': 'circular',
        'authority_rank': 50,
        'effective_from': '2022-01-01',
        'content': '''Thông tư 100/2021/TT-BTC ngày 15/11/2021 sửa đổi, bổ sung một số điều của Thông tư số 40/2021/TT-BTC ngày 01/6/2021 hướng dẫn thuế giá trị gia tăng, thuế thu nhập cá nhân và quản lý thuế đối với hộ kinh doanh, cá nhân kinh doanh.

Điều 1. Sửa đổi, bổ sung Thông tư số 40/2021/TT-BTC.
Tổ chức bao gồm cả chủ sở hữu Sàn giao dịch thương mại điện tử thực hiện việc khai thuế thay, nộp thuế thay cho cá nhân trên cơ sở ủy quyền theo quy định của pháp luật dân sự.
Trường hợp cá nhân kinh doanh thông qua Sàn giao dịch thương mại điện tử không ủy quyền khai thuế thay, nộp thuế thay, thì Sàn giao dịch thương mại điện tử có trách nhiệm cung cấp thông tin liên quan đến hoạt động kinh doanh của cá nhân thông qua sàn cho cơ quan thuế.

Cá nhân chỉ có hoạt động cho thuê tài sản và thời gian cho thuê không trọn năm, nếu doanh thu cho thuê từ 100 triệu đồng/năm trở xuống thì thuộc diện không phải nộp thuế GTGT, không phải nộp thuế TNCN. Trường hợp bên thuê trả tiền thuê tài sản trước cho nhiều năm thì mức doanh thu để xác định cá nhân phải nộp thuế hay không nộp thuế là doanh thu trả tiền một lần được phân bổ theo năm dương lịch.''',
        'relations': [
            {'target': 'tt_40_2021_tt_btc', 'type': 'modifies'}
        ]
    },
    {
        'key': 'CV_2155_TCT_CS_2024',
        'title': 'Công văn 2155/TCT-CS năm 2024 — Thuế GTGT dịch vụ khu chế xuất',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-05-21',
        'content': '''Công văn số 2155/TCT-CS ngày 21/05/2024 của Tổng cục Thuế hướng dẫn về chính sách thuế GTGT đối với dịch vụ cung cấp cho doanh nghiệp chế xuất.

Nội dung chính:
Theo quy định tại Điều 9 Thông tư 219/2013/TT-BTC, thuế suất 0% áp dụng đối với dịch vụ xuất khẩu, dịch vụ cung cấp trực tiếp cho tổ chức, cá nhân ở khu phi thuế quan và tiêu dùng trong khu phi thuế quan.
Trường hợp cơ sở kinh doanh cung cấp dịch vụ (ví dụ: dịch vụ sửa chữa máy móc, thiết bị) cho doanh nghiệp chế xuất, nhưng dịch vụ đó được thực hiện và tiêu dùng bên ngoài khu phi thuế quan (tại cơ sở kinh doanh cung cấp dịch vụ ở nội địa) thì không đáp ứng điều kiện tiêu dùng trong khu phi thuế quan.
Do đó, dịch vụ này không thuộc đối tượng áp dụng mức thuế suất thuế GTGT 0% mà phải áp dụng mức thuế suất tương ứng như dịch vụ cung cấp và tiêu dùng tại nội địa (thường là 10% hoặc 8% theo quy định giảm thuế). Các doanh nghiệp cần căn cứ vào hợp đồng, địa điểm thực hiện dịch vụ thực tế để lập hóa đơn GTGT chính xác.''',
        'relations': [
            {'target': 'TT_219_2013', 'type': 'implements'}
        ]
    },
    {
        'key': 'CV_2204_TCT_CS_2024',
        'title': 'Công văn 2204/TCT-CS năm 2024 — Thuế GTGT xuất khẩu phần mềm',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-05-24',
        'content': '''Công văn số 2204/TCT-CS ngày 24/05/2024 của Tổng cục Thuế hướng dẫn về chính sách thuế GTGT đối với hoạt động xuất khẩu phần mềm và chuyển nhượng quyền sở hữu trí tuệ ra nước ngoài.

Nội dung chính:
1. Đối với sản phẩm, dịch vụ phần mềm xuất khẩu:
Căn cứ Điều 9 Thông tư 219/2013/TT-BTC, nếu doanh nghiệp kinh doanh xuất khẩu phần mềm, dịch vụ phần mềm ra nước ngoài (cung cấp cho tổ chức, cá nhân ở nước ngoài và tiêu dùng ngoài Việt Nam), có hợp đồng xuất khẩu và chứng từ thanh toán qua ngân hàng theo đúng quy định, thì thuộc đối tượng áp dụng thuế suất thuế GTGT 0%. Doanh nghiệp được kê khai khấu trừ, hoàn thuế GTGT đầu vào nếu đáp ứng đủ điều kiện.

2. Đối với hoạt động chuyển nhượng quyền sở hữu trí tuệ:
Căn cứ Khoản 2 Điều 1 Thông tư 130/2016/TT-BTC, đối với hoạt động chuyển nhượng quyền sở hữu trí tuệ, bản quyền ra nước ngoài, nếu được cơ quan nhà nước có thẩm quyền xác định đúng bản chất là hoạt động chuyển nhượng quyền sở hữu trí tuệ thì không thuộc đối tượng áp dụng thuế suất thuế GTGT 0%. Hoạt động này thuộc đối tượng không chịu thuế GTGT.''',
        'relations': [
            {'target': 'TT_219_2013', 'type': 'implements'},
            {'target': 'TT_130_2016_TT_BTC', 'type': 'implements'}
        ]
    },
    {
        'key': 'CV_3543_TCT_CS_2024',
        'title': 'Công văn 3543/TCT-CS năm 2024 — Xử lý HĐĐT có sai sót',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-08-13',
        'content': '''Công văn số 3543/TCT-CS do Tổng cục Thuế ban hành ngày 13/08/2024 hướng dẫn xử lý đối với trường hợp doanh nghiệp lập hóa đơn điện tử có sai sót.

Nội dung chính:
Trường hợp doanh nghiệp lập hóa đơn điện tử để điều chỉnh hoặc thay thế cho nhiều hóa đơn điện tử đã lập có sai sót của cùng một người mua hàng:
Căn cứ điểm b khoản 2 Điều 19 Nghị định số 123/2020/NĐ-CP và khoản 1 Điều 7 Thông tư số 78/2021/TT-BTC, pháp luật hiện hành quy định nguyên tắc xử lý cho từng hóa đơn điện tử có sai sót (một hóa đơn điều chỉnh/thay thế cho một hóa đơn sai sót).
Tuy nhiên, để tạo điều kiện thuận lợi cho người nộp thuế, trong trường hợp xuất hiện nhiều hóa đơn đã lập có sai sót về cùng một nội dung (ví dụ sai đơn giá, sai tên hàng hóa) đối với cùng một khách hàng, doanh nghiệp có thể lập hóa đơn điện tử điều chỉnh hoặc thay thế, xử lý theo quy định tại khoản 2 Điều 19 Nghị định 123/2020/NĐ-CP.
Trên hóa đơn điều chỉnh/thay thế phải ghi rõ nội dung điều chỉnh hoặc thay thế cho các hóa đơn số..., ký hiệu..., ngày... tháng... năm... để phục vụ công tác đối chiếu, kiểm tra của cơ quan thuế. Đối với đề xuất cho phép 1 hóa đơn điều chỉnh cho nhiều hóa đơn khác nhau, Tổng cục Thuế ghi nhận để nghiên cứu trình cấp có thẩm quyền sửa đổi Nghị định 123.''',
        'relations': [
            {'target': 'nd_123_2020_ndcp', 'type': 'implements'},
            {'target': 'VB_78_2021_TT_BTC', 'type': 'implements'}
        ]
    },
    {
        'key': 'CV_573_TCT_CS_2024',
        'title': 'Công văn 573/TCT-CS năm 2024 — HĐĐT cho hàng hóa xuất khẩu',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-02-20',
        'content': '''Công văn 573/TCT-CS ngày 20/02/2024 của Tổng cục Thuế hướng dẫn về việc lập hóa đơn điện tử đối với hàng hóa xuất khẩu.

Nội dung chính:
Căn cứ khoản 1 Điều 8 và khoản 3 Điều 13 Nghị định số 123/2020/NĐ-CP:
1. Cơ sở kinh doanh có hàng hóa xuất khẩu thực hiện lập hóa đơn giá trị gia tăng (GTGT) điện tử đối với trường hợp kê khai nộp thuế theo phương pháp khấu trừ hoặc lập hóa đơn bán hàng điện tử đối với trường hợp kê khai theo phương pháp trực tiếp.
2. Thời điểm lập hóa đơn điện tử đối với hàng hóa xuất khẩu là sau khi người khai hải quan hoàn thành thủ tục hải quan xuất khẩu. Thời điểm hoàn thành thủ tục hải quan xác định theo quy định pháp luật về hải quan.
3. Về việc khấu trừ, hoàn thuế GTGT đầu vào cho hàng hóa xuất khẩu: Cơ quan thuế địa phương có trách nhiệm căn cứ hồ sơ hải quan, chứng từ thanh toán qua ngân hàng, hóa đơn điện tử xuất khẩu và các quy định tại Điều 16 Thông tư 219/2013/TT-BTC để kiểm tra, giải quyết hoàn thuế theo quy định của pháp luật. Người nộp thuế phải hoàn toàn chịu trách nhiệm trước pháp luật về tính hợp pháp, chính xác của hồ sơ xuất khẩu.''',
        'relations': [
            {'target': 'nd_123_2020_ndcp', 'type': 'implements'}
        ]
    },
    {
        'key': 'CV_4019_TCT_CS_2024',
        'title': 'Công văn 4019/TCT-CS năm 2024 — Thuế GTGT hóa đơn hợp đồng xây dựng',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-09-11',
        'content': '''Công văn số 4019/TCT-CS ngày 11/09/2024 của Tổng cục Thuế hướng dẫn về thuế GTGT đối với hóa đơn trong hợp đồng xây dựng, lắp đặt.

Nội dung chính:
Việc xác định giá tính thuế trên hóa đơn GTGT đối với hoạt động xây dựng, lắp đặt căn cứ vào hợp đồng kinh tế đã ký kết giữa chủ đầu tư và nhà thầu.
Trường hợp giá trúng thầu, giá hợp đồng xây dựng là giá chưa bao gồm thuế GTGT thì khi lập hóa đơn, nhà thầu xây dựng xác định giá tính thuế là giá trị công việc hoàn thành bàn giao chưa có thuế GTGT (theo Bảng khối lượng nghiệm thu thực tế, có điều chỉnh theo quy định của hợp đồng), sau đó cộng thêm thuế suất thuế GTGT theo quy định hiện hành (thông thường là 10% hoặc 8% theo chính sách giảm thuế của Chính phủ).
Chủ đầu tư và nhà thầu phải chịu trách nhiệm về số liệu thanh toán, quyết toán công trình. Cơ quan thuế khi thanh tra, kiểm tra sẽ đối chiếu khối lượng nghiệm thu thực tế với giá trị ghi trên hóa đơn GTGT để xử lý nghĩa vụ thuế. Trường hợp phát hiện lập hóa đơn khống khối lượng hoặc ghi sai giá trị công trình để trốn thuế, sẽ bị xử lý vi phạm hành chính hoặc hình sự theo Luật Quản lý thuế.''',
        'relations': [
            {'target': 'TT_219_2013', 'type': 'implements'}
        ]
    }
]

if __name__ == "__main__":
    ingestor = TaxKnowledgeIngestor()
    try:
        for doc in docs_to_ingest:
            ingestor.ingest(doc)
        ingestor.finalize_relations()
        print(f"Đã ingest {len(docs_to_ingest)} văn bản thành công!")
    finally:
        ingestor.close()
