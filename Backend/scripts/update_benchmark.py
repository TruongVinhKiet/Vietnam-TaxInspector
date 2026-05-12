import os
import re

citizen_file = "e:/TaxInspector/Backend/ml_engine/tax_agent_citizen_legal.py"
with open(citizen_file, "r", encoding="utf-8") as f:
    content = f.read()

new_snippets = """    CitizenLegalSnippet(
        key="vat_import",
        title="Thuế GTGT hàng nhập khẩu",
        legal_reference="Luật Thuế GTGT; Thông tư 219/2013/TT-BTC",
        keywords=("gtgt nhap khau", "hang nhap khau", "hai quan", "nhap khau", "khau tru gtgt nhap khau"),
        text="Thuế GTGT hàng nhập khẩu do cơ quan Hải quan thu khi làm thủ tục nhập khẩu. Doanh nghiệp có thể kê khai khấu trừ thuế GTGT hàng nhập khẩu nếu có chứng từ nộp tiền thuế GTGT khâu nhập khẩu hợp pháp.",
        next_steps=("Lưu giữ chứng từ nộp thuế GTGT khâu nhập khẩu.", "Kê khai khấu trừ vào tờ khai mẫu 01/GTGT kỳ tương ứng.")
    ),
    CitizenLegalSnippet(
        key="foreign_contractor_tax",
        title="Thuế nhà thầu nước ngoài",
        legal_reference="Thông tư 103/2014/TT-BTC",
        keywords=("nha thau nuoc ngoai", "fct", "nop thay", "nha thau", "chuyen tien nuoc ngoai"),
        text="Khi doanh nghiệp Việt Nam ký hợp đồng mua dịch vụ từ tổ chức/cá nhân nước ngoài không có hiện diện thương mại tại VN, bên VN phải khấu trừ, nộp thay thuế nhà thầu (gồm GTGT và TNDN/TNCN) trước khi thanh toán.",
        next_steps=("Xác định tỷ lệ % thuế GTGT và TNDN áp dụng cho ngành nghề.", "Kê khai và nộp thay thuế nhà thầu tại cơ quan thuế quản lý.", "Cấp chứng từ khấu trừ thuế cho nhà thầu nước ngoài nếu có yêu cầu.")
    ),
    CitizenLegalSnippet(
        key="corporate_tax_incentives",
        title="Ưu đãi thuế TNDN cho doanh nghiệp mới thành lập",
        legal_reference="Luật Thuế TNDN; Thông tư 78/2014/TT-BTC; Thông tư 96/2015/TT-BTC",
        keywords=("uu dai thue", "tndn", "doanh nghiep moi", "mien giam thue", "khuyen khich dau tu"),
        text="Doanh nghiệp thành lập mới từ dự án đầu tư tại địa bàn kinh tế xã hội khó khăn, đặc biệt khó khăn, hoặc lĩnh vực công nghệ cao, giáo dục, y tế có thể được hưởng thuế suất ưu đãi, miễn thuế hoặc giảm thuế TNDN có thời hạn.",
        next_steps=("Xác định địa bàn hoặc lĩnh vực đầu tư có thuộc diện ưu đãi không.", "Hạch toán riêng thu nhập từ hoạt động được ưu đãi và không được ưu đãi.", "Tự xác định mức ưu đãi và kê khai vào phụ lục ưu đãi thuế TNDN khi quyết toán.")
    ),
    CitizenLegalSnippet(
        key="real_estate_transfer_individual",
        title="Thuế chuyển nhượng bất động sản cá nhân",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("chuyen nhuong bat dong san", "ban nha", "ban dat", "tncn chuyen nhuong", "thue dat"),
        text="Cá nhân chuyển nhượng bất động sản phải nộp thuế TNCN với thuế suất 2% trên giá chuyển nhượng. Giá chuyển nhượng do các bên thỏa thuận nhưng không được thấp hơn giá đất do UBND cấp tỉnh quy định.",
        next_steps=("Chuẩn bị hồ sơ khai thuế TNCN và nộp cùng hồ sơ sang tên.", "Nếu giá hợp đồng thấp hơn giá nhà nước quy định, tính thuế theo giá nhà nước.", "Trường hợp bán nhà ở duy nhất, chuẩn bị cam kết để được miễn thuế TNCN.")
    ),
    CitizenLegalSnippet(
        key="stock_investment_tax",
        title="Thuế thu nhập từ đầu tư chứng khoán",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("chung khoan", "co phieu", "co tuc", "ban co phieu", "dau tu", "tncn chung khoan"),
        text="Cá nhân chuyển nhượng chứng khoán nộp thuế TNCN 0.1% trên giá bán từng lần chuyển nhượng. Nhận cổ tức bằng tiền mặt hoặc cổ phiếu đều phải nộp thuế TNCN 5% từ đầu tư vốn.",
        next_steps=("Công ty chứng khoán thường tự động khấu trừ 0.1% khi bán.", "Khi nhận cổ tức tiền mặt, công ty cũng tự động khấu trừ 5%.", "Lưu lại chứng từ khấu trừ thuế nếu cần quyết toán tổng thu nhập.")
    ),
    CitizenLegalSnippet(
        key="prize_winning_tax",
        title="Thuế thu nhập từ trúng thưởng",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("trung thuong", "xo so", "khuyen mai", "giai thuong", "tncn trung thuong"),
        text="Cá nhân trúng thưởng xổ số, khuyến mại, casino... nếu phần giá trị trúng thưởng vượt trên 10 triệu đồng cho mỗi lần trúng thưởng thì phải nộp thuế TNCN với thuế suất 10% trên phần vượt 10 triệu.",
        next_steps=("Tổ chức trả thưởng có trách nhiệm khấu trừ 10% trước khi trả thưởng.", "Cá nhân nhận thưởng yêu cầu tổ chức trả thưởng cấp chứng từ khấu trừ thuế TNCN.", "Thu nhập này được tính thuế từng lần phát sinh, không gộp vào quyết toán cuối năm.")
    ),
    CitizenLegalSnippet(
        key="freelancer_tax",
        title="Thuế TNCN cho freelancer/tự do",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("freelancer", "tu do", "cong tac vien", "10%", "khau tru 10", "chua ky hop dong"),
        text="Cá nhân làm nghề tự do, cộng tác viên (không ký HĐLĐ hoặc ký HĐLĐ dưới 3 tháng) có thu nhập từ 2 triệu đồng/lần trở lên sẽ bị tổ chức trả thu nhập khấu trừ 10% thuế TNCN trước khi trả.",
        next_steps=("Nếu ước tính tổng thu nhập trong năm chưa đến mức phải nộp thuế, có thể làm cam kết 08/CK-TNCN để tạm thời không bị khấu trừ 10%.", "Yêu cầu tổ chức trả thu nhập xuất chứng từ khấu trừ thuế.", "Cuối năm nếu có nhiều nguồn thu nhập, phải tự đi quyết toán thuế TNCN.")
    ),
    CitizenLegalSnippet(
        key="etax_electronic_declaration",
        title="Kê khai thuế điện tử eTax",
        legal_reference="Luật Quản lý thuế 2019; Thông tư 19/2021/TT-BTC",
        keywords=("etax", "ke khai dien tu", "nop thue dien tu", "chu ky so", "cong thong tin"),
        text="Hệ thống Thuế điện tử (eTax) hỗ trợ người nộp thuế thực hiện các thủ tục đăng ký, kê khai, nộp thuế và hoàn thuế hoàn toàn trực tuyến. Cá nhân sử dụng tài khoản eTax Mobile hoặc mã số thuế, doanh nghiệp sử dụng chữ ký số để đăng nhập.",
        next_steps=("Đăng ký tài khoản giao dịch thuế điện tử với cơ quan thuế.", "Đối với doanh nghiệp, cần mua chữ ký số (Token) và đăng ký tài khoản ngân hàng để nộp thuế điện tử.", "Thực hiện kê khai và nộp thuế đúng hạn trên cổng eTax.")
    ),
    CitizenLegalSnippet(
        key="personal_tax_code_registration",
        title="Đăng ký mã số thuế cá nhân",
        legal_reference="Luật Quản lý thuế 2019; Thông tư 105/2020/TT-BTC",
        keywords=("dang ky mst", "ma so thue", "mst ca nhan", "chua co mst", "cccd", "dang ky thue"),
        text="Cá nhân có phát sinh thu nhập chịu thuế hoặc có nghĩa vụ nộp ngân sách phải đăng ký mã số thuế. Thường công ty chi trả thu nhập sẽ đăng ký thay cho người lao động, hoặc cá nhân tự đăng ký trực tuyến qua eTax/Cổng DVC Quốc gia.",
        next_steps=("Kiểm tra xem mình đã có MST chưa bằng CCCD/CMND trên eTax.", "Nếu chưa có, cung cấp bản sao CCCD cho công ty để họ đăng ký thay.", "Hoặc tự đăng ký trên trang Thuế điện tử và nộp hồ sơ tại cơ quan thuế nếu là cá nhân tự do.")
    ),
    CitizenLegalSnippet(
        key="foreign_income_tax",
        title="Thuế với thu nhập từ nước ngoài",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("thu nhap nuoc ngoai", "youtube", "google adsense", "nhan tien nuoc ngoai", "tncn nuoc ngoai"),
        text="Cá nhân cư trú tại Việt Nam phải nộp thuế TNCN đối với thu nhập phát sinh trong và ngoài lãnh thổ Việt Nam. Ví dụ thu nhập từ Youtube, Google, Apple, Upwork... Nếu đã nộp thuế ở nước ngoài, có thể được trừ số thuế đã nộp tùy Hiệp định tránh đánh thuế hai lần.",
        next_steps=("Theo dõi sát sao các khoản tiền nhận từ nước ngoài.", "Đăng ký mã số thuế, kê khai nộp thuế tự nguyện (thường là 7% với dịch vụ sản xuất nội dung số).", "Thu thập chứng từ nộp thuế ở nước ngoài nếu muốn khấu trừ.")
    ),
    CitizenLegalSnippet(
        key="einvoice_correction",
        title="Hoá đơn điện tử bị sai phải điều chỉnh",
        legal_reference="Nghị định 123/2020/NĐ-CP; Thông tư 78/2021/TT-BTC",
        keywords=("sai hoa don", "dieu chinh hoa don", "thay the", "huy hoa don", "hoa don dien tu sai"),
        text="Khi hóa đơn điện tử đã lập có sai sót, tùy từng trường hợp mà có cách xử lý khác nhau: sai tên/địa chỉ (chưa giao khách/đã giao khách), sai mã số thuế, sai số lượng/thành tiền. Có thể phải lập biên bản, hủy và lập hóa đơn mới, hoặc lập hóa đơn điều chỉnh.",
        next_steps=("Kiểm tra kỹ xem sai sót là ở chỉ tiêu nào (trọng yếu hay không trọng yếu).", "Thỏa thuận với bên mua lập biên bản ghi nhận sai sót nếu cần.", "Lập hóa đơn thay thế hoặc điều chỉnh tùy theo tình huống và gửi lên CQT cấp mã.")
    ),
    CitizenLegalSnippet(
        key="vat_export_zero_percent",
        title="Thuế GTGT với hàng xuất khẩu (thuế suất 0%)",
        legal_reference="Luật Thuế GTGT; Thông tư 219/2013/TT-BTC",
        keywords=("xuat khau", "thue 0%", "gtgt 0%", "hoan thue xuat khau", "hai quan xuat khau"),
        text="Hàng hóa, dịch vụ xuất khẩu ra nước ngoài hoặc xuất vào khu phi thuế quan được áp dụng thuế suất GTGT 0% nếu đáp ứng đủ điều kiện: có hợp đồng xuất khẩu, có chứng từ thanh toán qua ngân hàng, có tờ khai hải quan (đối với hàng hóa).",
        next_steps=("Đảm bảo tờ khai hải quan đã thông quan và có đóng dấu xác nhận.", "Lưu giữ chứng từ thanh toán từ nước ngoài qua tài khoản ngân hàng của công ty.", "Tập hợp hồ sơ để kê khai thuế 0% và đề nghị hoàn thuế GTGT đầu vào nếu đủ điều kiện.")
    ),
    CitizenLegalSnippet(
        key="fixed_asset_depreciation",
        title="Khấu hao tài sản cố định",
        legal_reference="Thông tư 45/2013/TT-BTC",
        keywords=("khau hao", "tai san co dinh", "tscd", "trich khau hao", "chi phi khau hao"),
        text="Tài sản cố định (TSCĐ) phục vụ sản xuất kinh doanh phải được trích khấu hao để tính vào chi phí được trừ khi xác định thuế TNDN. Phải đăng ký phương pháp trích khấu hao với cơ quan thuế trước khi thực hiện.",
        next_steps=("Xác định thời gian trích khấu hao theo khung quy định tại Thông tư 45.", "Đăng ký phương pháp khấu hao (đường thẳng, số dư giảm dần...) với CQT.", "Lưu giữ hóa đơn, chứng từ hợp pháp chứng minh quyền sở hữu TSCĐ.")
    ),
    CitizenLegalSnippet(
        key="interest_expense_deduction",
        title="Chi phí lãi vay được trừ khi tính thuế TNDN",
        legal_reference="Luật Thuế TNDN; Nghị định 132/2020/NĐ-CP",
        keywords=("lai vay", "vay von", "chi phi lai", "tndn", "giao dich lien ket", "vay ca nhan"),
        text="Chi phí lãi vay phục vụ sản xuất kinh doanh được tính vào chi phí được trừ nếu vốn điều lệ đã góp đủ. Đối với doanh nghiệp có giao dịch liên kết, tổng chi phí lãi vay thuần được trừ không vượt quá 30% tổng lợi nhuận thuần từ hoạt động kinh doanh cộng chi phí lãi vay, chi phí khấu hao (EBITDA).",
        next_steps=("Kiểm tra xem doanh nghiệp đã góp đủ vốn điều lệ chưa.", "Tính toán chỉ số EBITDA để xác định mức trần chi phí lãi vay được trừ (nếu có GDLK).", "Đối với phần lãi vay vượt mức 30%, ghi nhận chuyển sang kỳ sau (tối đa 5 năm).")
    ),
    CitizenLegalSnippet(
        key="capital_transfer_tax",
        title="Thuế chuyển nhượng vốn/cổ phần",
        legal_reference="Luật Thuế TNDN; Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("chuyen nhuong von", "ban co phan", "tncn chuyen nhuong von", "tndn chuyen nhuong von"),
        text="Khi cá nhân chuyển nhượng phần vốn góp (công ty TNHH), nộp thuế TNCN 20% trên thu nhập (chênh lệch giá bán và giá mua). Khi chuyển nhượng cổ phần (công ty cổ phần), nộp 0.1% trên giá bán. Doanh nghiệp chuyển nhượng vốn nộp thuế TNDN 20% trên thu nhập.",
        next_steps=("Lập hợp đồng chuyển nhượng vốn và thực hiện thanh toán qua ngân hàng.", "Kê khai và nộp thuế chuyển nhượng vốn trong vòng 10 ngày kể từ ngày hợp đồng có hiệu lực.", "Thay đổi giấy chứng nhận đăng ký doanh nghiệp tại Sở Kế hoạch và Đầu tư.")
    ),
    CitizenLegalSnippet(
        key="family_deduction_calculation",
        title="Giảm trừ gia cảnh - cách tính",
        legal_reference="Luật Thuế TNCN; Nghị quyết 954/2020/UBTVQH14",
        keywords=("giam tru gia canh", "ban than", "nguoi phu thuoc", "11 trieu", "4.4 trieu", "tncn"),
        text="Mức giảm trừ gia cảnh hiện tại là 11 triệu đồng/tháng cho bản thân người nộp thuế (132 triệu đồng/năm) và 4,4 triệu đồng/tháng cho mỗi người phụ thuộc. Phải đăng ký mã số thuế cho người phụ thuộc để được tính giảm trừ.",
        next_steps=("Cung cấp hồ sơ chứng minh người phụ thuộc (giấy khai sinh, CMND, xác nhận thu nhập...) cho công ty.", "Công ty đăng ký MST người phụ thuộc qua cổng eTax.", "Khi tính thuế hàng tháng/quyết toán, trừ đi các khoản giảm trừ này trước khi nhân thuế suất.")
    ),
    CitizenLegalSnippet(
        key="year_end_bonus_tax",
        title="Thuế với tiền thưởng Tết/lương tháng 13",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("thuong tet", "luong thang 13", "tien thuong", "tncn thuong", "dong thue thuong"),
        text="Tiền thưởng Tết, lương tháng 13 là khoản thu nhập chịu thuế TNCN từ tiền lương, tiền công. Khoản thưởng này sẽ được cộng gộp vào tổng thu nhập của tháng chi trả để tính thuế TNCN theo biểu thuế lũy tiến từng phần.",
        next_steps=("Nhận bảng lương chi tiết từ công ty để xem khoản thưởng và số thuế bị khấu trừ.", "Nếu số thuế khấu trừ trong tháng nhận thưởng quá cao, cuối năm có thể làm quyết toán để nhận hoàn thuế.", "Thưởng bằng hiện vật (trừ một số khoản thưởng đặc biệt) cũng phải quy đổi ra tiền để tính thuế.")
    ),
    CitizenLegalSnippet(
        key="investment_project_vat_refund",
        title="Hoàn thuế GTGT cho dự án đầu tư",
        legal_reference="Luật Thuế GTGT; Thông tư 219/2013/TT-BTC; Thông tư 130/2016/TT-BTC",
        keywords=("hoan thue gtgt", "du an dau tu", "hoan dau tu", "hoan thue", "lap du an"),
        text="Doanh nghiệp đang hoạt động có dự án đầu tư mới hoặc cơ sở kinh doanh mới thành lập từ dự án đầu tư có số thuế GTGT đầu vào của hàng hóa, dịch vụ sử dụng cho đầu tư chưa được khấu trừ hết từ 300 triệu đồng trở lên có thể được hoàn thuế GTGT.",
        next_steps=("Đảm bảo dự án đầu tư có đầy đủ giấy tờ phê duyệt, giấy phép xây dựng...", "Tách riêng số thuế GTGT đầu vào của dự án đầu tư để kê khai trên tờ khai mẫu 02/GTGT.", "Lập hồ sơ đề nghị hoàn thuế gửi cơ quan thuế khi đủ điều kiện.")
    ),
    CitizenLegalSnippet(
        key="vehicle_rental_tax",
        title="Thuế đối với hoạt động cho thuê xe/phương tiện",
        legal_reference="Thông tư 40/2021/TT-BTC",
        keywords=("cho thue xe", "thue oto", "ca nhan cho thue", "thue kinh doanh xe", "nop thue cho thue xe"),
        text="Cá nhân cho thuê xe (ô tô, xe máy, phương tiện vận tải) phải kê khai nộp thuế GTGT (5%) và thuế TNCN (2%) nếu tổng doanh thu từ hoạt động kinh doanh (bao gồm cho thuê tài sản) trong năm dương lịch trên 100 triệu đồng.",
        next_steps=("Xác định tổng doanh thu cả năm, nếu >100tr/năm thì phải nộp thuế.", "Kê khai thuế theo từng lần phát sinh hoặc theo kỳ (tháng/quý) tại cơ quan thuế nơi cư trú.", "Có thể yêu cầu cơ quan thuế cấp hóa đơn lẻ để giao cho bên thuê (thường là công ty cần hóa đơn).")
    ),
    CitizenLegalSnippet(
        key="tax_audit_reassessment",
        title="Xử lý khi bị truy thu thuế sau thanh tra",
        legal_reference="Luật Quản lý thuế 2019",
        keywords=("truy thu thue", "thanh tra thue", "kiem tra thue", "phat thue", "bien ban thanh tra"),
        text="Sau khi thanh tra/kiểm tra thuế, nếu phát hiện khai thiếu thuế hoặc trốn thuế, cơ quan thuế sẽ ra Quyết định xử phạt vi phạm hành chính, bao gồm: truy thu số thuế thiếu, phạt tiền (10%-20% số thiếu hoặc 1-3 lần số trốn) và tính tiền chậm nộp.",
        next_steps=("Đọc kỹ Biên bản thanh tra, nếu có điểm chưa đồng ý có thể giải trình bảo vệ số liệu trước khi ra Quyết định.", "Nộp đủ số tiền thuế truy thu, tiền phạt và tiền chậm nộp theo thời hạn ghi trên Quyết định.", "Nếu vẫn không đồng ý với Quyết định xử phạt, có quyền khiếu nại hoặc khởi kiện ra tòa án hành chính.")
    ),
    CitizenLegalSnippet(
        key="seasonal_contract_tax",
        title="Thuế với thu nhập từ hợp đồng thời vụ",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC",
        keywords=("thoi vu", "hop dong thoi vu", "thu viec", "tncn thoi vu", "khau tru 10%"),
        text="Người lao động ký hợp đồng thời vụ, hợp đồng giao khoán dưới 3 tháng hoặc hợp đồng thử việc có thu nhập từ 2 triệu đồng/lần trở lên bị khấu trừ 10% thuế TNCN trước khi nhận lương.",
        next_steps=("Tương tự freelancer, nếu tổng thu nhập trong năm chưa đến mức đóng thuế, có thể nộp bản cam kết 08/CK-TNCN.", "Cung cấp MST cá nhân cho công ty để làm cam kết 08.", "Lưu ý nếu có nhiều nguồn thu nhập cùng lúc thì không được làm cam kết 08.")
    ),
)"""

new_content = content.replace(")\n\n\ndef retrieve_citizen_legal_snippets(", ")\n" + new_snippets.replace('    CitizenLegalSnippet', '    CitizenLegalSnippet', 1) + "\n\n\ndef retrieve_citizen_legal_snippets(")
# To avoid missing the replace, let's just use regex
new_content = re.sub(r'    \),\n\)\n\n\ndef retrieve_citizen_legal_snippets\(', '    ),\n' + new_snippets + '\n\n\ndef retrieve_citizen_legal_snippets(', content)

with open(citizen_file, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Updated tax_agent_citizen_legal.py")

eval_file = "e:/TaxInspector/Backend/scripts/run_experimental_evaluation.py"
with open(eval_file, "r", encoding="utf-8") as f:
    eval_content = f.read()

# We need to change:
#     templates = [
#         "Em hỏi về {topic}, cần làm gì cho đúng?",
#         "Cho tôi hỏi {topic} thì có phải nộp thuế hoặc bị phạt không?",
#         "Trường hợp {topic}, hướng dẫn từng bước giúp tôi.",
#         "{topic} theo quy định thuế xử lý ra sao?",
#     ]
# to the 10 templates.
new_templates = """    templates = [
        "Em hỏi về {topic}, cần làm gì cho đúng?",
        "Cho tôi hỏi {topic} thì có phải nộp thuế hoặc bị phạt không?",
        "Trường hợp {topic}, hướng dẫn từng bước giúp tôi.",
        "{topic} theo quy định thuế xử lý ra sao?",
        "Tôi muốn biết về {topic}, có quy định nào hướng dẫn không?",
        "Xin hỏi {topic} thì tôi phải nộp những gì?",
        "Doanh nghiệp tôi gặp vấn đề {topic}, xử lý thế nào?",
        "Hướng dẫn {topic} cho người mới bắt đầu.",
        "{topic} - mức phạt và cách xử lý?",
        "Ai chịu trách nhiệm khi {topic}?",
    ]"""
eval_content = re.sub(r'    templates = \[\n(?:        ".*?",\n)*    \]', new_templates, eval_content)

with open(eval_file, "w", encoding="utf-8") as f:
    f.write(eval_content)
print("Updated run_experimental_evaluation.py")
