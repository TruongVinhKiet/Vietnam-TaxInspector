"""Citizen-facing Vietnamese tax guidance fallback for the Tax Agent.

This module is deliberately deterministic and local. It does not replace the
GraphRAG legal knowledge base; it supplies narrowly-scoped, practical snippets
for common everyday questions when the KB has too few matches.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


def normalize_text(value: str) -> str:
    raw = unicodedata.normalize("NFD", value or "")
    raw = "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")
    raw = raw.replace("đ", "d").replace("Đ", "D")
    raw = re.sub(r"[^a-zA-Z0-9]+", " ", raw.lower())
    return re.sub(r"\s+", " ", raw).strip()


@dataclass(frozen=True)
class CitizenLegalSnippet:
    key: str
    title: str
    legal_reference: str
    keywords: tuple[str, ...]
    text: str
    next_steps: tuple[str, ...]


SNIPPETS: tuple[CitizenLegalSnippet, ...] = (
    CitizenLegalSnippet(
        key="tncn_salary_dependent_refund",
        title="Thuế TNCN tiền lương, người phụ thuộc và hoàn thuế",
        legal_reference="Luật Thuế TNCN; Thông tư 111/2013/TT-BTC; Thông tư 80/2021/TT-BTC",
        keywords=("tncn", "luong", "khau tru", "nguoi phu thuoc", "con nho", "hoan thue", "dong thua"),
        text=(
            "Nếu tiền lương đã bị khấu trừ thuế TNCN cao hơn số phải nộp sau khi tính giảm trừ bản thân, "
            "giảm trừ người phụ thuộc và các khoản bảo hiểm bắt buộc, người nộp thuế có thể quyết toán để bù trừ "
            "hoặc đề nghị hoàn phần nộp thừa. Trường hợp công ty khấu trừ chưa đúng, cần yêu cầu bảng lương, "
            "chứng từ khấu trừ thuế và hồ sơ đăng ký người phụ thuộc để đối chiếu."
        ),
        next_steps=(
            "Xin bảng lương, chứng từ khấu trừ thuế TNCN và thông tin đăng ký người phụ thuộc từ công ty.",
            "Tính lại thu nhập chịu thuế theo từng tháng/kỳ quyết toán, trừ giảm trừ gia cảnh và bảo hiểm bắt buộc.",
            "Nếu có nộp thừa, thực hiện quyết toán thuế TNCN và chọn bù trừ hoặc hoàn thuế theo hướng dẫn của cơ quan thuế.",
        ),
    ),
    CitizenLegalSnippet(
        key="ecommerce_individual_tax",
        title="Cá nhân bán hàng qua sàn thương mại điện tử",
        legal_reference="Luật Quản lý thuế 2019; Thông tư 40/2021/TT-BTC; Thông tư 100/2021/TT-BTC",
        keywords=("shopee", "tiktok", "ban hang online", "thuong mai dien tu", "tai khoan ca nhan", "doanh thu"),
        text=(
            "Cá nhân kinh doanh qua sàn thương mại điện tử cần theo dõi doanh thu thực nhận và nghĩa vụ kê khai/nộp thuế. "
            "Chính sách với hộ/cá nhân kinh doanh thường xét ngưỡng doanh thu năm và loại hoạt động để xác định VAT, "
            "thuế TNCN, lệ phí môn bài hoặc trách nhiệm khấu trừ của nền tảng. Cần đối chiếu theo năm tính thuế đang hỏi."
        ),
        next_steps=(
            "Tổng hợp doanh thu theo tháng/năm từ sàn và tài khoản nhận tiền.",
            "Xác định cá nhân kinh doanh thường xuyên hay phát sinh không thường xuyên.",
            "Liên hệ chi cục thuế nơi cư trú/kinh doanh hoặc cổng eTax để đăng ký, kê khai nếu vượt ngưỡng áp dụng.",
        ),
    ),
    CitizenLegalSnippet(
        key="household_business_small_revenue",
        title="Hộ kinh doanh nhỏ và ngưỡng doanh thu",
        legal_reference="Luật Quản lý thuế 2019; Thông tư 40/2021/TT-BTC",
        keywords=("ho kinh doanh", "tiem tap hoa", "doanh thu nam", "100 trieu", "khong dang ky cong ty"),
        text=(
            "Với hộ/cá nhân kinh doanh nhỏ, nghĩa vụ VAT và thuế TNCN thường phụ thuộc doanh thu năm, ngành nghề và "
            "hình thức kinh doanh. Nếu doanh thu cả năm không vượt ngưỡng chịu thuế theo chính sách hiện hành, thường "
            "không phát sinh VAT/TNCN khoán; tuy nhiên vẫn nên kiểm tra nghĩa vụ đăng ký kinh doanh, đăng ký thuế và "
            "lệ phí môn bài theo địa phương."
        ),
        next_steps=(
            "Ước tính doanh thu cả năm và lưu sổ bán hàng tối thiểu.",
            "Hỏi bộ phận một cửa/chi cục thuế về đăng ký hộ kinh doanh và lệ phí môn bài.",
            "Nếu doanh thu tăng vượt ngưỡng, kê khai điều chỉnh kịp thời để tránh bị truy thu.",
        ),
    ),
    CitizenLegalSnippet(
        key="late_filing_penalty",
        title="Nộp tờ khai thuế trễ và tiền chậm nộp",
        legal_reference="Luật Quản lý thuế 2019; Nghị định 125/2020/NĐ-CP",
        keywords=("nop to khai tre", "tre han", "cham nop", "phat bao nhieu", "gtgt thang", "xin giam"),
        text=(
            "Khi nộp tờ khai sau hạn, mức xử phạt phụ thuộc số ngày trễ, tình tiết giảm nhẹ/tăng nặng và việc đã phát sinh "
            "số thuế phải nộp hay chưa. Nếu có số thuế nộp chậm, ngoài phạt hành chính còn có tiền chậm nộp tính theo số ngày "
            "quá hạn. Việc tự giác nộp ngay và giải trình lý do có thể là căn cứ xem xét tình tiết giảm nhẹ."
        ),
        next_steps=(
            "Nộp ngay tờ khai còn thiếu và số thuế phát sinh nếu có.",
            "Lưu biên nhận nộp điện tử, chứng từ nộp tiền và văn bản giải trình nguyên nhân.",
            "Theo dõi thông báo xử phạt của cơ quan thuế và khiếu nại/giải trình nếu số liệu chưa đúng.",
        ),
    ),
    CitizenLegalSnippet(
        key="tax_debt_enforcement",
        title="Nợ thuế, cưỡng chế tài khoản và hạn chế xuất cảnh",
        legal_reference="Luật Quản lý thuế 2019; Nghị định 126/2020/NĐ-CP",
        keywords=("no thue", "cuong che", "khoa tai khoan", "trich tien", "cam xuat canh", "thue mon bai"),
        text=(
            "Cưỡng chế thuế không tự động xảy ra ngay khi trễ hạn một khoản nhỏ; cơ quan thuế thường phải xác định nợ, "
            "thông báo và áp dụng biện pháp theo trình tự. Các biện pháp có thể gồm trích tiền từ tài khoản, ngừng hóa đơn, "
            "thông báo hóa đơn không còn giá trị, hoặc hạn chế xuất cảnh với trường hợp đủ điều kiện theo quy định."
        ),
        next_steps=(
            "Tra cứu tình trạng nợ trên eTax hoặc liên hệ cơ quan thuế quản lý.",
            "Nộp khoản còn thiếu và tiền chậm nộp, sau đó lưu chứng từ.",
            "Nếu khó khăn dòng tiền, hỏi thủ tục gia hạn/nộp dần nếu thuộc trường hợp được xem xét.",
        ),
    ),
    CitizenLegalSnippet(
        key="einvoice_late_issue",
        title="Quên xuất hóa đơn điện tử và xử lý bổ sung",
        legal_reference="Nghị định 123/2020/NĐ-CP; Thông tư 78/2021/TT-BTC; Nghị định 125/2020/NĐ-CP",
        keywords=("quen xuat hoa don", "hoa don dien tu", "xuat bo sung", "khach doi hoa don", "may tinh tien"),
        text=(
            "Khi bán hàng/cung cấp dịch vụ mà chưa lập hóa đơn đúng thời điểm, người bán nên lập hóa đơn bổ sung ngay, "
            "kê khai điều chỉnh nếu cần và lưu hồ sơ giải trình. Rủi ro xử phạt phụ thuộc hành vi, thời điểm phát hiện, "
            "giá trị giao dịch và việc người bán tự khắc phục trước khi cơ quan thuế kiểm tra."
        ),
        next_steps=(
            "Lập hóa đơn điện tử bổ sung với thông tin giao dịch đúng thực tế.",
            "Đối chiếu kỳ kê khai VAT/TNDN để điều chỉnh nếu doanh thu đã ghi nhận sai kỳ.",
            "Lưu trao đổi với khách, chứng từ thanh toán và giải trình nguyên nhân chậm lập hóa đơn.",
        ),
    ),
    CitizenLegalSnippet(
        key="cash_payment_over_20m",
        title="Thanh toán tiền mặt cho hóa đơn từ 20 triệu đồng",
        legal_reference="Luật Thuế GTGT; Thông tư 219/2013/TT-BTC; Thông tư 96/2015/TT-BTC",
        keywords=("tien mat", "35 trieu", "20 trieu", "duoc tru", "khau tru vat", "chi phi hop ly"),
        text=(
            "Với hóa đơn mua hàng hóa/dịch vụ có giá trị từ ngưỡng quy định, điều kiện thanh toán không dùng tiền mặt là "
            "một điều kiện quan trọng để khấu trừ VAT đầu vào và tính chi phí được trừ khi xác định thuế TNDN. Nếu đã trả "
            "tiền mặt, cần xem có thể chuyển khoản lại theo thỏa thuận hợp pháp và xử lý chứng từ kế toán đúng bản chất hay không."
        ),
        next_steps=(
            "Kiểm tra giá trị từng hóa đơn, hợp đồng và chứng từ thanh toán.",
            "Trao đổi với bên bán/kế toán về phương án thanh toán không dùng tiền mặt nếu giao dịch còn có thể điều chỉnh hợp lệ.",
            "Không tự lập chứng từ khống; nếu đã hạch toán sai, thực hiện điều chỉnh sổ sách và kê khai.",
        ),
    ),
    CitizenLegalSnippet(
        key="rental_income_individual",
        title="Cá nhân cho thuê nhà",
        legal_reference="Thông tư 40/2021/TT-BTC; Luật Quản lý thuế 2019",
        keywords=("cho thue nha", "15 trieu thang", "thue nha", "truy thu", "ca nhan cho thue"),
        text=(
            "Cá nhân cho thuê tài sản cần xác định doanh thu cho thuê theo năm để xem có phát sinh VAT, thuế TNCN và lệ phí "
            "môn bài hay không. Nếu đã cho thuê nhiều năm nhưng chưa kê khai, cần rà soát hợp đồng, dòng tiền, thời điểm nhận "
            "tiền và chủ động kê khai bổ sung để giảm rủi ro bị truy thu, phạt và tiền chậm nộp."
        ),
        next_steps=(
            "Tổng hợp hợp đồng thuê, chứng từ nhận tiền và thời gian cho thuê từng năm.",
            "Tính doanh thu theo năm dương lịch và đối chiếu ngưỡng chịu thuế.",
            "Kê khai bổ sung các năm còn thiếu qua cơ quan thuế quản lý địa bàn có nhà cho thuê.",
        ),
    ),
    CitizenLegalSnippet(
        key="deductible_client_entertainment",
        title="Chi phí tiếp khách khi tính thuế TNDN",
        legal_reference="Luật Thuế TNDN; Thông tư 78/2014/TT-BTC; Thông tư 96/2015/TT-BTC",
        keywords=("tiep khach", "an uong", "chi phi duoc tru", "tndn", "chung tu", "hoa don"),
        text=(
            "Chi phí tiếp khách có thể được xem xét là chi phí được trừ nếu liên quan đến hoạt động sản xuất kinh doanh, "
            "có hóa đơn chứng từ hợp pháp và thanh toán không dùng tiền mặt khi thuộc ngưỡng bắt buộc. Cần tách phần phục vụ "
            "kinh doanh thật với chi phí cá nhân hoặc khoản không có chứng từ."
        ),
        next_steps=(
            "Lưu hóa đơn, chứng từ thanh toán, đề nghị thanh toán và mục đích tiếp khách.",
            "Ghi rõ khách hàng/dự án/hợp đồng liên quan để chứng minh phục vụ kinh doanh.",
            "Rà soát chính sách nội bộ để tránh chi phí cá nhân hóa hoặc vượt mức không hợp lý.",
        ),
    ),
    CitizenLegalSnippet(
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
)


def retrieve_citizen_legal_snippets(query: str, *, top_k: int = 3, min_score: float = 0.12) -> list[dict[str, Any]]:
    """Return GraphRAG-like hits for common citizen tax questions."""
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    query_tokens = set(normalized_query.split())
    scored: list[tuple[float, CitizenLegalSnippet]] = []
    for snippet in SNIPPETS:
        normalized_keywords = [normalize_text(k) for k in snippet.keywords]
        keyword_hits = sum(1 for k in normalized_keywords if k and k in normalized_query)
        title_tokens = set(normalize_text(snippet.title).split())
        body_tokens = set(normalize_text(snippet.text).split())
        lexical = len(query_tokens & (title_tokens | body_tokens)) / max(1, len(query_tokens))
        score = min(1.0, 0.2 * keyword_hits + 0.8 * lexical)
        if score >= min_score:
            scored.append((score, snippet))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits: list[dict[str, Any]] = []
    for rank, (score, snippet) in enumerate(scored[:top_k], start=1):
        digest = int(hashlib.sha1(snippet.key.encode("utf-8")).hexdigest()[:8], 16)
        next_steps = "\n".join(f"- {step}" for step in snippet.next_steps)
        full_text = f"{snippet.text}\n\nCác bước xử lý thực tế:\n{next_steps}"
        hits.append(
            {
                "chunk_id": -digest,
                "chunk_key": f"citizen_tax_faq:{snippet.key}",
                "title": snippet.title,
                "doc_type": "citizen_tax_guidance",
                "text": full_text[:900],
                "full_text": full_text,
                "score": round(float(score), 6),
                "rerank_tier": "citizen_legal_fallback",
                "components": {"keyword_rank": rank, "fallback": 1.0},
                "document_key": "citizen_tax_guidance_v1",
                "corpus_version": "citizen_tax_guidance_v1:2026-05-10",
                "content_hash": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
                "citation_key": f"citizen-faq-{snippet.key}",
                "legal_reference": snippet.legal_reference,
                "citation_spans": [],
                "authority_path": [
                    {
                        "display_name": snippet.legal_reference,
                        "entity_type": "guideline",
                        "authority_rank": 30,
                    }
                ],
                "effective_status": {
                    "state": "active",
                    "dominant_state": "active",
                    "as_of": "2026-05-10",
                    "is_usable": True,
                    "note": "Fallback FAQ; verify against current official text before final administrative action.",
                },
                "official_letter_scope": {
                    "is_official_letter": False,
                    "binding_level": "guidance_not_normative",
                    "scope": "common_citizen_question",
                    "warnings": [
                        "Fallback FAQ is practical guidance and must be checked against official legal documents for binding decisions.",
                    ],
                },
                "relation_path": [],
                "legal_metadata": {
                    "fallback_type": "citizen_tax_guidance",
                    "legal_reference": snippet.legal_reference,
                },
            }
        )
    return hits
