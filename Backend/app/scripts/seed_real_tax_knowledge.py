from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text
from app.database import SessionLocal
from app.scripts.ingest_tax_knowledge import ingest_document

def main() -> None:
    print("Ingesting real-world Vietnamese tax knowledge...")
    
    docs = [
        {
            "key": "nd_72_2024",
            "title": "Nghị định 72/2024/NĐ-CP: Quy định chính sách giảm thuế giá trị gia tăng theo Nghị quyết số 142/2024/QH15",
            "doc_type": "decree",
            "authority": "Chính phủ",
            "source_uri": "https://chinhphu.vn/nd_72_2024",
            "content": (
                "Điều 1. Giảm thuế giá trị gia tăng\n\n"
                "1. Giảm thuế giá trị gia tăng đối với các nhóm hàng hóa, dịch vụ đang áp dụng mức thuế suất 10%, trừ nhóm hàng hóa, dịch vụ sau:\n"
                "a) Viễn thông, hoạt động tài chính, ngân hàng, chứng khoán, bảo hiểm, kinh doanh bất động sản, kim loại và sản phẩm từ kim loại đúc sẵn, sản phẩm khai khoáng (không kể khai thác than), than cốc, dầu mỏ tinh chế, sản phẩm hóa chất. Chi tiết tại Phụ lục I ban hành kèm theo Nghị định này.\n"
                "b) Sản phẩm hàng hóa và dịch vụ chịu thuế tiêu thụ đặc biệt. Chi tiết tại Phụ lục II ban hành kèm theo Nghị định này.\n"
                "c) Công nghệ thông tin theo pháp luật về công nghệ thông tin. Chi tiết tại Phụ lục III ban hành kèm theo Nghị định này.\n\n"
                "2. Mức giảm thuế giá trị gia tăng:\n"
                "a) Cơ sở kinh doanh tính thuế giá trị gia tăng theo phương pháp khấu trừ được áp dụng mức thuế suất thuế giá trị gia tăng 8% đối với hàng hóa, dịch vụ quy định tại khoản 1 Điều này.\n"
                "b) Cơ sở kinh doanh (bao gồm cả hộ kinh doanh, cá nhân kinh doanh) tính thuế giá trị gia tăng theo phương pháp tỷ lệ % trên doanh thu được giảm 20% mức tỷ lệ % để tính thuế giá trị gia tăng khi thực hiện xuất hóa đơn đối với hàng hóa, dịch vụ được giảm thuế giá trị gia tăng quy định tại khoản 1 Điều này.\n\n"
                "Điều 2. Hiệu lực thi hành\n"
                "Nghị định này có hiệu lực thi hành từ ngày 01 tháng 07 năm 2024 đến hết ngày 31 tháng 12 năm 2024."
            ),
            "metadata": {
                "effective_status": "Còn hiệu lực",
                "official_letter_scope": "Toàn quốc",
                "authority_path": "Chính phủ"
            }
        },
        {
            "key": "cv_2721_2024",
            "title": "Công văn 2721/TCT-CS năm 2024: Về ưu đãi thuế TNDN đối với dự án đầu tư",
            "doc_type": "official_letter",
            "authority": "Tổng cục Thuế",
            "source_uri": "https://gdt.gov.vn/cv_2721_2024",
            "content": (
                "Căn cứ khoản 4 Điều 18 Luật Thuế thu nhập doanh nghiệp số 14/2008/QH12 quy định:\n\n"
                "'Doanh nghiệp có dự án đầu tư phát triển dự án đầu tư đang hoạt động thuộc lĩnh vực, địa bàn ưu đãi thuế thu nhập doanh nghiệp mở rộng quy mô sản xuất, nâng cao công suất, đổi mới công nghệ sản xuất nếu đáp ứng một trong ba tiêu chí quy định tại khoản này thì được lựa chọn hưởng ưu đãi thuế theo dự án đang hoạt động cho thời gian còn lại (nếu có) hoặc được áp dụng thời gian miễn thuế, giảm thuế đối với phần thu nhập tăng thêm do đầu tư mở rộng mang lại...'\n\n"
                "Theo đó, Tổng cục Thuế hướng dẫn Cục Thuế các tỉnh, thành phố trực thuộc Trung ương như sau:\n\n"
                "Trường hợp doanh nghiệp có dự án đầu tư mở rộng thỏa mãn các tiêu chí về vốn đầu tư, tỷ lệ tăng trưởng doanh thu hoặc số lượng lao động tăng thêm theo quy định thì doanh nghiệp được quyền lựa chọn hưởng ưu đãi thuế TNDN.\n"
                "Tuy nhiên, nếu dự án mở rộng này thực hiện tại địa bàn không thuộc danh mục địa bàn ưu đãi thuế, thì phần thu nhập tăng thêm từ dự án mở rộng sẽ không được hưởng ưu đãi thuế TNDN theo diện ưu đãi địa bàn."
            ),
            "metadata": {
                "effective_status": "Còn hiệu lực",
                "official_letter_scope": "Toàn quốc",
                "authority_path": "Bộ Tài chính > Tổng cục Thuế"
            }
        },
        {
            "key": "cv_6141_2024",
            "title": "Công văn 6141/TCT-CS năm 2024: Về việc áp dụng thuế suất thuế GTGT 8%",
            "doc_type": "official_letter",
            "authority": "Tổng cục Thuế",
            "source_uri": "https://gdt.gov.vn/cv_6141_2024",
            "content": (
                "Trả lời công văn của Cục Thuế tỉnh Kiên Giang về việc áp dụng thuế suất thuế GTGT theo Nghị định số 72/2024/NĐ-CP, Tổng cục Thuế có ý kiến như sau:\n\n"
                "Căn cứ Nghị định số 72/2024/NĐ-CP ngày 30/06/2024 của Chính phủ quy định chính sách giảm thuế GTGT:\n\n"
                "Trường hợp Công ty TNHH MTV thương mại dịch vụ xuất hóa đơn bán hàng hóa cho người mua, nếu hàng hóa bán ra không thuộc danh mục hàng hóa, dịch vụ quy định tại các Phụ lục I, II, III ban hành kèm theo Nghị định 72/2024/NĐ-CP thì được áp dụng mức thuế suất thuế GTGT 8%.\n\n"
                "Lưu ý: Đối với các dịch vụ bốc xếp hàng hóa tại cảng, nếu xác định đây không phải là dịch vụ trực tiếp hỗ trợ vận tải (mã ngành 522), thì được áp dụng thuế suất 8%."
            ),
            "metadata": {
                "effective_status": "Còn hiệu lực",
                "official_letter_scope": "Cục Thuế tỉnh Kiên Giang",
                "authority_path": "Bộ Tài chính > Tổng cục Thuế"
            }
        },
        {
            "key": "cv_4781_2024",
            "title": "Công văn 4781/TCT-CS năm 2024: Hoàn thuế GTGT đối với dự án đầu tư",
            "doc_type": "official_letter",
            "authority": "Tổng cục Thuế",
            "source_uri": "https://gdt.gov.vn/cv_4781_2024",
            "content": (
                "Về việc hoàn thuế GTGT dự án đầu tư, Tổng cục Thuế hướng dẫn như sau:\n\n"
                "Theo quy định tại khoản 3 Điều 1 Luật số 106/2016/QH13 sửa đổi, bổ sung một số điều của Luật Thuế GTGT: 'Cơ sở kinh doanh nộp thuế GTGT theo phương pháp khấu trừ thuế có dự án đầu tư (trừ dự án đầu tư xây dựng nhà để bán hoặc cho thuê mà không hình thành tài sản cố định) đang trong giai đoạn đầu tư có số thuế giá trị gia tăng của hàng hóa, dịch vụ mua vào sử dụng cho đầu tư mà chưa được khấu trừ hết và có số thuế còn lại từ 300 triệu đồng trở lên thì được hoàn thuế giá trị gia tăng.'\n\n"
                "Cơ quan thuế quản lý trực tiếp chịu trách nhiệm kiểm tra thực tế dự án đầu tư, đảm bảo dự án có đủ giấy phép xây dựng và giấy chứng nhận đăng ký đầu tư trước khi giải quyết hồ sơ hoàn thuế. Đặc biệt lưu ý các hóa đơn đầu vào phát sinh trước khi có giấy phép đầu tư cần được rà soát kỹ về tính hợp lý, hợp lệ."
            ),
            "metadata": {
                "effective_status": "Còn hiệu lực",
                "official_letter_scope": "Toàn quốc",
                "authority_path": "Bộ Tài chính > Tổng cục Thuế"
            }
        },
        {
            "key": "cv_1122_2023",
            "title": "Công văn 1122/TCT-CS năm 2023: Về hóa đơn điện tử khởi tạo từ máy tính tiền",
            "doc_type": "official_letter",
            "authority": "Tổng cục Thuế",
            "source_uri": "https://gdt.gov.vn/cv_1122_2023",
            "content": (
                "Triển khai hóa đơn điện tử khởi tạo từ máy tính tiền theo Thông tư 78/2021/TT-BTC:\n\n"
                "1. Đối tượng áp dụng: Các doanh nghiệp, hộ kinh doanh nộp thuế theo phương pháp kê khai hoạt động kinh doanh bán lẻ trực tiếp đến người tiêu dùng (trung tâm thương mại, siêu thị, bán lẻ thuốc tân dược, dịch vụ ăn uống, nhà hàng, khách sạn, dịch vụ vui chơi giải trí...).\n\n"
                "2. Lợi ích: Hóa đơn điện tử khởi tạo từ máy tính tiền có kết nối chuyển dữ liệu với cơ quan thuế giúp doanh nghiệp chủ động 24/7 trong việc lập hóa đơn, không cần ký số trên hóa đơn, tiết kiệm thời gian và chi phí, đồng thời hóa đơn này được sử dụng để làm căn cứ xác định chi phí hợp lý khi tính thuế TNDN.\n\n"
                "Các cơ quan thuế địa phương cần tích cực đôn đốc, hướng dẫn các cơ sở kinh doanh dịch vụ ăn uống chuyển đổi sang sử dụng hóa đơn điện tử từ máy tính tiền."
            ),
            "metadata": {
                "effective_status": "Hết hiệu lực",
                "official_letter_scope": "Toàn quốc",
                "authority_path": "Bộ Tài chính > Tổng cục Thuế"
            }
        }
    ]

    for doc in docs:
        print(f"Ingesting: {doc['title']}...")
        ingest_document(
            document_key=doc["key"],
            title=doc["title"],
            doc_type=doc["doc_type"],
            authority=doc["authority"],
            source_uri=doc["source_uri"],
            version_tag="v1",
            content=doc["content"],
            metadata=doc["metadata"],
        )

    print("All real-world tax knowledge seeded successfully!")

if __name__ == "__main__":
    main()
