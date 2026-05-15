import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ingest_tax_knowledge import TaxKnowledgeIngestor

docs_to_ingest = [
    {
        'key': 'CV_4846_TCT_DNNCN_2024',
        'title': 'Công văn 4846/TCT-DNNCN năm 2024: Về điều kiện ủy quyền quyết toán thuế TNCN',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-10-25',
        'content': '''Công văn số 4846/TCT-DNNCN được Tổng cục Thuế ban hành ngày 25/10/2024 về chính sách thuế thu nhập cá nhân (TNCN).
        
Về điều kiện ủy quyền quyết toán thuế TNCN:
Trường hợp trong năm tính thuế, người nộp thuế có thu nhập từ hai nơi trở lên, trong đó có thu nhập vãng lai chưa được khấu trừ thuế TNCN theo tỷ lệ 10% thì không đủ điều kiện để ủy quyền quyết toán thuế cho tổ chức hoặc cá nhân khác.
Trong trường hợp này, người nộp thuế phải trực tiếp khai quyết toán thuế TNCN với cơ quan thuế. Khi thực hiện khai quyết toán trực tiếp, người nộp thuế cần kê khai đầy đủ các khoản thu nhập chịu thuế nhận được trong kỳ theo quy định pháp luật.
        
Người nộp thuế có trách nhiệm tự tổng hợp thu nhập, các khoản giảm trừ gia cảnh để xác định chính xác số thuế thu nhập cá nhân phải nộp hoặc số thuế nộp thừa để đề nghị hoàn thuế. Cơ quan thuế các cấp có trách nhiệm hướng dẫn, hỗ trợ người nộp thuế thực hiện quyết toán thuế trực tuyến qua cổng eTax Mobile hoặc website của Tổng cục Thuế để tiết kiệm thời gian và chi phí.'''
    },
    {
        'key': 'CV_1569_TCT_KK_2024',
        'title': 'Công văn 1569/TCT-KK năm 2024: Về việc chuyển số thuế GTGT còn được khấu trừ sau khi chi nhánh chấm dứt hoạt động',
        'type': 'official_letter',
        'authority_rank': 40,
        'effective_from': '2024-04-15',
        'content': '''Công văn số 1569/TCT-KK do Tổng cục Thuế ban hành ngày 15/04/2024 hướng dẫn xử lý số thuế GTGT đầu vào chưa khấu trừ hết của các chi nhánh khi chi nhánh đó chấm dứt hoạt động.

Về việc chuyển số thuế GTGT còn được khấu trừ:
Trường hợp Chi nhánh là đơn vị phụ thuộc của công ty mẹ, nếu đã chấm dứt hoạt động và số thuế GTGT đầu vào vẫn còn được khấu trừ (đáp ứng đầy đủ các điều kiện khấu trừ theo quy định pháp luật hiện hành về thuế Giá trị gia tăng), thì Chi nhánh được chuyển số thuế GTGT đầu vào chưa khấu trừ hết cho Công ty mẹ để Công ty mẹ tiếp tục kê khai, khấu trừ theo quy định.

Trình tự, thủ tục:
1. Chi nhánh phải hoàn thành nghĩa vụ nộp thuế trước khi chấm dứt hoạt động.
2. Lập bảng kê số thuế GTGT chưa khấu trừ hết gửi cơ quan thuế quản lý trực tiếp của chi nhánh để xác nhận.
3. Sau khi có xác nhận, Công ty mẹ thực hiện kê khai bổ sung số thuế GTGT đầu vào này vào kỳ tính thuế hiện tại của Công ty mẹ. Cơ quan thuế quản lý Công ty mẹ sẽ căn cứ vào hồ sơ để kiểm tra và chấp thuận việc khấu trừ theo đúng quy định.'''
    }
]

if __name__ == "__main__":
    ingestor = TaxKnowledgeIngestor()
    try:
        for doc in docs_to_ingest:
            ingestor.ingest(doc)
        ingestor.finalize_relations()
        print("Thêm công văn thành công!")
    finally:
        ingestor.close()
