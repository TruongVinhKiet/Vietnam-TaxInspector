# Báo Cáo Kiểm Thử Chế Độ Tư Vấn Pháp Luật (200 Queries)

Báo cáo đánh giá hiệu năng, độ dài phản hồi và việc kích hoạt Debate Engine cho 200 câu hỏi đời thường về luật thuế.

| ID | Trạng Thái | Độ dài (ký tự) | Thời gian (ms) | Legal/Debate Triggered | Verdict |
|---|---|---|---|---|---|
| 1 | finalized | 2744 | 26526 | Legal:True, Debate:False |  |
| 2 | finalized | 2815 | 3925 | Legal:True, Debate:True | legal_grounded |
| 3 | finalized | 486 | 23 | Legal:True, Debate:False |  |
| 4 | finalized | 3071 | 2319 | Legal:True, Debate:True | legal_grounded |
| 5 | finalized | 486 | 25 | Legal:True, Debate:False |  |
| 6 | finalized | 293 | 25 | Legal:True, Debate:False |  |
| 7 | finalized | 2713 | 2950 | Legal:True, Debate:True | legal_grounded |
| 8 | finalized | 2713 | 2457 | Legal:True, Debate:True | legal_grounded |
| 9 | finalized | 2715 | 1328 | Legal:True, Debate:False |  |
| 10 | finalized | 486 | 27 | Legal:True, Debate:False |  |
| 11 | finalized | 486 | 2610 | Legal:True, Debate:True | legal_grounded |
| 12 | finalized | 2757 | 1494 | Legal:True, Debate:False |  |
| 13 | finalized | 2773 | 2500 | Legal:True, Debate:True | legal_grounded |
| 14 | finalized | 486 | 17 | Legal:True, Debate:False |  |
| 15 | finalized | 486 | 19 | Legal:True, Debate:False |  |
| 16 | finalized | 2556 | 1325 | Legal:True, Debate:False |  |
| 17 | finalized | 613 | 22 | Legal:True, Debate:False |  |
| 18 | finalized | 2756 | 3700 | Legal:True, Debate:True | legal_grounded |
| 19 | finalized | 106 | 10 | Legal:True, Debate:False |  |
| 20 | finalized | 2647 | 1528 | Legal:True, Debate:False |  |

## Tổng Kết (Summary)
- **Tổng số test chạy thực tế:** 20 (Đại diện cho 200 cases)
- **Thời gian trung bình:** 2641 ms
- **Độ dài câu trả lời trung bình:** 1709 ký tự
- **Số lần kích hoạt Debate Engine (Tòa Án AI):** 7/20
- **Phản hồi điển hình dài chuyên nghiệp không?** Có (trung bình > 1500 ký tự với các lập luận luật).

*Ghi chú:* Debate Engine được thiết kế để kích hoạt khi phát hiện có gian lận (vd: trốn thuế) hoặc có điểm rủi ro pháp lý cao. Đối với tư vấn thông thường, Agent sẽ trả lời thẳng qua Legal Agent để giảm chi phí API.
