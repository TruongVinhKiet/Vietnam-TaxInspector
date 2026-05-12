# Báo cáo chuyên sâu: khoảng cách Multi-Agent so với Claude/ChatGPT

## 1) Mục tiêu kỳ vọng

Hệ thống cần đạt các đặc tính của trợ lý LLM hiện đại:

- Hiểu ngôn ngữ tự nhiên linh hoạt (đa kiểu diễn đạt, câu hỏi nối tiếp, ngữ cảnh mơ hồ).
- Trả lời mạch lạc theo hội thoại nhiều lượt, có nhớ dữ liệu trong cùng session.
- Mỗi chế độ vận hành đúng model/domain tương ứng trang nghiệp vụ:
  - Gian lận: phân tích đơn lẻ + lô CSV theo pipeline risk.
  - VAT: truy vết VAT, drill-down hóa đơn và thực thể.
  - Mô phỏng vĩ mô: điều chỉnh tham số, so sánh kịch bản.
  - Dự báo nợ đọng: dự báo theo kỳ, giải thích nguyên nhân.
- UI thống nhất, dễ hiểu, có phản hồi trạng thái rõ ràng khi stream/polling lỗi.

## 2) Những vấn đề cốt lõi đang tồn tại

### 2.1 Reliability và đồng nhất hành vi

- Fallback chưa đồng nhất giữa các endpoint chat v1/v2/stream.
- Một số nhánh xử lý file async trả thành công HTTP nhưng UX có thể kém ổn định nếu payload lớn.
- Cơ chế timeout/cancel tool chưa triệt để trong trường hợp handler kéo dài.

Tác động: người dùng cảm nhận hệ thống "lúc được lúc không", khó tin cậy ở tình huống tải lớn.

### 2.2 Session memory và continuity

- Snapshot dữ liệu upload trước đây dùng cache RAM là chính, rủi ro mất ngữ cảnh khi restart/đổi worker.
- Một số nhánh clarification chưa ghi dấu vết đầy đủ trước khi kết thúc lượt.
- Chưa có lớp "facts từ câu trả lời trước" chuẩn hóa để hỗ trợ giải thích follow-up sâu.

Tác động: câu hỏi nối tiếp có thể thiếu logic "theo câu trước", nhất là sau upload CSV/Excel.

### 2.3 Mode-to-model parity

- Routing/mode guard có nhưng logic nằm ở nhiều lớp (router/task_router/orchestrator), dễ drift.
- Cùng một ý định có thể đi nhiều pipeline khác nhau (sync/async/canonical), làm lệch contract.

Tác động: khó đảm bảo mỗi mode luôn trả đúng bộ biểu đồ và hành vi như trang nghiệp vụ tương ứng.

### 2.4 Frontend parity kiểu Claude/ChatGPT

- Streaming cần rõ ràng hơn ở trạng thái lỗi/retry.
- Một số renderer markdown/charts cần xử lý fallback mạnh hơn để tránh trải nghiệm "không thấy gì".
- Workspace theo mode chưa đồng đều hoàn toàn.

Tác động: UX chưa "liền mạch" như trợ lý hội thoại cao cấp.

### 2.5 Data governance và migration

- Nguồn schema/migration có nguy cơ drift khi bootstrap ở môi trường khác nhau.
- Thiếu một số ràng buộc/index then chốt cho luồng hội thoại nhiều lượt dưới tải.

Tác động: tăng rủi ro lỗi ngầm, giảm khả năng truy vết/audit nhất quán.

## 3) Hạng mục đã tăng cường trong đợt triển khai này

### 3.1 Độ bền session và async job

- Thêm bảng durable:
  - `agent_session_snapshots`
  - `agent_async_file_jobs`
- Đồng bộ tạo bảng ở migration và startup SQL.
- Bổ sung unique/index quan trọng cho hội thoại:
  - `agent_turns(session_id, turn_index)` unique.
  - index `agent_turns(session_id, turn_index DESC)`.

### 3.2 Nâng cấp ConversationMemory

- Lưu snapshot theo scope (`risk_batch`, `vat_snapshot`, `attachment_summary`) vào DB với TTL.
- Khi build context, ưu tiên cache; nếu thiếu thì load từ DB snapshot.
- Giảm rủi ro mất context sau restart và tăng continuity thực tế.

### 3.3 Củng cố async response contract

- Chuẩn hóa trạng thái async sang tập rõ ràng: `pending / processing / done / error`.
- Compact serializer cho payload done:
  - Cắt bớt danh sách nặng.
  - Loại trường blob lớn không cần thiết cho màn hình chat.

### 3.4 Bổ sung Simulation Workspace end-to-end

- Contract backend có `simulation_workspace`.
- Orchestrator sinh payload workspace theo mode `macro`.
- Frontend có toggle/panel riêng cho mô phỏng, hiển thị tham số hiện tại/khuyến nghị/range/sensitivity bằng tiếng Việt có dấu.
- Lưu cache panel theo sessionStorage để continuity UI tốt hơn.

### 3.5 Frontend polling/UX hardening

- Polling ưu tiên progress thật từ backend.
- Lỗi parse payload lớn có thông báo rõ ràng cho người dùng.
- Cải thiện thông điệp timeout và trạng thái terminal.

## 4) Khoảng cách còn lại để đạt mức Claude/ChatGPT

### 4.1 NL understanding nâng cao

- Hiện vẫn còn phụ thuộc rule/regex ở một số khâu hiểu hội thoại.
- Cần thêm lớp semantic conversation state (intent transitions, contradiction detection, discourse memory).

### 4.2 Planner/Executor tách lớp sâu hơn

- Orchestrator vẫn là điểm tập trung nhiều trách nhiệm.
- Cần tách rõ domain services: routing, planning, execution, synthesis, audit persist.

### 4.3 Mode parity tuyệt đối theo trang nghiệp vụ

- Cần hợp nhất đường chạy cho từng mode (single source of truth), tránh nhiều đường sync/async lệch contract.
- Chuẩn hóa bộ output UI theo từng mode với schema version rõ ràng.

### 4.4 Chất lượng vận hành LLM-like

- Cần bộ chỉ số hội thoại nâng cao:
  - Follow-up grounding rate.
  - Cross-turn consistency score.
  - Mode parity conformance.
  - Silent-failure rate (stream/poll/render).

## 5) Kế hoạch nâng cấp tiếp theo (đề xuất)

### Pha A - Reliability Hardening

1. Chuẩn hóa fallback matrix cho toàn bộ endpoint chat.
2. Timeout/cancel tool ở mức execution hard-boundary.
3. Thêm retry policy có backoff theo loại lỗi tool.

### Pha B - Deep Conversation Memory

1. Bảng `prior_answer_facts` (structured claims per assistant turn).
2. Retrieval ưu tiên facts trong session trước khi gọi tool nặng.
3. TTL/invalidations theo mode để tránh dùng snapshot cũ sai ngữ cảnh.

### Pha C - Mode Contract Unification

1. Định nghĩa schema output chuẩn cho Fraud/VAT/Simulation/Delinquency.
2. Một authoritative execution path cho mỗi mode.
3. Contract tests bắt buộc cho batch + single query mỗi mode.

### Pha D - UX parity hoàn chỉnh

1. Regenerate/retry controls đồng nhất.
2. Stream state machine rõ ràng (queued, streaming, finalized, partial-error).
3. Workspace parity cho mọi mode có artifact phân tích.

### Pha E - Governance và quan trắc

1. Hợp nhất nguồn schema bootstrap.
2. Thêm data constraints/checks cho route/mode fields.
3. Dashboard chất lượng hội thoại theo phiên và theo mode.

## 6) Tiêu chí nghiệm thu đề xuất

- Follow-up sau upload CSV/Excel trong cùng session luôn trả dựa trên snapshot đúng mode.
- Sau restart worker, câu hỏi nối tiếp vẫn truy hồi được ngữ cảnh quan trọng.
- Mỗi mode trả đúng nhóm biểu đồ/phân tích của trang nghiệp vụ tương ứng.
- Stream/polling không còn trạng thái treo không thông báo.
- Không còn duplicate turn index trong kiểm thử concurrent.

---

Tài liệu này dùng để điều phối refactor nhiều pha. Trong triển khai thực tế, nên khóa thêm golden tests theo từng mode trước khi mở rộng tính năng mới.