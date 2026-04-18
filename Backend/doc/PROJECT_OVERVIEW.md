# 📚 TaxInspector - Architecture Overview & Master Plan

Tài liệu này lưu trữ toàn bộ thông tin về kiến trúc tổng thể, nghiệp vụ, và định hướng công nghệ của siêu dự án **TaxInspector**. Tài liệu dùng làm kim chỉ nam để quá trình lập trình xuyên suốt, trơn tru.

---

## 1. Giới thiệu chung
**TaxInspector** là một hệ thống phân tích, giám sát và cảnh báo rủi ro về Thuế. Hệ thống vận dụng sức mạnh của Data Science & Machine Learning chuyên sâu để hỗ trợ cơ quan ban ngành tự động phát hiện các hình thức gian lận thuế tinh vi.

---

## 2. Các Modules Nghiệp Vụ Cốt Lõi (Core Features)

### 📌 Module Bảo mật Tài khoản: Quên mật khẩu / Đổi mật khẩu
*   **Mô tả**: Bổ sung luồng tự khôi phục mật khẩu và đổi mật khẩu trong phiên đăng nhập.
*   **Luồng khôi phục**:
    *   Người dùng gửi yêu cầu tại trang đăng nhập (email công vụ).
    *   Backend phát hành reset token một lần, thời hạn ngắn.
    *   Trong local mode, reset link ghi vào `Backend/.otp_outbox.log` để kiểm thử không cần mailbox thật.
    *   Người dùng mở `reset-password.html` để đặt mật khẩu mới.
*   **Luồng đổi mật khẩu**:
    *   Thực hiện tại trang Tài khoản, yêu cầu mật khẩu hiện tại + mật khẩu mới.
*   **Tiêu chí an toàn**:
    *   Không lộ thông tin email tồn tại/không tồn tại.
    *   Token reset chỉ dùng một lần, có hạn dùng, lưu dạng hash trong DB.
    *   Mọi thao tác được ghi audit log.

### 📌 Tab 1: Hệ thống Chấm điểm Rủi ro (Fraud Risk Scoring)
*   **Mô tả**: Tự động đánh giá hàng triệu tờ khai thuế, phát hiện "xào nấu" số liệu (doanh thu cao, chi phí đột biến, v.v.).
*   **Điểm kỹ thuật yếu tố**:
    *   **Feature Engineering**: Chìa khóa quyết định. Tạo các "red flags" về tỷ suất lợi nhuận, chi phí phát sinh bất thường, mức độ biến động YoY/QoQ.
    *   **Thuật toán ML**: 
        *   Phát hiện bất thường: `Isolation Forest`, `One-Class SVM`.
        *   Phân loại và chấm điểm: `XGBoost`, `LightGBM`.
*   **Giao diện chỉ định**: Widget nhập Mã Số Thuế -> API trả về Thanh đo rủi ro (0-100), màu cảnh báo và danh sách "Red flags".

### 📌 Tab 2: Phân tích Đồ thị Hóa đơn Ảo (VAT Invoice Graph)
*   **Mô tả**: Khai phá và triệt phá các đường dây "công ty ma" mua bán hóa đơn GTGT lòng vòng nhằm trục lợi khấu trừ thuế.
*   **Điểm kỹ thuật yếu tố**:
    *   **Mô hình hóa dữ liệu**: Node = Công ty, Edge = Flow giao dịch mua bán.
    *   **Thuật toán ML**: Phân tích mạng lưới bằng thư viện `NetworkX` (Python). Định hướng sử dụng GNN (Graph Neural Networks) để nhận diện cụm giao dịch vòng tròn đáng ngờ (Circular Trading Rings).
    *   **Tối ưu truy vấn**: Cần tối ưu SQL / Graph query để tránh nghẽn cổ chai với lượng dữ liệu khổng lồ.
*   **Giao diện chỉ định**: Canvas vẽ mạng nhện đồ thị sử dụng thư viện `Vis.js` hoặc `D3.js`. Nhấn vào Node để highlight đường đi dòng tiền.

### 📌 Tab 3: Dự báo Trễ hạn Nộp thuế (Tax Delinquency Prediction)
*   **Mô tả**: Phát hiện sớm nguy cơ doanh nghiệp chây ì, trễ nộp hoặc làm mất khả năng thanh toán.
*   **Điểm kỹ thuật yếu tố**:
    *   **Thuật toán ML**: 
        *   Phân cụm hành vi nộp thuế: `K-Means`, `DBSCAN`.
        *   Dự báo chuỗi thời gian: `Time-series forecasting` + `Random Forest` / `Logistic Regression`.
    *   **Thử nghiệm tự động**: Cấu hình pipeline mô phỏng đẩy hàng nghìn hồ sơ vào hệ thống xử lý tính năng chịu tải (Stress test). 
    *   *Nguồn Data giả lập*: Sử dụng Synthetic Financial Data / Credit Card Fraud Detection (Kaggle) để tinh chỉnh đổi tên column giả lập thuế.
*   **Giao diện chỉ định**: Bảng Dashboard phân tích Batch Processing hiển thị Top 50 doanh nghiệp có xác suất trễ hạn nộp cao nhất.

---

## 3. Tech Stack Master (Công Nghệ)

*   **Backend Application**: 
    *   `FastAPI` (Python) - Hiệu năng cao, xử lý async xuất sắc, tự động sinh Swagger UI dọc theo mã nguồn.
*   **Machine Learning / Data Processing**: 
    *   `Scikit-Learn`, `XGBoost`, `NetworkX`, `Pandas`.
*   **Database**: 
    *   `PostgreSQL` với tính năng RDBMS cực mạnh cho khối dữ liệu quan hệ (Có định hướng mở rộng PostGIS nếu cần map khu vực địa lý).
*   **Frontend Interface**:
    *   **Kiến trúc**: Single Page Application (SPA) qua Vanilla JS. Fetch API async trực tiếp.
    *   **Giao diện/Styling**: HTML5, `Tailwind CSS`, Google Fonts (`Inter`, `Material Symbols`).
    *   **Thư viện Biểu đồ**: `Chart.js` (Biến động doanh thu/thuế), `Vis.js` (Đồ thị mạng lưới gian lận).

---

## 4. Kiến Trúc Cơ Sở Dữ Liệu Lõi (PostgreSQL Schema Plan)

Sẽ được khởi tạo với 4 bảng trung tâm:
1.  **`users`**: Hệ thống quản lý truy cập (Role-based access).
2.  **`companies`**: Dữ liệu gốc pháp nhân (`tax_code`, `name`, `industry`, `registration_date`).
3.  **`tax_returns`**: Lịch sử nộp khai báo thuế (`quarter`, `revenue`, `expenses`, `tax_paid`, `status`) - *Nguồn nuôi ML Tab 1 & Tab 3.*
4.  **`invoices`**: Sao kê hóa đơn xuất/nhập (`seller`, `buyer`, `amount`, `date`) - *Nguồn nuôi ML Tab 2.*

---

## 5. UI/UX Guidelines (Tiêu chí Giao diện)
*   **Màu sắc chủ đạo**: Deep Blue (Bảo mật/Chuyên nghiệp), Emerald (An toàn), Amber/Red (Cảnh báo).
*   **Kiến trúc Layout**:
    *   **Sidebar Cố định**: Logo quốc huy, Thông tin Admin, Menu Điều hướng gồm Tống quan + 3 Tính năng ML.
    *   **Top Bar**: Thanh tìm kiếm mã số thuế + Notify.
    *   **Main Canvas Chuyển động**: Khi chuyển Router phải có hiệu ứng `<Slide-Up>` và `<Fade-In>` mượt mà. Nút bấm mang các yếu tố nổi khối nhẹ (Glassmorphism + Subtle shadow hover).

---

## 6. Completion Roadmap (Model + UI + Ops)

### 6.1 Trạng thái hiện tại
- **Model tracks**
    - Audit Value: Model-backed v1, quality gate operational.
    - VAT Refund: Model-backed v1, quality gate operational.
    - Pilot: model-vs-heuristic report đã có, theo dõi thêm chu kỳ để ổn định.
    - Go/No-Go: tự động hóa hoàn chỉnh qua script chuyên biệt + history JSONL.
- **UI/Frontend**
    - Dashboard đã có KPI Split Trigger Governance widget.
    - Dashboard đã có Specialized Rollout Status widget (gates, pilot deltas, decision, actions).
- **Ops/Governance**
    - Monitoring endpoint tổng hợp rollout: `/api/monitoring/specialized_rollout_status`.
    - Chính sách KPI + snapshot + alerting + cooldown semantics đã hoạt động.
    - Regression test coverage đã bao gồm monitoring workflow và go/no-go logic.

### 6.2 Kế hoạch hoàn thiện 30 ngày
1. Chạy cadence pipeline specialized (ít nhất 2-3 chu kỳ/tuần) để tích lũy bằng chứng ổn định pilot.
2. Review `specialized_go_no_go_history.jsonl` hàng tuần để xác nhận stability gate.
3. Chuẩn hóa policy threshold (nếu cần) dựa trên drift/pass-rate thực tế từ dashboard.
4. Khi decision chuyển `go_phase_d_candidate`, mở scope Phase D tách workbench theo rollout batches.

### 6.3 Exit Criteria để đóng dự án V1
1. Audit + VAT quality reports duy trì `overall_pass=true` theo chu kỳ vận hành.
2. Pilot reports duy trì sample đủ lớn và không có regression nghiêm trọng kéo dài.
3. Go/No-Go decision có lịch sử ổn định, sẵn sàng cho quyết định Phase D.
4. Dashboard và monitoring endpoint phản ánh đúng trạng thái release readiness.
5. Regression test suite chính (`test_monitoring_kpi_workflow.py`, `test_specialized_go_no_go_review.py`) luôn pass.

*Document v1.1 - Updated with completion roadmap.*
