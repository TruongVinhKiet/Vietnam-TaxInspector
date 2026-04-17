# 🇻🇳 Vietnam TaxInspector
### AI-Powered Sovereign Tax Surveillance & Risk Analytics Suite

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![TailwindCSS](https://img.shields.io/badge/UI-TailwindCSS-06B6D4?style=flat&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

**Vietnam TaxInspector** là hệ thống phân tích dữ liệu thuế tiên tiến, kết hợp giữa Trí tuệ nhân tạo (Machine Learning) và Phân tích Đồ thị (Graph Analytics) nhằm hỗ trợ cơ quan Nhà nước trong việc phát hiện gian lận và tối ưu hóa quản lý nợ đọng thuế.

---

## 🌟 Tính Năng Cốt Lõi (Core Features)

### 📊 1. Trung Tâm Giám Sát (Unified Dashboard)
Giao diện quản lý tập trung hiển thị biến động doanh thu, chỉ số rủi ro hệ thống và các cảnh báo khẩn cấp theo thời gian thực từ 63 tỉnh thành.

### 🛡️ 2. Chấm Điểm Rủi Ro (Fraud Scoring)
Hệ thống **Security Analytical Suite** phân tích hành vi doanh nghiệp dựa trên hóa đơn điện tử và dòng tiền. Sử dụng các mô hình AI để gán nhãn mức độ nghi vấn rủi ro thuế (Thấp - Trung bình - Cao).

### 🕸️ 3. Mạng Lưới Điều Tra (Forensic Network Mapping)
Truy vết các chuỗi hóa đơn xoay vòng, công ty "ma" thông qua thuật toán đồ thị (Graph Algorithms). Phát hiện các cụm giao dịch bất thường trong mạng lưới hàng triệu mã số thuế.

### 📉 4. Dự Báo Nợ Đọng (Delinquency Prediction)
Dự báo khả năng trễ hạn nộp thuế trong quý tới của doanh nghiệp bằng mô hình AI Sovereign v2.4, giúp cán bộ thuế chủ động trong việc đốc thúc và quản lý dòng tiền ngân sách.

---

## 📸 Hình Ảnh Giao Diện (Preview)

| Dashboard | Mạng Lưới VAT |
| :---: | :---: |
| ![Dashboard](Frontend/Frontend_Example/dashboard.png) | ![VAT Graph](Frontend/Frontend_Example/Invoice_Graph.png) |

| Chấm Điểm Rủi Ro | Dự Báo Nợ Đọng |
| :---: | :---: |
| ![Fraud Scoring](Frontend/Frontend_Example/Fraud_Scoring.png) | ![Delinquency](Frontend/Frontend_Example/Delinquency_Prediction.png) |

---

## 🛠️ Kiến Trúc Công Nghệ (Tech Stack)

*   **Backend:** Python 3.9+, FastAPI, SQLAlchemy (ORM).
*   **Database:** PostgreSQL (Lưu trữ quan hệ và dữ liệu hóa đơn).
*   **Frontend:** HTML5, Vanilla JavaScript, TailwindCSS (Modern "Office White" Design System).
*   **Analytics:** NetworkX (Xử lý đồ thị), Scikit-learn/XGBoost (Mô hình dự báo).

---

## 🚀 Hướng Dẫn Cài Đặt (Quick Start)

### 1. Chuẩn bị Cơ sở dữ liệu
Đảm bảo bạn đã cài đặt **PostgreSQL** và tạo một database tên là `TaxInspector`. Sử dụng file schema tại `Database/init_db.sql` (nếu có) để khởi tạo cấu trúc.

### 2. Thiết lập Backend
```bash
cd Backend
python -m venv venv
source venv/bin/activate  # Trên Windows: .\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Khởi tạo Frontend
Vì lý do bảo mật trình duyệt, hãy chạy một local server nhẹ:
```bash
cd Frontend
python -m http.server 3000
```
Truy cập: **http://localhost:3000** để bắt đầu.

### 4. Quên mật khẩu / Đổi mật khẩu (Local mode)
- Tại trang đăng nhập có nút **Quên mật khẩu**.
- Hệ thống tạo link reset và ghi vào file nội bộ: `Backend/.otp_outbox.log` (dùng để test khi chưa có mailbox `@gdt.gov.vn` thật).
- Mở link trong outbox để vào trang `reset-password.html` và đặt mật khẩu mới.
- Sau khi đăng nhập, vào trang **Tài khoản** để dùng chức năng **Đổi mật khẩu**.

### 5. Test Delinquency End-to-End (có lệnh seed mock riêng)
`data/seed_db.py` chỉ tạo dữ liệu cho `companies` + `tax_returns`.
Để test đầy đủ Delinquency, cần seed thêm bảng `tax_payments` bằng script mới:

```bash
# Từ thư mục gốc dự án
python Backend/data/generate_mock_data.py
python Backend/data/seed_db.py
python Backend/data/seed_tax_payments.py --reset --companies 5000 --seed 42
```

Sau đó chạy batch predict để làm đầy cache dự báo:

```bash
curl -X POST "http://localhost:8000/api/delinquency/predict-batch" \
	-H "Authorization: Bearer <JWT_TOKEN>" \
	-H "Content-Type: application/json" \
	-d '{"limit": 500, "refresh_existing": false}'
```

Nếu cần refresh toàn bộ cache khi dữ liệu lớn (ví dụ 5000+ MST), dùng script chunk tự động (khuyến nghị):

```bash
cd Backend
python -m app.scripts.refresh_delinquency_cache --base-url http://127.0.0.1:8000 --chunk-size 500 --refresh-existing
```

Lý do dùng script: API `predict-batch` giới hạn `limit <= 2000`, nên full refresh phải chia nhiều lượt theo `tax_codes`.

Theo dõi sức khỏe cache Delinquency (coverage + freshness + model versions):

```bash
curl "http://localhost:8000/api/delinquency/health/cache?fresh_days=7&stale_days=30"
```

Dashboard `pages/dashboard.html` da duoc gan realtime widget cho health endpoint nay (polling mac dinh 60s).

Tự động refresh định kỳ bằng scheduler script:

```bash
cd Backend
python -m app.scripts.schedule_delinquency_refresh --interval-minutes 180 --chunk-size 500 --no-refresh-existing
```

Chạy 1 vòng duy nhất (phù hợp cron/Task Scheduler):

```bash
cd Backend
python -m app.scripts.schedule_delinquency_refresh --once --chunk-size 500 --refresh-existing
```

Backend co event hook canh bao tu dong khi `coverage` hoac `stale_ratio` vuot nguong:
- `delinquency_cache_health_threshold_breach` (warning/critical, co cooldown 5 phut de tranh spam log)
- `delinquency_cache_health_ok`

Checklist smoke test Delinquency:
- `GET /api/delinquency?page=1&page_size=20` trả về danh sách có `freshness`, `score_source`, `prediction_age_days`.
- `GET /api/delinquency?page=1&page_size=20&freshness=stale` lọc đúng theo freshness.
- `POST /api/delinquency/predict-batch` trả về `processed/created/updated/failed` hợp lệ.
- `GET /api/delinquency/health/cache` trả về `coverage/freshness/sources/model_versions` và cảnh báo vận hành.
- Mở `Frontend/pages/delinquency.html` để kiểm tra dropdown freshness + nút Batch Predict + metadata badge trên bảng.

Test backend nhanh:

```bash
python -m pytest Backend/tests/test_delinquency_contract_helpers.py -q
python -m pytest Backend/tests/test_delinquency_batch_api.py -q
```

Biến môi trường tùy chọn:
- `RESET_TOKEN_EXPIRE_MINUTES` (mặc định: 30)
- `FRONTEND_RESET_URL` (mặc định: http://localhost:3000/pages/reset-password.html)
- `PASSWORD_OUTBOX_PATH` (mặc định: Backend/.otp_outbox.log)
- `SMTP_ENABLED` (mặc định: false)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`, `SMTP_USE_TLS`

---

## 🔒 Bảo Mật (Security)
Dự án được thiết kế với các quy tắc bảo mật nghiêm ngặt:
- Mã hóa mật khẩu PBKDF2.
- Xác thực phiên làm việc bằng JWT (JSON Web Token).
- Hệ thống `.gitignore` chặn các file cấu hình nhạy cảm `.env` chứa thông tin kết nối database.

---
**Phát triển bởi:** [TruongVinhKiet](https://github.com/TruongVinhKiet)
*Dự án phục vụ mục đích nghiên cứu và demo giải pháp công nghệ số trong quản lý công.*
