# Hệ Thống Giám Sát Thuế - TaxInspector

TaxInspector là một dự án ứng dụng trí tuệ nhân tạo (Machine Learning) và trực quan hóa dữ liệu (Data Visualization) để hỗ trợ Cơ quan Thuế phát hiện hành vi gian lận (Fraud Detection), truy xuất chuỗi hóa đơn rủi ro (VAT Network) và cảnh báo nợ đọng (Delinquency Prediction).

## Yêu cầu Hệ thống (Prerequisites)
- **Python 3.9+** (Dành cho Backend)
- **PostgreSQL 14+** (Dành cho Database)
- **Trình duyệt Web hiện đại** (Chrome/Edge/Firefox)

---

## 1. Khởi động Cơ sở Dữ liệu (Database)
Nếu bạn đã khởi chạy file `.sql` ban đầu, bạn chỉ cần đảm bảo **Dịch vụ PostgreSQL đang chạy** (có thể check qua _pgAdmin_ hoặc ứng dụng _Services_ trên Windows).

*Thông số mặc định trong dự án:*
- Host/Port: `localhost:5432`
- User/Pass: `postgres` / `Kiet2004`
- Database: `TaxInspector`

---

## 2. Khởi động Backend (FastAPI Server)

Backend chịu trách nhiệm cung cấp API Machine Learning, truy xuất Cơ sở dữ liệu và xử lý bảo mật JWT Auth.

**Mở Terminal (khuyên dùng PowerShell hoặc CMD), di chuyển vào đúng thư mục Backend và chạy lệnh sau:**

```powershell
# 1. Di chuyển vào thư mục Backend
cd e:\TaxInspector\Backend

# 2. Kích hoạt môi trường ảo (Virtual Environment)
.\venv\Scripts\activate

# 3. Chạy Server FastAPI với cổng 8000
uvicorn app.main:app --reload --port 8000
```
> **Trạng thái thành công:** Bạn sẽ thấy dòng chữ xanh lá hiển thị `Application startup complete.` Hãy thu nhỏ cửa sổ này lại (đừng tắt) để server tiếp tục chạy ngầm.

---

## 3. Khởi động Frontend (Web Interface)

Giao diện Web được build nguyên bản bằng CSS/JS thuần tốc độ cao. Bạn cần chạy một HTTP Server đơn giản để tránh các lỗi dính dáng tới tính năng "CORS - Bảo mật tên miền chép file".

**Mở thêm 1 cửa sổ Terminal MỚI, di chuyển vào thư mục Frontend và chạy lệnh sau:**

```powershell
# 1. Di chuyển vào thư mục Frontend
cd e:\TaxInspector\Frontend

# 2. Khởi tạo một Web-Server nhẹ tại cổng 3000
python -m http.server 3000
```
> **Trạng thái thành công:** Giao diện Command prompt sẽ báo đang Serving ở cổng 3000. Bạn cũng thu nhỏ (đừng tắt) cửa sổ này lại.

---

## 4. Truy cập và Trải Nghiệm hệ thống

Sau khi cả 2 Terminal (Backend và Frontend) đều đang báo trạng thái chạy (Running):
1. Mở trình duyệt bất kỳ.
2. Truy cập vào địa chỉ: **[http://localhost:3000](http://localhost:3000)**
3. Giao diện Đăng nhập cán bộ sẽ xuất hiện.

### Gợi ý Account Test:
- Bấm vào Tab **Đăng ký**, tạo một tài khoản (Ví dụ: Mã số `VTI-8888`, Mật khẩu `tax12345`).
- Sử dụng tài khoản vừa tạo để **Đăng nhập** và truy cập vào Dashboard hệ thống!
