# TaxInspector Enterprise Multi-Agent Architecture
**Date**: April 2026

Tài liệu này ghi chú toàn bộ kiến trúc lõi của Hệ thống **Trợ lý Thuế Đa năng (Multi-Agent AI)** được xây dựng trực tiếp vào nền tảng TaxInspector. Tài liệu dành cho các Developer kế nhiệm nắm bắt luồng dữ liệu (Data Pipeline), mô hình Machine Learning, và cấu trúc Frontend để bảo trì và tiếp tục nâng cấp.

---

## 1. Tổng quan Hệ thống (System Overview)
Hệ thống AI không chỉ là một Chatbot thông thường. Đây là một Orchestration Multi-Agent (Hệ thống điều phối đa đặc vụ) với khả năng tiếp nhận yêu cầu, **lập kế hoạch (DAG)**, phân rã công việc cho các node AI chuyên biệt, đối chiếu luật thuế, và trả về dữ liệu đa phương tiện (Biểu đồ, Báo cáo rủi ro).

### Cơ chế hoạt động:
1. **Frontend (Voice/Text) -> Backend API (`/api/tax-agent/chat/stream`)**
2. **Intent Classification & Entity Recognition:** Nhận diện ý định (Legal, Analytics, Fraud) và bóc tách MST/Doanh nghiệp.
3. **Orchestrator Planner:** Tạo Directed Acyclic Graph (DAG) quy định trình tự gọi các Sub-Agent.
4. **Execution & RAG:** Vector Search trong CSDL Luật Thuế và GNN/Isolation Forest (nếu hỏi về rủi ro).
5. **Grounded Synthesis (LLM):** LLM tổng hợp các sự kiện thu thập được thành câu trả lời tự nhiên.
6. **Telemetry & Compliance Gate:** Chặn các Prompt Injection (Adversarial attacks) và đo đạc độ trễ trước khi push SSE stream về Client.

---

## 2. Kiến trúc Backend M-Agent (Phases 1-6)

### A. Sub-Agents & Orchestrator
- **`tax_agent_orchestrator.py`**: Trình điều phối trung tâm. Nhận Context, tạo `ExecutionPlan`, và gọi tuần tự các Agent.
- **`tax_agent_legal.py`**: Node xử lý RAG. Sử dụng Sentence-transformers truy xuất văn bản pháp quy từ File JSONL (hoặc Vector DB).
- **`tax_agent_analytics.py`**: Node chuyên biệt phân tích số liệu nợ đọng, thuế TNDN/VAT bằng SQL nội suy hoặc DataFrames.
- **`tax_agent_investigation.py`**: Node kết nối trực tiếp với GNN model báo cáo các mạng lưới gian lận (Ghost node).

### B. Môi trường LLM cục bộ (Local Fine-Tuning)
Thay vì phụ thuộc vào GPT-4/Claude, hệ thống được trang bị engine On-premise thông qua kiến trúc **LoRA (Low-Rank Adaptation)**:
- **`tax_agent_llm_model.py`**: Load mô hình `Qwen/Qwen2.5-0.5B-Instruct` qua `transformers`. Cơ chế Auto-Upgrade: Tự động dùng Template cứng (Rule-based) nếu không có GPU/Adapter, và *Kích hoạt Suy Luận Fine-tunned (LoRA)* nếu thư mục `data/models/tax_llm_lora` tồn tại.
- **`tax_agent_llm_data_pipeline.py` & `run_llm_training.py`**: Engine tự động thu gom nhật ký chat (Audit Logs), lọc dữ liệu đủ KPI (Golden datasets), generate augmentation bằng thư viện NLP cục bộ và ghi ra file ShareGPT dạng `.jsonl`. Sau đó Huggingface Trainer sẽ huấn luyện và xuất adapter cho model SFT.

### C. Governance & Red-Team (Bảo mật AI)
- **`tax_agent_telemetry.py`**: Ghi chép 100% Request/Response vào SQlite tables (`agent_entity_memory`, `agent_quality_metrics`).
- **`tax_agent_evaluator.py`**: Benchmarking suite tích hợp `RedTeam`. 16 test cases tự động để đánh giá mức độ bị bẻ khóa (Adversarial attacks, Jailbreak).

---

## 3. Kiến trúc Frontend & Giao diện (UI/UX)
Triết lý thiết kế Frontend là **Immersive AI (Tương tác Toàn hình)** và **Global Floating**.

### A. Floating Widget (`js/agent_widget.js`)
- Một Widget Avatar AI lơ lửng, theo quy chuẩn ở góc phải dưới (`bottom:30px; right:30px`) được nạp tự động (Global Injection) vào tất cả `*.html` trong thư mục `pages/`.
- Cung cấp hiệu ứng Sóng lan (Splash Transition) - Biến màn hình dashboard mượt mà thành màn hình Agent.
- Cơ chế giải quyết Bug White Screen (bfcache) khi dùng phím Back: Bắt sự kiện `pageshow` để dọn dẹp overlay dư thừa.

### B. Màn hình Immersive AI (`pages/agent.html` & `css/agent.css`)
- Không gian 100% Full-screen, lược bỏ Sidebar và Header truyền thống nhường chỗ cho Chart và Data Cards.
- **Web Speech API**: Hệ thống tích hợp Native Voice Input (Sóng âm Siri bằng CSS Keyframes). Tự nhận diện tiếng Việt.
- **Status Indicator**: Đèn LED giao diện phản hồi sáng lập lòe khi AI đang đọc Luật (Pháp Lý) hoặc Kiểm tra đồ thị (Không gian Mạng) thông qua `agent_ui.js`.
- Bắt kết nối SSE fetch (Server-Sent Events) để in thông điệp giả lập stream.

### C. Wardrobe Switcher (Hệ thống Tủ đồ Đa chiều)
Một tính năng tinh tế trong UX giúp đổi Theme trang phục cho Trợ lý Ảo:
- Modal Cài đặt Glassmorphism tại `agent.html`.
- Cơ chế đồng bộ hóa: Lưu trữ avatar (`ai_avatar.png`, `avatar_satin.png`, v.v.) vào `localStorage('taxAgentTheme')`.
- Thay biến lập tức `img src` cho tất cả bong bóng chat (chat bubble), header và Dashboard Widget. Không cần Load lại Backend.

---

## Tổng kết
TaxInspector Multi-Agent là một tuyệt tác được thiết kế không chỉ để show-off mà còn để triển khai thực tiễn (Enterprise-grade). Nó đã hoàn thiện đầy đủ luồng từ: Mắt (Voice/UI) -> Não bộ (DAG Planner/Sub-Agents) -> Trí nhớ (RAG/Entity DB) -> Học tập (LoRA Fine-tuning). Dev tiếp quản có thể dễ dàng cắm (Plug & Play) thêm Sub-agent mới vào `Orchestrator` hoặc đổi LLM model khác tại `tax_agent_llm_model.py`.
