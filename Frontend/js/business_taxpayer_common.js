(function () {
    const cardClass = "bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4";

    function root() {
        return document.querySelector("main .max-w-6xl") || document.querySelector("main");
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    async function api(path, options = {}) {
        const url = `${API_BASE}/taxpayer${path}`;
        const res = await secureFetch(url, options);
        const contentType = res.headers.get("content-type") || "";
        if (!res.ok) {
            let message = `API ${res.status}`;
            try {
                const data = contentType.includes("application/json") ? await res.json() : await res.text();
                message = data.detail || data.message || data || message;
            } catch (e) {}
            throw new Error(message);
        }
        if (contentType.includes("application/json")) return res.json();
        return res;
    }

    function json(method, path, body = {}) {
        return api(path, {
            method,
            body: JSON.stringify(body),
        });
    }

    function fmtVnd(value) {
        const n = Number(value || 0);
        return n.toLocaleString("vi-VN") + " VND";
    }

    function statusBadge(status) {
        const key = String(status || "").toLowerCase();
        let cls = "bg-slate-100 text-slate-600";
        if (["success", "submitted", "accepted", "valid", "deductible", "paid"].some((x) => key.includes(x))) {
            cls = "bg-emerald-100 text-emerald-800";
        } else if (["warning", "soon", "needs", "pending", "queued", "draft"].some((x) => key.includes(x))) {
            cls = "bg-amber-100 text-amber-800";
        } else if (["overdue", "critical", "invalid", "risky", "non"].some((x) => key.includes(x))) {
            cls = "bg-rose-100 text-rose-800";
        }
        return `<span class="px-2 py-0.5 ${cls} font-bold rounded text-[9px] uppercase tracking-wider">${escapeHtml(status || "unknown")}</span>`;
    }

    function toast(message, type = "success") {
        let box = document.getElementById("taxpayer-toast");
        if (!box) {
            box = document.createElement("div");
            box.id = "taxpayer-toast";
            box.className = "fixed right-6 bottom-6 z-[9999] max-w-sm rounded-lg px-4 py-3 text-xs font-bold shadow-xl transition-all";
            document.body.appendChild(box);
        }
        box.textContent = message;
        box.className = `fixed right-6 bottom-6 z-[9999] max-w-sm rounded-lg px-4 py-3 text-xs font-bold shadow-xl transition-all ${
            type === "error" ? "bg-rose-600 text-white" : type === "warn" ? "bg-amber-500 text-white" : "bg-emerald-600 text-white"
        }`;
        window.clearTimeout(box._timer);
        box._timer = window.setTimeout(() => box.remove(), 3600);
    }

    function panel(id, title, icon, bodyHtml, options = {}) {
        const container = root();
        if (!container) return null;
        let el = document.getElementById(id);
        if (!el) {
            el = document.createElement("section");
            el.id = id;
            el.className = options.grid ? "grid grid-cols-1 lg:grid-cols-3 gap-6" : cardClass;
            if (options.prepend) container.prepend(el);
            else container.appendChild(el);
        }
        if (options.raw) {
            el.innerHTML = bodyHtml;
            return el;
        }
        el.innerHTML = `
            <div class="flex items-center justify-between gap-3">
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-emerald-500">${escapeHtml(icon)}</span>
                    <h4 class="text-xs font-black uppercase tracking-wider text-slate-700">${escapeHtml(title)}</h4>
                </div>
                ${options.action || ""}
            </div>
            <div class="text-xs text-slate-600">${bodyHtml}</div>
        `;
        return el;
    }

    function readValue(id, fallback = "") {
        const el = document.getElementById(id);
        if (!el) return fallback;
        if (el.type === "checkbox") return el.checked;
        return el.value ?? fallback;
    }

    function downloadText(filename, content, type = "text/plain") {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    const navLabels = {
        "business_dashboard.html": ["dashboard", "Tổng quan", "Tổng quan hộ kinh doanh"],
        "business_registration.html": ["app_registration", "Nhận diện & đăng ký", "Nhận diện và đăng ký thuế"],
        "business_calculator.html": ["calculate", "Tính thuế tự động", "Tính toán thuế tự động"],
        "business_calendar.html": ["calendar_month", "Lịch & deadline", "Lịch trình và hạn nộp thuế"],
        "business_invoices.html": ["receipt_long", "Hóa đơn điện tử", "Quản lý hóa đơn điện tử"],
        "business_filing.html": ["edit_document", "Kê khai & nộp thuế", "Kê khai và nộp thuế"],
        "business_debts.html": ["account_balance_wallet", "Nghĩa vụ & nợ thuế", "Tra cứu nghĩa vụ và nợ thuế"],
        "business_legal.html": ["chat", "Tra cứu & AI hỏi đáp", "Tra cứu pháp luật và hỏi đáp AI"],
        "business_growth.html": ["published_with_changes", "Thay đổi mô hình", "Phát triển và thay đổi mô hình"],
        "business_accounting.html": ["menu_book", "Kế toán & sổ sách", "Kế toán và sổ sách"],
        "business_expenses.html": ["payments", "Chi phí được trừ", "Quản lý chi phí được trừ"],
        "business_claims.html": ["balance", "Quyền lợi & khiếu nại", "Bảo vệ quyền lợi và khiếu nại"],
        "business_profile.html": ["person", "Hồ sơ", "Hồ sơ người nộp thuế"],
    };

    function currentPage() {
        return window.location.pathname.split("/").pop() || "business_dashboard.html";
    }

    function enhanceTaxpayerShell() {
        if (!currentPage().startsWith("business_")) return;
        const sidebar = document.querySelector("aside");
        const nav = document.getElementById("sidebar-nav");
        if (!sidebar || !nav) return;

        document.body.classList.add("taxpayer-shell-v2");
        const subtitle = document.getElementById("sidebar-subtitle");
        if (subtitle) subtitle.textContent = "Người nộp thuế";

        nav.querySelectorAll("a[data-page]").forEach((link) => {
            const page = link.getAttribute("data-page");
            const item = navLabels[page];
            if (!item) return;
            const [icon, label, title] = item;
            link.setAttribute("data-title", title);
            const iconEl = link.querySelector(".material-symbols-outlined");
            const labelEl = link.querySelector("span:last-child");
            if (iconEl) iconEl.textContent = icon;
            if (labelEl) labelEl.textContent = label;
        });

        const sectionLabel = nav.querySelector("div");
        if (sectionLabel) sectionLabel.textContent = "Nhóm nghiệp vụ";

        const currentMeta = navLabels[currentPage()];
        if (currentMeta) document.title = `TaxInspector - ${currentMeta[2]}`;

        if (!document.getElementById("taxpayer-shell-toggle")) {
            const toggle = document.createElement("button");
            toggle.id = "taxpayer-shell-toggle";
            toggle.type = "button";
            toggle.className = "taxpayer-shell-toggle";
            toggle.setAttribute("aria-label", "Mở menu taxpayer");
            toggle.innerHTML = '<span class="material-symbols-outlined">menu</span>';
            document.body.appendChild(toggle);

            const backdrop = document.createElement("button");
            backdrop.id = "taxpayer-shell-backdrop";
            backdrop.type = "button";
            backdrop.className = "taxpayer-shell-backdrop";
            backdrop.setAttribute("aria-label", "Đóng menu taxpayer");
            document.body.appendChild(backdrop);

            const close = () => document.body.classList.remove("taxpayer-sidebar-open");
            toggle.addEventListener("click", () => document.body.classList.toggle("taxpayer-sidebar-open"));
            backdrop.addEventListener("click", close);
            nav.addEventListener("click", (event) => {
                if (event.target.closest("a")) close();
            });
        }
    }

    async function boot(callback) {
        enhanceTaxpayerShell();
        try {
            await api("/init");
        } catch (e) {
            console.warn("[TaxpayerUI] init skipped:", e);
        }
        try {
            await callback();
        } catch (e) {
            console.error(e);
            toast(e.message || "Không thể tải dữ liệu.", "error");
        }
    }

    window.TaxpayerUI = {
        api,
        get: (path) => api(path),
        post: (path, body) => json("POST", path, body),
        put: (path, body) => json("PUT", path, body),
        patch: (path, body) => json("PATCH", path, body),
        fmtVnd,
        statusBadge,
        toast,
        panel,
        readValue,
        downloadText,
        escapeHtml,
        enhanceTaxpayerShell,
        boot,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", enhanceTaxpayerShell, { once: true });
    } else {
        enhanceTaxpayerShell();
    }
})();
