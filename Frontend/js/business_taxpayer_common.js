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

    async function boot(callback) {
        try {
            await api("/init");
        } catch (e) {
            console.warn("[TaxpayerUI] init skipped:", e);
        }
        try {
            await callback();
        } catch (e) {
            console.error(e);
            toast(e.message || "Khong the tai du lieu.", "error");
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
        boot,
    };
})();
