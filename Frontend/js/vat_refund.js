const API_BASE_URL = typeof API_BASE !== 'undefined' ? API_BASE : "http://localhost:8000/api";

function formatMoney(n) {
    return Number(n || 0).toLocaleString("vi-VN");
}

function getStatusBadge(status) {
    const s = String(status).toLowerCase();
    if (s.includes("submitted")) {
        return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-blue-50 text-blue-600 text-[10px] font-bold uppercase border border-blue-200"><span class="material-symbols-outlined text-[12px]">send</span> Đã đệ trình</span>`;
    }
    if (s.includes("approved")) {
        return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-emerald-50 text-emerald-600 text-[10px] font-bold uppercase border border-emerald-200"><span class="material-symbols-outlined text-[12px]">check_circle</span> Đã duyệt</span>`;
    }
    if (s.includes("rejected")) {
        return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-rose-50 text-rose-600 text-[10px] font-bold uppercase border border-rose-200"><span class="material-symbols-outlined text-[12px]">cancel</span> Từ chối</span>`;
    }
    return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-100 text-slate-600 text-[10px] font-bold uppercase border border-slate-200">${status}</span>`;
}

function getRiskBadge(riskScore, riskLevel) {
    const score = Number(riskScore || 0).toFixed(1);
    const lvl = String(riskLevel).toLowerCase();
    
    let colorClass = "bg-slate-100 text-slate-600 border-slate-200";
    if (lvl === "high") colorClass = "bg-rose-50 text-rose-600 border-rose-200";
    else if (lvl === "medium") colorClass = "bg-amber-50 text-amber-600 border-amber-200";
    else if (lvl === "low") colorClass = "bg-emerald-50 text-emerald-600 border-emerald-200";
    
    return `<span class="inline-flex items-center justify-center px-2 py-0.5 rounded ${colorClass} text-xs font-bold border">${score} - ${String(riskLevel).toUpperCase()}</span>`;
}

async function loadCases() {
    const taxCode = String(document.getElementById("tax-code-input")?.value || "").trim();
    const params = new URLSearchParams();
    if (taxCode) params.set("tax_code", taxCode);
    params.set("limit", "100");
    const res = await secureFetch(`${API_BASE_URL}/vat-refund/cases?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

async function loadRisk(caseId) {
    const res = await secureFetch(`${API_BASE_URL}/vat-refund/cases/${encodeURIComponent(caseId)}/risk`);
    if (!res.ok) return null;
    return res.json();
}

async function render() {
    const tbody = document.querySelector("#cases-table tbody");
    const loadBtn = document.getElementById("load-cases-btn");
    if (!tbody || !loadBtn) return;
    
    const originalText = loadBtn.innerHTML;
    loadBtn.innerHTML = `<span class="material-symbols-outlined text-[16px] animate-spin">refresh</span> Đang quét...`;
    loadBtn.disabled = true;
    
    tbody.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-slate-400">Đang phân tích rủi ro hoàn thuế...</td></tr>`;
    
    try {
        const payload = await loadCases();
        const items = Array.isArray(payload?.items) ? payload.items : [];
        if (!items.length) {
            tbody.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-slate-400">Không có hồ sơ.</td></tr>`;
            return;
        }
        tbody.innerHTML = "";
        for (const item of items) {
            const risk = await loadRisk(item.case_id);
            const tr = document.createElement("tr");
            tr.className = "hover:bg-slate-50 transition-colors";
            
            const riskHtml = risk?.available ? getRiskBadge(risk.risk_score, risk.risk_level) : '<span class="text-xs text-slate-400 italic">Chưa có điểm</span>';
            
            tr.innerHTML = `
                <td class="px-4 py-3">
                    <div class="font-bold text-primary-fixed-dim">${item.case_id || ""}</div>
                    <div class="text-xs text-slate-500 font-mono">MST: ${item.tax_code || ""}</div>
                </td>
                <td class="px-4 py-3 text-center font-medium">${item.period || ""}</td>
                <td class="px-4 py-3 text-right font-medium text-emerald-600">${formatMoney(item.requested_amount)}</td>
                <td class="px-4 py-3 text-center">${getStatusBadge(item.status || "")}</td>
                <td class="px-4 py-3 text-center">${riskHtml}</td>
            `;
            tbody.appendChild(tr);
        }
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Lỗi tải dữ liệu: ${String(err)}</td></tr>`;
    } finally {
        loadBtn.innerHTML = originalText;
        loadBtn.disabled = false;
    }
}

document.getElementById("load-cases-btn")?.addEventListener("click", render);

document.addEventListener("DOMContentLoaded", () => {
    const loadBtn = document.getElementById("load-cases-btn");
    if (loadBtn) loadBtn.click();
});
