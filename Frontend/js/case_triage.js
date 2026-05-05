const API = typeof API_BASE !== 'undefined' ? API_BASE : (window.API_BASE_URL || "http://localhost:8000/api");

function getUrgencyDot(urgency) {
    const u = String(urgency || "").toLowerCase();
    if (u === "high" || u === "cao") {
        return `<span class="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse" title="High Urgency"></span>`;
    }
    if (u === "medium" || u === "trung bình") {
        return `<span class="w-3 h-3 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" title="Medium Urgency"></span>`;
    }
    return `<span class="w-3 h-3 rounded-full bg-blue-400" title="Low Urgency"></span>`;
}

function getStatusBadge(status) {
    const s = String(status || "").toLowerCase();
    if (s.includes("pending")) return `<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-600 text-[10px] font-bold uppercase border border-slate-200">Pending</span>`;
    if (s.includes("progress")) return `<span class="px-2 py-0.5 rounded bg-blue-50 text-blue-600 text-[10px] font-bold uppercase border border-blue-200">In Progress</span>`;
    if (s.includes("escalated")) return `<span class="px-2 py-0.5 rounded bg-red-50 text-red-600 text-[10px] font-bold uppercase border border-red-200">Escalated</span>`;
    return `<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-600 text-[10px] font-bold uppercase border border-slate-200">${status}</span>`;
}

const loadBtn = document.getElementById("load");
if (loadBtn) {
    loadBtn.addEventListener("click", async () => {
        const out = document.getElementById("out");
        const statusInput = document.getElementById("status");
        
        if (out) out.innerHTML = `<tr><td colspan="6" class="px-4 py-8 text-center text-slate-400">Đang tải danh sách hàng đợi...</td></tr>`;
        
        const status = String(statusInput?.value || "").trim();
        const q = status ? `?status=${encodeURIComponent(status)}` : "";
        
        const originalText = loadBtn.innerHTML;
        loadBtn.innerHTML = `<span class="material-symbols-outlined text-[16px] animate-spin">refresh</span> Đang tải...`;
        loadBtn.disabled = true;

        try {
            const res = await secureFetch(`${API}/case-triage/queue${q}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            
            if (!data.items || data.items.length === 0) {
                if (out) out.innerHTML = `<tr><td colspan="6" class="px-4 py-8 text-center text-slate-400">Không có hồ sơ nào trong hàng đợi.</td></tr>`;
            } else {
                if (out) {
                    out.innerHTML = data.items.map(item => `
                        <tr class="hover:bg-slate-50 transition-colors">
                            <td class="px-4 py-3 border-r border-outline-variant/10 text-center">${getUrgencyDot(item.urgency_level || 'low')}</td>
                            <td class="px-4 py-3">
                                <div class="font-bold text-primary-fixed-dim">${item.case_id}</div>
                                <div class="text-xs text-slate-500 font-mono">${item.entity_id}</div>
                            </td>
                            <td class="px-4 py-3">
                                <div class="font-semibold ${item.sla_due_at ? 'text-slate-700' : 'text-slate-400'}">${item.sla_due_at ? new Date(item.sla_due_at).toLocaleDateString("vi-VN") : 'Chưa xác định'}</div>
                                <div class="text-[10px] text-slate-400 uppercase tracking-wider">${item.urgency_level || 'Low'} Mức độ</div>
                            </td>
                            <td class="px-4 py-3">
                                <div class="text-sm text-slate-700 font-medium mb-1">${item.case_type || 'Unknown'}</div>
                                ${getStatusBadge(item.status || 'pending')}
                            </td>
                            <td class="px-4 py-3 text-center">
                                <span class="inline-flex items-center justify-center px-2 py-1 bg-surface-dim/20 text-on-surface-variant font-bold text-xs rounded border border-outline-variant/30">
                                    ${Number(item.priority_score || 0).toFixed(1)}
                                </span>
                            </td>
                            <td class="px-4 py-3 text-slate-600 font-medium">${item.routing_team || 'General Team'}</td>
                        </tr>
                    `).join('');
                }
            }
        } catch (err) {
            if (out) out.innerHTML = `<tr><td colspan="6" class="px-4 py-8 text-center text-red-500">Lỗi tải queue: ${String(err)}</td></tr>`;
        } finally {
            loadBtn.innerHTML = originalText;
            loadBtn.disabled = false;
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const autoLoadBtn = document.getElementById("load");
    if (autoLoadBtn) autoLoadBtn.click();
});
