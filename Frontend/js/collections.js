const API = typeof API_BASE !== 'undefined' ? API_BASE : (window.API_BASE_URL || "http://localhost:8000/api");

function formatMoney(n) {
    return Number(n || 0).toLocaleString("vi-VN");
}

function getActionBadge(action) {
    const a = (action || "").toLowerCase();
    if (a.includes('seize') || a.includes('cưỡng chế')) {
        return `<span class="px-2.5 py-1 bg-rose-50 text-rose-600 border border-rose-200 text-xs font-bold rounded flex w-max items-center gap-1"><span class="material-symbols-outlined text-[14px]">gavel</span> ${action}</span>`;
    }
    if (a.includes('letter') || a.includes('công văn')) {
        return `<span class="px-2.5 py-1 bg-amber-50 text-amber-600 border border-amber-200 text-xs font-bold rounded flex w-max items-center gap-1"><span class="material-symbols-outlined text-[14px]">mail</span> ${action}</span>`;
    }
    return `<span class="px-2.5 py-1 bg-emerald-50 text-emerald-600 border border-emerald-200 text-xs font-bold rounded flex w-max items-center gap-1"><span class="material-symbols-outlined text-[14px]">call</span> ${action}</span>`;
}

function getConfidenceBar(conf) {
    const c = String(conf || "").toLowerCase();
    let pct = 50;
    let color = "bg-slate-400";
    if (c === "high" || c === "cao") { pct = 90; color = "bg-green-500"; }
    else if (c === "medium" || c === "trung bình") { pct = 60; color = "bg-yellow-500"; }
    else if (c === "low" || c === "thấp") { pct = 30; color = "bg-red-500"; }
    else { pct = Number(conf) || 50; color = "bg-blue-500"; }
    
    return `
        <div class="flex items-center gap-2">
            <div class="w-16 bg-slate-200 rounded-full h-1.5"><div class="${color} h-1.5 rounded-full" style="width: ${pct}%"></div></div>
            <span class="text-xs text-slate-500 uppercase font-semibold">${conf}</span>
        </div>
    `;
}

const loadBtn = document.getElementById("load");
if (loadBtn) {
    loadBtn.addEventListener("click", async () => {
        const out = document.getElementById("out");
        const taxInput = document.getElementById("tax");
        
        if (out) out.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-slate-400">Đang xuất khuyến nghị NBA...</td></tr>`;
        
        const tax = String(taxInput?.value || "").trim();
        const q = tax ? `?tax_code=${encodeURIComponent(tax)}` : "";
        
        const originalText = loadBtn.innerHTML;
        loadBtn.innerHTML = `<span class="material-symbols-outlined text-[16px] animate-spin">refresh</span> Đang tải...`;
        loadBtn.disabled = true;

        try {
            const res = await secureFetch(`${API}/collections/next-best-action${q}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            
            if (!data.items || data.items.length === 0) {
                if (out) out.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-slate-400">Không tìm thấy khuyến nghị nào phù hợp.</td></tr>`;
            } else {
                if (out) {
                    out.innerHTML = data.items.map(item => `
                        <tr class="hover:bg-slate-50 transition-colors">
                            <td class="px-4 py-3 font-semibold text-primary-container">${item.tax_code}</td>
                            <td class="px-4 py-3">${getActionBadge(item.recommended_action || "Theo dõi thêm")}</td>
                            <td class="px-4 py-3 text-right">
                                <span class="inline-flex items-center px-1.5 py-0.5 rounded-md bg-green-50 text-green-700 text-xs font-bold font-mono">
                                    +${Number(item.uplift_pp || 0).toFixed(2)}%
                                </span>
                            </td>
                            <td class="px-4 py-3 text-right font-medium text-emerald-600">${formatMoney(item.expected_collection)}</td>
                            <td class="px-4 py-3">${getConfidenceBar(item.confidence || "Medium")}</td>
                        </tr>
                    `).join('');
                }
            }
        } catch (err) {
            if (out) out.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Lỗi xuất NBA: ${String(err)}</td></tr>`;
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
