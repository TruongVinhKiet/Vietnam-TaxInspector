const API = typeof API_BASE !== 'undefined' ? API_BASE : "http://localhost:8000/api";

function formatMoney(n) {
    return Number(n || 0).toLocaleString("vi-VN") + " VNĐ";
}

const loadBtn = document.getElementById("load");
if (loadBtn) {
    loadBtn.addEventListener("click", async () => {
        const out = document.getElementById("out");
        const budgetInput = document.getElementById("budget");
        
        if (out) out.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-slate-400">Đang phân tích tối ưu hóa...</td></tr>`;
        
        const budget = Number(budgetInput?.value || 400);
        const originalText = loadBtn.innerHTML;
        loadBtn.innerHTML = `<span class="material-symbols-outlined text-[16px] animate-spin">refresh</span> Đang xử lý...`;
        loadBtn.disabled = true;

        try {
            const res = await secureFetch(`${API}/audit/shortlist?budget_hours=${encodeURIComponent(budget)}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            
            // Update stats
            const bh = document.getElementById("budget-hours");
            const uh = document.getElementById("used-hours");
            const sc = document.getElementById("selected-count");
            if (bh) bh.textContent = data.budget_hours;
            if (uh) uh.textContent = data.used_hours;
            if (sc) sc.textContent = data.total;
            
            const pct = Math.min((data.used_hours / data.budget_hours) * 100, 100);
            const bar = document.getElementById("budget-progress-bar");
            if (bar) {
                bar.style.width = pct + "%";
                if (pct < 50) bar.className = "h-2 rounded-full bg-green-500";
                else if (pct < 85) bar.className = "h-2 rounded-full bg-yellow-500";
                else bar.className = "h-2 rounded-full bg-red-500";
            }

            if (!data.items || data.items.length === 0) {
                if (out) out.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-slate-400">Không tìm thấy hồ sơ nào phù hợp với ngân sách.</td></tr>`;
            } else {
                if (out) {
                    out.innerHTML = data.items.map(item => `
                        <tr class="hover:bg-slate-50 transition-colors">
                            <td class="px-4 py-3 font-semibold text-primary-container">${item.tax_code}</td>
                            <td class="px-4 py-3 text-center">
                                <span class="inline-flex items-center justify-center px-2 py-1 rounded bg-surface-dim/20 text-on-surface-variant text-xs font-bold w-12">
                                    ${Number(item.priority_score || 0).toFixed(1)}
                                </span>
                            </td>
                            <td class="px-4 py-3 text-right font-medium text-emerald-600">${formatMoney(item.expected_recovery)}</td>
                            <td class="px-4 py-3 text-center">
                                <div class="flex items-center justify-center gap-2">
                                    <div class="w-16 bg-slate-200 rounded-full h-1.5"><div class="bg-blue-500 h-1.5 rounded-full" style="width: ${Number(item.prob_recovery || 0)*100}%"></div></div>
                                    <span class="text-xs text-slate-500">${(Number(item.prob_recovery || 0)*100).toFixed(0)}%</span>
                                </div>
                            </td>
                            <td class="px-4 py-3 text-center text-rose-600 font-semibold">${Number(item.expected_effort || 0).toFixed(1)} h</td>
                        </tr>
                    `).join('');
                }
            }
        } catch (err) {
            if (out) out.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-red-500">Lỗi tải shortlist: ${String(err)}</td></tr>`;
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
