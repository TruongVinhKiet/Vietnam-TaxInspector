(function () {
    const UI = window.TaxpayerUI;

    async function loadDashboard() {
        const [summary, threshold, debt, invoices] = await Promise.all([
            UI.get("/profile/summary"),
            UI.get("/calendar/revenue-threshold"),
            UI.get("/debts/summary"),
            UI.get("/invoices/log"),
        ]);
        UI.panel("dashboard-live-taxpayer-panel", "Tong quan backend taxpayer nhom 3-11", "dashboard_customize", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nhom HKD</p>
                    <p class="font-black text-slate-800">${UI.escapeHtml(summary.profile.group_info.label)}</p>
                    <p class="text-[10px] text-slate-500">${UI.escapeHtml(summary.profile.group_info.threshold_label)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Doanh thu luy ke</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(threshold.cumulative_revenue)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">No thue</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(debt.total_debt)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Hoa don</p>
                    <p class="font-black text-slate-800">${invoices.invoices?.length || 0}</p>
                </div>
            </div>
            <div class="mt-3 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div class="h-full bg-emerald-500" style="width:${Math.round((threshold.progress_ratio || 0) * 100)}%"></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-[10px]">
                ${summary.next_deadlines.map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><b>${UI.escapeHtml(item.due_date)}</b><br>${UI.escapeHtml(item.title)}</div>`).join("")}
            </div>
        `, { prepend: true });
    }

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadDashboard));
})();
