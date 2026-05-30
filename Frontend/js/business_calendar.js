(function () {
    const UI = window.TaxpayerUI;

    function deadlineRow(item) {
        const color = item.status === "overdue" ? "rose" : item.status === "soon" ? "amber" : item.status === "due_today" ? "rose" : "slate";
        const icon = item.status === "overdue" || item.status === "due_today" ? "error_outline" : item.status === "soon" ? "notification_important" : "schedule";
        return `
            <div class="flex gap-4 relative">
                <div class="w-9 h-9 rounded-full bg-${color}-100 border border-${color}-300 flex items-center justify-center text-${color}-700 z-10 flex-shrink-0">
                    <span class="material-symbols-outlined text-base">${icon}</span>
                </div>
                <div class="flex-1 pb-4 border-b border-slate-100">
                    <div class="flex justify-between items-start gap-3">
                        <h5 class="text-xs font-bold text-slate-800">${UI.escapeHtml(item.title)}</h5>
                        ${UI.statusBadge(item.badge)}
                    </div>
                    <p class="text-[10px] text-slate-500 mt-1">${UI.escapeHtml(item.description)}</p>
                    <div class="mt-2 text-[10px] text-slate-400 flex gap-4">
                        <span>Han cuoi: ${UI.escapeHtml(item.due_date)}</span>
                        <span>Mau: ${UI.escapeHtml(item.form_code)}</span>
                    </div>
                </div>
            </div>
        `;
    }

    async function loadCalendar() {
        const data = await UI.get(`/calendar/deadlines?year=${new Date().getFullYear()}`);
        const target = document.querySelector("main .lg\\:col-span-2 .space-y-4.pt-2");
        if (target && data.deadlines) {
            target.innerHTML = data.deadlines.map(deadlineRow).join("");
        }
        const threshold = await UI.get("/calendar/revenue-threshold");
        UI.panel("calendar-revenue-threshold-panel", "Theo doi doanh thu luy ke", "monitoring", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <p class="text-[9px] font-bold uppercase text-slate-400">Doanh thu luy ke</p>
                    <p class="text-lg font-black text-slate-800">${UI.fmtVnd(threshold.cumulative_revenue)}</p>
                </div>
                <div class="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <p class="text-[9px] font-bold uppercase text-slate-400">Nguong ke tiep</p>
                    <p class="text-lg font-black text-slate-800">${UI.fmtVnd(threshold.next_threshold)}</p>
                </div>
                <div class="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <p class="text-[9px] font-bold uppercase text-slate-400">Canh bao</p>
                    <p class="text-sm font-black text-emerald-700">${UI.escapeHtml(threshold.alert)}</p>
                </div>
            </div>
            <div class="mt-3 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div class="h-full bg-emerald-500" style="width:${Math.round((threshold.progress_ratio || 0) * 100)}%"></div>
            </div>
        `);
    }

    window.syncCalendar = async function syncCalendar() {
        try {
            const data = await UI.post("/calendar/sync", { year: new Date().getFullYear() });
            if (data.download_path) {
                window.open(`${API_BASE.replace(/\/api$/, "/api")}/taxpayer/calendar/export.ics`, "_blank");
            }
            UI.toast(data.message || "Da dong bo lich thue.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.saveNotificationSettings = async function saveNotificationSettings() {
        try {
            const days = Number(UI.readValue("alert-days", 7));
            await UI.put("/calendar/settings", {
                sms_enabled: UI.readValue("toggle-sms", true),
                email_enabled: UI.readValue("toggle-email", true),
                days_before: [days, 3, 0],
            });
            UI.toast("Da luu cau hinh nhac nho.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadCalendar));
})();
