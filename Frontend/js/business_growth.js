(function () {
    const UI = window.TaxpayerUI;

    async function loadGrowth() {
        const data = await UI.get("/growth/readiness");
        UI.panel("growth-readiness-panel", "San sang thay doi quy mo", "trending_up", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nhom HKD</p>
                    <p class="font-black text-slate-800">${UI.escapeHtml(data.household_group.label)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Hoa don dien tu</p>
                    <p class="font-black text-slate-800">${UI.escapeHtml(data.einvoice.status)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Ton kho dau ky</p>
                    <p class="font-black text-slate-800">${data.inventory_opening_required ? "Bat buoc" : "Chua bat buoc"}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">TNHH uoc tinh</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(data.llc_comparison.llc_total_tax)}</p>
                </div>
            </div>
        `);
        UI.panel("growth-extra-events-panel", "Thay doi dia diem, nganh nghe va ton kho", "edit_location_alt", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button id="relocate-btn" class="py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Thong bao doi dia diem</button>
                <button id="industry-btn" class="py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Bo sung nganh nghe</button>
                <button id="inventory-btn" class="py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Khai ton kho 31/12</button>
            </div>
            <div id="growth-event-result" class="mt-3 text-[11px] text-slate-500"></div>
        `);
        document.getElementById("relocate-btn").onclick = () => submitEvent("relocate", { address: "Dia diem moi sandbox" });
        document.getElementById("industry-btn").onclick = () => submitEvent("industry_change", { industry: "service" });
        document.getElementById("inventory-btn").onclick = () => submitEvent("opening_inventory", { inventory_date: "2025-12-31", note: "Bang ke ton kho dau ky" });
    }

    async function submitEvent(event_type, payload = {}) {
        const data = await UI.post("/growth/event", { event_type, ...payload });
        const result = document.getElementById("growth-event-result");
        if (result) result.textContent = `Da ghi nhan: ${data.event.external_ref}`;
        UI.toast("Da gui yeu cau sandbox.");
    }

    window.requestUpgrade = async function requestUpgrade() {
        try {
            await submitEvent("upgrade_to_llc", { target_model: "limited_liability_company" });
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.requestClosure = async function requestClosure() {
        try {
            const type = UI.readValue("stop-type", "suspend");
            const start = UI.readValue("stop-start") || new Date().toISOString().slice(0, 10);
            const end = UI.readValue("stop-end") || null;
            await submitEvent(type === "close" ? "closure" : "temporary_suspension", {
                start_date: start,
                end_date: end,
            });
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadGrowth));
})();
