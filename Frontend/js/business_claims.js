(function () {
    const UI = window.TaxpayerUI;

    window.recalculateRisk = function recalculateRisk() {
        const checks = ["risk-cb-1", "risk-cb-2", "risk-cb-3"].filter((id) => document.getElementById(id)?.checked);
        const score = checks.length * 30;
        const scoreEl = document.getElementById("risk-score");
        const textEl = document.getElementById("risk-text");
        const barEl = document.getElementById("risk-bar");
        if (scoreEl) scoreEl.textContent = `${score} / 100`;
        if (barEl) {
            barEl.style.width = `${score}%`;
            barEl.className = `h-full rounded-full transition-all duration-300 ${score >= 70 ? "bg-rose-500" : score >= 40 ? "bg-amber-500" : "bg-emerald-500"}`;
        }
        if (textEl) textEl.textContent = score >= 70 ? "Trang thai: RUI RO CAO - Nen chuan bi ho so" : score >= 40 ? "Trang thai: RUI RO TRUNG BINH" : "Trang thai: RUI RO THAP - An toan";
    };

    async function loadClaims() {
        const [rights, timeline] = await Promise.all([
            UI.get("/claims/rights"),
            UI.get("/claims/timeline"),
        ]);
        UI.panel("claim-rights-panel", "Quyen cua nguoi nop thue", "shield", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                ${rights.rights.map((right) => `<div class="p-3 rounded-lg bg-slate-50 border border-slate-200 text-[11px] font-semibold text-slate-700">${UI.escapeHtml(right)}</div>`).join("")}
            </div>
            <div class="mt-3 flex flex-wrap gap-2">
                <button id="complaint-btn" class="px-3 py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">To cao sach nhieu</button>
                <button id="appointment-btn" class="px-3 py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Dat lich gap can bo thue</button>
                <span class="px-3 py-2 rounded-lg bg-emerald-50 text-emerald-700 text-[10px] font-bold">Hotline ${UI.escapeHtml(rights.hotline)}</span>
            </div>
        `);
        document.getElementById("complaint-btn").onclick = submitComplaint;
        document.getElementById("appointment-btn").onclick = bookAppointment;
        UI.panel("claim-timeline-panel", "Timeline ho so khieu nai", "timeline", `
            <div class="space-y-2">
                ${(timeline.claims || []).map((item) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between gap-3">
                        <div>
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.claim_type)} · ${UI.escapeHtml(item.decision_no || item.external_ref)}</p>
                            <p class="text-[10px] text-slate-500">${UI.escapeHtml(item.description || "")}</p>
                        </div>
                        ${UI.statusBadge(item.status)}
                    </div>
                `).join("") || `<p class="text-slate-400">Chua co ho so khieu nai.</p>`}
            </div>
        `);
    }

    window.submitAppeal = async function submitAppeal() {
        try {
            const payload = {
                claim_type: "appeal",
                decision_no: UI.readValue("appeal-case-id"),
                description: UI.readValue("appeal-desc"),
            };
            const data = await UI.post("/claims/appeal", payload);
            UI.toast(`Da gui ho so ${data.claim.external_ref}`);
            await loadClaims();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    async function submitComplaint() {
        const data = await UI.post("/claims/complaint", { description: "Phan anh sach nhieu sandbox" });
        UI.toast(`Da ghi nhan phan anh ${data.claim.external_ref}`);
        await loadClaims();
    }

    async function bookAppointment() {
        const data = await UI.post("/claims/appointment", { appointment_date: new Date().toISOString().slice(0, 10), purpose: "Tu van HKD" });
        UI.toast(`Da dat lich ${data.appointment_ref}`);
    }

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadClaims));
})();
