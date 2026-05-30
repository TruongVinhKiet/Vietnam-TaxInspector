(function () {
    const UI = window.TaxpayerUI;
    let lastFilingId = null;
    let lastPaymentRef = null;

    function currentDraftPayload(extra = {}) {
        const revenue = Number(UI.readValue("file-gtgt-rev", 0));
        return {
            revenue,
            gtgt_revenue: revenue,
            tncn_revenue: Number(UI.readValue("file-tncn-rev", revenue)),
            period: "2026-Q1",
            form_code: "01/CNKD",
            industry: "commerce",
            ...extra,
        };
    }

    async function ensureDraft(extra = {}) {
        const data = await UI.post("/filings/draft", currentDraftPayload(extra));
        lastFilingId = data.filing.id;
        updateTaxNumbers(data.filing);
        return data.filing;
    }

    function updateTaxNumbers(filing) {
        const gtgt = document.getElementById("file-gtgt-val");
        const tncn = document.getElementById("file-tncn-val");
        const total = document.getElementById("file-total-val");
        if (gtgt) gtgt.textContent = UI.fmtVnd(filing.gtgt_tax);
        if (tncn) tncn.textContent = UI.fmtVnd(filing.tncn_tax);
        if (total) total.textContent = UI.fmtVnd(filing.total_tax);
        const payAmount = document.getElementById("pay-amount");
        if (payAmount) payAmount.value = Math.round(Number(filing.total_tax || 0));
    }

    async function loadFilings() {
        const data = await UI.get("/filings/status");
        const rows = data.filings || [];
        UI.panel("filing-status-panel", "Trang thai ho so va to khai bo sung", "folder_managed", `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400">
                        <tr><th class="py-2">Mau</th><th>Ky</th><th>Loai</th><th>So thue</th><th>Trang thai</th></tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100">
                        ${rows.slice(0, 6).map((item) => `
                            <tr>
                                <td class="py-2 font-mono font-bold">${UI.escapeHtml(item.form_code)}</td>
                                <td>${UI.escapeHtml(item.period)}</td>
                                <td>${UI.escapeHtml(item.filing_type)}</td>
                                <td>${UI.fmtVnd(item.total_tax)}</td>
                                <td>${UI.statusBadge(item.status)}</td>
                            </tr>
                        `).join("") || `<tr><td colspan="5" class="py-3 text-slate-400">Chua co to khai nao.</td></tr>`}
                    </tbody>
                </table>
            </div>
            <button id="filing-amend-btn" class="mt-3 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-[10px] font-bold">
                Tao to khai bo sung tu ban moi nhat
            </button>
        `);
        const btn = document.getElementById("filing-amend-btn");
        if (btn) btn.onclick = amendLatest;
    }

    async function amendLatest() {
        try {
            const filing = await ensureDraft({ filing_type: "amendment", idempotency_key: `amend-${Date.now()}` });
            UI.toast(`Da tao to khai bo sung #${filing.id}`);
            await loadFilings();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    }

    window.exportXml = async function exportXml() {
        try {
            const filing = await ensureDraft();
            window.open(`${API_BASE}/taxpayer/filings/${filing.id}/xml`, "_blank");
            UI.toast("Da tao file XML to khai.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.signFiling = async function signFiling() {
        try {
            const filing = await ensureDraft();
            const data = await UI.post(`/filings/${filing.id}/submit`, { signature: "sandbox-signature" });
            UI.toast(data.gateway.message || "Da nop to khai.");
            await loadFilings();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.generateQr = async function generateQr() {
        try {
            const amount = Number(UI.readValue("pay-amount", 0));
            const data = await UI.post("/filings/payment-qr", { filing_id: lastFilingId, period: "2026-Q1", amount });
            lastPaymentRef = data.qr.payment_ref;
            const qrBox = document.getElementById("qr-box");
            if (qrBox) {
                qrBox.classList.remove("hidden");
                const note = qrBox.querySelector("p");
                if (note) note.textContent = data.qr.qr_payload;
            }
            UI.toast("Da tao QR thanh toan sandbox.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.confirmPayment = async function confirmPayment() {
        try {
            const data = await UI.post("/filings/payment-confirm", { payment_ref: lastPaymentRef });
            UI.toast(data.idempotent ? "Thanh toan da duoc xac nhan truoc do." : "Da xac nhan thanh toan.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(async () => {
        await ensureDraft();
        await loadFilings();
    }));
})();
