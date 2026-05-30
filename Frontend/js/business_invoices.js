(function () {
    const UI = window.TaxpayerUI;

    function invoiceRow(item) {
        const directionLabel = item.direction === "in" ? "Dau vao" : "Dau ra";
        const directionClass = item.direction === "in" ? "text-blue-600" : "text-emerald-600";
        return `
            <tr>
                <td class="px-6 py-3 font-bold ${directionClass}">${directionLabel}</td>
                <td class="px-6 py-3 font-mono font-bold">${UI.escapeHtml(item.invoice_number)}</td>
                <td class="px-6 py-3">${UI.escapeHtml(item.issue_date || "")}</td>
                <td class="px-6 py-3">${UI.escapeHtml(item.partner_name || "Chua co ten")}</td>
                <td class="px-6 py-3 font-mono">${UI.escapeHtml(item.buyer_tax_code || item.seller_tax_code || "")}</td>
                <td class="px-6 py-3 text-right font-bold">${UI.fmtVnd(item.total_amount || item.amount)}</td>
                <td class="px-6 py-3 text-center">${UI.statusBadge(item.status)}</td>
            </tr>
        `;
    }

    async function loadInvoices() {
        const [log, req] = await Promise.all([
            UI.get("/invoices/log"),
            UI.get("/invoices/einvoice-requirement"),
        ]);
        const body = document.getElementById("invoice-table-body");
        if (body && log.invoices?.length) {
            body.innerHTML = log.invoices.map(invoiceRow).join("");
        }
        UI.panel("invoice-requirement-panel", "Nguong bat buoc hoa don dien tu", "rule", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Trang thai</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml(req.requirement.label)}</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nguong tu nguyen</p>
                    <p class="text-sm font-black text-slate-800">${UI.fmtVnd(req.requirement.thresholds.voluntary_from)}</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nguong bat buoc</p>
                    <p class="text-sm font-black text-slate-800">${UI.fmtVnd(req.requirement.thresholds.mandatory_from)}</p>
                </div>
            </div>
            <p class="mt-3 text-[11px] text-slate-500">${UI.escapeHtml(req.requirement.action)}</p>
        `);
    }

    window.issueInvoice = async function issueInvoice() {
        try {
            const payload = {
                buyer_tax_code: UI.readValue("buyer-mst"),
                buyer_name: UI.readValue("buyer-name"),
                item_description: UI.readValue("item-desc"),
                vat_rate: Number(UI.readValue("item-vat", 8)),
                unit_price: Number(UI.readValue("item-price", 0)),
                quantity: Number(UI.readValue("item-qty", 1)),
            };
            const data = await UI.post("/invoices/issue", payload);
            UI.toast(`Da phat hanh ${data.invoice.invoice_number}`);
            await loadInvoices();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.scanInvoice = async function scanInvoice() {
        try {
            const seller = UI.readValue("scan-mst");
            const data = await UI.post("/invoices/scan", { seller_tax_code: seller, tax_code: seller });
            const flags = data.scan.risk_flags?.length ? data.scan.risk_flags.join(", ") : "Khong co co rui ro sandbox";
            UI.panel("invoice-scan-result-panel", "Ket qua ra soat hoa don dau vao", "fact_check", `
                <div class="flex items-center justify-between gap-3">
                    <div>
                        <p class="font-bold text-slate-800">${UI.escapeHtml(data.scan.message)}</p>
                        <p class="text-[11px] text-slate-500 mt-1">${UI.escapeHtml(flags)}</p>
                    </div>
                    ${UI.statusBadge(data.scan.status)}
                </div>
            `);
            await loadInvoices();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadInvoices));
})();
