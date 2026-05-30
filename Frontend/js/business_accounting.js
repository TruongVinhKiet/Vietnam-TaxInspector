(function () {
    const UI = window.TaxpayerUI;

    async function loadAccountingPanels() {
        const [revenue, docs] = await Promise.all([
            UI.get("/accounting/revenue"),
            UI.get("/accounting/documents"),
        ]);
        UI.panel("accounting-entry-panel", "Ghi nhan doanh thu, tai san va chung tu", "post_add", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="space-y-2">
                    <p class="text-[10px] font-bold uppercase text-slate-400">Doanh thu ngay</p>
                    <input id="acc-revenue-amount" type="number" class="w-full rounded-lg border-slate-200 text-xs" placeholder="So tien">
                    <input id="acc-revenue-desc" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Mo ta">
                    <button id="acc-revenue-btn" class="w-full py-2 rounded-lg bg-emerald-500 text-white text-[10px] font-bold">Ghi doanh thu</button>
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-bold uppercase text-slate-400">Tai san co dinh</p>
                    <input id="asset-name" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Ten tai san">
                    <input id="asset-cost" type="number" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Nguyen gia">
                    <button id="asset-btn" class="w-full py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Tinh khau hao</button>
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-bold uppercase text-slate-400">Kho chung tu</p>
                    <p class="text-sm font-black text-slate-800">${docs.documents?.length || 0} tep da luu</p>
                    <input id="doc-upload" type="file" class="w-full text-[10px] text-slate-500">
                    <button id="doc-upload-btn" class="w-full py-2 rounded-lg bg-emerald-500 text-white text-[10px] font-bold">Tai chung tu len</button>
                    <a href="${API_BASE}/taxpayer/accounting/report.xlsx" target="_blank" class="block text-center w-full py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tai Excel tong hop</a>
                    <a href="${API_BASE}/taxpayer/accounting/report.pdf" target="_blank" class="block text-center w-full py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tai PDF tom tat</a>
                </div>
            </div>
            <div class="mt-3 text-[11px] text-slate-500">Da ghi nhan ${revenue.entries?.length || 0} dong doanh thu trong nam.</div>
        `);
        document.getElementById("acc-revenue-btn").onclick = addRevenueEntry;
        document.getElementById("asset-btn").onclick = addAssetEntry;
        document.getElementById("doc-upload-btn").onclick = uploadDocument;
    }

    async function addRevenueEntry() {
        const data = await UI.post("/accounting/revenue", {
            amount: Number(UI.readValue("acc-revenue-amount", 0)),
            description: UI.readValue("acc-revenue-desc"),
            channel: "direct",
        });
        UI.toast(`Da ghi doanh thu ${UI.fmtVnd(data.entry.amount)}`);
        await loadAccountingPanels();
    }

    async function addAssetEntry() {
        const data = await UI.post("/accounting/assets", {
            asset_name: UI.readValue("asset-name", "Tai san co dinh"),
            cost: Number(UI.readValue("asset-cost", 0)),
            purchase_date: new Date().toISOString().slice(0, 10),
            useful_life_months: 36,
        });
        UI.toast(`Khau hao thang: ${UI.fmtVnd(data.depreciation.monthly_depreciation)}`);
    }

    async function uploadDocument() {
        const input = document.getElementById("doc-upload");
        const file = input?.files?.[0];
        if (!file) {
            UI.toast("Chon tep chung tu truoc.", "warn");
            return;
        }
        const form = new FormData();
        form.append("file", file);
        const res = await secureFetch(`${API_BASE}/taxpayer/accounting/documents?doc_type=evidence`, {
            method: "POST",
            body: form,
        });
        if (!res.ok) throw new Error("Khong the tai chung tu.");
        UI.toast("Da luu chung tu so hoa.");
        await loadAccountingPanels();
    }

    window.exportBook = async function exportBook(bookCode) {
        try {
            const data = await UI.get(`/accounting/books/${encodeURIComponent(bookCode)}`);
            const csv = [
                Object.keys(data.rows[0] || { book_code: bookCode }).join(","),
                ...data.rows.map((row) => Object.values(row).map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",")),
            ].join("\n");
            UI.downloadText(`${bookCode}.csv`, csv, "text/csv");
            UI.toast(`Da xuat so ${bookCode}.`);
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadAccountingPanels));
})();
