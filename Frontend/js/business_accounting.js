(function () {
    const UI = window.TaxpayerUI;

    async function loadAccountingPanels() {
        const [revenue, docs] = await Promise.all([
            UI.get("/accounting/revenue"),
            UI.get("/accounting/documents"),
        ]);
        UI.panel("accounting-entry-panel", "Ghi nhận doanh thu, tài sản và chứng từ", "post_add", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="space-y-2">
                    <p class="text-[10px] font-bold uppercase text-slate-400">Doanh thu ngày</p>
                    <input id="acc-revenue-amount" type="number" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Số tiền">
                    <input id="acc-revenue-desc" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Mô tả">
                    <button id="acc-revenue-btn" class="w-full py-2 rounded-lg bg-emerald-500 text-white text-[10px] font-bold">Ghi doanh thu</button>
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-bold uppercase text-slate-400">Tài sản cố định</p>
                    <input id="asset-name" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Tên tài sản">
                    <input id="asset-cost" type="number" class="w-full rounded-lg border-slate-200 text-xs" placeholder="Nguyên giá">
                    <button id="asset-btn" class="w-full py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Tính khấu hao</button>
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-bold uppercase text-slate-400">Kho chứng từ</p>
                    <p class="text-sm font-black text-slate-800">${docs.documents?.length || 0} tệp đã lưu</p>
                    <input id="doc-upload" type="file" class="w-full text-[10px] text-slate-500">
                    <button id="doc-upload-btn" class="w-full py-2 rounded-lg bg-emerald-500 text-white text-[10px] font-bold">Tải chứng từ lên</button>
                    <a href="${API_BASE}/taxpayer/accounting/report.xlsx" target="_blank" class="block text-center w-full py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tải Excel tổng hợp</a>
                    <a href="${API_BASE}/taxpayer/accounting/report.pdf" target="_blank" class="block text-center w-full py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tải PDF tóm tắt</a>
                </div>
            </div>
            <div class="mt-3 text-[11px] text-slate-500">Đã ghi nhận ${revenue.entries?.length || 0} dòng doanh thu trong năm.</div>
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
        UI.toast(`Đã ghi doanh thu ${UI.fmtVnd(data.entry.amount)}`);
        await loadAccountingPanels();
    }

    async function addAssetEntry() {
        const data = await UI.post("/accounting/assets", {
            asset_name: UI.readValue("asset-name", "Tài sản cố định"),
            cost: Number(UI.readValue("asset-cost", 0)),
            purchase_date: new Date().toISOString().slice(0, 10),
            useful_life_months: 36,
        });
        UI.toast(`Khấu hao tháng: ${UI.fmtVnd(data.depreciation.monthly_depreciation)}`);
    }

    async function uploadDocument() {
        const input = document.getElementById("doc-upload");
        const file = input?.files?.[0];
        if (!file) {
            UI.toast("Chọn tệp chứng từ trước.", "warn");
            return;
        }
        const form = new FormData();
        form.append("file", file);
        const res = await secureFetch(`${API_BASE}/taxpayer/accounting/documents?doc_type=evidence`, {
            method: "POST",
            body: form,
        });
        if (!res.ok) throw new Error("Không thể tải chứng từ.");
        UI.toast("Đã lưu chứng từ số hóa.");
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
            UI.toast(`Đã xuất sổ ${bookCode}.`);
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadAccountingPanels));
})();
