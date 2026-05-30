(function () {
    const UI = window.TaxpayerUI;

    function expenseRow(item) {
        return `
            <tr>
                <td class="px-6 py-3">${UI.escapeHtml(item.description || item.category)}</td>
                <td class="px-6 py-3 text-right font-mono font-bold">${UI.fmtVnd(item.amount)}</td>
                <td class="px-6 py-3">${UI.escapeHtml(item.payment_method)}</td>
                <td class="px-6 py-3 text-center">${UI.statusBadge(item.deductible_status)}</td>
            </tr>
        `;
    }

    async function loadExpenses() {
        const [summary, cases] = await Promise.all([
            UI.get("/expenses/summary"),
            UI.get("/expenses/no-invoice-cases"),
        ]);
        UI.panel("expense-rules-panel", "5 truong hop khong can hoa don nhung can bang ke", "receipt", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                ${cases.cases.map((item) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                        <p class="font-bold text-slate-800">${UI.escapeHtml(item.label)}</p>
                        <p class="text-[10px] text-slate-500 mt-1">${item.required_evidence.map(UI.escapeHtml).join(" · ")}</p>
                    </div>
                `).join("")}
            </div>
        `);
        UI.panel("expense-summary-panel", "Tong hop chi phi va canh bao chung tu", "analytics", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${(summary.summary || []).map((row) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                        <p class="text-[9px] uppercase font-bold text-slate-400">${UI.escapeHtml(row.deductible_status || "unknown")}</p>
                        <p class="font-black text-slate-800">${UI.fmtVnd(row.total)}</p>
                        <p class="text-[10px] text-slate-500">${row.count} khoan</p>
                    </div>
                `).join("") || `<p class="text-slate-400">Chua co du lieu chi phi.</p>`}
            </div>
            <button id="bhxh-btn" class="mt-3 px-3 py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tinh BHXH uoc tinh</button>
            <div id="bhxh-result" class="mt-2 text-[11px] text-slate-500"></div>
        `);
        document.getElementById("bhxh-btn").onclick = async () => {
            const data = await UI.post("/expenses/bhxh", { salary_base: 6000000, employees: 2, owner_voluntary: 1000000 });
            document.getElementById("bhxh-result").textContent = `BHXH DN uoc tinh: ${UI.fmtVnd(data.employer_bhxh_estimate)}; chu ho toi da: ${UI.fmtVnd(data.owner_voluntary_deduction_cap)}`;
        };
    }

    window.addExpense = async function addExpense() {
        try {
            const method = UI.readValue("exp-method", "bank_transfer");
            const payload = {
                description: UI.readValue("exp-desc"),
                amount: Number(UI.readValue("exp-amount", 0)),
                payment_method: method === "bank" ? "bank_transfer" : "cash",
                has_invoice: true,
                category: "operating",
            };
            const data = await UI.post("/accounting/expense", payload);
            const body = document.getElementById("expenses-table-body");
            if (body) body.insertAdjacentHTML("afterbegin", expenseRow(data.entry));
            UI.toast(data.evaluation.deductible ? "Chi phi da duoc ghi nhan." : "Chi phi bi canh bao khong duoc tru.", data.evaluation.deductible ? "success" : "warn");
            await loadExpenses();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadExpenses));
})();
