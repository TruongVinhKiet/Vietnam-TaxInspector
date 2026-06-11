(function () {
    const UI = window.TaxpayerUI;

    function expenseRow(item) {
        return `
            <tr>
                <td class="px-6 py-3">${UI.escapeHtml(item.description || item.category)}</td>
                <td class="px-6 py-3 text-right font-mono font-bold">${UI.fmtVnd(item.amount)}</td>
                <td class="px-6 py-3">${UI.escapeHtml(item.payment_method === 'bank_transfer' ? 'Chuyển khoản' : 'Tiền mặt')}</td>
                <td class="px-6 py-3 text-center">${UI.statusBadge(item.deductible_status)}</td>
            </tr>
        `;
    }

    async function loadExpenses() {
        const [summary, cases] = await Promise.all([
            UI.get("/expenses/summary"),
            UI.get("/expenses/no-invoice-cases"),
        ]);
        UI.panel("expense-rules-panel", "5 trường hợp không cần hóa đơn nhưng cần bảng kê (Mẫu 01/TNDN)", "receipt", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                ${cases.cases.map((item) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                        <p class="font-bold text-slate-800">${UI.escapeHtml(item.label)}</p>
                        <p class="text-[10px] text-slate-500 mt-1">${item.required_evidence.map(UI.escapeHtml).join(" · ")}</p>
                    </div>
                `).join("")}
            </div>
        `);
        UI.panel("expense-summary-panel", "Tổng hợp chi phí và Cảnh báo chứng từ", "analytics", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${(summary.summary || []).map((row) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                        <p class="text-[9px] uppercase font-bold text-slate-400">${UI.escapeHtml(row.deductible_status === 'deductible' ? 'Được trừ' : row.deductible_status === 'non_deductible' ? 'Không được trừ' : 'Chưa phân loại')}</p>
                        <p class="font-black text-slate-800">${UI.fmtVnd(row.total)}</p>
                        <p class="text-[10px] text-slate-500">${row.count} khoản</p>
                    </div>
                `).join("") || `<p class="text-slate-400">Chưa có dữ liệu chi phí.</p>`}
            </div>
            <button id="bhxh-btn" class="mt-3 px-3 py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tính BHXH ước tính</button>
            <div id="bhxh-result" class="mt-2 text-[11px] text-slate-500"></div>
        `);
        document.getElementById("bhxh-btn").onclick = async () => {
            const data = await UI.post("/expenses/bhxh", { salary_base: 6000000, employees: 2, owner_voluntary: 1000000 });
            document.getElementById("bhxh-result").textContent = `BHXH DN ước tính: ${UI.fmtVnd(data.employer_bhxh_estimate)}; chủ hộ tối đa: ${UI.fmtVnd(data.owner_voluntary_deduction_cap)}`;
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
            UI.toast(data.evaluation.deductible ? "Chi phí đã được ghi nhận." : "Chi phí bị cảnh báo không được trừ.", data.evaluation.deductible ? "success" : "warn");
            await loadExpenses();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.runExpenseAnomalyDetection = async function runExpenseAnomalyDetection() {
        try {
            const data = await UI.get("/intelligence/isolation-forest-expenses");
            const summaryContainer = document.getElementById("anomaly-summary-container");
            const detailsContainer = document.getElementById("anomaly-details-container");
            
            if (!summaryContainer || !detailsContainer) return;
            
            summaryContainer.classList.remove("hidden");
            summaryContainer.innerHTML = `
                <div class="p-3 bg-white border border-slate-200 rounded-lg">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng số chi phí</p>
                    <p class="text-base font-black text-slate-800">${data.summary.total}</p>
                </div>
                <div class="p-3 bg-white border border-slate-200 rounded-lg">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Số lượng bất thường</p>
                    <p class="text-base font-black text-rose-600">${data.summary.flagged}</p>
                </div>
                <div class="p-3 bg-white border border-slate-200 rounded-lg">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tỷ lệ ô nhiễm</p>
                    <p class="text-base font-black text-slate-800">${(data.contamination * 100).toFixed(1)}%</p>
                </div>
                <div class="p-3 bg-white border border-slate-200 rounded-lg">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Mức độ rủi ro</p>
                    <p class="text-sm font-black text-amber-600 uppercase mt-1">${data.verdict}</p>
                </div>
            `;
            
            detailsContainer.classList.remove("hidden");
            if (data.anomalies.length === 0) {
                detailsContainer.innerHTML = `<div class="p-4 text-center text-slate-400 text-xs">Không phát hiện chi phí bất thường nào.</div>`;
            } else {
                detailsContainer.innerHTML = `
                    <table class="w-full text-left border-collapse text-xs">
                        <thead>
                            <tr class="bg-slate-50 text-slate-400 font-bold uppercase tracking-wider text-[9px] border-b border-slate-200">
                                <th class="px-4 py-2">Hạng mục</th>
                                <th class="px-4 py-2">Phương thức</th>
                                <th class="px-4 py-2 text-right">Số tiền</th>
                                <th class="px-4 py-2 text-center">Z-Score</th>
                                <th class="px-4 py-2 text-center">Điểm rủi ro AI</th>
                                <th class="px-4 py-2 text-center">Trạng thái</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-200 text-slate-700">
                            ${data.anomalies.map(item => `
                                <tr class="${item.is_anomaly ? 'bg-rose-50/30' : ''}">
                                    <td class="px-4 py-2 font-medium">${UI.escapeHtml(item.category)}</td>
                                    <td class="px-4 py-2">${UI.escapeHtml(item.payment_method === 'bank_transfer' ? 'Chuyển khoản' : 'Tiền mặt')}</td>
                                    <td class="px-4 py-2 text-right font-mono font-bold">${UI.fmtVnd(item.amount)}</td>
                                    <td class="px-4 py-2 text-center font-mono">${item.z_score}</td>
                                    <td class="px-4 py-2 text-center font-mono font-bold ${item.is_anomaly ? 'text-rose-600' : 'text-slate-500'}">${item.anomaly_score}</td>
                                    <td class="px-4 py-2 text-center">
                                        <span class="px-2 py-0.5 rounded text-[9px] font-bold uppercase ${item.is_anomaly ? 'bg-rose-100 text-rose-800' : 'bg-slate-100 text-slate-600'}">
                                            ${item.is_anomaly ? 'Bất thường' : 'Bình thường'}
                                        </span>
                                    </td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                    <div class="p-3 bg-amber-50 border-t border-slate-200 text-[10px] text-amber-800 flex items-start gap-1">
                        <span class="material-symbols-outlined text-xs">info</span>
                        <div>
                            <strong>Đề xuất AI:</strong> ${UI.escapeHtml(data.explanation.counterfactual.reduce_cash)}
                        </div>
                    </div>
                `;
            }
            UI.toast("Đã hoàn tất rà soát chi phí bằng Isolation Forest AI.", "success");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadExpenses));
})();
