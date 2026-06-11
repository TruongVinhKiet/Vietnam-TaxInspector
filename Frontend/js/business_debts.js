(function () {
    const UI = window.TaxpayerUI;

    async function loadDebtPanels() {
        const [summary, impersonation, history] = await Promise.all([
            UI.get("/debts/summary"),
            UI.get("/debts/impersonation-check"),
            UI.get("/debts/history"),
        ]);
        UI.panel("debt-live-summary-panel", "Nghĩa vụ và nợ thuế thời gian thực", "account_balance_wallet", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng nợ</p>
                    <p class="text-lg font-black text-slate-800">${UI.fmtVnd(summary.total_debt)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tạm hoãn xuất cảnh</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml(summary.passport_ban_risk.level)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Biên lai đã ghi nhận</p>
                    <p class="text-lg font-black text-slate-800">${history.payments?.length || 0}</p>
                </div>
            </div>
            <p class="mt-3 text-[11px] text-slate-500">${UI.escapeHtml(summary.passport_ban_risk.message)}</p>
        `);
        UI.panel("debt-impersonation-panel", "Kiểm tra mạo danh kê khai thu nhập", "person_search", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="font-bold text-slate-800">${UI.escapeHtml(impersonation.result.message)}</p>
                    <p class="text-[11px] text-slate-500 mt-1">Số tổ chức nghi vấn: ${impersonation.result.suspicious_payers?.length || 0}</p>
                </div>
                ${UI.statusBadge(impersonation.result.status)}
            </div>
        `);
        UI.panel("debt-relief-panel", "Hoàn thuế, bù trừ và phân kỳ", "request_quote", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button id="refund-offset-btn" class="py-2 rounded-lg bg-emerald-500 text-white text-[10px] font-bold">Yêu cầu hoàn/bù trừ</button>
                <button id="installment-btn" class="py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Xin phân kỳ 3 tháng</button>
                <button id="late-penalty-btn" class="py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Tính phạt chậm nộp</button>
            </div>
            <div id="debt-action-result" class="mt-3 text-[11px] text-slate-500"></div>
        `);
        document.getElementById("refund-offset-btn").onclick = requestRefundOffset;
        document.getElementById("installment-btn").onclick = requestInstallment;
        document.getElementById("late-penalty-btn").onclick = calculatePenalty;
    }

    async function requestRefundOffset() {
        const data = await UI.post("/debts/refund-offset", { amount: 1000000, preference: "offset_next_period" });
        document.getElementById("debt-action-result").textContent = data.message;
        UI.toast("Đã gửi yêu cầu hoàn/bù trừ.");
    }

    async function requestInstallment() {
        const data = await UI.post("/debts/installment", { amount: 50000000, months: 3 });
        document.getElementById("debt-action-result").textContent = `Dự kiến mỗi tháng: ${UI.fmtVnd(data.plan.monthly_amount)}`;
        UI.toast("Đã tạo phương án phân kỳ sandbox.");
    }

    async function calculatePenalty() {
        const data = await UI.get("/debts/late-penalty?amount=10000000&days=30");
        document.getElementById("debt-action-result").textContent = `Tiền chậm nộp ước tính: ${UI.fmtVnd(data.penalty.penalty)}`;
    }

    window.downloadReceipt = function downloadReceipt(code) {
        UI.downloadText(`${code}.txt`, `Biên lai sandbox ${code}\nTaxInspector`, "text/plain");
    };

    window.checkPassportBan = async function checkPassportBan() {
        try {
            const data = await UI.get("/debts/passport-ban-risk");
            UI.toast(data.message, data.level === "critical" ? "error" : data.level === "warning" ? "warn" : "success");
            await loadDebtPanels();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadDebtPanels));
})();
