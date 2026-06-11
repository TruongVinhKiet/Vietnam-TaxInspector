(function () {
    const UI = window.TaxpayerUI;
    const Registry = window.TaxpayerAIRegistry;
    if (!UI || !Registry) return;

    function confidenceLabel(data) {
        const value = data?.confidence || data?.model?.confidence || "low";
        const label = { high: "cao", medium: "trung bình", low: "thấp" }[String(value).toLowerCase()] || value;
        const cls = String(value).toLowerCase() === "high"
            ? "bg-emerald-100 text-emerald-700"
            : String(value).toLowerCase() === "medium"
                ? "bg-amber-100 text-amber-700"
                : "bg-slate-100 text-slate-600";
        return `<span class="px-2 py-1 rounded-md text-[10px] font-black uppercase ${cls}">Độ tin cậy ${UI.escapeHtml(label)}</span>`;
    }

    function sufficiency(data) {
        const score = Number(data?.data_sufficiency_score ?? data?.data_sufficiency?.score ?? 0);
        const tier = data?.data_sufficiency?.tier || (score >= 75 ? "rich" : score >= 50 ? "usable" : "thin");
        return { score, tier };
    }

    function metric(label, value, icon, tone = "slate") {
        return `
            <div class="taxpayer-ai-metric rounded-lg border border-${tone}-100 bg-${tone}-50 p-3">
                <div class="flex items-center justify-between gap-2">
                    <p class="text-[10px] font-black uppercase text-${tone}-600">${UI.escapeHtml(label)}</p>
                    <span class="material-symbols-outlined text-${tone}-600 text-base">${UI.escapeHtml(icon)}</span>
                </div>
                <p class="mt-2 text-lg font-black text-slate-900">${value}</p>
            </div>
        `;
    }

    function metaStrip(data) {
        const ds = sufficiency(data);
        return `
            <div class="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-3 text-[10px] text-slate-500">
                <span>Model: ${UI.escapeHtml(data?.model_name || data?.model?.model_name || "baseline")} / ${UI.escapeHtml(data?.model_version || data?.model?.model_version || "")}</span>
                <span>Data sufficiency: ${Math.round(ds.score)} (${UI.escapeHtml(ds.tier)})</span>
                ${confidenceLabel(data)}
            </div>
        `;
    }

    function emptyState(message) {
        return `<div class="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs font-semibold text-slate-500">${UI.escapeHtml(message)}</div>`;
    }

    function renderOverview(data, cfg) {
        if (!data || data.status === "error") {
            return UI.panel("taxpayer-ai-overview-panel", cfg.title, cfg.icon, emptyState(data?.message || "Chưa có dữ liệu AI."), { prepend: cfg.prepend });
        }
        const scores = data.scores || {};
        const alerts = data.alerts || [];
        const recs = data.top_recommendations || [];
        UI.panel("taxpayer-ai-overview-panel", cfg.title, cfg.icon, `
            <div class="grid grid-cols-1 gap-3 md:grid-cols-4">
                ${metric("Tài chính", `${Math.round(Number(scores.financial_health || 0))}/100`, "monitoring", "emerald")}
                ${metric("Tuân thủ", `${Math.round(Number(scores.compliance || 0))}/100`, "verified", "sky")}
                ${metric("Dòng tiền", `${Math.round(Number(scores.cashflow || 0))}/100`, "payments", "amber")}
                ${metric("Dữ liệu", `${Math.round(Number(scores.data_quality || 0))}/100`, "database", "indigo")}
            </div>
            <div class="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
                <div class="space-y-2">
                    <p class="text-[10px] font-black uppercase text-slate-400">Cảnh báo ưu tiên</p>
                    ${alerts.slice(0, 3).map((item) => `
                        <div class="rounded-lg border border-slate-200 bg-white p-3">
                            <div class="flex items-center justify-between gap-2">
                                <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                                ${UI.statusBadge(item.severity || "medium")}
                            </div>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p>
                        </div>
                    `).join("") || emptyState("Chưa có cảnh báo lớn trong dữ liệu hiện tại.")}
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-black uppercase text-slate-400">Việc nên làm tiếp theo</p>
                    ${recs.slice(0, 3).map((item) => `
                        <div class="rounded-lg border border-slate-200 bg-white p-3">
                            <div class="flex items-center justify-between gap-2">
                                <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                                ${UI.statusBadge(item.priority || "medium")}
                            </div>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.reason)}</p>
                        </div>
                    `).join("") || emptyState("Chưa có khuyến nghị mới.")}
                </div>
            </div>
            ${metaStrip(data)}
        `, { prepend: cfg.prepend });
    }

    function renderForecast(data, cfg) {
        if (!data || data.status === "error") return renderGeneric("forecast", data, cfg);
        const probs = data.threshold_probabilities || {};
        const months = data.forecast_months || [];
        UI.panel("taxpayer-ai-forecast-panel", "Dự báo doanh thu và ngưỡng thuế", "timeline", `
            <div class="grid grid-cols-1 gap-3 md:grid-cols-3">
                ${metric("Doanh thu cuối năm", UI.fmtVnd(data.projected_year_end_revenue), "trending_up", "emerald")}
                ${metric("Vượt 500 triệu", `${Math.round(Number(probs.taxable_500m || 0) * 100)}%`, "flag", "amber")}
                ${metric("Vượt 1 tỷ HĐĐT", `${Math.round(Number(probs.einvoice_1b || 0) * 100)}%`, "receipt_long", "sky")}
            </div>
            <div class="mt-3 grid grid-cols-2 gap-2 md:grid-cols-6">
                ${months.slice(0, 6).map((item) => `
                    <div class="rounded-lg border border-slate-200 bg-white p-2">
                        <p class="text-[10px] font-bold text-slate-400">${UI.escapeHtml(item.period)}</p>
                        <p class="mt-1 text-[11px] font-black text-slate-800">${UI.fmtVnd(item.revenue)}</p>
                    </div>
                `).join("")}
            </div>
            ${metaStrip(data)}
        `);
    }

    function renderReconciliationCases(data, cfg) {
        const cases = data?.cases || [];
        const summary = data?.summary || {};
        UI.panel("taxpayer-ai-reconciliation-cases-panel", "Case đối soát cần xử lý", "rule_folder", `
            <div class="grid grid-cols-1 gap-3 md:grid-cols-3">
                ${metric("Case đang mở", summary.open_case_count ?? cases.length, "pending_actions", "amber")}
                ${metric("Mức cao", summary.high_case_count ?? 0, "priority_high", "rose")}
                ${metric("Hành động", UI.escapeHtml(data?.next_action || "Kiểm tra case trước khi nộp hồ sơ."), "task_alt", "emerald")}
            </div>
            <div class="mt-3 space-y-2">
                ${cases.slice(0, 4).map((item) => `
                    <div class="rounded-lg border border-slate-200 bg-white p-3">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.title || item.case_key)}</p>
                            ${UI.statusBadge(item.severity || item.status || "open")}
                        </div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.description || "")}</p>
                    </div>
                `).join("") || emptyState("Chưa có case đối soát đang mở.")}
            </div>
            ${metaStrip(data)}
        `);
    }

    function renderNextBestAction(data, cfg) {
        const actions = data?.actions || data?.recommendations || [];
        UI.panel("taxpayer-ai-next-action-panel", "Việc nên làm hôm nay", "auto_awesome", `
            <div class="space-y-2">
                ${actions.slice(0, 5).map((item, index) => `
                    <div class="flex items-start gap-3 rounded-lg border border-slate-200 bg-white p-3">
                        <div class="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-100 text-xs font-black text-emerald-700">${index + 1}</div>
                        <div class="min-w-0 flex-1">
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.title || item.action || item.key)}</p>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.reason || item.message || "")}</p>
                        </div>
                        ${UI.statusBadge(item.priority || item.severity || "medium")}
                    </div>
                `).join("") || emptyState("Không có việc khẩn cấp trong hôm nay.")}
            </div>
            ${metaStrip(data)}
        `);
    }

    function renderCashflowRisk(data, cfg) {
        const schedule = data?.optimized_payment_schedule || data?.payment_plan || [];
        UI.panel("taxpayer-ai-cashflow-panel", "Dòng tiền nộp thuế", "payments", `
            <div class="grid grid-cols-1 gap-3 md:grid-cols-3">
                ${metric("Rủi ro", UI.escapeHtml(data?.risk_level || data?.risk_after_plan || "đang tính"), "warning", "amber")}
                ${metric("Nợ/thuế chờ xử lý", UI.fmtVnd(data?.pending_tax_and_debt || data?.projected_tax?.total_tax || 0), "account_balance", "rose")}
                ${metric("Dự phòng tháng", UI.fmtVnd(data?.monthly_reserve_amount || 0), "savings", "emerald")}
            </div>
            <div class="mt-3 space-y-2">
                ${schedule.slice(0, 3).map((item) => `
                    <div class="rounded-lg border border-slate-200 bg-white p-3 text-xs">
                        <b>${UI.fmtVnd(item.amount)}</b>
                        <span class="text-slate-500"> - ${UI.escapeHtml(item.objective || item.label || "")}</span>
                    </div>
                `).join("") || emptyState("Chưa có lịch thanh toán đề xuất.")}
            </div>
            ${metaStrip(data)}
        `);
    }

    function renderGeneric(key, data, cfg) {
        const cap = Registry.getCapability(key) || {};
        if (!data || data.status === "error") {
            return UI.panel(`taxpayer-ai-${key}-panel`, cap.label || cfg.title, cfg.icon || "auto_awesome", emptyState(data?.message || "Chưa có dữ liệu phù hợp."));
        }
        const reasonCodes = data.reason_codes || data.explanation?.reason_codes || [];
        const human = data.needs_human_confirmation ? "Cần xác nhận người dùng trước khi áp dụng." : "Có thể dùng như insight tham khảo.";
        UI.panel(`taxpayer-ai-${key}-panel`, cap.label || cfg.title, cfg.icon || "auto_awesome", `
            <div class="grid grid-cols-1 gap-3 md:grid-cols-3">
                ${metric("Trạng thái", UI.escapeHtml(data.status || "success"), "check_circle", "emerald")}
                ${metric("Human confirm", UI.escapeHtml(data.needs_human_confirmation ? "có" : "không"), "verified_user", data.needs_human_confirmation ? "amber" : "emerald")}
                ${metric("Cache", UI.escapeHtml(data.cache_status || "live"), "cached", "sky")}
            </div>
            <div class="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p class="font-bold text-slate-800">${UI.escapeHtml(human)}</p>
                <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(reasonCodes.slice(0, 4).join(", ") || "rule/statistical baseline")}</p>
            </div>
            ${metaStrip(data)}
        `);
    }

    window.TaxpayerAIPanelsCore = {
        renderOverview,
        renderForecast,
        renderReconciliationCases,
        renderNextBestAction,
        renderCashflowRisk,
        renderGeneric,
        metaStrip,
        confidenceLabel,
    };
})();
