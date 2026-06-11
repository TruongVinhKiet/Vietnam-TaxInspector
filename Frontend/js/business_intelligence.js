(function () {
    const UI = window.TaxpayerUI;
    if (!UI) return;

    const page = window.location.pathname.split("/").pop() || "business_dashboard.html";
    const pageConfig = {
        "business_dashboard.html": { title: "AI Điều hành Kinh doanh", icon: "psychology", mode: "full", prepend: true },
        "business_calendar.html": { title: "AI Dự báo Trễ hạn và Ngưỡng Doanh thu", icon: "event_upcoming", mode: "forecast" },
        "business_invoices.html": { title: "AI Rà soát Hóa đơn và Đối tác", icon: "document_scanner", mode: "invoice" },
        "business_filing.html": { title: "AI Kiểm tra Tờ khai trước khi Nộp", icon: "fact_check", mode: "filing" },
        "business_debts.html": { title: "AI Dự báo Nợ thuế và Dòng tiền", icon: "account_balance", mode: "debt" },
        "business_legal.html": { title: "AI Pháp lý có Trích dẫn (GraphRAG)", icon: "gavel", mode: "legal" },
        "business_growth.html": { title: "AI Gợi ý Thay đổi Mô hình Kinh doanh", icon: "trending_up", mode: "growth" },
        "business_accounting.html": { title: "AI Phân tích Sổ sách và Chứng từ", icon: "auto_stories", mode: "accounting" },
        "business_expenses.html": { title: "AI Phân loại Chi phí được trừ", icon: "rule", mode: "expense" },
        "business_claims.html": { title: "AI Đánh giá Hồ sơ Khiếu nại", icon: "balance", mode: "claim" },
        "business_profile.html": { title: "AI Đánh giá Chất lượng Dữ liệu Hồ sơ", icon: "manage_accounts", mode: "profile" },
        "business_registration.html": { title: "AI Chuẩn bị Đăng ký Thuế", icon: "how_to_reg", mode: "registration" },
        "business_calculator.html": { title: "AI Mô phỏng Thuế và Lợi nhuận", icon: "query_stats", mode: "calculator" },
    };

    const cfg = pageConfig[page] || pageConfig["business_dashboard.html"];

    function scoreColor(value) {
        const n = Number(value || 0);
        if (n >= 75) return "emerald";
        if (n >= 50) return "amber";
        return "rose";
    }

    function priorityBadge(priority) {
        const key = String(priority || "medium").toLowerCase();
        const cls = key === "high" ? "bg-rose-100 text-rose-700" : key === "low" ? "bg-slate-100 text-slate-600" : "bg-amber-100 text-amber-700";
        return `<span class="px-2 py-0.5 rounded text-[9px] font-black uppercase ${cls}">${UI.escapeHtml(priority || "medium")}</span>`;
    }

    function scoreCard(label, value, icon) {
        const color = scoreColor(value);
        return `
            <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                <div class="flex items-center justify-between">
                    <p class="text-[9px] uppercase font-bold text-slate-400">${UI.escapeHtml(label)}</p>
                    <span class="material-symbols-outlined text-${color}-600 text-base">${icon}</span>
                </div>
                <p class="mt-2 text-lg font-black text-slate-800">${Math.round(Number(value || 0))}/100</p>
                <div class="mt-2 h-1.5 bg-white rounded-full overflow-hidden border border-slate-100">
                    <div class="h-full bg-${color}-500" style="width:${Math.max(0, Math.min(100, Number(value || 0)))}%"></div>
                </div>
            </div>
        `;
    }

    function renderOverview(data) {
        const scores = data.scores || {};
        const alerts = data.alerts || [];
        const recs = data.top_recommendations || [];
        const model = data.model || {};
        UI.panel("taxpayer-ai-overview-panel", cfg.title, cfg.icon, `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                ${scoreCard("Tài chính", scores.financial_health, "monitoring")}
                ${scoreCard("Tuân thủ", scores.compliance, "verified")}
                ${scoreCard("Dòng tiền", scores.cashflow, "payments")}
                ${scoreCard("Dữ liệu", scores.data_quality, "database")}
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
                <div class="space-y-2">
                    <p class="text-[10px] font-black uppercase text-slate-400">Cảnh báo thông minh</p>
                    ${alerts.slice(0, 3).map((item) => `
                        <div class="p-3 rounded-lg bg-white border border-slate-200">
                            <div class="flex items-center justify-between gap-2">
                                <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                                ${priorityBadge(item.severity)}
                            </div>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p>
                        </div>
                    `).join("") || `<div class="p-3 rounded-lg bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chưa có cảnh báo lớn trong dữ liệu hiện tại.</div>`}
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-black uppercase text-slate-400">Khuyến nghị hành động</p>
                    ${recs.slice(0, 3).map((item) => `
                        <div class="p-3 rounded-lg bg-white border border-slate-200">
                            <div class="flex items-center justify-between gap-2">
                                <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                                ${priorityBadge(item.priority)}
                            </div>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.reason)}</p>
                            <div class="mt-2 flex items-center gap-2">
                                <a href="${UI.escapeHtml(item.target_page || page)}" class="px-2 py-1 rounded bg-slate-900 text-white text-[10px] font-bold">${UI.escapeHtml(item.action_label || "Mở")}</a>
                                <button class="ai-feedback px-2 py-1 rounded bg-slate-100 text-slate-600 text-[10px] font-bold" data-target="${UI.escapeHtml(item.key)}" data-signal="helpful">Hữu ích</button>
                                <button class="ai-feedback px-2 py-1 rounded bg-slate-100 text-slate-600 text-[10px] font-bold" data-target="${UI.escapeHtml(item.key)}" data-signal="not_relevant">Chưa đúng</button>
                            </div>
                        </div>
                    `).join("")}
                </div>
            </div>
            <div class="mt-3 flex items-center justify-between text-[10px] text-slate-400">
                <span>Mô hình AI: ${UI.escapeHtml(model.model_name || "baseline")} / Phiên bản: ${UI.escapeHtml(model.model_version || "")}</span>
                <span>Độ tin cậy: ${UI.escapeHtml(model.confidence || "thấp")}</span>
            </div>
        `, { prepend: cfg.prepend });
        bindFeedback();
    }

    function renderForecast(data) {
        const months = data.forecast_months || [];
        const probs = data.threshold_probabilities || {};
        if (!["full", "forecast", "growth", "debt", "calculator"].includes(cfg.mode)) return;
        UI.panel("taxpayer-ai-forecast-panel", "AI Dự báo 6 tháng và Ngưỡng rủi ro", "timeline", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Doanh thu cuối năm (Dự phóng)</p>
                    <p class="text-lg font-black text-slate-800">${UI.fmtVnd(data.projected_year_end_revenue)}</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Xác suất vượt ngưỡng 1 tỷ</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(probs.einvoice_1b || 0) * 100)}%</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Dòng tiền dự kiến 90 ngày</p>
                    <p class="text-lg font-black text-slate-800">${UI.fmtVnd((data.cashflow_30_60_90 || {}).days_90)}</p>
                </div>
            </div>
            <div class="mt-3 grid grid-cols-2 md:grid-cols-6 gap-2">
                ${months.slice(0, 6).map((item) => `
                    <div class="p-2 rounded bg-white border border-slate-200">
                        <p class="text-[9px] font-bold text-slate-400">${UI.escapeHtml(item.period)}</p>
                        <p class="text-[11px] font-black text-slate-700">${UI.fmtVnd(item.revenue)}</p>
                    </div>
                `).join("")}
            </div>
        `);
    }

    function renderBenchmark(data) {
        if (!["full", "expense", "accounting", "growth"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;

        const metrics = data.taxpayer_metrics || {};
        const signals = data.signals || {};
        const similarities = data.peer_similarities || [];
        const closestPeer = data.closest_peer_label || "Chưa xác định";

        UI.panel("taxpayer-ai-benchmark-panel", "So sánh Hiệu năng Cùng ngành (KNN Similarity)", "bar_chart", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Biên lợi nhuận</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${Math.round(Number(metrics.profit_margin || 0) * 100)}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tỷ lệ chi phí</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${Math.round(Number(metrics.expense_ratio || 0) * 100)}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tỷ lệ thuế hiệu dụng</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${(Number(metrics.tax_ratio || 0) * 100).toFixed(1)}%</p>
                </div>
                <div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200">
                    <p class="text-[9px] uppercase font-bold text-indigo-600">Nhóm ngành tương đồng nhất</p>
                    <p class="text-xs font-black text-indigo-800 mt-1">${UI.escapeHtml(closestPeer)}</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <!-- KNN Similary List -->
                <div class="bg-white rounded-lg border border-slate-200 p-4 space-y-3">
                    <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2">Độ tương đồng tài chính (KNN Distance)</h5>
                    <div class="space-y-3">
                        ${similarities.map(s => {
                            const isMatch = s.label === closestPeer;
                            const barColor = isMatch ? "bg-indigo-600" : "bg-slate-400";
                            return `
                                <div class="space-y-1">
                                    <div class="flex justify-between text-[10px] ${isMatch ? 'font-bold text-indigo-900' : 'text-slate-600'}">
                                        <span>${UI.escapeHtml(s.label)}</span>
                                        <span>${s.similarity_score}%</span>
                                    </div>
                                    <div class="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                        <div class="h-full ${barColor} transition-all" style="width: ${s.similarity_score}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join("")}
                    </div>
                </div>

                <!-- Peer signals -->
                <div class="space-y-3">
                    <div class="p-3 rounded-lg bg-white border border-slate-200 space-y-2">
                        <p class="text-[9px] uppercase font-bold text-slate-400">Kết quả đối sánh pháp lý</p>
                        <div class="space-y-2 text-xs">
                            <div class="flex justify-between items-center py-1 border-b border-slate-50">
                                <span class="text-slate-500">Vị thế biên lợi nhuận:</span>
                                <span class="font-bold text-${signals.margin_position === 'below_peer_range' ? 'amber' : 'emerald'}-600">
                                    ${signals.margin_position === 'below_peer_range' ? 'Thấp hơn trung bình' : 'Đạt chuẩn ngành'}
                                </span>
                            </div>
                            <div class="flex justify-between items-center py-1 border-b border-slate-50">
                                <span class="text-slate-500">Cảnh báo tỷ lệ chi phí vượt ngưỡng:</span>
                                <span class="font-bold text-${signals.expense_ratio_flag ? 'rose' : 'emerald'}-600">
                                    ${signals.expense_ratio_flag ? 'Cảnh báo đỏ' : 'An toàn'}
                                </span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="p-3 rounded-lg bg-indigo-50/50 border border-indigo-100">
                        <p class="text-[9px] uppercase font-bold text-indigo-600">Khuyến nghị tối ưu hóa KNN</p>
                        <p class="text-xs text-indigo-950 mt-1">${UI.escapeHtml((data.explanation || {}).counterfactual?.align_expenses || '')}</p>
                    </div>
                </div>
            </div>
        `);
    }

    function renderCharts(data) {
        if (!["full", "forecast", "accounting", "expense", "growth", "calculator"].includes(cfg.mode)) return;
        const months = data.monthly_series || [];
        const channels = data.channel_breakdown || [];
        const expenses = data.expense_breakdown || [];
        const maxRevenue = Math.max(1, ...months.map((item) => Number(item.revenue || 0)));
        UI.panel("taxpayer-ai-chart-panel", "Biểu đồ Dữ liệu Kinh doanh", "stacked_line_chart", `
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div class="lg:col-span-2 p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400 mb-3">Doanh thu - Chi phí theo Tháng</p>
                    <div class="flex items-end gap-1 h-32">
                        ${months.map((item) => `
                            <div class="flex-1 flex flex-col justify-end gap-1 min-w-0">
                                <div title="${UI.escapeHtml(item.period)}: ${UI.fmtVnd(item.revenue)}" class="bg-emerald-500 rounded-t" style="height:${Math.max(4, Number(item.revenue || 0) / maxRevenue * 100)}px"></div>
                                <div title="Chi phí: ${UI.fmtVnd(item.expense)}" class="bg-amber-400 rounded-t" style="height:${Math.max(3, Number(item.expense || 0) / maxRevenue * 100)}px"></div>
                                <p class="truncate text-[8px] text-slate-400 text-center">${String(item.period || "").slice(5)}</p>
                            </div>
                        `).join("")}
                    </div>
                    <div class="mt-3 flex items-center gap-4 text-[9px] font-bold text-slate-500">
                        <span><i class="inline-block w-2.5 h-2 bg-emerald-500 rounded-sm"></i> Doanh thu</span>
                        <span><i class="inline-block w-2.5 h-2 bg-amber-400 rounded-sm"></i> Chi phí</span>
                    </div>
                </div>
                <div class="space-y-3">
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <p class="text-[9px] uppercase font-bold text-slate-400 mb-2">Kênh Doanh thu</p>
                        ${channels.slice(0, 4).map((item) => `
                            <div class="mb-2">
                                <div class="flex justify-between text-[10px] font-bold"><span>${UI.escapeHtml(item.label)}</span><span>${Math.round(Number(item.share || 0) * 100)}%</span></div>
                                <div class="h-1.5 bg-white rounded overflow-hidden"><div class="h-full bg-emerald-500" style="width:${Math.round(Number(item.share || 0) * 100)}%"></div></div>
                            </div>
                        `).join("") || `<p class="text-[11px] text-slate-400">Chưa có dữ liệu kênh.</p>`}
                    </div>
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <p class="text-[9px] uppercase font-bold text-slate-400 mb-2">Nhóm Chi phí</p>
                        ${expenses.slice(0, 4).map((item) => `
                            <div class="flex justify-between text-[10px] py-1 border-b border-white last:border-0">
                                <span class="font-bold text-slate-600">${UI.escapeHtml(item.label)}</span>
                                <span>${UI.fmtVnd(item.value)}</span>
                            </div>
                        `).join("") || `<p class="text-[11px] text-slate-400">Chưa có dữ liệu chi phí.</p>`}
                    </div>
                </div>
            </div>
        `);
    }

    function renderAnomalies(data) {
        if (!["full", "invoice", "filing", "debt", "accounting", "expense", "claim"].includes(cfg.mode)) return;
        const anomalies = data.anomalies || [];
        UI.panel("taxpayer-ai-anomaly-panel", "Phát hiện Bất thường (Anomaly Detection)", "radar", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${anomalies.slice(0, 6).map((item) => `
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-800 text-xs">${UI.escapeHtml(item.title)}</p>
                            ${priorityBadge(item.severity)}
                        </div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.description)}</p>
                        <p class="mt-2 text-[10px] font-bold text-emerald-700">${UI.escapeHtml(item.recommended_action)}</p>
                    </div>
                `).join("")}
            </div>
        `);
    }

    function renderCatalog(data) {
        if (!["full", "legal", "profile"].includes(cfg.mode)) return;
        const items = data.items || [];
        UI.panel("taxpayer-ai-catalog-panel", "Nang luc AI dang khai thac tu TaxInspector", "model_training", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${items.slice(0, 6).map((item) => `
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.name)}</p>
                            ${UI.statusBadge(item.status)}
                        </div>
                        <p class="text-[10px] text-slate-400 mt-1 font-mono">${UI.escapeHtml(item.taxinspector_module)}</p>
                        <p class="text-[11px] text-slate-600 mt-2">${UI.escapeHtml(item.taxpayer_use)}</p>
                    </div>
                `).join("")}
            </div>
        `);
    }

    function renderScenario(data) {
        if (!["full", "growth", "profile", "registration"].includes(cfg.mode)) return;
        const persona = data.persona || {};
        const cards = data.strategy_cards || [];
        const heatmap = data.risk_heatmap || [];
        UI.panel("taxpayer-ai-scenario-panel", "Ban do chien luoc taxpayer", "dashboard", `
            <div class="grid grid-cols-1 lg:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nhom du bao</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml((persona.household_group || {}).label || "N/A")}</p>
                    <p class="mt-1 text-[10px] text-slate-500">${UI.fmtVnd(persona.projected_year_end_revenue)}</p>
                </div>
                ${cards.slice(0, 3).map((item) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2">
                            <p class="text-[10px] font-black text-slate-700">${UI.escapeHtml(item.title)}</p>
                            ${UI.statusBadge(item.status)}
                        </div>
                        <p class="mt-2 text-[11px] text-slate-500">${UI.escapeHtml(item.action)}</p>
                    </div>
                `).join("")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-5 gap-2">
                ${heatmap.map((item) => `
                    <div class="p-2 rounded bg-white border border-slate-200">
                        <div class="flex justify-between text-[9px] font-bold text-slate-500"><span>${UI.escapeHtml(item.label)}</span><span>${Math.round(Number(item.score || 0))}</span></div>
                        <div class="mt-1 h-1.5 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-${scoreColor(100 - Number(item.score || 0))}-500" style="width:${Math.min(100, Number(item.score || 0))}%"></div></div>
                    </div>
                `).join("")}
            </div>
        `);
    }

    function renderCashflowRisk(data) {
        if (!["full", "forecast", "debt", "filing"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;

        // Upgraded structure
        const isUpgraded = Array.isArray(data.series);
        const score = data.risk_score !== undefined ? data.risk_score : (data.risk_score_pct || 0);
        const verdict = data.verdict || data.risk_level || "low";
        const verdictColor = data.verdict_color || (verdict === "high" ? "rose" : verdict === "medium" ? "amber" : "emerald");
        const verdictLabel = data.verdict_label || (verdict === "high" ? "Rủi ro Cao" : verdict === "medium" ? "Rủi ro Trung bình" : "An toàn");
        
        let initialReserve = 50000000;
        let finalReserve = data.final_reserve || 0;
        let series = [];
        let insights = data.insights || [];
        let counterfactual = (data.explanation || {}).counterfactual || {};

        if (isUpgraded) {
            series = data.series;
            initialReserve = data.initial_reserve;
        } else {
            // Backward compatibility adapter
            const flow = data.cashflow_30_60_90 || {};
            finalReserve = flow.days_30 || 0;
            series = [
                { period: "30 Ngày", inflow: flow.days_30 || 0, outflow: (flow.days_30 || 0) * 0.8, closing_reserve: flow.days_30 || 0, safety_ratio: 1.0, is_critical: false },
                { period: "60 Ngày", inflow: flow.days_60 || 0, outflow: (flow.days_60 || 0) * 0.8, closing_reserve: flow.days_60 || 0, safety_ratio: 1.0, is_critical: false },
                { period: "90 Ngày", inflow: flow.days_90 || 0, outflow: (flow.days_90 || 0) * 0.8, closing_reserve: flow.days_90 || 0, safety_ratio: 1.0, is_critical: false }
            ];
        }

        const maxFlow = Math.max(...series.map(s => Math.max(s.inflow, s.outflow, s.closing_reserve)), 1000000);

        UI.panel("taxpayer-ai-cashflow-risk-panel", "Dự báo Dòng tiền & Rủi ro Chậm nộp Thuế (Time-series RNN)", "account_balance_wallet", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${verdictColor}-50 border border-${verdictColor}-200">
                    <p class="text-[9px] uppercase font-bold text-${verdictColor}-600">Đánh giá rủi ro</p>
                    <p class="text-sm font-black text-${verdictColor}-800 mt-1">${UI.escapeHtml(verdictLabel)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Điểm Rủi ro Dòng tiền</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${Math.round(score)}/100</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Dự trữ Tiền mặt Đầu kỳ</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(initialReserve)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Dự trữ khả dụng Cuối kỳ</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(finalReserve)}</p>
                </div>
            </div>

            <!-- Projection chart -->
            <div class="bg-white rounded-lg border border-slate-200 p-4 space-y-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2">Biểu đồ Trải Dòng tiền & Cân đối Thuế</h5>
                
                <div class="flex gap-4 h-32 pt-2 border-b border-slate-100 pb-2 relative items-end">
                    ${series.map(s => {
                        const hInflow = (s.inflow / maxFlow) * 100;
                        const hOutflow = (s.outflow / maxFlow) * 100;
                        const hReserve = (Math.max(0, s.closing_reserve) / maxFlow) * 100;
                        
                        return `
                            <div class="flex-1 h-full flex flex-col justify-end gap-1 relative group" title="${UI.escapeHtml(s.period)}\n- Dòng vào: ${UI.fmtVnd(s.inflow)}\n- Dòng ra (gồm thuế): ${UI.fmtVnd(s.outflow)}\n- Dự trữ: ${UI.fmtVnd(s.closing_reserve)}">
                                <div class="flex items-end gap-1.5 h-full w-full justify-center">
                                    <!-- Inflow bar -->
                                    <div class="w-2.5 bg-emerald-500 rounded-t transition-all" style="height: ${hInflow}%"></div>
                                    <!-- Outflow bar -->
                                    <div class="w-2.5 bg-amber-400 rounded-t transition-all" style="height: ${hOutflow}%"></div>
                                    <!-- Reserve line representation -->
                                    <div class="w-1 bg-indigo-600 rounded transition-all" style="height: ${hReserve}%"></div>
                                </div>
                                <span class="text-[8px] text-slate-400 text-center block mt-1 truncate">${UI.escapeHtml(s.period)}</span>
                            </div>
                        `;
                    }).join("")}
                </div>

                <div class="flex items-center gap-4 mt-3 flex-wrap text-[9px] font-bold text-slate-500">
                    <span><i class="inline-block w-2.5 h-2 bg-emerald-500 rounded-sm"></i> Dòng tiền vào (Thu)</span>
                    <span><i class="inline-block w-2.5 h-2 bg-amber-400 rounded-sm"></i> Dòng tiền ra (Chi & Thuế)</span>
                    <span><i class="inline-block w-1.5 h-2 bg-indigo-600 rounded-sm"></i> Dự trữ khả dụng</span>
                </div>
            </div>

            <!-- Insights list -->
            ${insights.length > 0 ? `
                <div class="mt-3 space-y-2">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Đánh giá khả năng thanh khoản</p>
                    ${insights.map(item => `
                        <div class="p-2.5 rounded-lg border bg-white border-slate-200 flex items-start gap-2">
                            <span class="material-symbols-outlined text-${item.severity === 'high' ? 'rose' : item.severity === 'medium' ? 'amber' : 'sky'}-500 text-sm mt-0.5">
                                ${item.severity === 'high' ? 'error' : item.severity === 'medium' ? 'warning' : 'info'}
                            </span>
                            <p class="text-[11px] text-slate-700">${UI.escapeHtml(item.message)}</p>
                        </div>
                    `).join("")}
                </div>
            ` : ''}

            ${counterfactual.optimize_receivables ? `
                <div class="mt-3 p-3 rounded-lg bg-indigo-50 border border-indigo-100">
                    <p class="text-[9px] uppercase font-bold text-indigo-600">Đề xuất quản lý tài chính</p>
                    <p class="text-xs text-indigo-950 mt-1">${UI.escapeHtml(counterfactual.optimize_receivables)}</p>
                </div>
            ` : ''}
        `);
    }

    function renderSupplierGraph(data) {
        if (!["full", "invoice", "accounting"].includes(cfg.mode)) return;
        const summary = data.summary || {};
        const risks = data.top_risks || [];
        
        function getTrustBadge(score) {
            if (score >= 80) return `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">Tin cậy cao (${score}%)</span>`;
            if (score >= 50) return `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-sky-100 text-sky-800 border border-sky-200">Trung bình (${score}%)</span>`;
            return `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-rose-100 text-rose-800 border border-rose-200">Rủi ro cao (${score}%)</span>`;
        }

        UI.panel("taxpayer-ai-supplier-graph-panel", "Phân tích Chuỗi Cung ứng & Độ Tin cậy Đối tác (PageRank Centrality)", "hub", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng số đối tác giao dịch</p>
                    <p class="text-lg font-black text-slate-800 mt-1">${summary.supplier_count || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Liên kết hóa đơn đầu vào/ra</p>
                    <p class="text-lg font-black text-slate-800 mt-1">${summary.edge_count || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-rose-50 border border-rose-200">
                    <p class="text-[9px] uppercase font-bold text-rose-600">Đối tác rủi ro cao cần lưu ý</p>
                    <p class="text-lg font-black text-rose-800 mt-1">${summary.high_risk_count || 0}</p>
                </div>
            </div>

            <!-- PageRank Trust Table -->
            <div class="bg-white rounded-lg border border-slate-200 p-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Xếp hạng uy tín đối tác chuỗi cung ứng</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-[9px] uppercase text-slate-400 border-b border-slate-100">
                            <tr>
                                <th class="py-2.5">Đối tác (MST)</th>
                                <th>Số HĐ</th>
                                <th>Doanh số lũy kế</th>
                                <th>PageRank Trung tâm</th>
                                <th>Chỉ số Tin cậy</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100">
                            ${risks.slice(0, 6).map((item) => `
                                <tr>
                                    <td class="py-3 font-bold text-slate-700">
                                        <p>${UI.escapeHtml(item.partner_name || item.tax_code)}</p>
                                        <span class="text-[9px] font-normal text-slate-400">${UI.escapeHtml(item.tax_code)}</span>
                                    </td>
                                    <td>${item.invoice_count || 0}</td>
                                    <td class="font-semibold text-slate-800">${UI.fmtVnd(item.amount)}</td>
                                    <td>
                                        <div class="flex items-center gap-2">
                                            <div class="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden">
                                                <div class="bg-indigo-600 h-1.5" style="width: ${item.centrality_score || 50}%"></div>
                                            </div>
                                            <span class="text-[10px] font-mono text-slate-500">${(item.centrality_score || 50).toFixed(1)}%</span>
                                        </div>
                                    </td>
                                    <td>${getTrustBadge(item.trust_score || 50)}</td>
                                </tr>
                            `).join("") || `<tr><td colspan="5" class="py-4 text-center text-slate-400">Chưa ghi nhận đối tác giao dịch trong kỳ.</td></tr>`}
                        </tbody>
                    </table>
                </div>
            </div>
        `);
    }

    function renderUpgradeReadiness(data) {
        if (!["full", "growth", "profile"].includes(cfg.mode)) return;
        const components = data.components || {};
        const missing = data.missing_capabilities || [];
        UI.panel("taxpayer-ai-upgrade-panel", "San sang chuyen doi len doanh nghiep", "rocket_launch", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Diem san sang</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(data.readiness_score || 0))}/100</p>
                    ${UI.statusBadge(data.readiness_level)}
                </div>
                ${Object.entries(components).slice(0, 3).map(([key, value]) => `
                    <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                        <p class="text-[9px] uppercase font-bold text-slate-400">${UI.escapeHtml(key)}</p>
                        <p class="text-sm font-black text-slate-800">${Math.round(Number(value || 0))}/100</p>
                    </div>
                `).join("")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${missing.slice(0, 4).map((item) => `<div class="p-2 rounded bg-white border border-slate-200 text-[11px] font-bold text-slate-600">${UI.escapeHtml(item)}</div>`).join("") || `<div class="p-2 rounded bg-emerald-50 border border-emerald-100 text-[11px] font-bold text-emerald-700">Nen tang toc ho so chuyen doi khi co nhu cau mo rong.</div>`}
            </div>
        `);
    }

    function renderAdvancedDashboard(data) {
        if (!["full", "profile"].includes(cfg.mode)) return;
        const center = data.command_center || {};
        const heatmap = data.risk_heatmap || [];
        const actions = data.top_actions || [];
        UI.panel("taxpayer-ai-advanced-command-panel", "Trung tâm Điều hành AI Nâng cao (Command Center)", "memory", `
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
                ${scoreCard("Tài chính", center.financial_health, "monitoring")}
                ${scoreCard("Tuân thủ", center.compliance, "verified_user")}
                ${scoreCard("Rủi ro mạng lưới", 100 - Number(center.graph_risk || 0), "hub")}
                ${scoreCard("Quản trị dữ liệu", center.data_quality, "admin_panel_settings")}
            </div>
            <div class="mt-4 grid grid-cols-1 lg:grid-cols-5 gap-2">
                ${heatmap.map((item) => `
                    <div class="p-2 rounded bg-white border border-slate-200">
                        <div class="flex justify-between text-[9px] font-bold text-slate-500">
                            <span>${UI.escapeHtml(item.label)}</span>
                            <span>${Math.round(Number(item.score || 0))}</span>
                        </div>
                        <div class="mt-1 h-1.5 bg-slate-100 rounded overflow-hidden">
                            <div class="h-full bg-${scoreColor(100 - Number(item.score || 0))}-500" style="width:${Math.min(100, Number(item.score || 0))}%"></div>
                        </div>
                    </div>
                `).join("")}
            </div>
            <div class="mt-4 grid grid-cols-1 md:grid-cols-3 gap-2">
                ${actions.slice(0, 3).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-800 text-xs">${UI.escapeHtml(item.title)}</p>
                            ${priorityBadge(item.priority || item.risk_level)}
                        </div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.expected_impact || item.reason)}</p>
                    </div>
                `).join("")}
            </div>
        `, { prepend: cfg.prepend });
    }

    function renderProbabilisticForecast(data) {
        if (!["full", "forecast", "growth", "debt", "calculator"].includes(cfg.mode)) return;
        const intervals = data.intervals || [];
        const probs = data.threshold_probabilities || {};
        const maxP90 = Math.max(1, ...intervals.map((item) => Number(item.p90 || 0)));
        UI.panel("taxpayer-ai-probabilistic-panel", "Dự báo Xác suất Ngưỡng Doanh thu (P10/P50/P90)", "candlestick_chart", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Vượt 500 triệu</p><p class="text-lg font-black">${Math.round(Number(probs.taxable_500m || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Vượt 1 tỷ</p><p class="text-lg font-black">${Math.round(Number(probs.einvoice_1b || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Vượt 3 tỷ</p><p class="text-lg font-black">${Math.round(Number(probs.group3_3b || 0) * 100)}%</p></div>
            </div>
            <div class="mt-3 flex items-end gap-2 h-36">
                ${intervals.slice(0, 6).map((item) => `
                    <div class="flex-1 flex flex-col items-center justify-end gap-1 min-w-0">
                        <div class="w-full rounded bg-slate-100 relative" style="height:${Math.max(12, Number(item.p90 || 0) / maxP90 * 120)}px">
                            <div class="absolute left-1/2 -translate-x-1/2 bottom-0 w-2 bg-emerald-500 rounded-t" style="height:${Math.max(6, Number(item.p50 || 0) / maxP90 * 120)}px"></div>
                        </div>
                        <p class="truncate text-[8px] text-slate-400">${UI.escapeHtml(String(item.period || "").slice(5))}</p>
                    </div>
                `).join("")}
            </div>
            <p class="mt-2 text-[10px] text-slate-400">Lưu ý: Khoảng rộng thể hiện độ bất định cao; cột xanh đại diện cho giá trị trung vị P50.</p>
        `);
    }

    function renderGraphRisk(data) {
        if (!["full", "invoice", "accounting", "expense"].includes(cfg.mode)) return;
        const summary = data.summary || {};
        const centrality = data.centrality || [];
        UI.panel("taxpayer-ai-graph-risk-panel", "Rủi ro Mạng lưới Đối tác (Heterogeneous Graph Risk)", "account_tree", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Số nút (Node)</p><p class="text-lg font-black">${summary.node_count || 0}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Số cạnh (Edge)</p><p class="text-lg font-black">${summary.edge_count || 0}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Mật độ mạng</p><p class="text-lg font-black">${Number(summary.density || 0).toFixed(3)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Điểm rủi ro mạng</p><p class="text-lg font-black">${Math.round(Number(summary.graph_risk_score || 0))}/100</p></div>
            </div>
            <div class="mt-3 overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Mã nút (Node Key)</th><th>Bậc kết nối (Degree)</th><th>Rủi ro nút</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${centrality.slice(0, 5).map((item) => `<tr><td class="py-2 font-bold">${UI.escapeHtml(item.node_key)}</td><td>${item.degree || 0}</td><td>${Math.round(Number(item.risk_score || 0))}</td></tr>`).join("") || `<tr><td colspan="3" class="py-3 text-slate-400">Chưa có dữ liệu cấu trúc mạng.</td></tr>`}
                    </tbody>
                </table>
            </div>
        `);
    }

    function renderNextBestAction(data) {
        const actions = data.actions || [];
        if (!actions.length) return;
        
        const policy = data.ranking_policy || {};
        const policyLabel = policy.method_stack ? policy.method_stack[0] : "contextual_bandit";
        const exploration = policy.exploration_rate !== undefined ? (policy.exploration_rate * 100) : 5;

        UI.panel("taxpayer-ai-nba-panel", "Hành động Gợi ý Tối ưu (Causal AI Next-Best-Action)", "moving", `
            <div class="mb-3 flex items-center justify-between text-[10px] text-slate-400 border-b border-slate-100 pb-2">
                <span>Chính sách xếp hạng: <b class="text-slate-600 font-mono">${UI.escapeHtml(policyLabel)}</b></span>
                <span>Exploration Rate: <b class="text-slate-600">${exploration}%</b></span>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${actions.slice(0, 6).map((item) => `
                    <div class="p-3 rounded-lg bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all flex flex-col justify-between">
                        <div>
                            <div class="flex items-start justify-between gap-2">
                                <p class="font-bold text-slate-800 text-xs">${UI.escapeHtml(item.title)}</p>
                                <span class="px-2 py-0.5 rounded text-[9px] font-black bg-emerald-50 text-emerald-700 border border-emerald-100 flex items-center gap-0.5" title="Uplift Score: Thể hiện giá trị gia tăng của quyết định đối với hồ sơ tuân thủ">
                                    <i class="material-symbols-outlined text-[10px]">trending_up</i>
                                    +${Math.round(Number(item.uplift_score || 0))}
                                </span>
                            </div>
                            <p class="mt-2 text-[11px] text-slate-600 leading-relaxed">${UI.escapeHtml(item.expected_impact || item.reason)}</p>
                        </div>
                        <div class="mt-3 pt-2 border-t border-slate-50 flex items-center justify-between">
                            <span class="text-[9px] text-slate-400 font-mono">Policy: ${UI.escapeHtml(item.policy || "bandit")}</span>
                            <a href="${UI.escapeHtml(item.target_page || "#")}" class="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-white text-[10px] font-bold transition-colors">
                                ${UI.escapeHtml(item.action_label || "Thực hiện")}
                            </a>
                        </div>
                    </div>
                `).join("")}
            </div>

            ${policy.fairness_guard ? `
                <div class="mt-3 text-[9px] text-slate-400 italic">
                    * ${UI.escapeHtml(policy.fairness_guard)}
                </div>
            ` : ''}
        `);
    }

    function renderGovernance(data) {
        if (!["full", "legal", "profile"].includes(cfg.mode)) return;
        const drift = data.drift || {};
        const privacy = data.privacy || {};
        const feedback = data.feedback_quality || {};
        const cards = data.model_cards || [];

        function getDriftBadge(level) {
            if (level === "low") return `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">Ổn định (Psi < 0.1)</span>`;
            if (level === "medium") return `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-amber-100 text-amber-800 border border-amber-200">Cảnh báo (Psi < 0.2)</span>`;
            return `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-rose-100 text-rose-800 border border-rose-200">Lệch nghiêm trọng</span>`;
        }

        UI.panel("taxpayer-ai-governance-panel", "Giám sát Mô hình & Độ lệch Dữ liệu (AI Governance & PSI Drift)", "shield", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chỉ số Lệch Dữ liệu (PSI)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${Math.round(Number(drift.score || 0))}/100</p>
                    <div class="mt-1">${getDriftBadge(drift.level)}</div>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Phản hồi của người dùng</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${feedback.feedback_count || 0} lượt</p>
                    <span class="text-[9px] text-slate-400 font-mono">Tích cực: ${feedback.helpful_count || 0}</span>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Đồng thuận Ngân hàng</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.escapeHtml(privacy.bank_training_consent || "chưa cấp")}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Federated Learning</p>
                    <p class="text-sm font-black text-emerald-800 mt-1">${privacy.federated_learning_ready ? "Kích hoạt" : "Vô hiệu"}</p>
                </div>
            </div>

            <!-- Model Registry Table -->
            <div class="bg-white rounded-lg border border-slate-200 p-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Danh mục mô hình đang tải & Rủi ro kiểm soát</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-[9px] uppercase text-slate-400 border-b border-slate-100">
                            <tr>
                                <th class="py-2.5">Tên Mô Hình</th>
                                <th>Trạng thái đăng ký</th>
                                <th>Cấp độ rủi ro</th>
                                <th>Cần duyệt thủ công</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-slate-700">
                            ${cards.map(card => `
                                <tr>
                                    <td class="py-3 font-bold text-indigo-950">${UI.escapeHtml(card.model)}</td>
                                    <td><span class="px-1.5 py-0.5 rounded text-[9px] bg-slate-100 border border-slate-200 text-slate-600">${UI.escapeHtml(card.status)}</span></td>
                                    <td>
                                        <span class="px-1.5 py-0.5 rounded text-[9px] font-bold ${
                                            card.risk === 'high' ? 'bg-rose-50 text-rose-800 border border-rose-100' :
                                            card.risk === 'medium' ? 'bg-amber-50 text-amber-800 border border-amber-100' :
                                            'bg-emerald-50 text-emerald-800 border border-emerald-100'
                                        }">
                                            ${UI.escapeHtml(card.risk.toUpperCase())}
                                        </span>
                                    </td>
                                    <td class="font-semibold ${card.human_review_required ? 'text-rose-600' : 'text-slate-500'}">
                                        ${card.human_review_required ? 'Yêu cầu' : 'Không'}
                                    </td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            </div>
        `);
    }

    function renderProductionReconciliation(data, caseData) {
        if (!["full", "invoice", "filing", "accounting"].includes(cfg.mode)) return;
        const summary = data.summary || {};
        const totals = data.totals || {};
        const cases = data.cases || (caseData || {}).cases || [];
        const explanation = data.explanation || {};
        const counterfactual = explanation.counterfactual || {};

        function getStatusIcon(valL, valR) {
            const delta = Math.abs(valL - valR);
            const base = Math.max(valL, valR, 1.0);
            if (delta / base < 0.08) return `<span class="material-symbols-outlined text-emerald-500 text-sm">check_circle</span>`;
            if (delta / base < 0.18) return `<span class="material-symbols-outlined text-amber-500 text-sm">warning</span>`;
            return `<span class="material-symbols-outlined text-rose-500 text-sm">error</span>`;
        }

        UI.panel("taxpayer-ai-production-reconcile-panel", "Đối soát 4 Chiều (Ngân hàng - Hóa đơn - Sổ sách - Tờ khai)", "account_tree", `
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Điểm khớp dữ liệu</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${summary.reconciliation_score || 100}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Lệch tài chính đang mở</p>
                    <p class="text-sm font-black text-rose-800 mt-1">${summary.open_case_count || cases.length || 0} trường hợp</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Doanh thu sàn TMĐT</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(totals.platform_gross || 0)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nguồn dữ liệu tích hợp</p>
                    <p class="text-[10px] font-semibold text-slate-600 mt-1">5 nguồn (API & File)</p>
                </div>
            </div>

            <!-- Reconciliation Grid -->
            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Bảng kiểm kê đối soát chéo</h5>
                <div class="space-y-3">
                    <!-- Row 1: Bank In vs Book Revenue -->
                    <div class="flex items-center justify-between text-xs py-1.5 border-b border-slate-100">
                        <span class="w-1/3 font-bold text-slate-700">Dòng tiền vào vs Sổ Doanh thu</span>
                        <div class="w-1/2 flex items-center justify-between font-mono text-[11px] text-slate-600">
                            <span>Ngân hàng: ${UI.fmtVnd(totals.bank_in || 0)}</span>
                            <span>Sổ sách: ${UI.fmtVnd(totals.book_revenue || 0)}</span>
                        </div>
                        <div class="w-10 text-right">${getStatusIcon(totals.bank_in || 0, totals.book_revenue || 0)}</div>
                    </div>
                    <!-- Row 2: Invoice Out vs Book Revenue -->
                    <div class="flex items-center justify-between text-xs py-1.5 border-b border-slate-100">
                        <span class="w-1/3 font-bold text-slate-700">Hóa đơn đầu ra vs Sổ Doanh thu</span>
                        <div class="w-1/2 flex items-center justify-between font-mono text-[11px] text-slate-600">
                            <span>Hóa đơn: ${UI.fmtVnd(totals.invoice_out || 0)}</span>
                            <span>Sổ sách: ${UI.fmtVnd(totals.book_revenue || 0)}</span>
                        </div>
                        <div class="w-10 text-right">${getStatusIcon(totals.invoice_out || 0, totals.book_revenue || 0)}</div>
                    </div>
                    <!-- Row 3: Bank Out vs Book Expense -->
                    <div class="flex items-center justify-between text-xs py-1.5 border-b border-slate-100">
                        <span class="w-1/3 font-bold text-slate-700">Dòng tiền ra vs Sổ Chi phí</span>
                        <div class="w-1/2 flex items-center justify-between font-mono text-[11px] text-slate-600">
                            <span>Ngân hàng: ${UI.fmtVnd(totals.bank_out || 0)}</span>
                            <span>Sổ sách: ${UI.fmtVnd(totals.book_expense || 0)}</span>
                        </div>
                        <div class="w-10 text-right">${getStatusIcon(totals.bank_out || 0, totals.book_expense || 0)}</div>
                    </div>
                    <!-- Row 4: Declared vs Book Revenue -->
                    <div class="flex items-center justify-between text-xs py-1.5 border-b border-slate-100">
                        <span class="w-1/3 font-bold text-slate-700">Tờ khai thuế vs Sổ Doanh thu</span>
                        <div class="w-1/2 flex items-center justify-between font-mono text-[11px] text-slate-600">
                            <span>Đã khai: ${UI.fmtVnd(totals.declared_revenue || 0)}</span>
                            <span>Sổ sách: ${UI.fmtVnd(totals.book_revenue || 0)}</span>
                        </div>
                        <div class="w-10 text-right">${getStatusIcon(totals.declared_revenue || 0, totals.book_revenue || 0)}</div>
                    </div>
                </div>
            </div>

            <!-- Exception Cases -->
            <div class="mt-3 space-y-2">
                <p class="text-[9px] uppercase font-bold text-slate-400">Các điểm lệch cần xử lý bổ sung</p>
                ${cases.slice(0, 4).map((item) => `
                    <div class="p-3 rounded-lg bg-white border border-slate-200 shadow-sm flex items-start justify-between gap-3">
                        <div class="space-y-1">
                            <div class="flex items-center gap-2">
                                <p class="font-bold text-slate-800 text-xs">${UI.escapeHtml(item.title || item.case_key)}</p>
                                <span class="px-1.5 py-0.5 rounded text-[8px] font-bold bg-${item.severity === 'high' ? 'rose' : 'amber'}-50 text-${item.severity === 'high' ? 'rose' : 'amber'}-700 border border-${item.severity === 'high' ? 'rose' : 'amber'}-100">
                                    Độ lệch: ${item.score || 0}%
                                </span>
                            </div>
                            <p class="text-[11px] text-slate-600">${UI.escapeHtml(item.description || "")}</p>
                            <div class="flex flex-wrap gap-1.5 mt-2">
                                ${(item.suggested_actions || []).map(act => `
                                    <span class="px-1.5 py-0.5 rounded bg-slate-50 border border-slate-100 text-[10px] text-slate-500">&bull; ${UI.escapeHtml(act)}</span>
                                `).join("")}
                            </div>
                        </div>
                    </div>
                `).join("") || `<div class="p-3 rounded-lg bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold text-xs">Chúc mừng! Hệ thống đối soát 4 chiều không phát hiện sai lệch đáng kể nào.</div>`}
            </div>

            ${counterfactual.import_missing_sources ? `
                <div class="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-100">
                    <p class="text-[9px] uppercase font-bold text-sky-600">Khuyến nghị hoàn thiện dữ liệu</p>
                    <p class="text-xs text-slate-700 mt-1">${UI.escapeHtml(counterfactual.import_missing_sources)}</p>
                </div>
            ` : ''}
        `, { prepend: cfg.mode === "full" });
    }

    function renderChannelAttribution(data) {
        if (!["full", "invoice", "filing", "accounting", "growth"].includes(cfg.mode)) return;
        const rows = data.attribution || [];
        UI.panel("taxpayer-ai-channel-attribution-panel", "Phân bổ Doanh thu Đa kênh", "call_split", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Tổng dòng tiền vào</p><p class="font-black">${UI.fmtVnd(data.bank_in)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Đã nhận diện kênh</p><p class="font-black">${UI.fmtVnd(data.recognized_revenue)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Chưa phân bổ</p><p class="font-black">${UI.fmtVnd(data.missing_unattributed_revenue)}</p></div>
            </div>
            <div class="mt-3 space-y-2">
                ${rows.slice(0, 6).map((item) => `
                    <div class="flex items-center gap-2 text-xs">
                        <span class="w-24 font-bold text-slate-600 truncate">${UI.escapeHtml(item.channel)}</span>
                        <div class="flex-1 h-2 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-emerald-500" style="width:${Math.min(100, Number(item.share || 0) * 100)}%"></div></div>
                        <span class="w-24 text-right font-bold">${UI.fmtVnd(item.amount)}</span>
                    </div>
                `).join("") || `<p class="text-[11px] text-slate-400">Chưa có dữ liệu phân bổ.</p>`}
            </div>
        `);
    }

    function renderTaxReserve(data) {
        if (!["full", "forecast", "filing", "debt", "calculator"].includes(cfg.mode)) return;
        const schedule = data.optimized_payment_schedule || [];
        const fan = data.cash_fan_chart || [];
        UI.panel("taxpayer-ai-tax-reserve-panel", "Tối ưu hóa Dự phòng Thuế (Tax Reserve Optimizer)", "savings", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Tỷ lệ dự phòng đề xuất</p><p class="text-lg font-black">${Math.round(Number(data.recommended_reserve_rate || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Dự phòng hàng tháng</p><p class="text-lg font-black">${UI.fmtVnd(data.monthly_reserve_amount)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Giảm phạt nộp chậm (Ước tính)</p><p class="text-lg font-black">${UI.fmtVnd(data.expected_penalty_avoided)}</p></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${schedule.map((item) => `<div class="p-2 rounded bg-white border border-slate-200 text-xs"><b>T+${item.date_offset_days || 0} ngày:</b> ${UI.fmtVnd(item.amount)}<p class="text-[10px] text-slate-500">${UI.escapeHtml(item.objective || "")}</p></div>`).join("")}
                ${fan.slice(0, 4).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200 text-xs"><b>Lũy kế ${item.horizon_days} ngày (P50):</b> ${UI.fmtVnd(item.cash_p50)}</div>`).join("")}
            </div>
        `);
    }

    function renderSupplierAccountRisk(data) {
        if (!["full", "invoice", "accounting", "expense"].includes(cfg.mode)) return;
        const alerts = data.account_change_alerts || [];
        UI.panel("taxpayer-ai-supplier-account-risk-panel", "Rủi ro Thay đổi Tài khoản Đối tác (Account Change Alerts)", "hub", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${(data.recommended_controls || []).slice(0, 3).map((item) => `<div class="p-3 rounded bg-slate-50 border border-slate-200 text-[11px] font-bold text-slate-600">${UI.escapeHtml(item)}</div>`).join("")}
            </div>
            <div class="mt-3 overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Nhà cung cấp</th><th>Số TK GD</th><th>Doanh số thanh toán</th><th>Mức độ rủi ro</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${alerts.slice(0, 6).map((item) => `<tr><td class="py-2 font-bold">${UI.escapeHtml(item.partner_name || item.supplier_key)}</td><td>${item.account_count || 0}</td><td>${UI.fmtVnd(item.payment_amount)}</td><td>${priorityBadge(item.severity)}</td></tr>`).join("") || `<tr><td colspan="4" class="py-3 text-slate-400">Chưa ghi nhận cảnh báo đổi tài khoản.</td></tr>`}
                    </tbody>
                </table>
            </div>
        `);
    }

    function renderInventoryAI(data) {
        if (!["full", "accounting", "expense", "growth"].includes(cfg.mode)) return;
        const alerts = data.alerts || [];
        UI.panel("taxpayer-ai-inventory-cogs-panel", "Phân tích Giá vốn & Tồn kho (COGS Intelligence)", "inventory_2", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Giá vốn ước tính</p><p class="font-black">${UI.fmtVnd(data.cogs_estimate)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Biên lợi nhuận gộp</p><p class="font-black">${Math.round(Number(data.gross_margin || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Số cảnh báo tồn kho</p><p class="font-black">${alerts.length}</p></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${alerts.slice(0, 4).map((item) => `<div class="p-2 rounded bg-white border border-slate-200"><div class="flex justify-between gap-2"><b>${UI.escapeHtml(item.type)}</b>${priorityBadge(item.severity)}</div><p class="text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p></div>`).join("") || `<div class="p-2 rounded bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chưa phát hiện bất thường tồn kho/giá vốn.</div>`}
            </div>
        `);
    }

    function renderEvidenceBundle(data) {
        if (!["full", "filing", "legal", "claim", "expense"].includes(cfg.mode)) return;
        const sections = data.sections || [];
        UI.panel("taxpayer-ai-evidence-bundle-panel", "Bộ Hồ sơ Chứng minh Tự động (Evidence Bundle AI)", "folder_managed", `
            <div class="flex items-center justify-between gap-3">
                <div><p class="text-[9px] uppercase font-bold text-slate-400">Điểm hoàn thiện hồ sơ</p><p class="text-lg font-black">${Math.round(Number(data.bundle_score || 0))}/100</p></div>
                ${UI.statusBadge(data.readiness || "needs_review")}
            </div>
            <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
                ${sections.slice(0, 8).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><p class="text-[10px] font-bold text-slate-600">${UI.escapeHtml(item.title)}</p><p class="text-sm font-black">${item.item_count || 0} tài liệu</p></div>`).join("")}
            </div>
        `);
    }

    function renderLegalChangeImpact(data) {
        if (!["full", "legal", "growth", "calculator"].includes(cfg.mode)) return;
        const alerts = data.change_alerts || data.impacts || [];
        UI.panel("taxpayer-ai-legal-change-panel", "Đánh giá Tác động Luật Thuế mới (Legal Change Impact)", "policy", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${alerts.slice(0, 4).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2"><p class="font-bold text-slate-800 text-xs">${UI.escapeHtml(item.title)}</p>${priorityBadge(item.severity)}</div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message || item.action || "")}</p>
                    </div>
                `).join("")}
            </div>
            <div class="mt-3 flex flex-wrap gap-2">
                ${(data.citations || []).slice(0, 4).map((item) => `<a class="px-2 py-1 rounded bg-white border border-slate-200 text-[10px] font-bold text-slate-600" href="${UI.escapeHtml(item.source_url || "#")}" target="_blank">${UI.escapeHtml(item.article_ref || item.title || "nguồn dẫn")}</a>`).join("")}
            </div>
        `);
    }

    function renderProductionGovernance(data) {
        if (!["full", "profile", "legal"].includes(cfg.mode)) return;
        const gates = data.production_gates || [];
        const readiness = data.connector_readiness || {};
        UI.panel("taxpayer-ai-production-governance-panel", "Kiểm soát Vận hành Mô hình (Production MLOps Gates)", "admin_panel_settings", `
            <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                ${Object.entries(readiness).map(([key, value]) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">${UI.escapeHtml(key)}</p><p class="font-black">${value || 0}</p></div>`).join("")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${gates.map((item) => `<div class="p-2 rounded border ${item.pass ? "border-emerald-100 bg-emerald-50 text-emerald-700" : "border-amber-100 bg-amber-50 text-amber-700"} text-[11px] font-bold">${item.pass ? "ĐẠT (PASS)" : "CHỜ DUYỆT"} - ${UI.escapeHtml(item.gate)}</div>`).join("")}
            </div>
        `);
    }

    function renderCopilot() {
        UI.panel("taxpayer-ai-copilot-panel", "Trợ lý AI Đa ngữ cảnh (AI Copilot)", "support_agent", `
            <div class="flex flex-col md:flex-row gap-2">
                <input id="taxpayer-ai-copilot-input" class="flex-1 rounded-lg border-slate-200 text-xs" placeholder="Hỏi về dòng tiền, hóa đơn, chính sách thuế, chuyển đổi mô hình..." />
                <button id="taxpayer-ai-copilot-btn" class="px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-bold">Gửi câu hỏi</button>
            </div>
            <div id="taxpayer-ai-copilot-result" class="mt-3 text-[11px] text-slate-600"></div>
        `);
        const input = document.getElementById("taxpayer-ai-copilot-input");
        const btn = document.getElementById("taxpayer-ai-copilot-btn");
        const send = async () => {
            const question = input ? input.value : "";
            if (!question.trim()) return;
            const resultBox = document.getElementById("taxpayer-ai-copilot-result");
            if (resultBox) resultBox.innerHTML = `<p class="font-bold text-slate-400">Đang phân tích dữ liệu...</p>`;
            try {
                const data = await UI.post("/intelligence/copilot", { question, page });
                if (resultBox) {
                    resultBox.innerHTML = `
                        <p class="font-bold text-slate-800">${UI.escapeHtml(data.answer)}</p>
                        <div class="mt-2 flex flex-wrap gap-2">
                            ${(data.actions || []).map((item) => `<a class="px-2 py-1 rounded bg-slate-100 text-slate-700 font-bold" href="${UI.escapeHtml(item.target_page || page)}">${UI.escapeHtml(item.label || "Mở")}</a>`).join("")}
                        </div>
                    `;
                }
            } catch (e) {
                if (resultBox) resultBox.innerHTML = `<p class="font-bold text-rose-600">${UI.escapeHtml(e.message || "Không thể kết nối đến Trợ lý AI.")}</p>`;
            }
        };
        if (btn) btn.onclick = send;
        if (input) input.onkeydown = (event) => { if (event.key === "Enter") send(); };
    }

    async function bindFeedback() {
        document.querySelectorAll(".ai-feedback").forEach((btn) => {
            btn.onclick = async () => {
                try {
                    await UI.post("/intelligence/feedback", {
                        target_type: "recommendation",
                        target_id: btn.dataset.target,
                        signal: btn.dataset.signal,
                        page,
                    });
                    UI.toast("Da ghi nhan phan hoi AI.");
                } catch (e) {
                    UI.toast(e.message || "Khong luu duoc phan hoi.", "error");
                }
            };
        });
    }

    function renderBenfordAnalysis(data) {
        if (!["full", "accounting", "invoice", "expense"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const digits = data.digits || [];
        const flagged = data.flagged_digits || [];
        const verdict = data.verdict || "unknown";
        const verdictColor = verdict === "conforming" ? "emerald" : verdict === "significant_deviation" ? "rose" : verdict === "moderate_deviation" ? "amber" : "slate";
        const maxPct = Math.max(...digits.map(d => Math.max(d.observed_pct || 0, d.expected_pct || 0)), 35);

        UI.panel("taxpayer-ai-benford-panel", "Luật Benford — Kiểm định Gian lận Số liệu", "search_insights", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${verdictColor}-50 border border-${verdictColor}-200 md:col-span-2">
                    <p class="text-[9px] uppercase font-bold text-${verdictColor}-600">Kết luận kiểm định</p>
                    <p class="text-sm font-black text-${verdictColor}-800">${UI.escapeHtml(data.verdict_label || "Đang phân tích...")}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chi-square</p>
                    <p class="text-lg font-black text-slate-800">${Number(data.chi_square || 0).toFixed(2)}</p>
                    <p class="text-[9px] text-slate-400">Bậc tự do (df)=${data.degrees_of_freedom || 8}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">p-value</p>
                    <p class="text-lg font-black text-slate-800">${Number(data.p_value || 0).toFixed(4)}</p>
                    <p class="text-[9px] text-slate-400">Mẫu n=${data.sample_size || 0}</p>
                </div>
            </div>
            <div class="bg-white rounded-lg border border-slate-200 p-4">
                <p class="text-[9px] uppercase font-bold text-slate-400 mb-3">Phân phối chữ số đầu tiên: Thực tế vs. Benford</p>
                <div class="flex items-end gap-1" style="height:180px">
                    ${digits.map(d => {
                        const obsH = Math.max(2, (d.observed_pct / maxPct) * 160);
                        const expH = Math.max(2, (d.expected_pct / maxPct) * 160);
                        const isFlagged = flagged.some(f => f.digit === d.digit);
                        const barColor = isFlagged ? (flagged.find(f => f.digit === d.digit).severity === "high" ? "bg-rose-500" : "bg-amber-400") : "bg-sky-500";
                        return `<div class="flex-1 flex flex-col items-center gap-0.5">
                            <div class="w-full flex justify-center items-end gap-px" style="height:164px">
                                <div class="${barColor} rounded-t w-[40%] transition-all" style="height:${obsH}px" title="Thực tế: ${d.observed_pct}%"></div>
                                <div class="bg-slate-300 rounded-t w-[40%] transition-all" style="height:${expH}px" title="Benford: ${d.expected_pct}%"></div>
                            </div>
                            <span class="text-[10px] font-bold ${isFlagged ? 'text-rose-600' : 'text-slate-600'}">${d.digit}</span>
                        </div>`;
                    }).join("")}
                </div>
                <div class="flex items-center gap-4 mt-3">
                    <div class="flex items-center gap-1"><div class="w-3 h-3 bg-sky-500 rounded"></div><span class="text-[9px] text-slate-500">Thực tế</span></div>
                    <div class="flex items-center gap-1"><div class="w-3 h-3 bg-slate-300 rounded"></div><span class="text-[9px] text-slate-500">Benford (Lý thuyết)</span></div>
                    <div class="flex items-center gap-1"><div class="w-3 h-3 bg-rose-500 rounded"></div><span class="text-[9px] text-slate-500">Sai lệch lớn</span></div>
                    <div class="flex items-center gap-1"><div class="w-3 h-3 bg-amber-400 rounded"></div><span class="text-[9px] text-slate-500">Sai lệch vừa</span></div>
                </div>
            </div>
            ${flagged.length ? `
                <div class="mt-3 space-y-2">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chữ số bất thường</p>
                    ${flagged.map(f => `
                        <div class="p-2 rounded-lg bg-${f.severity === 'high' ? 'rose' : 'amber'}-50 border border-${f.severity === 'high' ? 'rose' : 'amber'}-200">
                            <div class="flex items-center justify-between">
                                <span class="font-bold text-slate-800">Chữ số ${f.digit}</span>
                                ${priorityBadge(f.severity)}
                            </div>
                            <p class="text-[11px] text-slate-600 mt-1">${UI.escapeHtml(f.message)}</p>
                        </div>
                    `).join("")}
                </div>
            ` : ''}
            <div class="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-100">
                <p class="text-[9px] uppercase font-bold text-sky-600">Phương pháp luận</p>
                <p class="text-[11px] text-sky-800 mt-1">${UI.escapeHtml((data.explanation || {}).methodology || '')}</p>
            </div>
        `);
    }

    function renderSeasonalDecomposition(data) {
        if (!["full", "forecast", "growth", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;

        const series = data.series || [];
        const strength = data.seasonal_strength || 0;
        const trendDir = data.trend_direction || "stable";
        const trendChange = data.trend_change_pct || 0;
        const peak = data.peak_month;
        const trough = data.trough_month;
        const insights = data.insights || [];

        // Colors based on trend direction
        const trendColor = trendDir === "up" ? "emerald" : trendDir === "down" ? "rose" : "slate";
        const trendIcon = trendDir === "up" ? "trending_up" : trendDir === "down" ? "trending_down" : "trending_flat";
        const trendText = trendDir === "up" ? "Tăng trưởng" : trendDir === "down" ? "Suy giảm" : "Ổn định";

        // Find max values for scaling the bars
        const maxVal = Math.max(...series.map(s => Math.max(s.original || 0, s.trend || 0, s.expense || 0)), 1);
        const maxSeasonal = Math.max(...series.map(s => Math.abs(s.seasonal || 0)), 1);

        UI.panel("taxpayer-ai-seasonal-panel", "Phân tích Mùa vụ & Xu hướng (STL Decomposition)", "insights", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Sức mạnh mùa vụ</p>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span class="text-lg font-black text-slate-800">${strength}%</span>
                        <span class="text-[10px] font-bold text-emerald-600">${UI.escapeHtml(data.seasonal_label || "")}</span>
                    </div>
                    <div class="mt-2 h-1 bg-slate-200 rounded-full overflow-hidden">
                        <div class="h-full bg-emerald-500" style="width: ${strength}%"></div>
                    </div>
                </div>

                <div class="p-3 rounded-lg bg-${trendColor}-50 border border-${trendColor}-200">
                    <p class="text-[9px] uppercase font-bold text-${trendColor}-600">Xu hướng dài hạn</p>
                    <div class="flex items-center gap-1 mt-1">
                        <span class="material-symbols-outlined text-${trendColor}-600 text-lg">${trendIcon}</span>
                        <span class="text-sm font-black text-${trendColor}-800">${trendText} (${trendChange >= 0 ? '+' : ''}${trendChange}%)</span>
                    </div>
                </div>

                <div class="p-3 rounded-lg bg-amber-50 border border-amber-200">
                    <p class="text-[9px] uppercase font-bold text-amber-600">Tháng cao điểm (Peak)</p>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span class="text-lg font-black text-amber-800">${peak ? 'Tháng ' + peak : 'N/A'}</span>
                        <span class="text-[9px] text-amber-600">Doanh thu lớn nhất</span>
                    </div>
                </div>

                <div class="p-3 rounded-lg bg-rose-50 border border-rose-200">
                    <p class="text-[9px] uppercase font-bold text-rose-600">Tháng thấp điểm (Trough)</p>
                    <div class="flex items-baseline gap-1 mt-1">
                        <span class="text-lg font-black text-rose-800">${trough ? 'Tháng ' + trough : 'N/A'}</span>
                        <span class="text-[9px] text-rose-600">Doanh thu thấp nhất</span>
                    </div>
                </div>
            </div>

            <!-- Interactive STL components visual representation -->
            <div class="bg-white rounded-lg border border-slate-200 p-4 space-y-4">
                <div>
                    <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2 flex items-center justify-between">
                        <span>1. Doanh thu Thực tế (Xanh) vs. Xu hướng (Xám)</span>
                        <span class="text-[8px] font-normal lowercase text-slate-400">Scale cực đại: ${UI.fmtVnd(maxVal)}</span>
                    </h5>
                    <div class="flex items-end gap-1.5 h-20 pt-2 border-b border-slate-100 pb-1">
                        ${series.map(s => {
                            const actH = (s.original / maxVal) * 100;
                            const trdH = (s.trend / maxVal) * 100;
                            return `
                                <div class="flex-1 flex items-end gap-px h-full group relative" title="T${s.month}: ${UI.fmtVnd(s.original)}">
                                    <div class="w-1/2 bg-sky-400 group-hover:bg-sky-500 rounded-t transition-all" style="height: ${Math.max(2, actH)}%"></div>
                                    <div class="w-1/2 bg-slate-300 group-hover:bg-slate-400 rounded-t transition-all" style="height: ${Math.max(2, trdH)}%"></div>
                                </div>
                            `;
                        }).join("")}
                    </div>
                </div>

                <div>
                    <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2 flex items-center justify-between">
                        <span>2. Biến động Mùa vụ (Seasonal component)</span>
                        <span class="text-[8px] font-normal lowercase text-slate-400">Độ lệch cực đại: ±${UI.fmtVnd(maxSeasonal)}</span>
                    </h5>
                    <div class="relative h-20 pt-1 pb-1 border-b border-slate-100">
                        <div class="absolute top-1/2 left-0 right-0 h-px bg-slate-200"></div>
                        <div class="absolute inset-0 flex items-end gap-1.5">
                            ${series.map(s => {
                                const val = s.seasonal || 0;
                                const barH = Math.abs(val / maxSeasonal) * 50; // Max 50% height either way
                                const isPos = val >= 0;
                                return `
                                    <div class="flex-1 flex flex-col h-full relative group" title="S${s.month}: ${val >= 0 ? '+' : ''}${UI.fmtVnd(val)}">
                                        <div class="absolute left-0 right-0 w-full rounded transition-all ${isPos ? 'bottom-1/2 bg-emerald-400 group-hover:bg-emerald-500' : 'top-1/2 bg-amber-400 group-hover:bg-amber-500'}" 
                                             style="height: ${Math.max(2, barH)}%"></div>
                                    </div>
                                `;
                            }).join("")}
                        </div>
                    </div>
                </div>

                <div class="flex items-center justify-between text-[8px] text-slate-400 px-1">
                    ${series.map(s => `<span>T${s.month}</span>`).join("")}
                </div>
            </div>

            <!-- Insights list -->
            <div class="mt-3 space-y-2">
                <p class="text-[9px] uppercase font-bold text-slate-400">Gợi ý từ mô hình phân tích STL</p>
                ${insights.map(item => `
                    <div class="p-2.5 rounded-lg border bg-white border-slate-200 flex items-start gap-2">
                        <span class="material-symbols-outlined text-${item.severity === 'high' ? 'rose' : item.severity === 'medium' ? 'amber' : 'sky'}-500 text-sm mt-0.5">
                            ${item.severity === 'high' ? 'warning' : 'info'}
                        </span>
                        <p class="text-[11px] text-slate-700">${UI.escapeHtml(item.message)}</p>
                    </div>
                `).join("")}
            </div>

            <div class="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-100">
                <p class="text-[9px] uppercase font-bold text-sky-600">Phương pháp đồng dạng</p>
                <p class="text-[11px] text-sky-800 mt-1">${UI.escapeHtml((data.explanation || {}).methodology || '')}</p>
            </div>
        `);
    }

    function renderSurvivalAnalysis(data) {
        if (!["full", "debt"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;

        const series = data.series || [];
        const index = data.survival_index || 100;
        const verdict = data.verdict || "low_risk";
        const verdictLabel = data.verdict_label || "Bình thường";
        const verdictColor = data.verdict_color || "slate";
        const medianVal = data.median_survival_months || ">12 tháng";
        const hr = data.hazard_ratio || 1.0;
        const insights = data.insights || [];

        UI.panel("taxpayer-ai-survival-panel", "Mô hình Phân tích Sinh tồn & Chậm nộp Thuế", "timer", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${verdictColor}-50 border border-${verdictColor}-200 md:col-span-2">
                    <p class="text-[9px] uppercase font-bold text-${verdictColor}-600">Trạng thái rủi ro</p>
                    <p class="text-sm font-black text-${verdictColor}-800">${UI.escapeHtml(verdictLabel)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chỉ số Sinh tồn (12T)</p>
                    <p class="text-lg font-black text-slate-800">${index}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tỷ lệ rủi ro (Hazard Ratio)</p>
                    <div class="flex items-baseline gap-1">
                        <span class="text-lg font-black text-slate-800">${hr}x</span>
                        <span class="text-[9px] text-slate-400">vs. baseline</span>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div class="bg-white rounded-lg border border-slate-200 p-4 lg:col-span-2">
                    <p class="text-[9px] uppercase font-bold text-slate-400 mb-3">Biểu đồ Xác suất đóng thuế đúng hạn (Survival Curve) & Nguy cơ chậm nộp (Hazard Rate)</p>
                    <div class="flex items-end gap-1.5 h-28 pt-2 border-b border-slate-100 pb-1">
                        ${series.map(s => {
                            const survH = s.survival_probability_pct; // 0 to 100
                            const hazH = s.hazard_rate_pct * 4; // scaled for visibility
                            return `
                                <div class="flex-1 flex items-end justify-center gap-px h-full group relative" title="Tháng ${s.month}\n- Xác suất duy trì: ${s.survival_probability_pct}%\n- Nguy cơ xảy ra chậm nộp: ${s.hazard_rate_pct}%">
                                    <div class="w-1/2 bg-indigo-500 group-hover:bg-indigo-600 rounded-t transition-all" style="height: ${Math.max(4, survH)}%"></div>
                                    <div class="w-1/2 bg-rose-400 group-hover:bg-rose-500 rounded-t transition-all" style="height: ${Math.max(2, hazH)}%"></div>
                                </div>
                            `;
                        }).join("")}
                    </div>
                    <div class="flex items-center justify-between text-[8px] text-slate-400 px-1 mt-1">
                        ${series.map(s => `<span>T${s.month}</span>`).join("")}
                    </div>
                    <div class="flex items-center gap-4 mt-3">
                        <div class="flex items-center gap-1"><div class="w-3 h-1.5 bg-indigo-500 rounded"></div><span class="text-[9px] text-slate-500">Xác suất Sinh tồn (Không chậm nộp)</span></div>
                        <div class="flex items-center gap-1"><div class="w-3 h-1.5 bg-rose-400 rounded"></div><span class="text-[9px] text-slate-500">Tỷ lệ nguy hiểm (Hazard)</span></div>
                    </div>
                </div>

                <div class="space-y-3">
                    <div class="p-3 bg-white border border-slate-200 rounded-lg">
                        <p class="text-[9px] uppercase font-bold text-slate-400">Thời gian trung vị đến trễ hạn (Median Survival)</p>
                        <p class="text-sm font-black text-slate-700 mt-1">${UI.escapeHtml(medianVal)}</p>
                        <p class="text-[10px] text-slate-400 leading-normal mt-1">Ước tính khoảng thời gian bình quân trước khi phát sinh chậm nộp thuế nếu giữ nguyên trạng thái tài chính.</p>
                    </div>

                    <div class="p-3 bg-indigo-50/50 border border-indigo-100 rounded-lg">
                        <p class="text-[9px] uppercase font-bold text-indigo-700">Ý nghĩa chỉ số</p>
                        <p class="text-[10.5px] text-indigo-900 leading-relaxed mt-1">
                            Nếu chỉ số dưới 50%, doanh nghiệp có nguy cơ cao phát sinh nợ thuế lớn trong vòng 3-6 tháng tới. Cần có kế hoạch giãn nợ hoặc thanh toán ưu tiên.
                        </p>
                    </div>
                </div>
            </div>

            <!-- Insights list -->
            <div class="mt-3 space-y-2">
                <p class="text-[9px] uppercase font-bold text-slate-400">Phân tích hành vi & Gợi ý từ mô hình</p>
                ${insights.map(item => `
                    <div class="p-2.5 rounded-lg border bg-white border-slate-200 flex items-start gap-2">
                        <span class="material-symbols-outlined text-${item.severity === 'high' ? 'rose' : item.severity === 'medium' ? 'amber' : 'sky'}-500 text-sm mt-0.5">
                            ${item.severity === 'high' ? 'error' : item.severity === 'medium' ? 'warning' : 'info'}
                        </span>
                        <p class="text-[11px] text-slate-700">${UI.escapeHtml(item.message)}</p>
                    </div>
                `).join("")}
            </div>

            <div class="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-100">
                <p class="text-[9px] uppercase font-bold text-sky-600">Giải thích Mô hình AI</p>
                <p class="text-[11px] text-sky-800 mt-1">${UI.escapeHtml((data.explanation || {}).methodology || '')}</p>
            </div>
        `);
    }

    function renderBayesianForecast(data) {
        if (!["full", "forecast", "growth"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;

        const series = data.series || [];
        const histCount = data.historical_months_count || 0;
        const postMean = data.posterior_mean || 0;
        const postStd = data.posterior_std || 0;
        const trend = data.estimated_monthly_trend_pct || 0;
        const confidence = data.confidence || "medium";
        const insights = data.insights || [];

        // Find max value to scale the fan chart
        const maxVal = Math.max(...series.map(s => s.hdi_95_upper), 1000000);

        // Colors based on trend direction
        const trendColor = trend >= 0 ? "emerald" : "rose";
        const trendIcon = trend >= 0 ? "trending_up" : "trending_down";

        UI.panel("taxpayer-ai-bayesian-forecast-panel", "Dự báo Doanh thu Bayesian & Khoảng bất định (HDI)", "analytics", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Doanh thu Kỳ vọng (Trung vị)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(postMean)}</p>
                </div>
                <div class="p-3 rounded-lg bg-${trendColor}-50 border border-${trendColor}-200">
                    <p class="text-[9px] uppercase font-bold text-${trendColor}-600">Xu hướng hàng tháng</p>
                    <div class="flex items-center gap-1 mt-1">
                        <span class="material-symbols-outlined text-${trendColor}-600 text-base">${trendIcon}</span>
                        <span class="text-xs font-black text-${trendColor}-800">${trend >= 0 ? '+' : ''}${trend}% / tháng</span>
                    </div>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Độ tin cậy mô hình</p>
                    <p class="text-sm font-black text-slate-800 mt-1 uppercase text-${confidence === 'high' ? 'emerald' : confidence === 'medium' ? 'indigo' : 'amber'}-600">${confidence}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Số lượng tháng lịch sử</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${histCount} tháng</p>
                </div>
            </div>

            <!-- Bayesian Fan Chart -->
            <div class="bg-white rounded-lg border border-slate-200 p-4 space-y-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2 flex items-center justify-between">
                    <span>Biểu đồ Phễu Bất định Dự báo Doanh thu (6 Tháng tới)</span>
                    <span class="text-[8px] font-normal lowercase text-slate-400">Scale max: ${UI.fmtVnd(maxVal)}</span>
                </h5>
                
                <div class="flex gap-2 h-36 pt-2 border-b border-slate-100 pb-2 relative">
                    ${series.map(s => {
                        const m95_l = (s.hdi_95_lower / maxVal) * 100;
                        const m95_u = (s.hdi_95_upper / maxVal) * 100;
                        const m80_l = (s.hdi_80_lower / maxVal) * 100;
                        const m80_u = (s.hdi_80_upper / maxVal) * 100;
                        const m_expected = (s.expected_mean / maxVal) * 100;
                        
                        return `
                            <div class="flex-1 h-full relative group" title="Tháng dự báo thứ ${s.month}\n- Kỳ vọng: ${UI.fmtVnd(s.expected_mean)}\n- Khoảng 80% HDI: ${UI.fmtVnd(s.hdi_80_lower)} - ${UI.fmtVnd(s.hdi_80_upper)}\n- Khoảng 95% HDI: ${UI.fmtVnd(s.hdi_95_lower)} - ${UI.fmtVnd(s.hdi_95_upper)}">
                                <!-- 95% HDI Zone -->
                                <div class="absolute left-1 right-1 bg-indigo-50 border-l border-r border-indigo-100 rounded opacity-60" style="bottom: ${m95_l}%; height: ${m95_u - m95_l}%"></div>
                                <!-- 80% HDI Zone -->
                                <div class="absolute left-2 right-2 bg-indigo-100 border-l border-r border-indigo-200 rounded" style="bottom: ${m80_l}%; height: ${m80_u - m80_l}%"></div>
                                <!-- Expected Mean Marker -->
                                <div class="absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-indigo-600 rounded-full border border-white shadow-sm z-10" style="bottom: calc(${m_expected}% - 4px)"></div>
                            </div>
                        `;
                    }).join("")}
                </div>

                <div class="flex items-center justify-between text-[8px] text-slate-400 px-1">
                    ${series.map(s => `<span>Tháng +${s.month}</span>`).join("")}
                </div>

                <div class="flex items-center gap-4 mt-3 flex-wrap">
                    <div class="flex items-center gap-1.5"><div class="w-3 h-3 bg-indigo-600 rounded-full"></div><span class="text-[9px] text-slate-500">Giá trị kỳ vọng (Expected Mean)</span></div>
                    <div class="flex items-center gap-1.5"><div class="w-3.5 h-2.5 bg-indigo-100 rounded"></div><span class="text-[9px] text-slate-500">Khoảng tin cậy 80% HDI (Khả thi cao)</span></div>
                    <div class="flex items-center gap-1.5"><div class="w-3.5 h-2.5 bg-indigo-50 rounded border border-indigo-100"></div><span class="text-[9px] text-slate-500">Khoảng tin cậy 95% HDI (Biên độ dao động cực đại)</span></div>
                </div>
            </div>

            <!-- Insights list -->
            <div class="mt-3 space-y-2">
                <p class="text-[9px] uppercase font-bold text-slate-400">Nhận định & Khuyến nghị quản trị</p>
                ${insights.map(item => `
                    <div class="p-2.5 rounded-lg border bg-white border-slate-200 flex items-start gap-2">
                        <span class="material-symbols-outlined text-${item.severity === 'high' ? 'rose' : item.severity === 'medium' ? 'amber' : 'sky'}-500 text-sm mt-0.5">
                            ${item.severity === 'high' ? 'error' : item.severity === 'medium' ? 'warning' : 'info'}
                        </span>
                        <p class="text-[11px] text-slate-700">${UI.escapeHtml(item.message)}</p>
                    </div>
                `).join("")}
            </div>

            <div class="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-100">
                <p class="text-[9px] uppercase font-bold text-sky-600">Phương pháp luận toán học</p>
                <p class="text-[11px] text-sky-800 mt-1">${UI.escapeHtml((data.explanation || {}).methodology || '')}</p>
                <p class="text-[10px] text-sky-600 mt-1">Gợi ý tối ưu: ${UI.escapeHtml((data.explanation || {}).counterfactual.more_history || '')}</p>
            </div>
        `);
    }

    function renderExplainability(data) {
        if (!["full", "forecast", "accounting", "growth"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;

        const contributions = data.contributions || [];
        const baseVal = data.base_value || 0;
        const totalRisk = data.compliance_risk_score || 0;
        const counterfactual = (data.explanation || {}).counterfactual || {};

        // Find max SHAP value to normalize the bars
        const maxShap = Math.max(...contributions.map(c => Math.abs(c.shap_value)), 1);

        UI.panel("taxpayer-ai-explainability-panel", "Giải thích Mô hình AI & Phân bổ Rủi ro (SHAP Attribution)", "psychology", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Rủi ro cơ sở (Base Value)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${baseVal}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Điểm Rủi ro Tổng hợp</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${totalRisk}%</p>
                </div>
                <div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200">
                    <p class="text-[9px] uppercase font-bold text-indigo-600">Thuật toán Giải thích</p>
                    <p class="text-xs font-black text-indigo-800 mt-1">SHAP (Kernel Explainer)</p>
                </div>
            </div>

            <!-- SHAP Force Plot / Bar chart -->
            <div class="bg-white rounded-lg border border-slate-200 p-4 space-y-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2">Đóng góp của các đặc trưng tài chính vào Điểm Rủi ro</h5>
                
                <div class="space-y-4">
                    ${contributions.map(c => {
                        const isRisk = c.direction === "risk";
                        const pctWidth = (Math.abs(c.shap_value) / maxShap) * 50; // Max width is 50% from center
                        const barColor = isRisk ? "bg-rose-500" : "bg-emerald-500";
                        const alignmentStyle = isRisk 
                            ? `left: 50%; width: ${pctWidth}%;` 
                            : `right: 50%; width: ${pctWidth}%;`;

                        return `
                            <div class="grid grid-cols-12 gap-2 items-center text-xs">
                                <!-- Feature Name -->
                                <div class="col-span-4 min-w-0">
                                    <p class="font-bold text-slate-700 truncate">${UI.escapeHtml(c.feature_label)}</p>
                                    <span class="text-[9px] text-slate-400 font-mono">${UI.escapeHtml(c.feature_value)}</span>
                                </div>
                                
                                <!-- SHAP Bar Track -->
                                <div class="col-span-5 h-6 relative bg-slate-50 rounded overflow-hidden border border-slate-100">
                                    <!-- Center line (baseline representation) -->
                                    <div class="absolute left-1/2 top-0 bottom-0 w-px bg-slate-300 z-10"></div>
                                    <!-- SHAP value bar -->
                                    <div class="absolute top-1 bottom-1 ${barColor} rounded-sm transition-all" style="${alignmentStyle}"></div>
                                    <!-- Value label -->
                                    <span class="absolute top-1/2 -translate-y-1/2 text-[9px] font-mono font-bold z-20 ${isRisk ? 'left-[calc(50%+4px)] text-rose-800' : 'right-[calc(50%+4px)] text-emerald-800'}">
                                        ${c.shap_value >= 0 ? '+' : ''}${c.shap_value.toFixed(1)}%
                                    </span>
                                </div>
                                
                                <!-- Description -->
                                <div class="col-span-3 text-[10px] text-slate-500 leading-tight">
                                    ${UI.escapeHtml(c.description)}
                                </div>
                            </div>
                        `;
                    }).join("")}
                </div>

                <div class="flex items-center justify-between text-[8px] text-slate-400 border-t border-slate-100 pt-2 px-1">
                    <span>&larr; Giảm rủi ro (Compliance Anchor)</span>
                    <span class="font-mono">Điểm trung tuyến (Baseline)</span>
                    <span>Tăng rủi ro (Risk Amplification) &rarr;</span>
                </div>
            </div>

            <!-- Counterfactual / Optimization -->
            ${counterfactual.reduce_risk ? `
                <div class="mt-3 p-3 rounded-lg bg-emerald-50 border border-emerald-100">
                    <p class="text-[9px] uppercase font-bold text-emerald-600">Đề xuất cải thiện điểm tuân thủ (Counterfactual Action)</p>
                    <p class="text-xs text-emerald-950 mt-1">${UI.escapeHtml(counterfactual.reduce_risk)}</p>
                </div>
            ` : ''}

            <div class="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-100">
                <p class="text-[9px] uppercase font-bold text-sky-600">Giải thích phương pháp SHAP</p>
                <p class="text-[11px] text-sky-800 mt-1">${UI.escapeHtml((data.explanation || {}).methodology || '')}</p>
            </div>
        `);
    }

    function renderAutoencoderAnomaly(data) {
        if (!["full", "accounting", "invoice"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const anomalies = data.anomalies || [];
        const summary = data.summary || {};
        const threshold = data.threshold || 0;
        
        UI.panel("taxpayer-ai-autoencoder-panel", "Phát hiện bất thường giao dịch ngân hàng (Autoencoder DL)", "savings", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng giao dịch</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${summary.total || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Giao dịch nghi vấn</p>
                    <p class="text-sm font-black text-rose-600 mt-1">${summary.flagged || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Ngưỡng tái cấu trúc (Threshold)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(threshold)}</p>
                </div>
                <div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200">
                    <p class="text-[9px] uppercase font-bold text-indigo-600">Chiều không gian ẩn (Latent)</p>
                    <p class="text-xs font-black text-indigo-800 mt-1">${data.latent_dim || 8} dimensions</p>
                </div>
            </div>
            
            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Danh sách giao dịch có lỗi tái cấu trúc cao</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-[9px] uppercase text-slate-400 border-b border-slate-100">
                            <tr>
                                <th class="py-2">Ngày GD</th>
                                <th>Đối tác</th>
                                <th>Loại</th>
                                <th class="text-right">Số tiền</th>
                                <th class="text-center">Reconstruction Error</th>
                                <th class="text-center">Trạng thái</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-slate-700">
                            ${anomalies.slice(0, 6).map(a => `
                                <tr>
                                    <td class="py-2.5 font-mono">${UI.escapeHtml(a.date || "")}</td>
                                    <td>${UI.escapeHtml(a.counterparty)}</td>
                                    <td class="font-bold uppercase ${a.direction === 'out' ? 'text-rose-600' : 'text-emerald-600'}">${a.direction === 'out' ? 'Chi' : 'Thu'}</td>
                                    <td class="text-right font-mono font-bold">${UI.fmtVnd(a.amount)}</td>
                                    <td class="text-center font-mono">${a.reconstruction_error}%</td>
                                    <td class="text-center">
                                        <span class="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                                            a.is_anomaly ? 'bg-rose-100 text-rose-800' : 'bg-slate-100 text-slate-700'
                                        }">
                                            ${a.is_anomaly ? 'Nghi vấn' : 'Bình thường'}
                                        </span>
                                    </td>
                                </tr>
                            `).join("") || `<tr><td colspan="6" class="py-3 text-center text-slate-400">Không phát hiện giao dịch bất thường.</td></tr>`}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Đề xuất Autoencoder:</strong> <span>${data.explanation?.counterfactual?.recurring_payments || ""}</span>
                </div>
            </div>
        `);
    }

    function renderRfmSegmentation(data) {
        if (!["full", "growth", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const customers = data.customers || [];
        const summary = data.summary || {};
        
        UI.panel("taxpayer-ai-rfm-panel", "Phân khúc khách hàng & CLV (RFM Clustering)", "group", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng số khách hàng</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${summary.total_customers || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                    <p class="text-[9px] uppercase font-bold text-emerald-600">Khách hàng VIP (Champions)</p>
                    <p class="text-sm font-black text-emerald-800 mt-1">${summary.champions || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-amber-50 border border-amber-200">
                    <p class="text-[9px] uppercase font-bold text-amber-600">Khách hàng rủi ro rời bỏ</p>
                    <p class="text-sm font-black text-amber-800 mt-1">${summary.at_risk || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-rose-50 border border-rose-200">
                    <p class="text-[9px] uppercase font-bold text-rose-600">Khách hàng đã mất</p>
                    <p class="text-sm font-black text-rose-800 mt-1">${summary.lost || 0}</p>
                </div>
            </div>
            
            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Xếp hạng giá trị vòng đời khách hàng (CLV)</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-[9px] uppercase text-slate-400 border-b border-slate-100">
                            <tr>
                                <th class="py-2">Khách hàng</th>
                                <th class="text-center">Gần đây (Ngày)</th>
                                <th class="text-center">Tần suất</th>
                                <th class="text-right">Tổng giá trị mua</th>
                                <th class="text-center">Phân khúc</th>
                                <th class="text-right">Ước tính CLV (12T)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-slate-700">
                            ${customers.slice(0, 6).map(c => `
                                <tr>
                                    <td class="py-2.5 font-bold">${UI.escapeHtml(c.customer)}</td>
                                    <td class="text-center font-mono">${c.recency_days}d</td>
                                    <td class="text-center font-mono">${c.frequency} lần</td>
                                    <td class="text-right font-mono font-bold">${UI.fmtVnd(c.monetary)}</td>
                                    <td class="text-center">
                                        <span class="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                                            c.segment === 'Champions' ? 'bg-emerald-100 text-emerald-800' :
                                            c.segment === 'Loyal' ? 'bg-sky-100 text-sky-800' :
                                            c.segment === 'Potential' ? 'bg-indigo-100 text-indigo-800' :
                                            c.segment === 'At Risk' ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-800'
                                        }">
                                            ${c.segment}
                                        </span>
                                    </td>
                                    <td class="text-right font-mono font-bold text-emerald-600">${UI.fmtVnd(c.clv_estimate)}</td>
                                </tr>
                            `).join("") || `<tr><td colspan="6" class="py-3 text-center text-slate-400">Không có dữ liệu khách hàng.</td></tr>`}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Chiến dịch gợi ý:</strong> <span>${data.explanation?.counterfactual?.retain_at_risk || ""}</span>
                </div>
            </div>
        `);
    }

    function renderWorkingCapital(data) {
        if (!["full", "calculator", "profile"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const actions = data.action_plan || [];
        
        UI.panel("taxpayer-ai-working-capital-panel", "Tối ưu hóa Vốn lưu động (Working Capital Optimization)", "monetization_on", `
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chu kỳ chuyển tiền (CCC)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${data.ccc || 0} ngày</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Thời gian thu nợ (DSO)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${data.dso || 0} ngày</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Thời gian trả nợ (DPO)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${data.dpo || 0} ngày</p>
                </div>
                <div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200">
                    <p class="text-[9px] uppercase font-bold text-indigo-600">Điểm thanh khoản (Liquidity)</p>
                    <p class="text-sm font-black text-indigo-800 mt-1">${data.liquidity_score || 0}/100</p>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div class="bg-white rounded-lg border border-slate-200 p-4">
                    <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2">Thông tin tài chính</h5>
                    <div class="space-y-2 mt-2">
                        <div class="flex justify-between text-xs py-1 border-b border-slate-100">
                            <span class="text-slate-500">Khoản phải thu:</span>
                            <span class="font-bold text-slate-700">${UI.fmtVnd(data.receivable_total)}</span>
                        </div>
                        <div class="flex justify-between text-xs py-1 border-b border-slate-100">
                            <span class="text-slate-500">Khoản phải trả:</span>
                            <span class="font-bold text-slate-700">${UI.fmtVnd(data.payable_total)}</span>
                        </div>
                        <div class="flex justify-between text-xs py-1 border-b border-slate-100">
                            <span class="text-slate-500">Dự phòng tối ưu:</span>
                            <span class="font-bold text-emerald-600">${UI.fmtVnd(data.optimal_cash_buffer)}</span>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg border border-slate-200 p-4">
                    <h5 class="text-[10px] font-black uppercase text-slate-500 mb-2">Đề xuất hành động</h5>
                    <div class="space-y-2 overflow-y-auto max-h-40">
                        ${actions.map(act => `
                            <div class="p-2 rounded bg-slate-50 border border-slate-200 text-xs">
                                <div class="flex justify-between">
                                    <b class="text-slate-800">${UI.escapeHtml(act.action)}</b>
                                    ${priorityBadge(act.impact || "medium")}
                                </div>
                                <p class="text-[10.5px] text-slate-500 mt-1">${UI.escapeHtml(act.detail)}</p>
                            </div>
                        `).join("") || `<div class="text-xs text-emerald-700 font-bold p-2 bg-emerald-50 rounded">Vốn lưu động đang được vận hành tối ưu!</div>`}
                    </div>
                </div>
            </div>

            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Phân tích mô hình:</strong> <span>${data.explanation?.counterfactual?.reduce_ccc || ""}</span>
                </div>
            </div>
        `);
    }

    function renderRegulatoryChangeDiff(data) {
        if (!["full", "legal"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const changes = data.changes || [];
        
        UI.panel("taxpayer-ai-regulatory-diff-panel", "Thay đổi Quy định Pháp lý & So sánh văn bản (Regulatory Diff)", "policy", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng quy định ảnh hưởng</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${data.summary?.total_changes || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                    <p class="text-[9px] uppercase font-bold text-emerald-600">Đang có hiệu lực</p>
                    <p class="text-sm font-black text-emerald-800 mt-1">${data.summary?.active || 0}</p>
                </div>
                <div class="p-3 rounded-lg bg-rose-50 border border-rose-200">
                    <p class="text-[9px] uppercase font-bold text-rose-600">Tác động cao (High Impact)</p>
                    <p class="text-sm font-black text-rose-800 mt-1">${data.summary?.high_impact || 0}</p>
                </div>
            </div>

            <div class="space-y-3 mb-4">
                ${changes.map(c => `
                    <div class="p-4 bg-white border border-slate-200 rounded-lg shadow-sm">
                        <div class="flex items-start justify-between gap-2 border-b border-slate-100 pb-2">
                            <div>
                                <h5 class="text-xs font-bold text-slate-800">${UI.escapeHtml(c.title)}</h5>
                                <p class="text-[9px] text-slate-400 font-mono mt-0.5">Mã hiệu: ${UI.escapeHtml(c.id)} | Hiệu lực: ${UI.escapeHtml(c.effective_date)}</p>
                            </div>
                            <div class="flex gap-1">
                                <span class="px-1.5 py-0.5 rounded text-[8px] font-bold ${
                                    c.status === 'active' ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-700'
                                }">${c.status}</span>
                                ${priorityBadge(c.impact_level)}
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                            <div class="p-2.5 rounded bg-rose-50/50 border border-rose-100 text-xs">
                                <span class="text-[8px] uppercase font-bold text-rose-600 block mb-1">Quy định Cũ</span>
                                <p class="text-slate-600 text-[10.5px]">${UI.escapeHtml(c.old_text)}</p>
                            </div>
                            <div class="p-2.5 rounded bg-emerald-50/50 border border-emerald-100 text-xs">
                                <span class="text-[8px] uppercase font-bold text-emerald-600 block mb-1">Quy định Mới</span>
                                <p class="text-slate-800 text-[10.5px] font-bold">${UI.escapeHtml(c.new_text)}</p>
                            </div>
                        </div>

                        ${c.action_items?.length ? `
                            <div class="mt-3 bg-slate-50 p-2.5 rounded border border-slate-100">
                                <span class="text-[9px] uppercase font-bold text-slate-500 block mb-1">Hành động cần thực hiện</span>
                                <ul class="list-disc list-inside text-[10.5px] text-slate-600 space-y-0.5">
                                    ${c.action_items.map(item => `<li>${UI.escapeHtml(item)}</li>`).join("")}
                                </ul>
                            </div>
                        ` : ""}
                    </div>
                `).join("") || `<div class="text-xs text-slate-400 text-center py-4">Chưa có thay đổi pháp luật ảnh hưởng đến ngành nghề.</div>`}
            </div>

            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Khuyến nghị pháp lý:</strong> <span>${data.explanation?.counterfactual?.review_changes || ""}</span>
                </div>
            </div>
        `);
    }

    function renderComplianceRiskHeatmap(data) {
        if (!["full", "profile"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const dimensions = data.dimensions || [];
        const score = data.composite_score || 0;
        const level = data.composite_level || "low";
        const levelColor = level === "high" ? "rose" : level === "medium" ? "amber" : "emerald";
        const levelLabel = level === "high" ? "Nguy cơ cao" : level === "medium" ? "Trung bình" : "An toàn";
        
        UI.panel("taxpayer-ai-compliance-heatmap-panel", "Bản đồ nhiệt Rủi ro Tuân thủ 10 Chiều (Compliance Risk Heatmap)", "radar", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${levelColor}-50 border border-${levelColor}-200 md:col-span-2">
                    <p class="text-[9px] uppercase font-bold text-${levelColor}-600">Mức độ rủi ro tổng hợp</p>
                    <p class="text-sm font-black text-${levelColor}-800 mt-1">${levelLabel} (${score}/100)</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Số chiều phân tích</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${dimensions.length} chiều</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Đánh giá rủi ro đa chiều chi tiết</h5>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    ${dimensions.map(dim => {
                        const dimScore = Math.round(Number(dim.score || 0));
                        const dimColor = scoreColor(100 - dimScore); // inverse color because high score in risk means dangerous (red)
                        return `
                            <div class="p-2.5 rounded bg-slate-50 border border-slate-200">
                                <div class="flex justify-between items-center text-[11px] font-bold text-slate-700 mb-1">
                                    <span>${UI.escapeHtml(dim.label)}</span>
                                    <span class="font-mono text-${dimColor}-600">${dimScore}%</span>
                                </div>
                                <div class="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                                    <div class="h-full bg-${dimColor}-500 rounded-full" style="width: ${dimScore}%"></div>
                                </div>
                            </div>
                        `;
                    }).join("")}
                </div>
            </div>

            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Đề xuất hạ điểm rủi ro:</strong> <span>${data.explanation?.counterfactual?.reduce_top_risk || ""}</span>
                </div>
            </div>
        `);
    }

    function renderTaxCalendarOptimization(data) {
        if (!["full", "forecast"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const deadlines = data.deadlines || [];
        const impact = data.cashflow_impact || {};
        
        UI.panel("taxpayer-ai-calendar-optimization-panel", "Tối ưu hóa Lịch nộp Thuế bằng AI (Tax Calendar Optimizer)", "calendar_today", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                    <p class="text-[9px] uppercase font-bold text-emerald-600">Tiết kiệm phạt ước tính</p>
                    <p class="text-sm font-black text-emerald-800 mt-1">${UI.fmtVnd(data.total_penalty_savings)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Số dư khả dụng</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(impact.available)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng nghĩa vụ sắp tới</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${UI.fmtVnd(impact.total_obligations)}</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Lịch nộp thuế đề xuất từ AI</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-[9px] uppercase text-slate-400 border-b border-slate-100">
                            <tr>
                                <th class="py-2.5">Nghĩa vụ</th>
                                <th>Loại</th>
                                <th class="text-right">Ước tính thuế</th>
                                <th>Hạn gốc</th>
                                <th>Lịch đề xuất AI</th>
                                <th class="text-center">Độ ưu tiên</th>
                                <th class="text-right">Tiết kiệm</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-slate-700">
                            ${deadlines.map(d => `
                                <tr>
                                    <td class="py-2.5 font-bold">${UI.escapeHtml(d.label)}</td>
                                    <td><span class="px-1.5 py-0.5 rounded text-[9px] bg-slate-100 border border-slate-200 text-slate-600 font-mono">${UI.escapeHtml(d.tax_type)}</span></td>
                                    <td class="text-right font-mono font-bold">${UI.fmtVnd(d.estimated_amount)}</td>
                                    <td class="font-mono text-slate-500">${UI.escapeHtml(d.original_date)}</td>
                                    <td class="font-mono font-bold text-indigo-950">${UI.escapeHtml(d.optimized_date)}</td>
                                    <td class="text-center">${priorityBadge(d.priority)}</td>
                                    <td class="text-right font-mono font-bold text-emerald-600">${UI.fmtVnd(d.penalty_savings)}</td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Chiến lược nộp thuế:</strong> <span>${data.explanation?.counterfactual?.pay_early || ""}</span>
                </div>
            </div>
        `);
    }

    function renderCohortAnalysis(data) {
        if (!["full", "growth", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const cohorts = data.cohorts || [];
        const retentionMatrix = data.retention_matrix || [];
        
        UI.panel("taxpayer-ai-cohort-panel", "Phân tích Cohort duy trì doanh thu (Retention Heatmap)", "calendar_view_month", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tổng chu kỳ phân tích</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${data.summary?.total_periods || 0} tháng</p>
                </div>
                <div class="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                    <p class="text-[9px] uppercase font-bold text-emerald-600">Tỷ lệ duy trì trung bình</p>
                    <p class="text-sm font-black text-emerald-800 mt-1">${(data.summary?.avg_retention * 100).toFixed(1)}%</p>
                </div>
                <div class="p-3 rounded-lg bg-rose-50 border border-rose-200">
                    <p class="text-[9px] uppercase font-bold text-rose-600">Số kỳ suy giảm (>15%)</p>
                    <p class="text-sm font-black text-rose-800 mt-1">${data.summary?.declining_periods || 0} tháng</p>
                </div>
            </div>
            
            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Ma trận duy trì khách hàng/doanh thu (Cohort Matrix)</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs border-collapse">
                        <thead>
                            <tr class="bg-slate-100/50 text-slate-400 font-bold uppercase tracking-wider text-[9px] border-b border-slate-200">
                                <th class="px-3 py-2">Tháng bắt đầu</th>
                                <th class="px-3 py-2 text-center">M+0</th>
                                <th class="px-3 py-2 text-center">M+1</th>
                                <th class="px-3 py-2 text-center">M+2</th>
                                <th class="px-3 py-2 text-center">M+3</th>
                                <th class="px-3 py-2 text-center">M+4</th>
                                <th class="px-3 py-2 text-center">M+5</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-slate-700">
                            ${retentionMatrix.slice(0, 6).map(row => {
                                const m0 = row.periods["M+0"] !== undefined ? (row.periods["M+0"] * 100).toFixed(0) + "%" : "-";
                                const m1 = row.periods["M+1"] !== undefined ? (row.periods["M+1"] * 100).toFixed(0) + "%" : "-";
                                const m2 = row.periods["M+2"] !== undefined ? (row.periods["M+2"] * 100).toFixed(0) + "%" : "-";
                                const m3 = row.periods["M+3"] !== undefined ? (row.periods["M+3"] * 100).toFixed(0) + "%" : "-";
                                const m4 = row.periods["M+4"] !== undefined ? (row.periods["M+4"] * 100).toFixed(0) + "%" : "-";
                                const m5 = row.periods["M+5"] !== undefined ? (row.periods["M+5"] * 100).toFixed(0) + "%" : "-";
                                
                                function cellBg(val) {
                                    if (val === "-") return "bg-slate-50 text-slate-400";
                                    const n = parseFloat(val);
                                    if (n >= 90) return "bg-emerald-500 text-white font-bold";
                                    if (n >= 70) return "bg-emerald-300 text-emerald-950";
                                    if (n >= 50) return "bg-emerald-100 text-emerald-900";
                                    if (n >= 30) return "bg-amber-100 text-amber-900";
                                    return "bg-rose-100 text-rose-900";
                                }
                                
                                return `
                                    <tr>
                                        <td class="px-3 py-2 font-bold">${UI.escapeHtml(row.cohort)}</td>
                                        <td class="px-3 py-2 text-center ${cellBg(m0)}">${m0}</td>
                                        <td class="px-3 py-2 text-center ${cellBg(m1)}">${m1}</td>
                                        <td class="px-3 py-2 text-center ${cellBg(m2)}">${m2}</td>
                                        <td class="px-3 py-2 text-center ${cellBg(m3)}">${m3}</td>
                                        <td class="px-3 py-2 text-center ${cellBg(m4)}">${m4}</td>
                                        <td class="px-3 py-2 text-center ${cellBg(m5)}">${m5}</td>
                                    </tr>
                                `;
                            }).join("") || `<tr><td colspan="7" class="py-3 text-center text-slate-400">Không có dữ liệu cohort.</td></tr>`}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="p-3 bg-emerald-50 border border-emerald-100 text-[10px] text-emerald-800 flex items-start gap-1 rounded-lg">
                <span class="material-symbols-outlined text-xs mt-0.5">info</span>
                <div>
                    <strong>Cải thiện xu hướng:</strong> <span>${data.explanation?.counterfactual?.stabilize_revenue || ""}</span>
                </div>
            </div>
        `);
    }

    function renderEntropyRevenue(data) {
        if (!["full", "profile", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        const entropy = Number(data.entropy_value || 0).toFixed(4);
        const maxEntropy = Number(data.max_possible_entropy || 0).toFixed(4);
        const ratio = Math.round((entropy / (maxEntropy || 1)) * 100);
        const buckets = data.distribution_density || [];
        const maxDensity = Math.max(...buckets.map(b => b.density), 0.01);
        
        let ratioColor = "emerald";
        if (ratio > 80) ratioColor = "rose";
        else if (ratio > 50) ratioColor = "amber";

        UI.panel("taxpayer-ai-entropy-panel", "Shannon Entropy — Kiểm tra Chất lượng dữ liệu Doanh thu (F22)", "query_stats", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Entropy Thực tế (H(X))</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${entropy} bits</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Entropy Tối đa</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${maxEntropy} bits</p>
                </div>
                <div class="p-3 rounded-lg bg-${ratioColor}-50 border border-${ratioColor}-200">
                    <p class="text-[9px] uppercase font-bold text-${ratioColor}-600">Độ Bất Định / Nhiễu Dữ liệu</p>
                    <p class="text-sm font-black text-${ratioColor}-800 mt-1">${ratio}%</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Biểu đồ Mật độ Phân bố Doanh thu (Probability Density)</h5>
                <div class="flex items-end gap-1.5 h-24 pt-2 border-b border-slate-100 pb-1">
                    ${buckets.map(b => {
                        const h = (b.density / maxDensity) * 100;
                        return `
                            <div class="flex-1 h-full flex flex-col justify-end group relative" title="Khoảng: ${UI.fmtVnd(b.range_start || 0)} - ${UI.fmtVnd(b.range_end || 0)}\nTỷ lệ: ${(b.density * 100).toFixed(1)}%">
                                <div class="w-full bg-indigo-500 group-hover:bg-indigo-600 rounded-t transition-all" style="height: ${Math.max(3, h)}%"></div>
                                <span class="text-[7px] text-slate-400 text-center block mt-1 truncate">${Math.round((b.range_start || 0) / 1e6)}M</span>
                            </div>
                        `;
                    }).join("")}
                </div>
                <div class="text-center text-[9px] text-slate-400 mt-2">Đơn vị trục hoành: Triệu VND</div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Nhận định chất lượng khai trình:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Đầy đủ dữ liệu")}</p>
            </div>
        `);
    }

    function renderHmmFinancialState(data) {
        if (!["full", "forecast", "debt"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        
        const timeline = data.state_timeline || [];
        const currentState = data.current_state || "Healthy";
        const nextPred = data.next_month_prediction || {};
        
        const stateColors = {
            "Healthy": "emerald",
            "Stressed": "amber",
            "Crisis": "rose"
        };
        const stateLabels = {
            "Healthy": "Khỏe mạnh (Ổn định)",
            "Stressed": "Áp lực tài chính",
            "Crisis": "Rủi ro mất thanh khoản (Khủng hoảng)"
        };
        
        const currColor = stateColors[currentState] || "slate";

        UI.panel("taxpayer-ai-hmm-panel", "Hidden Markov Model — Cảnh báo sớm Trạng thái Tài chính (F23)", "hourglass_empty", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${currColor}-50 border border-${currColor}-200 md:col-span-1">
                    <p class="text-[9px] uppercase font-bold text-${currColor}-600">Trạng thái Hiện tại</p>
                    <p class="text-sm font-black text-${currColor}-800 mt-1">${stateLabels[currentState] || currentState}</p>
                </div>
                <div class="p-3 rounded-lg bg-white border border-slate-200 md:col-span-2">
                    <p class="text-[9px] uppercase font-bold text-slate-400 mb-2">Dự báo Xác suất Trạng thái Tháng tới</p>
                    <div class="grid grid-cols-3 gap-2">
                        ${Object.entries(nextPred).map(([state, prob]) => {
                            const pVal = Math.round(Number(prob) * 100);
                            const c = stateColors[state] || "slate";
                            return `
                                <div class="p-2 rounded bg-slate-50 border border-slate-100">
                                    <div class="flex justify-between text-[9px] font-bold text-slate-500">
                                        <span>${stateLabels[state] || state}</span>
                                        <span>${pVal}%</span>
                                    </div>
                                    <div class="mt-1 h-1 bg-slate-200 rounded-full overflow-hidden">
                                        <div class="h-full bg-${c}-500" style="width:${pVal}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join("")}
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Lịch sử Chuyển trạng thái 6 Tháng (Viterbi Path)</h5>
                <div class="flex justify-between items-center relative py-2">
                    <div class="absolute left-0 right-0 h-0.5 bg-slate-100 top-1/2 -translate-y-1/2 z-0"></div>
                    ${timeline.map((step, idx) => {
                        const state = step.state;
                        const c = stateColors[state] || "slate";
                        return `
                            <div class="flex flex-col items-center z-10 relative group">
                                <div class="w-7 h-7 rounded-full bg-${c}-500 text-white font-bold text-[9px] flex items-center justify-center border-4 border-white shadow-sm" title="${stateLabels[state] || state}\nHealthy: ${Math.round((step.probabilities?.Healthy || 0)*100)}%\nStressed: ${Math.round((step.probabilities?.Stressed || 0)*100)}%\nCrisis: ${Math.round((step.probabilities?.Crisis || 0)*100)}%">
                                    ${idx + 1}
                                </div>
                                <span class="text-[9px] font-black text-slate-700 mt-1">${UI.escapeHtml(step.period)}</span>
                                <span class="text-[8px] text-${c}-600 font-semibold text-center block whitespace-nowrap overflow-visible max-w-[50px] truncate-none">${stateLabels[state] || state}</span>
                            </div>
                        `;
                    }).join("")}
                </div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Nhận định & Khuyến nghị HMM:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Chưa phát hiện rủi ro ngắn hạn")}</p>
            </div>
        `);
    }

    function renderCusumChangeDetection(data) {
        if (!["full", "forecast", "growth", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        
        const pos = data.cusum_positive || [];
        const neg = data.cusum_negative || [];
        const cp = data.change_points || [];
        const maxVal = Math.max(1, ...pos, ...neg);
        
        UI.panel("taxpayer-ai-cusum-panel", "CUSUM Change-Point — Phát hiện Điểm Chuyển đổi Doanh thu (F24)", "alt_route", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Số điểm gãy được phát hiện</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${data.change_point_count || 0} điểm chuyển đổi cấu trúc</p>
                </div>
                <div class="p-3 rounded-lg bg-indigo-50 border border-indigo-200">
                    <p class="text-[9px] uppercase font-bold text-indigo-600">Thuật toán giám sát</p>
                    <p class="text-xs font-black text-indigo-800 mt-1">Cumulative Sum (CUSUM) với Slack=0.5σ, Hạn mức=4.0σ</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Đường tích lũy CUSUM & Sự dịch chuyển</h5>
                <div class="flex items-end gap-2 h-32 pt-2 border-b border-slate-100 pb-1">
                    ${pos.map((pVal, idx) => {
                        const nVal = neg[idx] || 0;
                        const hPos = (pVal / maxVal) * 100;
                        const hNeg = (nVal / maxVal) * 100;
                        const isCp = cp.includes(idx);
                        
                        return `
                            <div class="flex-1 h-full flex flex-col justify-end gap-0.5 relative group" title="Chu kỳ ${idx + 1}\nCUSUM Tăng (+): ${pVal.toFixed(2)}\nCUSUM Giảm (-): ${nVal.toFixed(2)}">
                                ${isCp ? `
                                    <div class="absolute inset-0 bg-rose-50 border-l border-r border-rose-200 opacity-60 z-0"></div>
                                    <span class="absolute top-1 left-1/2 -translate-x-1/2 text-[7px] font-black bg-rose-600 text-white px-1 rounded z-10 animate-pulse">BREAK</span>
                                ` : ''}
                                <div class="flex items-end gap-px h-full justify-center z-10">
                                    <div class="w-2.5 bg-emerald-500 rounded-t" style="height: ${Math.max(2, hPos)}%"></div>
                                    <div class="w-2.5 bg-rose-400 rounded-t" style="height: ${Math.max(2, hNeg)}%"></div>
                                </div>
                                <span class="text-[8px] text-slate-400 text-center block z-10">K${idx + 1}</span>
                            </div>
                        `;
                    }).join("")}
                </div>
                <div class="mt-3 flex items-center gap-4 text-[9px] font-bold text-slate-500">
                    <span><i class="inline-block w-2.5 h-2 bg-emerald-500 rounded-sm"></i> Xu hướng tăng tích lũy (S+)</span>
                    <span><i class="inline-block w-2.5 h-2 bg-rose-400 rounded-sm"></i> Xu hướng giảm tích lũy (S-)</span>
                    <span><i class="inline-block w-2.5 h-2 bg-rose-100 border border-rose-300 rounded-sm"></i> Điểm gãy xu hướng</span>
                </div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Kết luận kiểm định điểm gãy CUSUM:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Xu hướng kinh doanh ổn định")}</p>
            </div>
        `);
    }

    function renderSvdExpenseDecomposition(data) {
        if (!["full", "expense", "accounting"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        
        const singulars = data.singular_values || [];
        const weights1 = data.v1_weights || {};
        const weights2 = data.v2_weights || {};
        const projections = data.projections || [];
        
        UI.panel("taxpayer-ai-svd-panel", "Singular Value Decomposition (SVD) — Phân tích Cấu trúc Chi phí (F25)", "grid_view", `
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200 col-span-1">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Các Trị Số Riêng SVD (Singular Values)</p>
                    <div class="mt-2 space-y-2">
                        ${singulars.map((val, idx) => `
                            <div class="flex justify-between items-center text-xs">
                                <span class="font-bold text-slate-600">Trị số σ_${idx+1}:</span>
                                <span class="font-mono font-bold text-indigo-900">${val.toFixed(4)}</span>
                            </div>
                        `).join("")}
                    </div>
                </div>
                <div class="p-3 rounded-lg bg-white border border-slate-200 col-span-2">
                    <p class="text-[9px] uppercase font-bold text-slate-400 mb-2">Trọng số Thành phần Chi phí Phổ biến nhất (PC1 & PC2)</p>
                    <div class="grid grid-cols-2 gap-3 text-[11px]">
                        <div>
                            <span class="font-bold text-indigo-700 block mb-1">PC1 (Xu hướng chủ đạo):</span>
                            ${Object.entries(weights1).map(([cat, w]) => `
                                <div class="flex justify-between py-0.5 border-b border-slate-50 text-[10px]">
                                    <span class="text-slate-500">${cat}:</span>
                                    <span class="font-bold">${(w * 100).toFixed(1)}%</span>
                                </div>
                            `).join("")}
                        </div>
                        <div>
                            <span class="font-bold text-emerald-700 block mb-1">PC2 (Biến động phụ):</span>
                            ${Object.entries(weights2).map(([cat, w]) => `
                                <div class="flex justify-between py-0.5 border-b border-slate-50 text-[10px]">
                                    <span class="text-slate-500">${cat}:</span>
                                    <span class="font-bold">${(w * 100).toFixed(1)}%</span>
                                </div>
                            `).join("")}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Projections table -->
            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Không gian Chiếu Chi phí & Điểm Bất thường cấu trúc</h5>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-[9px] uppercase text-slate-400 border-b border-slate-100">
                            <tr>
                                <th class="py-2">Tháng</th>
                                <th class="text-center">PC1 Score</th>
                                <th class="text-center">PC2 Score</th>
                                <th class="text-right">Chỉ số Bất thường</th>
                                <th class="text-center">Nhận diện cấu trúc</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 text-slate-700">
                            ${projections.map(p => {
                                const score = p.anomaly_score || 0;
                                const isAnomaly = score > 1.5;
                                return `
                                    <tr>
                                        <td class="py-2.5 font-bold">${UI.escapeHtml(p.period)}</td>
                                        <td class="text-center font-mono">${p.pc1.toFixed(3)}</td>
                                        <td class="text-center font-mono">${p.pc2.toFixed(3)}</td>
                                        <td class="text-right font-mono font-bold text-indigo-700">${score.toFixed(2)}</td>
                                        <td class="text-center">
                                            <span class="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                                                isAnomaly ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'
                                            }">
                                                ${isAnomaly ? 'Bất thường' : 'Chuẩn hóa'}
                                            </span>
                                        </td>
                                    </tr>
                                `;
                            }).join("")}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Nhận định cấu trúc SVD:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Cấu trúc chi phí hợp lý")}</p>
            </div>
        `);
    }

    function renderWaveletRevenue(data) {
        if (!["full", "forecast", "growth", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        
        const periods = data.periods || [];
        const original = data.original_values || [];
        const trend = data.trend_component || [];
        const seasonal = data.seasonal_component || [];
        const noise = data.noise_component || [];
        
        const maxVal = Math.max(1, ...original);

        UI.panel("taxpayer-ai-wavelet-panel", "Haar Wavelet Multi-Resolution — Phân tách Xu hướng & Biến động Mùa vụ (F26)", "waves", `
            <div class="bg-indigo-50 border border-indigo-200 p-3 rounded-lg mb-4 text-xs">
                <p class="font-bold text-indigo-950">Giải thuật Phân rã Haar Wavelet (Cấp độ 2):</p>
                <p class="text-indigo-900 mt-1">Phép biến đổi Wavelet rời rạc (DWT) giúp chia chu kỳ doanh thu thành 3 thành phần tần số: Xu hướng dài hạn (Low-pass), Biến động chu kỳ (Mid-pass) và Nhiễu hệ thống (High-pass).</p>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Thành phần Tín hiệu sau khi Tách lọc</h5>
                <div class="space-y-4">
                    <!-- Original and Trend Chart -->
                    <div>
                        <span class="text-[9px] font-bold text-slate-500 uppercase">1. Tín hiệu gốc vs. Xu hướng chính (Trend)</span>
                        <div class="flex items-end gap-1.5 h-16 pt-2 border-b border-slate-100 pb-1">
                            ${original.map((val, idx) => {
                                const h1 = (val / maxVal) * 100;
                                const h2 = ((trend[idx] || 0) / maxVal) * 100;
                                return `
                                    <div class="flex-1 h-full flex items-end gap-px group relative" title="Gốc: ${UI.fmtVnd(val)}\nXu hướng: ${UI.fmtVnd(trend[idx] || 0)}">
                                        <div class="w-1/2 bg-slate-300 h-full" style="height: ${Math.max(2, h1)}%"></div>
                                        <div class="w-1/2 bg-indigo-600 h-full" style="height: ${Math.max(2, h2)}%"></div>
                                    </div>
                                `;
                            }).join("")}
                        </div>
                    </div>

                    <!-- Seasonal Component Chart -->
                    <div>
                        <span class="text-[9px] font-bold text-slate-500 uppercase">2. Biến động Mùa vụ chi tiết (Seasonal component)</span>
                        <div class="flex items-end gap-1.5 h-12 pt-2 border-b border-slate-100 pb-1">
                            ${seasonal.map((val, idx) => {
                                const absVal = Math.abs(val);
                                const h = (absVal / maxVal) * 200; // Scaled up for visibility
                                const isPos = val >= 0;
                                return `
                                    <div class="flex-1 h-full flex items-end justify-center group relative" title="Mùa vụ: ${UI.fmtVnd(val)}">
                                        <div class="w-3 rounded-t ${isPos ? 'bg-emerald-500' : 'bg-amber-400'}" style="height: ${Math.max(4, h)}%"></div>
                                    </div>
                                `;
                            }).join("")}
                        </div>
                    </div>

                    <!-- Noise Component Chart -->
                    <div>
                        <span class="text-[9px] font-bold text-slate-500 uppercase">3. Nhiễu giao dịch bất định (Noise/Residual component)</span>
                        <div class="flex items-end gap-1.5 h-10 pt-2 border-b border-slate-100 pb-1">
                            ${noise.map((val, idx) => {
                                const absVal = Math.abs(val);
                                const h = (absVal / maxVal) * 300; // Scaled up even more
                                return `
                                    <div class="flex-1 h-full flex items-end justify-center group relative" title="Nhiễu: ${UI.fmtVnd(val)}">
                                        <div class="w-2 bg-rose-500" style="height: ${Math.max(2, h)}%"></div>
                                    </div>
                                `;
                            }).join("")}
                        </div>
                    </div>
                </div>
                <div class="flex justify-between text-[8px] text-slate-400 mt-2 px-1">
                    ${periods.map(p => `<span>${UI.escapeHtml(p)}</span>`).join("")}
                </div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Nhận định & Khuyến nghị Wavelet:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Tách lọc tín hiệu hoàn tất")}</p>
            </div>
        `);
    }

    function renderAltmanZscore(data) {
        if (!["full", "growth", "profile", "calculator"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        
        const score = Number(data.z_score || 0).toFixed(3);
        const prob = Math.round(Number(data.probability_of_bankruptcy || 0) * 100);
        const zone = data.zone || "safe";
        
        const zoneColors = {
            "safe": "emerald",
            "grey": "amber",
            "distress": "rose"
        };
        const zoneLabels = {
            "safe": "Vùng An toàn (Safe Zone) — Ít khả năng mất thanh toán",
            "grey": "Vùng Cảnh báo (Grey Zone) — Rủi ro trung bình",
            "distress": "Vùng Nguy hiểm (Distress Zone) — Rủi ro cao"
        };
        
        const zColor = zoneColors[zone] || "slate";

        UI.panel("taxpayer-ai-altman-panel", "Altman Z-Score — Dự đoán & Đánh giá Sức khỏe Tài chính (F27)", "health_and_safety", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${zColor}-50 border border-${zColor}-200 col-span-2">
                    <p class="text-[9px] uppercase font-bold text-${zColor}-600">Chỉ số Z-Score</p>
                    <p class="text-sm font-black text-${zColor}-800 mt-1">${score} (${zoneLabels[zone] || zone})</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Xác suất Phá sản (12 Tháng)</p>
                    <p class="text-sm font-black text-slate-800 mt-1">${prob}%</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Thang đo Mức độ An toàn (Z-Score Gauge)</h5>
                <div class="relative w-full h-8 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                    <div class="absolute left-0 top-0 bottom-0 bg-rose-500" style="width: 35%" title="Vùng nguy hiểm (< 1.8)"></div>
                    <div class="absolute left-[35%] top-0 bottom-0 bg-amber-400" style="width: 25%" title="Vùng cảnh báo (1.8 - 2.9)"></div>
                    <div class="absolute left-[60%] top-0 bottom-0 bg-emerald-500" style="width: 40%" title="Vùng an toàn (> 2.9)"></div>
                    
                    ${(() => {
                        const scoreNum = Number(score);
                        const pct = Math.max(0, Math.min(100, (scoreNum / 5.0) * 100));
                        return `
                            <div class="absolute top-0 bottom-0 w-1 bg-slate-900 shadow z-10 transition-all" style="left: ${pct}%"></div>
                            <span class="absolute top-1/2 -translate-y-1/2 text-[9px] font-black bg-slate-900 text-white px-2 py-0.5 rounded shadow z-20" style="left: calc(${pct}% - 20px)">
                                ${score}
                            </span>
                        `;
                    })()}
                </div>
                <div class="flex justify-between text-[8px] text-slate-400 mt-2 px-1">
                    <span>0.0 (Nguy hiểm)</span>
                    <span>1.8 (Cảnh báo)</span>
                    <span>3.0 (An toàn)</span>
                    <span>5.0+</span>
                </div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Đánh giá rủi ro từ Hệ thống:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Khả năng chi trả tốt")}</p>
            </div>
        `);
    }

    function renderKMeansSupplierClustering(data) {
        if (!["full", "invoice", "accounting"].includes(cfg.mode)) return;
        if (!data || data.status === "error") return;
        
        const suppliers = data.suppliers || [];
        const clusterColors = ["emerald", "amber", "rose"];
        const clusterLabels = [
            "Ổn định & Tin cậy cao (Thành viên nhóm 1)",
            "Giao dịch thông thường (Thành viên nhóm 2)",
            "Cần kiểm soát rủi ro (Hóa đơn lớn / Tần suất thấp / Nhóm 3)"
        ];

        UI.panel("taxpayer-ai-kmeans-panel", "K-Means++ Supplier Clustering — Phân nhóm Rủi ro Đối tác (F28)", "group_work", `
            <div class="bg-indigo-50 border border-indigo-200 p-3 rounded-lg mb-4 text-xs">
                <p class="font-bold text-indigo-950">Giải thuật phân nhóm K-Means++:</p>
                <p class="text-indigo-900 mt-1">Hệ thống phân tích các nhà cung cấp/đối tác dựa trên 3 trục tọa độ: Tần suất xuất hóa đơn, Số tiền bình quân mỗi hóa đơn, và Độ lệch chuẩn (sự bất thường) của hóa đơn.</p>
            </div>

            <div class="grid grid-cols-1 gap-3 mb-4">
                ${[0, 1, 2].map(cIdx => {
                    const cSuppliers = suppliers.filter(s => s.cluster_index === cIdx);
                    const cColor = clusterColors[cIdx];
                    return `
                        <div class="bg-white border border-slate-200 rounded-lg p-3">
                            <div class="flex items-center justify-between border-b border-slate-100 pb-2 mb-2">
                                <span class="text-xs font-bold text-${cColor}-700 flex items-center gap-1">
                                    <span class="w-2.5 h-2.5 rounded-full bg-${cColor}-500 inline-block"></span>
                                    ${clusterLabels[cIdx]}
                                </span>
                                <span class="px-2 py-0.5 rounded-full text-[9px] font-bold bg-${cColor}-100 text-${cColor}-800">
                                    ${cSuppliers.length} đối tác
                                </span>
                            </div>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                                ${cSuppliers.map(s => `
                                    <div class="p-2 bg-slate-50 rounded border border-slate-100 text-xs flex justify-between items-center">
                                        <div>
                                            <p class="font-bold text-slate-800">${UI.escapeHtml(s.supplier_name)}</p>
                                            <p class="text-[9px] text-slate-400 mt-0.5">${s.frequency} hóa đơn | Avg: ${UI.fmtVnd(s.mean_amount)}</p>
                                        </div>
                                        <div class="text-right">
                                            <span class="px-1.5 py-0.5 rounded text-[9px] font-black bg-${cColor}-50 text-${cColor}-700 border border-${cColor}-100">
                                                Risk: ${s.risk_score}%
                                            </span>
                                        </div>
                                    </div>
                                `).join("") || `<p class="text-[10px] text-slate-400 italic py-1">Không có đối tác trong nhóm này.</p>`}
                            </div>
                        </div>
                    `;
                }).join("")}
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Báo cáo phân tích đối tác:</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Phân nhóm nhà cung cấp thành công")}</p>
            </div>
        `);
    }

    function renderCompositeRiskScore(data) {
        if (!data || data.status === "error") return;
        
        const score = Math.round(Number(data.composite_risk_score || 0));
        const health = Math.round(Number(data.health_score || 0));
        const ratings = data.ratings || {};
        
        let scoreColor = "emerald";
        if (score >= 70) scoreColor = "rose";
        else if (score >= 35) scoreColor = "amber";

        UI.panel("taxpayer-ai-composite-panel", "Gradient Boosting Composite Risk — Điểm Sức khỏe Thuế Tổng hợp (F29)", "shield_with_heart", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div class="p-3 rounded-lg bg-${scoreColor}-50 border border-${scoreColor}-200 md:col-span-2">
                    <p class="text-[9px] uppercase font-bold text-${scoreColor}-600">Chỉ số Rủi ro Tổng hợp (Composite Risk)</p>
                    <p class="text-lg font-black text-${scoreColor}-800 mt-1">${score}/100</p>
                    <div class="mt-2 h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div class="h-full bg-${scoreColor}-500" style="width: ${score}%"></div>
                    </div>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Điểm Sức khỏe Tài chính</p>
                    <p class="text-lg font-black text-emerald-800 mt-1">${health}/100</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Thuật toán tích hợp</p>
                    <p class="text-[10px] font-semibold text-slate-600 mt-1">Iterative Gradient Boosting (Lightweight residual fitting)</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-4">
                <h5 class="text-[10px] font-black uppercase text-slate-500 mb-3">Đánh giá 6 Trụ cột Sức khỏe & Tuân thủ Thuế</h5>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                    ${Object.entries(ratings).map(([key, rVal]) => {
                        const rScore = Math.round(Number(rVal || 0));
                        let c = "emerald";
                        if (rScore >= 70) c = "rose";
                        else if (rScore >= 35) c = "amber";
                        
                        const labels = {
                            "compliance": "Tuân thủ Quy định",
                            "financial": "Sức khỏe Tài chính",
                            "cashflow": "Dòng tiền & Thanh khoản",
                            "data_quality": "Chất lượng Chứng từ",
                            "solvency": "Khả năng thanh toán",
                            "operations": "Rủi ro Vận hành"
                        };
                        
                        return `
                            <div class="p-2.5 rounded bg-slate-50 border border-slate-200">
                                <div class="flex justify-between items-center text-[10px] font-bold text-slate-700 mb-1">
                                    <span>${labels[key] || key}</span>
                                    <span class="font-mono text-${c}-600">${rScore}%</span>
                                </div>
                                <div class="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                                    <div class="h-full bg-${c}-500 rounded-full" style="width: ${rScore}%"></div>
                                </div>
                            </div>
                        `;
                    }).join("")}
                </div>
            </div>

            <div class="p-3 bg-slate-50 border border-slate-200 text-[10px] text-slate-700 rounded-lg">
                <p class="font-bold text-slate-800">Ý kiến hội đồng chuyên môn (Gradient Boosting Verdict):</p>
                <p class="mt-1 leading-relaxed">${UI.escapeHtml(data.verdict || "Hồ sơ tuân thủ tốt")}</p>
            </div>
        `);
    }

    const registryLoadedCapabilities = new Set();

    function renderRegistryCapability(key, data, registryCfg) {
        const Core = window.TaxpayerAIPanelsCore;
        if (!Core) return;
        if (key === "overview") return Core.renderOverview(data, registryCfg);
        if (key === "forecast") return Core.renderForecast(data, registryCfg);
        if (key === "reconciliation_cases") return Core.renderReconciliationCases(data, registryCfg);
        if (key === "next_best_action") return Core.renderNextBestAction(data, registryCfg);
        if (["cashflow_delinquency", "cashflow_risk", "tax_reserve"].includes(key)) return Core.renderCashflowRisk(data, registryCfg);

        const legacyRenderers = {
            peer_benchmark: renderBenchmark,
            charts: renderCharts,
            anomalies: renderAnomalies,
            scenario_dashboard: renderScenario,
            advanced_dashboard: renderAdvancedDashboard,
            probabilistic_forecast: renderProbabilisticForecast,
            supplier_risk_graph: renderSupplierGraph,
            supplier_account_risk: renderSupplierAccountRisk,
            graph_risk: renderGraphRisk,
            business_upgrade_readiness: renderUpgradeReadiness,
            model_governance: renderGovernance,
            model_governance_production: renderProductionGovernance,
            regulatory_change_diff: renderRegulatoryChangeDiff,
            tax_calendar_optimization: renderTaxCalendarOptimization,
            evidence_bundle: renderEvidenceBundle,
            legal_change_impact: renderLegalChangeImpact,
            channel_attribution: renderChannelAttribution,
            inventory_analyze: renderInventoryAI,
            reconcile_4way: (result) => renderProductionReconciliation(result, null),
            benford_analysis: renderBenfordAnalysis,
            seasonal_decomposition: renderSeasonalDecomposition,
            survival_analysis: renderSurvivalAnalysis,
            bayesian_forecast: renderBayesianForecast,
            explainability: renderExplainability,
            autoencoder_bank_anomaly: renderAutoencoderAnomaly,
            rfm_customer_segmentation: renderRfmSegmentation,
            working_capital: renderWorkingCapital,
            compliance_risk_heatmap: renderComplianceRiskHeatmap,
            cohort_analysis: renderCohortAnalysis,
            entropy_revenue: renderEntropyRevenue,
            hmm_financial_state: renderHmmFinancialState,
            cusum_change_detection: renderCusumChangeDetection,
            svd_expense_decomposition: renderSvdExpenseDecomposition,
            wavelet_revenue: renderWaveletRevenue,
            altman_zscore: renderAltmanZscore,
            kmeans_supplier_clustering: renderKMeansSupplierClustering,
            composite_risk_score: renderCompositeRiskScore,
        };
        const renderer = legacyRenderers[key];
        if (renderer && data && data.status !== "error") return renderer(data);
        return Core.renderGeneric(key, data, registryCfg);
    }

    async function loadRegistryCapability(key, options = {}) {
        const Registry = window.TaxpayerAIRegistry;
        const Client = window.TaxpayerAIClient;
        if (!Registry || !Client) return null;
        const registryCfg = Registry.resolvePageConfig(page);
        const cap = Registry.getCapability(key);
        const payload = { ...(cap?.body || {}) };
        if (key === "evidence_bundle" && registryCfg.mode === "claim") payload.purpose = "appeal";
        const data = await Client.safeRequestCapability(key, payload);
        registryLoadedCapabilities.add(key);
        if (options.render !== false) renderRegistryCapability(key, data, registryCfg);
        return data;
    }

    async function loadRegistryIntelligence() {
        const Registry = window.TaxpayerAIRegistry;
        const Advanced = window.TaxpayerAIPanelsAdvanced;
        if (!Registry || !window.TaxpayerAIClient || !window.TaxpayerAIPanelsCore) return false;

        const registryCfg = Registry.resolvePageConfig(page);
        const budget = Registry.getRenderBudget();
        const primaryKeys = Registry.getPageCapabilities(page).slice(0, budget.primary);

        await Promise.all(primaryKeys.map((key) => loadRegistryCapability(key, { render: true })));
        if (Advanced) {
            Advanced.renderDeepAnalysisLauncher({
                cfg: registryCfg,
                loadedKeys: registryLoadedCapabilities,
                advancedKeys: Registry.getPageCapabilities(page, { includeAdvanced: true }),
                loadCapability: loadRegistryCapability,
            });
        }
        renderCopilot();
        return true;
    }

    async function loadIntelligence() {
        if (await loadRegistryIntelligence()) return;
        const [
            overview,
            forecast,
            benchmark,
            charts,
            anomalies,
            catalog,
            scenario,
            cashflow,
            suppliers,
            upgrade,
            advanced,
            probabilistic,
            graphRisk,
            cashflowDelinquency,
            nba,
            governance,
            reconcile4way,
            reconciliationCases,
            channelAttribution,
            taxReserve,
            supplierAccountRisk,
            inventoryAI,
            evidenceBundle,
            legalChange,
            productionGovernance,
            benfordAnalysis,
            seasonalDecomposition,
            survivalAnalysis,
            bayesianForecast,
            explainability,
            autoencoderBankAnomaly,
            rfmCustomerSegmentation,
            workingCapital,
            regulatoryChangeDiff,
            complianceRiskHeatmap,
            taxCalendarOptimization,
            cohortAnalysis,
            entropyRevenue,
            hmmFinancialState,
            cusumChangeDetection,
            svdExpenseDecomposition,
            waveletRevenue,
            altmanZscore,
            kmeansSupplierClustering,
            compositeRiskScore,
        ] = await Promise.all([
            UI.get("/intelligence/overview"),
            UI.get("/intelligence/forecast").catch(() => null),
            UI.get("/intelligence/peer-benchmark").catch(() => null),
            UI.get("/intelligence/charts").catch(() => null),
            UI.get("/intelligence/anomalies").catch(() => null),
            UI.get("/intelligence/model-catalog").catch(() => null),
            UI.get("/intelligence/scenario-dashboard").catch(() => null),
            UI.get("/intelligence/cashflow-risk").catch(() => null),
            UI.get("/intelligence/supplier-risk-graph").catch(() => null),
            UI.get("/intelligence/business-upgrade-readiness").catch(() => null),
            UI.get("/intelligence/advanced-dashboard").catch(() => null),
            UI.get("/intelligence/forecast/probabilistic").catch(() => null),
            UI.get("/intelligence/graph/risk").catch(() => null),
            UI.get("/intelligence/cashflow/delinquency").catch(() => null),
            UI.get("/intelligence/next-best-action").catch(() => null),
            UI.get("/intelligence/model-governance").catch(() => null),
            UI.post("/intelligence/reconcile/4way", {}).catch(() => null),
            UI.get("/intelligence/reconciliation-cases").catch(() => null),
            UI.post("/intelligence/channel-attribution", {}).catch(() => null),
            UI.post("/intelligence/tax-reserve/optimize", {}).catch(() => null),
            UI.get("/intelligence/supplier-account-risk").catch(() => null),
            UI.post("/intelligence/inventory/analyze", {}).catch(() => null),
            UI.post("/intelligence/evidence-bundle", { purpose: cfg.mode === "claim" ? "appeal" : "tax_audit_explanation" }).catch(() => null),
            UI.post("/intelligence/legal/change-impact", {}).catch(() => null),
            UI.get("/intelligence/model-governance/production").catch(() => null),
            UI.get("/intelligence/benford-analysis").catch(() => null),
            UI.get("/intelligence/seasonal-decomposition").catch(() => null),
            UI.get("/intelligence/survival-analysis").catch(() => null),
            UI.get("/intelligence/bayesian-forecast").catch(() => null),
            UI.get("/intelligence/explainability").catch(() => null),
            UI.get("/intelligence/autoencoder-bank-anomaly").catch(() => null),
            UI.get("/intelligence/rfm-customer-segmentation").catch(() => null),
            UI.get("/intelligence/working-capital").catch(() => null),
            UI.get("/intelligence/regulatory-change-diff").catch(() => null),
            UI.get("/intelligence/compliance-risk-heatmap").catch(() => null),
            UI.get("/intelligence/tax-calendar-optimization").catch(() => null),
            UI.get("/intelligence/cohort-analysis").catch(() => null),
            UI.get("/intelligence/entropy-revenue").catch(() => null),
            UI.get("/intelligence/hmm-financial-state").catch(() => null),
            UI.get("/intelligence/cusum-change-detection").catch(() => null),
            UI.get("/intelligence/svd-expense-decomposition").catch(() => null),
            UI.get("/intelligence/wavelet-revenue").catch(() => null),
            UI.get("/intelligence/altman-zscore").catch(() => null),
            UI.get("/intelligence/kmeans-supplier-clustering").catch(() => null),
            UI.get("/intelligence/composite-risk-score").catch(() => null),
        ]);
        if (advanced) renderAdvancedDashboard(advanced);
        if (reconcile4way) renderProductionReconciliation(reconcile4way, reconciliationCases);
        renderOverview(overview);
        if (scenario) renderScenario(scenario);
        if (probabilistic) renderProbabilisticForecast(probabilistic);
        if (forecast) renderForecast(forecast);
        if (channelAttribution) renderChannelAttribution(channelAttribution);
        if (taxReserve) renderTaxReserve(taxReserve);
        if (cashflowDelinquency) renderCashflowRisk(cashflowDelinquency);
        else if (cashflow) renderCashflowRisk(cashflow);
        if (benchmark) renderBenchmark(benchmark);
        if (suppliers) renderSupplierGraph(suppliers);
        if (graphRisk) renderGraphRisk(graphRisk);
        if (supplierAccountRisk) renderSupplierAccountRisk(supplierAccountRisk);
        if (inventoryAI) renderInventoryAI(inventoryAI);
        if (evidenceBundle) renderEvidenceBundle(evidenceBundle);
        if (legalChange) renderLegalChangeImpact(legalChange);
        if (nba) renderNextBestAction(nba);
        if (charts) renderCharts(charts);
        if (anomalies) renderAnomalies(anomalies);
        if (upgrade) renderUpgradeReadiness(upgrade);
        if (governance) renderGovernance(governance);
        if (productionGovernance) renderProductionGovernance(productionGovernance);
        if (benfordAnalysis) renderBenfordAnalysis(benfordAnalysis);
        if (seasonalDecomposition) renderSeasonalDecomposition(seasonalDecomposition);
        if (survivalAnalysis) renderSurvivalAnalysis(survivalAnalysis);
        if (bayesianForecast) renderBayesianForecast(bayesianForecast);
        if (explainability) renderExplainability(explainability);
        if (autoencoderBankAnomaly) renderAutoencoderAnomaly(autoencoderBankAnomaly);
        if (rfmCustomerSegmentation) renderRfmSegmentation(rfmCustomerSegmentation);
        if (workingCapital) renderWorkingCapital(workingCapital);
        if (regulatoryChangeDiff) renderRegulatoryChangeDiff(regulatoryChangeDiff);
        if (complianceRiskHeatmap) renderComplianceRiskHeatmap(complianceRiskHeatmap);
        if (taxCalendarOptimization) renderTaxCalendarOptimization(taxCalendarOptimization);
        if (cohortAnalysis) renderCohortAnalysis(cohortAnalysis);
        if (entropyRevenue) renderEntropyRevenue(entropyRevenue);
        if (hmmFinancialState) renderHmmFinancialState(hmmFinancialState);
        if (cusumChangeDetection) renderCusumChangeDetection(cusumChangeDetection);
        if (svdExpenseDecomposition) renderSvdExpenseDecomposition(svdExpenseDecomposition);
        if (waveletRevenue) renderWaveletRevenue(waveletRevenue);
        if (altmanZscore) renderAltmanZscore(altmanZscore);
        if (kmeansSupplierClustering) renderKMeansSupplierClustering(kmeansSupplierClustering);
        if (compositeRiskScore) renderCompositeRiskScore(compositeRiskScore);
        if (catalog) renderCatalog(catalog);
        renderCopilot();
    }

    window.loadTaxpayerAICapability = async function loadTaxpayerAICapability(key, payload = undefined) {
        const Registry = window.TaxpayerAIRegistry;
        const Client = window.TaxpayerAIClient;
        if (!Registry || !Client) throw new Error("Taxpayer AI registry chưa sẵn sàng.");
        const cap = Registry.getCapability(key);
        const body = payload ?? cap?.body ?? {};
        const data = await Client.requestCapability(key, body);
        renderRegistryCapability(key, data, Registry.resolvePageConfig(page));
        return data;
    };

    window.askTaxpayerLegalAI = async function askTaxpayerLegalAI(question) {
        const data = await UI.post("/intelligence/legal-chat", { question });
        return data.answer || data;
    };

    window.runTaxpayerOptimization = async function runTaxpayerOptimization(payload = {}) {
        const data = await UI.post("/intelligence/optimize-tax", payload);
        UI.panel("taxpayer-ai-optimization-panel", "Tối ưu hóa Phương pháp Tính thuế", "savings", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Phương pháp gợi ý</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml(data.preferred_method)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chênh lệch ước tính</p>
                    <p class="text-sm font-black text-slate-800">${UI.fmtVnd(data.estimated_saving)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Độ tin cậy</p>
                    <p class="text-sm font-black text-slate-800 text-indigo-600 capitalize">${UI.escapeHtml((data.model || {}).confidence || "Thấp")}</p>
                </div>
            </div>
            <p class="mt-3 text-[11px] text-slate-600">${UI.escapeHtml(data.recommendation)}</p>
        `);
        return data;
    };

    window.assistTaxpayerClaim = async function assistTaxpayerClaim(payload = {}) {
        const data = await UI.post("/intelligence/claim-assist", payload);
        UI.panel("taxpayer-ai-claim-assist-panel", "Đánh giá Hồ sơ Khiếu nại bằng AI (Claim Assist)", "assignment_late", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="text-[9px] uppercase font-bold text-slate-400">Mức sẵn sàng</p>
                    <p class="text-lg font-black text-slate-800">${UI.escapeHtml(data.readiness)} - ${Math.round(Number(data.readiness_score || 0))}/100</p>
                </div>
                ${UI.statusBadge(data.model?.confidence || "low")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[10px] font-black uppercase text-slate-400 mb-2">Thiếu sót cần bổ sung</p>
                    ${(data.evidence_gaps || []).map((item) => `<p class="text-[11px] text-slate-600 py-1 border-b border-white last:border-0">${UI.escapeHtml(item)}</p>`).join("") || `<p class="text-[11px] text-emerald-700 font-bold">Hồ sơ có cấu trúc tốt.</p>`}
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[10px] font-black uppercase text-slate-400 mb-2">Khung đơn đề xuất</p>
                    ${(data.draft_outline || []).slice(0, 5).map((item) => `<p class="text-[11px] text-slate-600 py-1 border-b border-white last:border-0">${UI.escapeHtml(item)}</p>`).join("")}
                </div>
            </div>
        `);
        return data;
    };

    window.precheckTaxReturnAI = async function precheckTaxReturnAI(payload = {}) {
        const data = await UI.post("/intelligence/tax-return-precheck", payload);
        UI.panel("taxpayer-ai-precheck-panel", "Kiểm tra Tờ khai trước khi nộp", "rule_folder", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="text-[9px] uppercase font-bold text-slate-400">Trạng thái tờ khai</p>
                    <p class="text-lg font-black text-slate-800">${UI.escapeHtml(data.readiness)} - ${Math.round(Number(data.readiness_score || 0))}/100</p>
                </div>
                ${UI.statusBadge((data.model || {}).confidence || "low")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${(data.issues || []).slice(0, 6).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><div class="flex justify-between gap-2"><b>${UI.escapeHtml(item.type)}</b>${priorityBadge(item.severity)}</div><p class="mt-1">${UI.escapeHtml(item.message)}</p></div>`).join("") || `<div class="p-2 rounded bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chưa phát hiện lỗi lớn nào.</div>`}
            </div>
        `);
        return data;
    };

    window.autoBookkeepingAI = async function autoBookkeepingAI(payload = {}) {
        const data = await UI.post("/intelligence/auto-bookkeeping", payload);
        UI.panel("taxpayer-ai-bookkeeping-panel", "Gợi ý Ghi sổ Tự động bằng AI", "edit_note", `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Mã sổ</th><th>Loại bút toán</th><th>Mô tả nghiệp vụ</th><th>Số tiền</th><th>Trạng thái thuế</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${(data.proposed_entries || []).slice(0, 8).map((item) => `
                            <tr>
                                <td class="py-2 font-mono font-bold">${UI.escapeHtml(item.book_code)}</td>
                                <td>${UI.escapeHtml(item.entry_type)}</td>
                                <td>${UI.escapeHtml(item.description)}</td>
                                <td>${UI.fmtVnd(item.amount)}</td>
                                <td>${UI.statusBadge(item.deductible_status || item.confidence)}</td>
                            </tr>
                        `).join("") || `<tr><td colspan="5" class="py-3 text-slate-400">Chưa có gợi ý ghi sổ nào.</td></tr>`}
                    </tbody>
                </table>
            </div>
        `);
        return data;
    };

    window.reconcileTaxpayerDocumentAI = async function reconcileTaxpayerDocumentAI(file, docType = "evidence") {
        const form = new FormData();
        if (file) form.append("file", file);
        form.append("doc_type", docType);
        const endpoint = docType === "legacy_ocr" ? "/intelligence/ocr-reconcile" : "/intelligence/document-ai/reconcile";
        const data = await UI.api(endpoint, { method: "POST", body: form });
        UI.panel("taxpayer-ai-reconcile-panel", "Đối soát Chứng từ bằng OCR", "document_scanner", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Trạng thái đối soát</p><p class="font-black">${UI.escapeHtml(data.reconciliation_status)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Khớp (Matches)</p><p class="font-black">${(data.reconciliation_matches || []).length}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Độ tin cậy</p><p class="font-black">${UI.escapeHtml((data.model || {}).confidence || "Thấp")}</p></div>
            </div>
        `);
        return data;
    };

    window.advancedDocumentExtractAI = async function advancedDocumentExtractAI(file, docType = "evidence") {
        const form = new FormData();
        if (file) form.append("file", file);
        form.append("doc_type", docType);
        const data = await UI.api("/intelligence/document-ai/extract", { method: "POST", body: form });
        UI.panel("taxpayer-ai-document-ai-panel", "Trích xuất Thông tin bằng Document AI", "document_scanner", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Phân loại đề xuất</p><p class="font-black">${UI.escapeHtml(data.suggested_category || data.doc_type)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Số trường trích xuất</p><p class="font-black">${Object.keys(data.extracted_fields || {}).length} trường</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Duyệt thủ công</p><p class="font-black">${(data.active_learning || {}).needs_human_review ? "YÊU CẦU" : "KHÔNG CẦN"}</p></div>
            </div>
        `);
        return data;
    };

    window.digitalTwinAI = async function digitalTwinAI(payload = {}) {
        const data = await UI.post("/intelligence/digital-twin/simulate", payload);
        UI.panel("taxpayer-ai-digital-twin-panel", "Mô phỏng Chuyển đổi Mô hình (Digital Twin HKD vs TNHH)", "schema", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${(data.variants || []).slice(0, 3).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <p class="font-bold text-slate-800">${UI.escapeHtml(item.label)}</p>
                        <p class="mt-1 text-[10px] text-slate-400">Ước tính Thuế: ${UI.fmtVnd(item.tax)}</p>
                        <p class="text-sm font-black text-emerald-700">${UI.fmtVnd(item.profit_after_tax)}</p>
                    </div>
                `).join("")}
            </div>
        `);
        return data;
    };

    window.advancedPrecheckTaxReturnAI = async function advancedPrecheckTaxReturnAI(payload = {}) {
        const data = await UI.post("/intelligence/filing/precheck-advanced", payload);
        UI.panel("taxpayer-ai-advanced-precheck-panel", "Kiểm tra Tờ khai Nâng cao (Advanced Precheck)", "rule_folder", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="text-[9px] uppercase font-bold text-slate-400">Điểm sẵn sàng nâng cao</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(data.advanced_readiness_score || 0))}/100</p>
                </div>
                ${UI.statusBadge(data.advanced_readiness || "needs_review")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${(data.issues || []).slice(0, 6).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><div class="flex justify-between gap-2"><b>${UI.escapeHtml(item.type)}</b>${priorityBadge(item.severity)}</div><p class="mt-1 text-slate-600 text-xs">${UI.escapeHtml(item.message)}</p></div>`).join("")}
            </div>
        `);
        return data;
    };

    window.ledgerAutopostAI = async function ledgerAutopostAI(payload = {}) {
        const data = await UI.post("/intelligence/ledger/autopost", payload);
        UI.panel("taxpayer-ai-ledger-autopost-panel", "Tự động Ghi Sổ Cái bằng AI (Ledger Autopost)", "post_add", `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Mã sổ</th><th>Tài khoản đối ứng</th><th>Mô tả bút toán</th><th>Số tiền</th><th>Độ tin cậy</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${(data.ledger_entries || []).slice(0, 8).map((item) => `<tr><td class="py-2 font-bold">${UI.escapeHtml(item.book_code)}</td><td>${UI.escapeHtml(item.account_code)}</td><td>${UI.escapeHtml(item.description)}</td><td>${UI.fmtVnd(item.amount)}</td><td>${Math.round(Number(item.confidence_score || 0) * 100)}%</td></tr>`).join("")}
                    </tbody>
                </table>
            </div>
        `);
        return data;
    };

    window.legalGraphRAGAI = async function legalGraphRAGAI(question, payload = {}) {
        const data = await UI.post("/intelligence/legal/graphrag", { ...payload, question });
        UI.panel("taxpayer-ai-graphrag-panel", "Truy vấn Pháp lý bằng Legal GraphRAG (Có nguồn dẫn)", "gavel", `
            <p class="font-bold text-slate-800 text-xs leading-relaxed">${UI.escapeHtml(data.answer)}</p>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${(data.citations || []).slice(0, 4).map((item) => `<a class="p-2 rounded bg-slate-50 border border-slate-200 text-[11px] font-bold text-slate-700 hover:bg-slate-100 transition-colors" href="${UI.escapeHtml(item.source_url || "#")}" target="_blank">${UI.escapeHtml(item.title || item.article_ref)}</a>`).join("")}
            </div>
        `);
        return data;
    };

    window.policyImpactAI = async function policyImpactAI(payload = {}) {
        const data = await UI.post("/intelligence/policy-impact", payload);
        UI.panel("taxpayer-ai-policy-impact-panel", "Đánh giá Tác động Chính sách Thuế theo Hồ sơ", "policy", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${(data.impacts || []).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2"><p class="font-bold text-slate-800 text-xs">${UI.escapeHtml(item.title)}</p>${priorityBadge(item.severity)}</div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p>
                    </div>
                `).join("")}
            </div>
        `);
        return data;
    };

    window.importBankConnectorAI = async function importBankConnectorAI(payload = {}) {
        const data = await UI.post("/connectors/bank/import", payload);
        UI.toast(`Đã nhập khẩu ${data.inserted_transactions || 0} giao dịch ngân hàng thành công.`);
        return data;
    };

    window.importEinvoiceConnectorAI = async function importEinvoiceConnectorAI(payload = {}) {
        const data = await UI.post("/connectors/einvoice/import", payload);
        UI.toast(`Đã nhập khẩu ${data.imported_invoices || 0} hóa đơn điện tử thành công.`);
        return data;
    };

    window.importEcommerceConnectorAI = async function importEcommerceConnectorAI(payload = {}) {
        const data = await UI.post("/connectors/ecommerce/import", payload);
        UI.toast(`Đã nhập khẩu ${data.imported_orders || 0} đơn hàng thành công.`);
        return data;
    };

    window.reconcile4WayAI = async function reconcile4WayAI(payload = {}) {
        const data = await UI.post("/intelligence/reconcile/4way", payload);
        renderProductionReconciliation(data, null);
        return data;
    };

    window.channelAttributionAI = async function channelAttributionAI(payload = {}) {
        const data = await UI.post("/intelligence/channel-attribution", payload);
        renderChannelAttribution(data);
        return data;
    };

    window.taxReserveOptimizerAI = async function taxReserveOptimizerAI(payload = {}) {
        const data = await UI.post("/intelligence/tax-reserve/optimize", payload);
        renderTaxReserve(data);
        return data;
    };

    window.supplierAccountRiskAI = async function supplierAccountRiskAI() {
        const data = await UI.get("/intelligence/supplier-account-risk");
        renderSupplierAccountRisk(data);
        return data;
    };

    window.inventoryAnalyzeAI = async function inventoryAnalyzeAI(payload = {}) {
        const data = await UI.post("/intelligence/inventory/analyze", payload);
        renderInventoryAI(data);
        return data;
    };

    window.evidenceBundleAI = async function evidenceBundleAI(payload = {}) {
        const data = await UI.post("/intelligence/evidence-bundle", payload);
        renderEvidenceBundle(data);
        return data;
    };

    window.legalChangeImpactAI = async function legalChangeImpactAI(payload = {}) {
        const data = await UI.post("/intelligence/legal/change-impact", payload);
        renderLegalChangeImpact(data);
        return data;
    };

    window.productionGovernanceAI = async function productionGovernanceAI() {
        const data = await UI.get("/intelligence/model-governance/production");
        renderProductionGovernance(data);
        return data;
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadIntelligence));
})();
