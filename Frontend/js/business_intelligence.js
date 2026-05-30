(function () {
    const UI = window.TaxpayerUI;
    if (!UI) return;

    const page = window.location.pathname.split("/").pop() || "business_dashboard.html";
    const pageConfig = {
        "business_dashboard.html": { title: "AI dieu hanh kinh doanh", icon: "psychology", mode: "full", prepend: true },
        "business_calendar.html": { title: "AI du bao tre han va nguong doanh thu", icon: "event_upcoming", mode: "forecast" },
        "business_invoices.html": { title: "AI ra soat hoa don va doi tac", icon: "document_scanner", mode: "invoice" },
        "business_filing.html": { title: "AI kiem tra to khai truoc khi nop", icon: "fact_check", mode: "filing" },
        "business_debts.html": { title: "AI du bao no thue va dong tien", icon: "account_balance", mode: "debt" },
        "business_legal.html": { title: "AI phap ly co trich dan", icon: "gavel", mode: "legal" },
        "business_growth.html": { title: "AI goi y thay doi mo hinh", icon: "trending_up", mode: "growth" },
        "business_accounting.html": { title: "AI phan tich so sach va chung tu", icon: "auto_stories", mode: "accounting" },
        "business_expenses.html": { title: "AI phan loai chi phi duoc tru", icon: "rule", mode: "expense" },
        "business_claims.html": { title: "AI danh gia ho so khieu nai", icon: "balance", mode: "claim" },
        "business_profile.html": { title: "AI chat luong du lieu ho so", icon: "manage_accounts", mode: "profile" },
        "business_registration.html": { title: "AI san sang dang ky thue", icon: "how_to_reg", mode: "registration" },
        "business_calculator.html": { title: "AI mo phong thue va loi nhuan", icon: "query_stats", mode: "calculator" },
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
                ${scoreCard("Tai chinh", scores.financial_health, "monitoring")}
                ${scoreCard("Tuan thu", scores.compliance, "verified")}
                ${scoreCard("Dong tien", scores.cashflow, "payments")}
                ${scoreCard("Du lieu", scores.data_quality, "database")}
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
                <div class="space-y-2">
                    <p class="text-[10px] font-black uppercase text-slate-400">Canh bao thong minh</p>
                    ${alerts.slice(0, 3).map((item) => `
                        <div class="p-3 rounded-lg bg-white border border-slate-200">
                            <div class="flex items-center justify-between gap-2">
                                <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                                ${priorityBadge(item.severity)}
                            </div>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p>
                        </div>
                    `).join("") || `<div class="p-3 rounded-lg bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chua co canh bao lon trong du lieu hien tai.</div>`}
                </div>
                <div class="space-y-2">
                    <p class="text-[10px] font-black uppercase text-slate-400">Khuyen nghi hanh dong</p>
                    ${recs.slice(0, 3).map((item) => `
                        <div class="p-3 rounded-lg bg-white border border-slate-200">
                            <div class="flex items-center justify-between gap-2">
                                <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                                ${priorityBadge(item.priority)}
                            </div>
                            <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.reason)}</p>
                            <div class="mt-2 flex items-center gap-2">
                                <a href="${UI.escapeHtml(item.target_page || page)}" class="px-2 py-1 rounded bg-slate-900 text-white text-[10px] font-bold">${UI.escapeHtml(item.action_label || "Mo")}</a>
                                <button class="ai-feedback px-2 py-1 rounded bg-slate-100 text-slate-600 text-[10px] font-bold" data-target="${UI.escapeHtml(item.key)}" data-signal="helpful">Huu ich</button>
                                <button class="ai-feedback px-2 py-1 rounded bg-slate-100 text-slate-600 text-[10px] font-bold" data-target="${UI.escapeHtml(item.key)}" data-signal="not_relevant">Chua dung</button>
                            </div>
                        </div>
                    `).join("")}
                </div>
            </div>
            <div class="mt-3 flex items-center justify-between text-[10px] text-slate-400">
                <span>Model: ${UI.escapeHtml(model.model_name || "baseline")} / ${UI.escapeHtml(model.model_version || "")}</span>
                <span>Do tin cay: ${UI.escapeHtml(model.confidence || "low")}</span>
            </div>
        `, { prepend: cfg.prepend });
        bindFeedback();
    }

    function renderForecast(data) {
        const months = data.forecast_months || [];
        const probs = data.threshold_probabilities || {};
        if (!["full", "forecast", "growth", "debt", "calculator"].includes(cfg.mode)) return;
        UI.panel("taxpayer-ai-forecast-panel", "Du bao 6 thang va nguong rui ro", "timeline", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Doanh thu cuoi nam</p>
                    <p class="text-lg font-black text-slate-800">${UI.fmtVnd(data.projected_year_end_revenue)}</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Xac suat cham 1 ty</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(probs.einvoice_1b || 0) * 100)}%</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Dong tien 90 ngay</p>
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
        const metrics = data.taxpayer_metrics || {};
        const signals = data.signals || {};
        UI.panel("taxpayer-ai-benchmark-panel", "Benchmark cung nganh", "bar_chart", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Bien loi nhuan</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(metrics.profit_margin || 0) * 100)}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Ty le chi phi</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(metrics.expense_ratio || 0) * 100)}%</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tin hieu</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml(signals.margin_position || "unknown")}</p>
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
        UI.panel("taxpayer-ai-chart-panel", "Bieu do du lieu kinh doanh", "stacked_line_chart", `
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div class="lg:col-span-2 p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400 mb-3">Doanh thu - chi phi theo thang</p>
                    <div class="flex items-end gap-1 h-32">
                        ${months.map((item) => `
                            <div class="flex-1 flex flex-col justify-end gap-1 min-w-0">
                                <div title="${UI.escapeHtml(item.period)}: ${UI.fmtVnd(item.revenue)}" class="bg-emerald-500 rounded-t" style="height:${Math.max(4, Number(item.revenue || 0) / maxRevenue * 100)}px"></div>
                                <div title="Chi phi: ${UI.fmtVnd(item.expense)}" class="bg-amber-400 rounded-t" style="height:${Math.max(3, Number(item.expense || 0) / maxRevenue * 100)}px"></div>
                                <p class="truncate text-[8px] text-slate-400 text-center">${String(item.period || "").slice(5)}</p>
                            </div>
                        `).join("")}
                    </div>
                    <div class="mt-3 flex items-center gap-4 text-[9px] font-bold text-slate-500">
                        <span><i class="inline-block w-2 h-2 bg-emerald-500 rounded-sm"></i> Doanh thu</span>
                        <span><i class="inline-block w-2 h-2 bg-amber-400 rounded-sm"></i> Chi phi</span>
                    </div>
                </div>
                <div class="space-y-3">
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <p class="text-[9px] uppercase font-bold text-slate-400 mb-2">Kenh doanh thu</p>
                        ${channels.slice(0, 4).map((item) => `
                            <div class="mb-2">
                                <div class="flex justify-between text-[10px] font-bold"><span>${UI.escapeHtml(item.label)}</span><span>${Math.round(Number(item.share || 0) * 100)}%</span></div>
                                <div class="h-1.5 bg-white rounded overflow-hidden"><div class="h-full bg-emerald-500" style="width:${Math.round(Number(item.share || 0) * 100)}%"></div></div>
                            </div>
                        `).join("") || `<p class="text-[11px] text-slate-400">Chua co du lieu kenh.</p>`}
                    </div>
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <p class="text-[9px] uppercase font-bold text-slate-400 mb-2">Nhom chi phi</p>
                        ${expenses.slice(0, 4).map((item) => `
                            <div class="flex justify-between text-[10px] py-1 border-b border-white last:border-0">
                                <span class="font-bold text-slate-600">${UI.escapeHtml(item.label)}</span>
                                <span>${UI.fmtVnd(item.value)}</span>
                            </div>
                        `).join("") || `<p class="text-[11px] text-slate-400">Chua co du lieu chi phi.</p>`}
                    </div>
                </div>
            </div>
        `);
    }

    function renderAnomalies(data) {
        if (!["full", "invoice", "filing", "debt", "accounting", "expense", "claim"].includes(cfg.mode)) return;
        const anomalies = data.anomalies || [];
        UI.panel("taxpayer-ai-anomaly-panel", "Phat hien bat thuong", "radar", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${anomalies.slice(0, 6).map((item) => `
                    <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
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
        const flow = data.cashflow_30_60_90 || {};
        const plan = data.payment_plan || [];
        UI.panel("taxpayer-ai-cashflow-risk-panel", "Du bao dong tien nop thue", "account_balance_wallet", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Rui ro</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(data.risk_score || 0))}/100</p>
                    ${priorityBadge(data.risk_level)}
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">30 ngay</p><p class="text-sm font-black text-slate-800">${UI.fmtVnd(flow.days_30)}</p></div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">60 ngay</p><p class="text-sm font-black text-slate-800">${UI.fmtVnd(flow.days_60)}</p></div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">90 ngay</p><p class="text-sm font-black text-slate-800">${UI.fmtVnd(flow.days_90)}</p></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2">
                ${plan.slice(0, 3).map((item) => `
                    <div class="p-3 rounded bg-white border border-slate-200">
                        <div class="flex items-center justify-between gap-2"><p class="font-bold text-slate-700">${UI.escapeHtml(item.action)}</p>${priorityBadge(item.priority)}</div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.fmtVnd(item.amount)}</p>
                    </div>
                `).join("")}
            </div>
        `);
    }

    function renderSupplierGraph(data) {
        if (!["full", "invoice", "accounting"].includes(cfg.mode)) return;
        const summary = data.summary || {};
        const risks = data.top_risks || [];
        UI.panel("taxpayer-ai-supplier-graph-panel", "Graph rui ro nha cung cap", "hub", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Doi tac</p><p class="text-lg font-black text-slate-800">${summary.supplier_count || 0}</p></div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Lien ket hoa don</p><p class="text-lg font-black text-slate-800">${summary.edge_count || 0}</p></div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Rui ro cao</p><p class="text-lg font-black text-slate-800">${summary.high_risk_count || 0}</p></div>
            </div>
            <div class="mt-3 overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Doi tac</th><th>Hoa don</th><th>Gia tri</th><th>Rui ro</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${risks.slice(0, 5).map((item) => `
                            <tr>
                                <td class="py-2 font-bold">${UI.escapeHtml(item.partner_name || item.tax_code)}</td>
                                <td>${item.invoice_count || 0}</td>
                                <td>${UI.fmtVnd(item.amount)}</td>
                                <td>${priorityBadge(item.risk_level)}</td>
                            </tr>
                        `).join("") || `<tr><td colspan="4" class="py-3 text-slate-400">Chua co du lieu doi tac.</td></tr>`}
                    </tbody>
                </table>
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
        UI.panel("taxpayer-ai-advanced-command-panel", "Advanced AI command center", "memory", `
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
                ${scoreCard("Tai chinh", center.financial_health, "monitoring")}
                ${scoreCard("Tuan thu", center.compliance, "verified_user")}
                ${scoreCard("Graph risk", 100 - Number(center.graph_risk || 0), "hub")}
                ${scoreCard("Governance", center.data_quality, "admin_panel_settings")}
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
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
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
        UI.panel("taxpayer-ai-probabilistic-panel", "Du bao xac suat P10/P50/P90", "candlestick_chart", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Vuot 500M</p><p class="text-lg font-black">${Math.round(Number(probs.taxable_500m || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Vuot 1B</p><p class="text-lg font-black">${Math.round(Number(probs.einvoice_1b || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Vuot 3B</p><p class="text-lg font-black">${Math.round(Number(probs.group3_3b || 0) * 100)}%</p></div>
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
            <p class="mt-2 text-[10px] text-slate-400">Khoang rong = bat dinh cao hon; cot xanh = P50.</p>
        `);
    }

    function renderGraphRisk(data) {
        if (!["full", "invoice", "accounting", "expense"].includes(cfg.mode)) return;
        const summary = data.summary || {};
        const centrality = data.centrality || [];
        UI.panel("taxpayer-ai-graph-risk-panel", "Heterogeneous graph risk", "account_tree", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Node</p><p class="text-lg font-black">${summary.node_count || 0}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Edge</p><p class="text-lg font-black">${summary.edge_count || 0}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Density</p><p class="text-lg font-black">${Number(summary.density || 0).toFixed(3)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Risk</p><p class="text-lg font-black">${Math.round(Number(summary.graph_risk_score || 0))}/100</p></div>
            </div>
            <div class="mt-3 overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Node</th><th>Degree</th><th>Risk</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${centrality.slice(0, 5).map((item) => `<tr><td class="py-2 font-bold">${UI.escapeHtml(item.node_key)}</td><td>${item.degree || 0}</td><td>${Math.round(Number(item.risk_score || 0))}</td></tr>`).join("") || `<tr><td colspan="3" class="py-3 text-slate-400">Chua co graph du lieu.</td></tr>`}
                    </tbody>
                </table>
            </div>
        `);
    }

    function renderNextBestAction(data) {
        const actions = data.actions || [];
        if (!actions.length) return;
        UI.panel("taxpayer-ai-nba-panel", "Next-best-action theo causal AI", "moving", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${actions.slice(0, 6).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>
                            <span class="text-[10px] font-black text-emerald-700">${Math.round(Number(item.uplift_score || 0))}</span>
                        </div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.expected_impact || item.reason)}</p>
                        <a href="${UI.escapeHtml(item.target_page || page)}" class="inline-block mt-2 px-2 py-1 rounded bg-slate-900 text-white text-[10px] font-bold">${UI.escapeHtml(item.action_label || "Mo")}</a>
                    </div>
                `).join("")}
            </div>
        `);
    }

    function renderGovernance(data) {
        if (!["full", "legal", "profile"].includes(cfg.mode)) return;
        const drift = data.drift || {};
        const privacy = data.privacy || {};
        const feedback = data.feedback_quality || {};
        UI.panel("taxpayer-ai-governance-panel", "Model governance va privacy", "shield", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Drift</p><p class="text-lg font-black">${Math.round(Number(drift.score || 0))}/100</p>${priorityBadge(drift.level)}</div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Feedback</p><p class="text-lg font-black">${feedback.feedback_count || 0}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Bank consent</p><p class="text-xs font-black">${UI.escapeHtml(privacy.bank_training_consent || "not_granted")}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">FL ready</p><p class="text-xs font-black">${privacy.federated_learning_ready ? "READY" : "NO"}</p></div>
            </div>
        `);
    }

    function renderProductionReconciliation(data, caseData) {
        if (!["full", "invoice", "filing", "accounting"].includes(cfg.mode)) return;
        const summary = data.summary || {};
        const totals = data.totals || {};
        const cases = data.cases || (caseData || {}).cases || [];
        UI.panel("taxpayer-ai-production-reconcile-panel", "4-way reconciliation Bank-HDDT-So-To khai", "account_tree", `
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
                ${scoreCard("Doi soat", summary.reconciliation_score, "sync_alt")}
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Case mo</p><p class="text-lg font-black">${summary.open_case_count || cases.length || 0}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Bank in</p><p class="text-sm font-black">${UI.fmtVnd(totals.bank_in)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Khai DT</p><p class="text-sm font-black">${UI.fmtVnd(totals.declared_revenue)}</p></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${cases.slice(0, 4).map((item) => `
                    <div class="p-3 rounded bg-white border border-slate-200">
                        <div class="flex items-center justify-between gap-2"><p class="font-bold text-slate-800">${UI.escapeHtml(item.title || item.case_key)}</p>${priorityBadge(item.severity)}</div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.description || "")}</p>
                    </div>
                `).join("") || `<div class="p-3 rounded bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chua co exception doi soat lon.</div>`}
            </div>
        `, { prepend: cfg.mode === "full" });
    }

    function renderChannelAttribution(data) {
        if (!["full", "invoice", "filing", "accounting", "growth"].includes(cfg.mode)) return;
        const rows = data.attribution || [];
        UI.panel("taxpayer-ai-channel-attribution-panel", "Phan bo doanh thu da kenh", "call_split", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Bank in</p><p class="font-black">${UI.fmtVnd(data.bank_in)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Da nhan dien</p><p class="font-black">${UI.fmtVnd(data.recognized_revenue)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Chua gan kenh</p><p class="font-black">${UI.fmtVnd(data.missing_unattributed_revenue)}</p></div>
            </div>
            <div class="mt-3 space-y-2">
                ${rows.slice(0, 6).map((item) => `
                    <div class="flex items-center gap-2 text-xs">
                        <span class="w-24 font-bold text-slate-600 truncate">${UI.escapeHtml(item.channel)}</span>
                        <div class="flex-1 h-2 bg-slate-100 rounded overflow-hidden"><div class="h-full bg-emerald-500" style="width:${Math.min(100, Number(item.share || 0) * 100)}%"></div></div>
                        <span class="w-24 text-right font-bold">${UI.fmtVnd(item.amount)}</span>
                    </div>
                `).join("") || `<p class="text-[11px] text-slate-400">Chua co du lieu kenh.</p>`}
            </div>
        `);
    }

    function renderTaxReserve(data) {
        if (!["full", "forecast", "filing", "debt", "calculator"].includes(cfg.mode)) return;
        const schedule = data.optimized_payment_schedule || [];
        const fan = data.cash_fan_chart || [];
        UI.panel("taxpayer-ai-tax-reserve-panel", "Tax reserve optimizer", "savings", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Ty le du phong</p><p class="text-lg font-black">${Math.round(Number(data.recommended_reserve_rate || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Hang thang</p><p class="text-lg font-black">${UI.fmtVnd(data.monthly_reserve_amount)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Giam phat uoc tinh</p><p class="text-lg font-black">${UI.fmtVnd(data.expected_penalty_avoided)}</p></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${schedule.map((item) => `<div class="p-2 rounded bg-white border border-slate-200 text-xs"><b>T+${item.date_offset_days || 0} ngay:</b> ${UI.fmtVnd(item.amount)}<p class="text-[10px] text-slate-500">${UI.escapeHtml(item.objective || "")}</p></div>`).join("")}
                ${fan.slice(0, 4).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200 text-xs"><b>${item.horizon_days}d cash P50:</b> ${UI.fmtVnd(item.cash_p50)}</div>`).join("")}
            </div>
        `);
    }

    function renderSupplierAccountRisk(data) {
        if (!["full", "invoice", "accounting", "expense"].includes(cfg.mode)) return;
        const alerts = data.account_change_alerts || [];
        UI.panel("taxpayer-ai-supplier-account-risk-panel", "Supplier/account-change graph risk", "hub", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${(data.recommended_controls || []).slice(0, 3).map((item) => `<div class="p-3 rounded bg-slate-50 border border-slate-200 text-[11px] font-bold text-slate-600">${UI.escapeHtml(item)}</div>`).join("")}
            </div>
            <div class="mt-3 overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">Nha cung cap</th><th>TK</th><th>Thanh toan</th><th>Risk</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${alerts.slice(0, 6).map((item) => `<tr><td class="py-2 font-bold">${UI.escapeHtml(item.partner_name || item.supplier_key)}</td><td>${item.account_count || 0}</td><td>${UI.fmtVnd(item.payment_amount)}</td><td>${priorityBadge(item.severity)}</td></tr>`).join("") || `<tr><td colspan="4" class="py-3 text-slate-400">Chua co canh bao doi tai khoan.</td></tr>`}
                    </tbody>
                </table>
            </div>
        `);
    }

    function renderInventoryAI(data) {
        if (!["full", "accounting", "expense", "growth"].includes(cfg.mode)) return;
        const alerts = data.alerts || [];
        UI.panel("taxpayer-ai-inventory-cogs-panel", "Inventory va COGS intelligence", "inventory_2", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Gia von uoc tinh</p><p class="font-black">${UI.fmtVnd(data.cogs_estimate)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Bien gop</p><p class="font-black">${Math.round(Number(data.gross_margin || 0) * 100)}%</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Canh bao</p><p class="font-black">${alerts.length}</p></div>
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${alerts.slice(0, 4).map((item) => `<div class="p-2 rounded bg-white border border-slate-200"><div class="flex justify-between gap-2"><b>${UI.escapeHtml(item.type)}</b>${priorityBadge(item.severity)}</div><p class="text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p></div>`).join("") || `<div class="p-2 rounded bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chua co bat thuong ton kho/gia von.</div>`}
            </div>
        `);
    }

    function renderEvidenceBundle(data) {
        if (!["full", "filing", "legal", "claim", "expense"].includes(cfg.mode)) return;
        const sections = data.sections || [];
        UI.panel("taxpayer-ai-evidence-bundle-panel", "Evidence bundle AI", "folder_managed", `
            <div class="flex items-center justify-between gap-3">
                <div><p class="text-[9px] uppercase font-bold text-slate-400">Diem ho so</p><p class="text-lg font-black">${Math.round(Number(data.bundle_score || 0))}/100</p></div>
                ${UI.statusBadge(data.readiness || "needs_review")}
            </div>
            <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
                ${sections.slice(0, 8).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><p class="text-[10px] font-bold text-slate-600">${UI.escapeHtml(item.title)}</p><p class="text-sm font-black">${item.item_count || 0}</p></div>`).join("")}
            </div>
        `);
    }

    function renderLegalChangeImpact(data) {
        if (!["full", "legal", "growth", "calculator"].includes(cfg.mode)) return;
        const alerts = data.change_alerts || data.impacts || [];
        UI.panel("taxpayer-ai-legal-change-panel", "Legal change impact GraphRAG", "policy", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${alerts.slice(0, 4).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2"><p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>${priorityBadge(item.severity)}</div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message || item.action || "")}</p>
                    </div>
                `).join("")}
            </div>
            <div class="mt-3 flex flex-wrap gap-2">
                ${(data.citations || []).slice(0, 4).map((item) => `<a class="px-2 py-1 rounded bg-white border border-slate-200 text-[10px] font-bold text-slate-600" href="${UI.escapeHtml(item.source_url || "#")}" target="_blank">${UI.escapeHtml(item.article_ref || item.title || "source")}</a>`).join("")}
            </div>
        `);
    }

    function renderProductionGovernance(data) {
        if (!["full", "profile", "legal"].includes(cfg.mode)) return;
        const gates = data.production_gates || [];
        const readiness = data.connector_readiness || {};
        UI.panel("taxpayer-ai-production-governance-panel", "Production MLOps gates", "admin_panel_settings", `
            <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                ${Object.entries(readiness).map(([key, value]) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">${UI.escapeHtml(key)}</p><p class="font-black">${value || 0}</p></div>`).join("")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${gates.map((item) => `<div class="p-2 rounded border ${item.pass ? "border-emerald-100 bg-emerald-50 text-emerald-700" : "border-amber-100 bg-amber-50 text-amber-700"} text-[11px] font-bold">${item.pass ? "PASS" : "WAIT"} - ${UI.escapeHtml(item.gate)}</div>`).join("")}
            </div>
        `);
    }

    function renderCopilot() {
        UI.panel("taxpayer-ai-copilot-panel", "AI Copilot ngu canh", "support_agent", `
            <div class="flex flex-col md:flex-row gap-2">
                <input id="taxpayer-ai-copilot-input" class="flex-1 rounded-lg border-slate-200 text-xs" placeholder="Hoi ve dong tien, hoa don, chinh sach, chuyen doi..." />
                <button id="taxpayer-ai-copilot-btn" class="px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-bold">Hoi AI</button>
            </div>
            <div id="taxpayer-ai-copilot-result" class="mt-3 text-[11px] text-slate-600"></div>
        `);
        const input = document.getElementById("taxpayer-ai-copilot-input");
        const btn = document.getElementById("taxpayer-ai-copilot-btn");
        const send = async () => {
            const question = input ? input.value : "";
            if (!question.trim()) return;
            const resultBox = document.getElementById("taxpayer-ai-copilot-result");
            if (resultBox) resultBox.innerHTML = `<p class="font-bold text-slate-400">Dang phan tich...</p>`;
            try {
                const data = await UI.post("/intelligence/copilot", { question, page });
                if (resultBox) {
                    resultBox.innerHTML = `
                        <p class="font-bold text-slate-800">${UI.escapeHtml(data.answer)}</p>
                        <div class="mt-2 flex flex-wrap gap-2">
                            ${(data.actions || []).map((item) => `<a class="px-2 py-1 rounded bg-slate-100 text-slate-700 font-bold" href="${UI.escapeHtml(item.target_page || page)}">${UI.escapeHtml(item.label || "Mo")}</a>`).join("")}
                        </div>
                    `;
                }
            } catch (e) {
                if (resultBox) resultBox.innerHTML = `<p class="font-bold text-rose-600">${UI.escapeHtml(e.message || "Khong goi duoc AI.")}</p>`;
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

    async function loadIntelligence() {
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
        if (catalog) renderCatalog(catalog);
        renderCopilot();
    }

    window.askTaxpayerLegalAI = async function askTaxpayerLegalAI(question) {
        const data = await UI.post("/intelligence/legal-chat", { question });
        return data.answer || data;
    };

    window.runTaxpayerOptimization = async function runTaxpayerOptimization(payload = {}) {
        const data = await UI.post("/intelligence/optimize-tax", payload);
        UI.panel("taxpayer-ai-optimization-panel", "Toi uu phuong phap tinh thue", "savings", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Phuong phap goi y</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml(data.preferred_method)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Chenh lech uoc tinh</p>
                    <p class="text-sm font-black text-slate-800">${UI.fmtVnd(data.estimated_saving)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Do tin cay</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml((data.model || {}).confidence || "low")}</p>
                </div>
            </div>
            <p class="mt-3 text-[11px] text-slate-600">${UI.escapeHtml(data.recommendation)}</p>
        `);
        return data;
    };

    window.assistTaxpayerClaim = async function assistTaxpayerClaim(payload = {}) {
        const data = await UI.post("/intelligence/claim-assist", payload);
        UI.panel("taxpayer-ai-claim-assist-panel", "Danh gia ho so khieu nai bang AI", "assignment_late", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="text-[9px] uppercase font-bold text-slate-400">Muc san sang</p>
                    <p class="text-lg font-black text-slate-800">${UI.escapeHtml(data.readiness)} - ${Math.round(Number(data.readiness_score || 0))}/100</p>
                </div>
                ${UI.statusBadge(data.model?.confidence || "low")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[10px] font-black uppercase text-slate-400 mb-2">Thieu sot can bo sung</p>
                    ${(data.evidence_gaps || []).map((item) => `<p class="text-[11px] text-slate-600 py-1 border-b border-white last:border-0">${UI.escapeHtml(item)}</p>`).join("") || `<p class="text-[11px] text-emerald-700 font-bold">Ho so co cau truc tot.</p>`}
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[10px] font-black uppercase text-slate-400 mb-2">Khung don de xuat</p>
                    ${(data.draft_outline || []).slice(0, 5).map((item) => `<p class="text-[11px] text-slate-600 py-1 border-b border-white last:border-0">${UI.escapeHtml(item)}</p>`).join("")}
                </div>
            </div>
        `);
        return data;
    };

    window.precheckTaxReturnAI = async function precheckTaxReturnAI(payload = {}) {
        const data = await UI.post("/intelligence/tax-return-precheck", payload);
        UI.panel("taxpayer-ai-precheck-panel", "Kiem tra to khai truoc khi nop", "rule_folder", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="text-[9px] uppercase font-bold text-slate-400">Trang thai</p>
                    <p class="text-lg font-black text-slate-800">${UI.escapeHtml(data.readiness)} - ${Math.round(Number(data.readiness_score || 0))}/100</p>
                </div>
                ${UI.statusBadge((data.model || {}).confidence || "low")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${(data.issues || []).slice(0, 6).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><div class="flex justify-between gap-2"><b>${UI.escapeHtml(item.type)}</b>${priorityBadge(item.severity)}</div><p class="mt-1">${UI.escapeHtml(item.message)}</p></div>`).join("") || `<div class="p-2 rounded bg-emerald-50 border border-emerald-100 text-emerald-700 font-bold">Chua phat hien loi lon.</div>`}
            </div>
        `);
        return data;
    };

    window.autoBookkeepingAI = async function autoBookkeepingAI(payload = {}) {
        const data = await UI.post("/intelligence/auto-bookkeeping", payload);
        UI.panel("taxpayer-ai-bookkeeping-panel", "Goi y ghi so tu dong", "edit_note", `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">So</th><th>Loai</th><th>Mo ta</th><th>So tien</th><th>Trang thai</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">
                        ${(data.proposed_entries || []).slice(0, 8).map((item) => `
                            <tr>
                                <td class="py-2 font-mono font-bold">${UI.escapeHtml(item.book_code)}</td>
                                <td>${UI.escapeHtml(item.entry_type)}</td>
                                <td>${UI.escapeHtml(item.description)}</td>
                                <td>${UI.fmtVnd(item.amount)}</td>
                                <td>${UI.statusBadge(item.deductible_status || item.confidence)}</td>
                            </tr>
                        `).join("") || `<tr><td colspan="5" class="py-3 text-slate-400">Chua co goi y ghi so.</td></tr>`}
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
        UI.panel("taxpayer-ai-reconcile-panel", "Doi soat chung tu OCR", "document_scanner", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Trang thai</p><p class="font-black">${UI.escapeHtml(data.reconciliation_status)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Match</p><p class="font-black">${(data.reconciliation_matches || []).length}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Tin cay</p><p class="font-black">${UI.escapeHtml((data.model || {}).confidence || "low")}</p></div>
            </div>
        `);
        return data;
    };

    window.advancedDocumentExtractAI = async function advancedDocumentExtractAI(file, docType = "evidence") {
        const form = new FormData();
        if (file) form.append("file", file);
        form.append("doc_type", docType);
        const data = await UI.api("/intelligence/document-ai/extract", { method: "POST", body: form });
        UI.panel("taxpayer-ai-document-ai-panel", "Document AI extraction", "document_scanner", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Loai</p><p class="font-black">${UI.escapeHtml(data.suggested_category || data.doc_type)}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Fields</p><p class="font-black">${Object.keys(data.extracted_fields || {}).length}</p></div>
                <div class="p-3 rounded bg-slate-50 border border-slate-200"><p class="text-[9px] uppercase font-bold text-slate-400">Review</p><p class="font-black">${(data.active_learning || {}).needs_human_review ? "CAN" : "OK"}</p></div>
            </div>
        `);
        return data;
    };

    window.digitalTwinAI = async function digitalTwinAI(payload = {}) {
        const data = await UI.post("/intelligence/digital-twin/simulate", payload);
        UI.panel("taxpayer-ai-digital-twin-panel", "Digital twin HKD vs TNHH", "schema", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                ${(data.variants || []).slice(0, 3).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <p class="font-bold text-slate-800">${UI.escapeHtml(item.label)}</p>
                        <p class="mt-1 text-[10px] text-slate-400">Thue: ${UI.fmtVnd(item.tax)}</p>
                        <p class="text-sm font-black text-emerald-700">${UI.fmtVnd(item.profit_after_tax)}</p>
                    </div>
                `).join("")}
            </div>
        `);
        return data;
    };

    window.advancedPrecheckTaxReturnAI = async function advancedPrecheckTaxReturnAI(payload = {}) {
        const data = await UI.post("/intelligence/filing/precheck-advanced", payload);
        UI.panel("taxpayer-ai-advanced-precheck-panel", "Advanced precheck to khai", "rule_folder", `
            <div class="flex items-center justify-between gap-3">
                <div>
                    <p class="text-[9px] uppercase font-bold text-slate-400">Diem advanced</p>
                    <p class="text-lg font-black text-slate-800">${Math.round(Number(data.advanced_readiness_score || 0))}/100</p>
                </div>
                ${UI.statusBadge(data.advanced_readiness || "needs_review")}
            </div>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${(data.issues || []).slice(0, 6).map((item) => `<div class="p-2 rounded bg-slate-50 border border-slate-200"><div class="flex justify-between gap-2"><b>${UI.escapeHtml(item.type)}</b>${priorityBadge(item.severity)}</div><p class="mt-1">${UI.escapeHtml(item.message)}</p></div>`).join("")}
            </div>
        `);
        return data;
    };

    window.ledgerAutopostAI = async function ledgerAutopostAI(payload = {}) {
        const data = await UI.post("/intelligence/ledger/autopost", payload);
        UI.panel("taxpayer-ai-ledger-autopost-panel", "Ledger autopost AI", "post_add", `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400"><tr><th class="py-2">So</th><th>TK</th><th>Mo ta</th><th>So tien</th><th>Tin cay</th></tr></thead>
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
        UI.panel("taxpayer-ai-graphrag-panel", "Legal GraphRAG co citation", "gavel", `
            <p class="font-bold text-slate-800">${UI.escapeHtml(data.answer)}</p>
            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${(data.citations || []).slice(0, 4).map((item) => `<a class="p-2 rounded bg-slate-50 border border-slate-200 text-[11px] font-bold text-slate-700" href="${UI.escapeHtml(item.source_url || "#")}" target="_blank">${UI.escapeHtml(item.title || item.article_ref)}</a>`).join("")}
            </div>
        `);
        return data;
    };

    window.policyImpactAI = async function policyImpactAI(payload = {}) {
        const data = await UI.post("/intelligence/policy-impact", payload);
        UI.panel("taxpayer-ai-policy-impact-panel", "Tac dong chinh sach theo ho so", "policy", `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${(data.impacts || []).map((item) => `
                    <div class="p-3 rounded bg-slate-50 border border-slate-200">
                        <div class="flex items-center justify-between gap-2"><p class="font-bold text-slate-800">${UI.escapeHtml(item.title)}</p>${priorityBadge(item.severity)}</div>
                        <p class="mt-1 text-[11px] text-slate-500">${UI.escapeHtml(item.message)}</p>
                    </div>
                `).join("")}
            </div>
        `);
        return data;
    };

    window.importBankConnectorAI = async function importBankConnectorAI(payload = {}) {
        const data = await UI.post("/connectors/bank/import", payload);
        UI.toast(`Da import ${data.inserted_transactions || 0} giao dich ngan hang.`);
        return data;
    };

    window.importEinvoiceConnectorAI = async function importEinvoiceConnectorAI(payload = {}) {
        const data = await UI.post("/connectors/einvoice/import", payload);
        UI.toast(`Da import ${data.imported_invoices || 0} hoa don dien tu.`);
        return data;
    };

    window.importEcommerceConnectorAI = async function importEcommerceConnectorAI(payload = {}) {
        const data = await UI.post("/connectors/ecommerce/import", payload);
        UI.toast(`Da import ${data.imported_orders || 0} don hang.`);
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
