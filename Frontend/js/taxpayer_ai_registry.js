(function () {
    const PAGE_CONFIGS = {
        "business_dashboard.html": {
            title: "AI điều hành kinh doanh",
            icon: "space_dashboard",
            mode: "dashboard",
            prepend: true,
            capabilities: ["overview", "reconciliation_cases", "next_best_action"],
            advanced: ["advanced_dashboard", "probabilistic_forecast", "graph_risk", "composite_risk_score", "model_governance_production"],
        },
        "business_calendar.html": {
            title: "AI lịch thuế thông minh",
            icon: "event_upcoming",
            mode: "calendar",
            capabilities: ["overview", "forecast", "tax_calendar_optimization"],
            advanced: ["survival_analysis", "bayesian_forecast"],
        },
        "business_invoices.html": {
            title: "AI hóa đơn và đối tác",
            icon: "document_scanner",
            mode: "invoice",
            capabilities: ["reconciliation_cases", "supplier_account_risk", "supplier_risk_graph"],
            advanced: ["graph_risk", "pagerank_supplier_trust", "kmeans_supplier_clustering", "reconcile_4way"],
        },
        "business_filing.html": {
            title: "AI kiểm tra tờ khai",
            icon: "fact_check",
            mode: "filing",
            capabilities: ["reconciliation_cases", "forecast", "model_governance"],
            advanced: ["filing_precheck_advanced", "reconcile_4way", "tax_return_precheck"],
        },
        "business_debts.html": {
            title: "AI nợ thuế và dòng tiền",
            icon: "account_balance",
            mode: "debt",
            capabilities: ["cashflow_delinquency", "survival_analysis", "working_capital"],
            advanced: ["tax_reserve", "outflow_stress", "monte_carlo_simulation"],
        },
        "business_legal.html": {
            title: "AI pháp lý có trích dẫn",
            icon: "gavel",
            mode: "legal",
            capabilities: ["regulatory_change_diff", "model_governance", "overview"],
            advanced: ["legal_graphrag", "legal_change_impact", "policy_impact"],
        },
        "business_growth.html": {
            title: "AI tăng trưởng và chuyển đổi mô hình",
            icon: "trending_up",
            mode: "growth",
            capabilities: ["business_upgrade_readiness", "forecast", "probabilistic_forecast"],
            advanced: ["digital_twin", "bayesian_forecast", "breakeven_analysis"],
        },
        "business_accounting.html": {
            title: "AI sổ sách và chứng từ",
            icon: "auto_stories",
            mode: "accounting",
            capabilities: ["charts", "anomalies", "reconciliation_cases"],
            advanced: ["ledger_autopost", "benford_analysis", "svd_expense_decomposition", "wavelet_revenue"],
        },
        "business_expenses.html": {
            title: "AI chi phí được trừ",
            icon: "rule",
            mode: "expense",
            capabilities: ["charts", "anomalies", "peer_benchmark"],
            advanced: ["inventory_analyze", "isolation_forest_expenses", "benford_analysis"],
        },
        "business_claims.html": {
            title: "AI hồ sơ khiếu nại",
            icon: "balance",
            mode: "claim",
            capabilities: ["evidence_bundle", "overview", "model_governance"],
            advanced: ["claim_assist", "legal_change_impact", "regulatory_change_diff"],
        },
        "business_profile.html": {
            title: "AI chất lượng dữ liệu hồ sơ",
            icon: "manage_accounts",
            mode: "profile",
            capabilities: ["overview", "model_governance", "model_governance_production"],
            advanced: ["explainability"],
        },
        "business_registration.html": {
            title: "AI chuẩn bị đăng ký thuế",
            icon: "how_to_reg",
            mode: "registration",
            capabilities: ["overview", "forecast", "model_governance"],
            advanced: ["policy_impact"],
        },
        "business_calculator.html": {
            title: "AI mô phỏng thuế và lợi nhuận",
            icon: "query_stats",
            mode: "calculator",
            capabilities: ["forecast", "probabilistic_forecast", "breakeven_analysis"],
            advanced: ["bayesian_forecast", "monte_carlo_simulation", "digital_twin"],
        },
    };

    const CAPABILITIES = {
        overview: { endpoint: "/intelligence/overview", method: "GET", label: "Điểm sức khỏe", cost: "low", defaultPanel: true },
        forecast: { endpoint: "/intelligence/forecast", method: "GET", label: "Dự báo doanh thu", cost: "low", defaultPanel: true },
        peer_benchmark: { endpoint: "/intelligence/peer-benchmark", method: "GET", label: "Benchmark ngành", cost: "low", defaultPanel: true },
        charts: { endpoint: "/intelligence/charts", method: "GET", label: "Biểu đồ số liệu", cost: "low", defaultPanel: true },
        anomalies: { endpoint: "/intelligence/anomalies", method: "GET", label: "Bất thường dữ liệu", cost: "medium", defaultPanel: true },
        model_governance: { endpoint: "/intelligence/model-governance", method: "GET", label: "Quản trị mô hình", cost: "low" },
        model_governance_production: { endpoint: "/intelligence/model-governance/production", method: "GET", label: "Production gates", cost: "medium" },
        scenario_dashboard: { endpoint: "/intelligence/scenario-dashboard", method: "GET", label: "Kịch bản chiến lược", cost: "medium" },
        advanced_dashboard: { endpoint: "/intelligence/advanced-dashboard", method: "GET", label: "Control Tower nâng cao", cost: "medium" },
        probabilistic_forecast: { endpoint: "/intelligence/forecast/probabilistic", method: "GET", label: "Dự báo P10/P50/P90", cost: "medium" },
        cashflow_risk: { endpoint: "/intelligence/cashflow-risk", method: "GET", label: "Rủi ro dòng tiền", cost: "medium" },
        cashflow_delinquency: { endpoint: "/intelligence/cashflow/delinquency", method: "GET", label: "Rủi ro chậm nộp", cost: "medium" },
        supplier_risk_graph: { endpoint: "/intelligence/supplier-risk-graph", method: "GET", label: "Graph nhà cung cấp", cost: "medium" },
        supplier_account_risk: { endpoint: "/intelligence/supplier-account-risk", method: "GET", label: "Đổi tài khoản đối tác", cost: "medium" },
        graph_risk: { endpoint: "/intelligence/graph/risk", method: "GET", label: "Graph risk", cost: "high" },
        reconciliation_cases: { endpoint: "/intelligence/reconciliation-cases", method: "GET", label: "Case đối soát", cost: "low", defaultPanel: true },
        reconcile_4way: { endpoint: "/intelligence/reconcile/4way", method: "POST", label: "Đối soát 4 chiều", cost: "high", body: {} },
        channel_attribution: { endpoint: "/intelligence/channel-attribution", method: "POST", label: "Phân bổ kênh bán", cost: "medium", body: {} },
        tax_reserve: { endpoint: "/intelligence/tax-reserve/optimize", method: "POST", label: "Tối ưu quỹ thuế", cost: "high", body: {} },
        inventory_analyze: { endpoint: "/intelligence/inventory/analyze", method: "POST", label: "Tồn kho và COGS", cost: "high", body: {} },
        evidence_bundle: { endpoint: "/intelligence/evidence-bundle", method: "POST", label: "Hồ sơ chứng cứ", cost: "medium", body: { purpose: "tax_audit_explanation" } },
        legal_change_impact: { endpoint: "/intelligence/legal/change-impact", method: "POST", label: "Tác động văn bản mới", cost: "medium", body: {} },
        legal_graphrag: { endpoint: "/intelligence/legal/graphrag", method: "POST", label: "GraphRAG pháp lý", cost: "high", body: { question: "Các chính sách mới ảnh hưởng gì đến hồ sơ hiện tại?" } },
        policy_impact: { endpoint: "/intelligence/policy-impact", method: "POST", label: "Policy impact", cost: "medium", body: {} },
        business_upgrade_readiness: { endpoint: "/intelligence/business-upgrade-readiness", method: "GET", label: "Sẵn sàng chuyển đổi", cost: "medium" },
        next_best_action: { endpoint: "/intelligence/next-best-action", method: "GET", label: "Việc nên làm hôm nay", cost: "medium" },
        tax_calendar_optimization: { endpoint: "/intelligence/tax-calendar-optimization", method: "GET", label: "Tối ưu lịch nhắc", cost: "medium" },
        regulatory_change_diff: { endpoint: "/intelligence/regulatory-change-diff", method: "GET", label: "Văn bản mới", cost: "medium" },
        survival_analysis: { endpoint: "/intelligence/survival-analysis", method: "GET", label: "Survival risk", cost: "medium" },
        working_capital: { endpoint: "/intelligence/working-capital", method: "GET", label: "Vốn lưu động", cost: "medium" },
        digital_twin: { endpoint: "/intelligence/digital-twin/simulate", method: "POST", label: "Digital twin", cost: "high", body: {} },
        tax_return_precheck: { endpoint: "/intelligence/tax-return-precheck", method: "POST", label: "Precheck tờ khai", cost: "medium", body: {} },
        filing_precheck_advanced: { endpoint: "/intelligence/filing/precheck-advanced", method: "POST", label: "Precheck nâng cao", cost: "high", body: {} },
        ledger_autopost: { endpoint: "/intelligence/ledger/autopost", method: "POST", label: "Auto-ledger", cost: "high", body: {} },
        claim_assist: { endpoint: "/intelligence/claim-assist", method: "POST", label: "Claim assist", cost: "medium", body: {} },
        benford_analysis: { endpoint: "/intelligence/benford-analysis", method: "GET", label: "Benford", cost: "medium" },
        seasonal_decomposition: { endpoint: "/intelligence/seasonal-decomposition", method: "GET", label: "Mùa vụ", cost: "medium" },
        monte_carlo_simulation: { endpoint: "/intelligence/monte-carlo-simulation", method: "GET", label: "Monte Carlo", cost: "high" },
        breakeven_analysis: { endpoint: "/intelligence/breakeven-analysis", method: "GET", label: "Điểm hòa vốn", cost: "medium" },
        bayesian_forecast: { endpoint: "/intelligence/bayesian-forecast", method: "GET", label: "Bayesian forecast", cost: "medium" },
        isolation_forest_expenses: { endpoint: "/intelligence/isolation-forest-expenses", method: "GET", label: "Isolation Forest", cost: "medium" },
        explainability: { endpoint: "/intelligence/explainability", method: "GET", label: "SHAP/giải thích", cost: "medium" },
        pagerank_supplier_trust: { endpoint: "/intelligence/pagerank-supplier-trust", method: "GET", label: "PageRank đối tác", cost: "medium" },
        autoencoder_bank_anomaly: { endpoint: "/intelligence/autoencoder-bank-anomaly", method: "GET", label: "Autoencoder ngân hàng", cost: "high" },
        rfm_customer_segmentation: { endpoint: "/intelligence/rfm-customer-segmentation", method: "GET", label: "RFM khách hàng", cost: "medium" },
        outflow_stress: { endpoint: "/intelligence/outflow-stress", method: "GET", label: "Stress dòng tiền ra", cost: "high" },
        svd_expense_decomposition: { endpoint: "/intelligence/svd-expense-decomposition", method: "GET", label: "SVD chi phí", cost: "high" },
        wavelet_revenue: { endpoint: "/intelligence/wavelet-revenue", method: "GET", label: "Wavelet doanh thu", cost: "high" },
        composite_risk_score: { endpoint: "/intelligence/composite-risk-score", method: "GET", label: "Composite risk", cost: "medium" },
        kmeans_supplier_clustering: { endpoint: "/intelligence/kmeans-supplier-clustering", method: "GET", label: "Phân cụm đối tác", cost: "medium" },
    };

    function currentPage() {
        return window.location.pathname.split("/").pop() || "business_dashboard.html";
    }

    function resolvePageConfig(page = currentPage()) {
        return PAGE_CONFIGS[page] || PAGE_CONFIGS["business_dashboard.html"];
    }

    function getPageCapabilities(page = currentPage(), options = {}) {
        const cfg = resolvePageConfig(page);
        const base = cfg.capabilities || [];
        if (!options.includeAdvanced) return base.slice();
        return [...base, ...(cfg.advanced || [])].filter((key, idx, arr) => arr.indexOf(key) === idx);
    }

    function getCapability(key) {
        return CAPABILITIES[key] || null;
    }

    function getRenderBudget() {
        return { primary: 3, advancedLazy: true };
    }

    window.TaxpayerAIRegistry = {
        pageConfigs: PAGE_CONFIGS,
        capabilities: CAPABILITIES,
        currentPage,
        resolvePageConfig,
        getPageCapabilities,
        getCapability,
        getRenderBudget,
    };
})();
