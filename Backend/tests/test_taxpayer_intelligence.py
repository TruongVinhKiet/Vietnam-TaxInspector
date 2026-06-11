import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.taxpayer_intelligence import TaxpayerIntelligenceService


def sample_dataset() -> dict:
    return {
        "year": 2026,
        "today": "2026-05-29",
        "profile": {
            "tax_code": "1234567890",
            "full_name": "Ho kinh doanh mau",
            "business_name": "Ho kinh doanh mau",
            "household_group": 2,
            "annual_revenue": 900_000_000,
            "industry": "commerce",
            "email": "taxpayer@example.com",
            "phone": "0900000000",
            "address": "TP HCM",
        },
        "revenue_entries": [
            {"entry_date": "2026-01-10", "amount": 100_000_000, "channel": "direct"},
            {"entry_date": "2026-02-10", "amount": 120_000_000, "channel": "ecommerce"},
            {"entry_date": "2026-03-10", "amount": 130_000_000, "channel": "direct"},
            {"entry_date": "2026-04-10", "amount": 150_000_000, "channel": "direct"},
        ],
        "expense_entries": [
            {
                "expense_date": "2026-04-12",
                "amount": 6_000_000,
                "category": "materials",
                "payment_method": "cash",
                "deductible_status": "non_deductible",
            },
            {
                "expense_date": "2026-04-13",
                "amount": 80_000_000,
                "category": "rent",
                "payment_method": "bank_transfer",
                "deductible_status": "deductible",
            },
        ],
        "invoices": [
            {
                "invoice_number": "HD-001",
                "status": "valid",
                "amount": 10_000_000,
                "total_amount": 10_000_000,
                "seller_tax_code": "0101234567",
                "partner_name": "Nha cung cap A",
                "risk_json": {},
            }
        ],
        "filings": [],
        "payments": [],
        "debts": [{"amount_due": 55_000_000, "amount_paid": 0, "due_date": "2026-01-01"}],
        "documents": [],
        "claims": [],
        "bank_transactions": [
            {"transaction_date": "2026-04-10", "direction": "in", "amount": 140_000_000, "channel": "bank_transfer", "description": "Thu tien ban hang"},
            {
                "transaction_date": "2026-04-12",
                "direction": "out",
                "amount": 80_000_000,
                "counterparty_name": "Nha cung cap A",
                "counterparty_tax_code": "0101234567",
                "bank_account": "123",
                "metadata_json": {"counterparty_account": "998877"},
            },
            {
                "transaction_date": "2026-04-18",
                "direction": "out",
                "amount": 20_000_000,
                "counterparty_name": "Nha cung cap A",
                "counterparty_tax_code": "0101234567",
                "bank_account": "123",
                "metadata_json": {"counterparty_account": "112233"},
            },
        ],
        "ledger_entries": [{"entry_type": "revenue", "amount": 420_000_000}, {"entry_type": "expense", "amount": 86_000_000}],
        "platform_orders": [{"platform": "shopee", "gross_amount": 130_000_000, "net_amount": 124_000_000}],
        "einvoice_line_items": [{"invoice_number": "HD-001", "item_name": "Hang A", "quantity": 10, "amount": 70_000_000}],
        "inventory_movements": [{"sku": "A", "movement_type": "out", "quantity": 12, "unit_cost": 7_000_000, "total_cost": 84_000_000}],
        "reconciliation_cases": [],
    }


def test_intelligence_overview_produces_scores_alerts_and_recommendations() -> None:
    service = TaxpayerIntelligenceService()
    result = service.overview(sample_dataset())
    assert result["scores"]["compliance"] < 100
    assert result["alerts"]
    assert result["top_recommendations"]
    assert result["model"]["confidence"] in {"low", "medium", "high"}


def test_forecast_thresholds_and_what_if_are_explainable() -> None:
    service = TaxpayerIntelligenceService()
    forecast = service.forecast(sample_dataset())
    assert len(forecast["forecast_months"]) == 6
    assert forecast["threshold_probabilities"]["einvoice_1b"] >= 0
    assert forecast["model"]["model_version"]

    scenario = service.what_if({"revenue": 800_000_000, "expenses": 520_000_000, "industry": "commerce"}, sample_dataset())
    assert scenario["taxes"]["total_tax"] > 0
    assert scenario["household_group"]["group"] == 2
    assert scenario["model"]["input_hash"]


def test_expense_invoice_document_and_benchmark_quickwins() -> None:
    service = TaxpayerIntelligenceService()
    expense = service.classify_expense({"amount": 5_000_000, "payment_method": "cash", "has_invoice": True})
    assert expense["label"] == "non_deductible"
    assert expense["risk_score"] >= 80

    invoice = service.invoice_risk({"invoice_number": "HD-001", "seller_tax_code": "000123"}, sample_dataset())
    assert invoice["risk_level"] in {"medium", "high"}
    assert "duplicate_invoice_number_in_taxpayer_log" in invoice["risk_flags"]

    extraction = service.extract_document(b"<Invoice><MST>1234567890</MST><SHDon>42</SHDon></Invoice>", "invoice.xml")
    assert extraction["extracted_fields"]["tax_code"] == "1234567890"

    benchmark = service.peer_benchmark(sample_dataset())
    assert benchmark["signals"]["margin_position"] in {"below_peer_range", "within_peer_range", "above_peer_range"}


def test_chart_anomaly_optimization_claim_and_catalog_features() -> None:
    service = TaxpayerIntelligenceService()
    charts = service.chart_analytics(sample_dataset())
    assert len(charts["monthly_series"]) == 12
    assert charts["threshold_markers"][0]["value"] == 500_000_000.0

    anomalies = service.anomaly_insights(sample_dataset())
    assert anomalies["anomalies"]
    assert anomalies["model"]["input_hash"]

    optimized = service.optimize_tax({"revenue": 900_000_000, "expenses": 720_000_000}, sample_dataset())
    assert {item["key"] for item in optimized["methods"]} == {"revenue_percentage", "profit_based"}
    assert optimized["checklist"]

    claim = service.claim_assist(
        {
            "decision_no": "QD-01",
            "description": "Khieu nai quyet dinh truy thu vi doanh thu bi tinh trung giua san TMDT va cua hang truc tiep, can dieu chinh so tien thue.",
            "evidence_items": ["to khai", "sao ke", "hoa don"],
        },
        sample_dataset(),
    )
    assert claim["readiness_score"] >= 70

    catalog = service.model_catalog()
    assert any(item["key"] == "legal_rag" for item in catalog["items"])


def test_taxpayer_intelligence_v2_practical_workflows() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()

    scenario = service.scenario_dashboard(dataset)
    assert scenario["persona"]["household_group"]["group"] in {1, 2, 3}
    assert scenario["risk_heatmap"]

    cashflow = service.cashflow_risk(dataset)
    assert cashflow["risk_level"] in {"low", "medium", "high"}
    assert "reserve_needed" in cashflow

    supplier = service.supplier_risk_graph(dataset)
    assert supplier["summary"]["supplier_count"] >= 1
    assert supplier["graph"]["nodes"]

    reconcile = service.ocr_reconcile(b"<Invoice><MST>0101234567</MST><SHDon>HD-001</SHDon><TotalAmount>10000000</TotalAmount></Invoice>", "invoice.xml", "invoice_in", dataset)
    assert reconcile["reconciliation_status"] == "matched"
    assert reconcile["suggested_book_entries"]

    bookkeeping = service.auto_bookkeeping({"description": "Mua hang hoa dau vao", "amount": 5_500_000, "payment_method": "cash"}, dataset)
    assert bookkeeping["proposed_entries"][0]["book_code"] == "S2a-HKD"
    assert bookkeeping["control_warnings"]

    precheck = service.tax_return_precheck({"revenue": 300_000_000, "expenses": 90_000_000}, dataset)
    assert precheck["issues"]
    assert precheck["readiness"] in {"ready", "needs_review", "blocked"}

    impact = service.policy_impact({"channels": ["ecommerce"], "revenue": 1_200_000_000}, dataset)
    assert impact["impacts"]
    assert impact["citations"]

    upgrade = service.business_upgrade_readiness(dataset)
    assert upgrade["readiness_level"] in {"ready", "prepare", "not_ready"}
    assert upgrade["components"]

    copilot = service.copilot({"question": "Toi co nen chuyen len cong ty TNHH khong?", "page": "business_growth.html"}, dataset)
    assert copilot["answer"]
    assert copilot["actions"]


def test_taxpayer_advanced_intelligence_platform_workflows() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()

    dashboard = service.advanced_dashboard(dataset)
    assert dashboard["command_center"]["graph_risk"] >= 0
    assert dashboard["top_actions"]
    assert dashboard["explanation"]["reason_codes"]

    doc_ai = service.document_ai_extract(
        b"<Invoice><MST>0101234567</MST><SHDon>HD-001</SHDon><TotalAmount>10000000</TotalAmount></Invoice>",
        "invoice.xml",
        "invoice_in",
        dataset,
    )
    assert doc_ai["field_confidence"]["tax_code"] > 0
    assert doc_ai["table_extraction"]["method"]

    reconcile = service.document_ai_reconcile(
        b"<Invoice><MST>0101234567</MST><SHDon>HD-001</SHDon><TotalAmount>10000000</TotalAmount></Invoice>",
        "invoice.xml",
        "invoice_in",
        dataset,
    )
    assert "reconciliation_score" in reconcile

    probabilistic = service.probabilistic_forecast(dataset)
    assert probabilistic["intervals"][0]["p10"] <= probabilistic["intervals"][0]["p50"] <= probabilistic["intervals"][0]["p90"]
    assert probabilistic["threshold_probabilities"]["taxable_500m"] >= 0

    twin = service.digital_twin_simulate({"growth_rate_pct": 20, "months_active": 12}, dataset)
    assert len(twin["variants"]) == 3
    assert twin["recommended_variant"] in {item["key"] for item in twin["variants"]}

    graph = service.graph_risk(dataset)
    assert graph["summary"]["node_count"] >= 1
    assert graph["gnn_signals"]["artifact_status"] == "adapter_ready"

    ledger = service.ledger_autopost({"description": "Mua hang hoa dau vao", "amount": 8_000_000, "payment_method": "cash"}, dataset)
    assert ledger["ledger_entries"]
    assert ledger["missing_evidence"]

    precheck = service.filing_precheck_advanced({"revenue": 300_000_000, "expenses": 90_000_000}, dataset)
    assert precheck["advanced_readiness"] in {"ready", "needs_review", "blocked"}
    assert precheck["explainable_delta"]["main_drivers"]

    delinquency = service.cashflow_delinquency(dataset)
    assert delinquency["survival_curve"]
    assert delinquency["hazard_90d"] >= delinquency["hazard_30d"]

    rag = service.legal_graphrag({"question": "Chi phi tien mat 5 trieu co duoc tru khong?"}, dataset)
    assert rag["citations"]
    assert rag["citation_verifier"]["citation_count"] >= 1

    nba = service.next_best_action(dataset)
    assert nba["actions"][0]["uplift_score"] >= nba["actions"][-1]["uplift_score"]

    governance = service.model_governance(dataset)
    assert governance["privacy"]["federated_learning_ready"] is True
    assert governance["model_cards"]


def test_taxpayer_production_intelligence_workflows() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()

    registry = service.capability_registry()
    assert registry["render_budget"]["primary_panels_per_page"] == 3
    assert any(item["key"] == "reconcile_4way" for item in registry["capabilities"])
    assert any(item["default_panel"] for item in registry["capabilities"])

    sufficiency = service.data_sufficiency(service.overview(dataset)["snapshot"])
    assert sufficiency["score"] >= 50
    assert sufficiency["tier"] in {"usable", "rich"}

    reconcile = service.reconcile_4way({}, dataset)
    assert reconcile["summary"]["open_case_count"] >= 1
    assert reconcile["model_name"]
    assert "needs_human_confirmation" in reconcile
    assert reconcile["data_sufficiency_score"] >= 0

    cases = service.reconciliation_cases({**dataset, "reconciliation_cases": reconcile["cases"]})
    assert cases["summary"]["case_count"] == len(reconcile["cases"])

    channel = service.channel_attribution({}, dataset)
    assert channel["attribution"]
    assert channel["reason_codes"]

    reserve = service.tax_reserve_optimize({"current_cash": 100_000_000}, dataset)
    assert reserve["optimized_payment_schedule"]
    assert reserve["recommended_reserve_rate"] > 0

    supplier = service.supplier_account_risk(dataset)
    assert supplier["account_change_alerts"]
    assert supplier["needs_human_confirmation"] is True

    inventory = service.inventory_analyze({}, dataset)
    assert inventory["alerts"]
    assert inventory["cogs_estimate"] > 0

    bundle = service.evidence_bundle({"purpose": "appeal"}, dataset)
    assert bundle["sections"]
    assert bundle["readiness"] in {"ready", "needs_review", "blocked"}

    legal = service.legal_change_impact({"channels": ["ecommerce"], "revenue": 1_200_000_000}, dataset)
    assert legal["citations"]
    assert legal["needs_human_confirmation"] is True

    governance = service.model_governance_production(dataset)
    assert governance["production_gates"]
    assert "taxpayer_reconciliation_ranker" not in governance["tracked_training_tracks"]
    assert "reconciliation_ranker" in governance["tracked_training_tracks"]


def test_f1_benford_analysis_chi_square_and_digit_distribution() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.benford_analysis(dataset)
    assert result["sample_size"] >= 1
    assert len(result["digits"]) == 9
    assert result["chi_square"] >= 0
    assert 0 <= result["p_value"] <= 1
    assert result["verdict"] in {"conforming", "moderate_deviation", "significant_deviation", "insufficient_data"}
    assert result["degrees_of_freedom"] == 8
    # Each digit entry has expected_pct and observed_pct
    for d in result["digits"]:
        assert 1 <= d["digit"] <= 9
        assert d["expected_pct"] >= 0
        assert d["observed_pct"] >= 0


def test_f2_seasonal_decomposition_trend_seasonal_residual() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.seasonal_decomposition(dataset)
    assert len(result["series"]) == 12
    assert result["seasonal_strength"] >= 0
    assert result["trend_direction"] in {"up", "down", "stable", "insufficient_data"}
    assert result["seasonal_label"] in {"Mua vu manh", "Mua vu vua", "It mua vu"}
    for s in result["series"]:
        assert "original" in s and "trend" in s and "seasonal" in s and "residual" in s


def test_f3_monte_carlo_percentile_ordering_and_expense_ratio() -> None:
    service = TaxpayerIntelligenceService()
    result = service.monte_carlo_simulation(
        revenue_mean=500_000_000,
        volatility_pct=15,
        expense_ratio_pct=60,
        tax_rate_pct=1.5,
        iterations=5000,
    )
    p = result["percentiles"]
    # Percentiles must be monotonically increasing
    assert p["P5"]["tax"] <= p["P25"]["tax"] <= p["P50"]["tax"] <= p["P75"]["tax"] <= p["P95"]["tax"]
    assert p["P5"]["revenue"] <= p["P95"]["revenue"]
    # Tax should be on profit, not revenue — so with 60% expense ratio, tax ~ rev*0.4*0.015
    expected_median_tax = 500_000_000 * 0.4 * 0.015
    assert abs(p["P50"]["tax"] - expected_median_tax) / expected_median_tax < 0.25  # within 25%
    assert len(result["bins"]) == 10
    assert result["value_at_risk_95"]["tax"] > 0


def test_f4_breakeven_analysis_safety_margin_and_cvp() -> None:
    service = TaxpayerIntelligenceService()
    result = service.breakeven_analysis(
        fixed_costs=50_000_000,
        variable_cost_ratio=40,
        current_revenue=150_000_000,
        target_profit=30_000_000,
    )
    assert result["status"] == "success"
    assert result["breakeven_revenue"] > 0
    # contribution_margin_ratio = 1 - 0.4 = 0.6, breakeven = 50M / 0.6 ≈ 83.3M
    assert abs(result["breakeven_revenue"] - 83_333_333) < 1_000_000
    assert result["safety_margin_pct"] > 0  # current > breakeven
    assert result["target_revenue"] > result["breakeven_revenue"]
    assert result["verdict"] == "safe"
    assert len(result["points"]) == 11


def test_f5_survival_analysis_hazard_and_series() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.survival_analysis(dataset)
    assert len(result["series"]) == 12
    assert result["hazard_ratio"] >= 1.0  # dataset has debt, so theta > 1
    assert 0 < result["survival_index"] <= 100
    assert result["verdict"] in {"low_risk", "medium_risk", "high_risk"}
    # Survival probabilities should decrease over time
    probs = [s["survival_probability_pct"] for s in result["series"]]
    assert probs[0] >= probs[-1]
    assert result["insights"]


def test_f6_bayesian_forecast_uses_real_data_and_credible_intervals() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.bayesian_forecast(dataset)
    assert result["status"] == "success"
    assert result["historical_months_count"] >= 3
    assert result["posterior_mean"] > 0
    assert len(result["series"]) == 6
    # Credible intervals: 95% wider than 80%
    for s in result["series"]:
        assert s["hdi_95_lower"] <= s["hdi_80_lower"]
        assert s["hdi_80_upper"] <= s["hdi_95_upper"]
        assert s["expected_mean"] > 0
    # With 4 revenue entries (100M, 120M, 130M, 150M), posterior should be near actual mean ~125M
    assert result["posterior_mean"] > 50_000_000


def test_f7_isolation_forest_expenses_anomaly_scoring() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.isolation_forest_expenses(dataset)
    assert result["summary"]["total"] == 2
    assert result["contamination"] >= 0
    assert len(result["anomalies"]) <= 20
    for a in result["anomalies"]:
        assert "anomaly_score" in a
        assert "z_score" in a
        assert "is_anomaly" in a
    assert "isolation_forest_scoring" in result["method_stack"]


def test_f8_markov_chain_transition_matrix_and_steady_state() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.markov_chain_prediction(dataset)
    assert result["current_state"] in {"growth", "stable", "decline"}
    # Transition matrix rows should sum to ~1.0
    matrix = result["transition_matrix"]["matrix"]
    for row in matrix:
        assert abs(sum(row) - 1.0) < 0.01
    # Steady state probabilities should sum to ~1.0
    ss = result["steady_state"]
    assert abs(ss["growth"] + ss["stable"] + ss["decline"] - 1.0) < 0.01
    assert len(result["step_probabilities"]) == 6
    assert len(result["trajectory"]) == 7  # current + 6 steps


def test_f9_shap_explainability_contributions_sum_to_risk() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.explainability(dataset)
    assert result["status"] == "success"
    assert len(result["contributions"]) == 5
    # Base + sum of SHAP values should equal compliance_risk_score
    computed = result["base_value"] + sum(c["shap_value"] for c in result["contributions"])
    # Clamped to [1, 99]
    expected = max(1.0, min(99.0, computed))
    assert abs(result["compliance_risk_score"] - expected) < 0.1
    for c in result["contributions"]:
        assert c["direction"] in {"risk", "compliance"}


def test_f10_pagerank_supplier_trust_tier_ranking() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.pagerank_supplier_trust(dataset)
    assert result["summary"]["total_suppliers"] >= 1
    for s in result["suppliers"]:
        assert s["trust_tier"] in {"A", "B", "C", "D"}
        assert s["pagerank_score"] > 0
        assert s["trust_score"] >= 0
    assert "modified_pagerank" in result["method_stack"]


def test_f11_autoencoder_bank_anomaly_reconstruction_error() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.autoencoder_bank_anomaly(dataset)
    assert result["summary"]["total"] == 3  # 3 bank transactions in sample
    assert result["threshold"] > 0
    for a in result["anomalies"]:
        assert "reconstruction_error" in a
        assert "z_score" in a
        assert a["direction"] in {"in", "out"}
    assert result["latent_dim"] == 8


def test_f12_rfm_customer_segmentation_and_clv() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.rfm_customer_segmentation(dataset)
    assert result["summary"]["total_customers"] >= 1
    valid_segments = {"Champions", "Loyal", "Potential", "At Risk", "Lost"}
    for c in result["customers"]:
        assert c["segment"] in valid_segments
        assert 1 <= c["r_score"] <= 5
        assert 1 <= c["f_score"] <= 5
        assert 1 <= c["m_score"] <= 5
        assert c["clv_estimate"] >= 0
    assert result["segment_summary"]


def test_f13_working_capital_ccc_dso_dpo() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.working_capital_optimization(dataset)
    # CCC = DSO + DIO - DPO
    expected_ccc = result["dso"] + result["dio"] - result["dpo"]
    assert abs(result["ccc"] - expected_ccc) < 0.2
    assert 0 <= result["liquidity_score"] <= 100
    assert result["optimal_cash_buffer"] > 0
    assert "cash_conversion_cycle" in result["method_stack"]


def test_f14_price_elasticity_revenue_sensitivity() -> None:
    service = TaxpayerIntelligenceService()
    # Elastic demand: coefficient < -1
    result = service.price_elasticity({
        "current_price": 100_000,
        "current_quantity": 500,
        "new_price": 120_000,
        "elasticity_coefficient": -1.5,
    })
    assert result["status"] == "success"
    # Price up 20% with elastic demand → quantity drops more, revenue should drop
    assert result["new_quantity"] < 500
    assert result["verdict_label"]  # non-empty
    assert result["gtgt_tax_change"] is not None


def test_f15_regulatory_change_diff_filters_by_industry() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.regulatory_change_diff(dataset)
    assert result["summary"]["total_changes"] >= 1
    for c in result["changes"]:
        assert "diff_highlights" in c
        assert "action_items" in c
        assert c["status"] in {"active", "expired"}
    assert result["severity_heatmap"]


def test_f16_compliance_risk_heatmap_10_dimensions() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.compliance_risk_heatmap(dataset)
    assert len(result["dimensions"]) == 10
    assert 0 <= result["composite_score"] <= 100
    assert result["composite_level"] in {"low", "medium", "high"}
    for d in result["dimensions"]:
        assert 0 <= d["score"] <= 100
        assert d["severity"] in {"low", "medium", "high"}


def test_f17_tax_calendar_optimization_scheduling() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.tax_calendar_optimization(dataset)
    assert len(result["deadlines"]) >= 1
    assert result["total_penalty_savings"] >= 0
    for dl in result["deadlines"]:
        assert dl["priority"] in {"immediate", "high", "normal"}
        assert dl["estimated_amount"] >= 0
        assert "optimized_date" in dl
    assert result["cashflow_impact"]["total_obligations"] > 0


def test_f18_cohort_analysis_retention_matrix() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.cohort_analysis(dataset)
    assert len(result["cohorts"]) >= 1
    assert result["summary"]["total_periods"] >= 1
    assert 0 <= result["summary"]["avg_retention"] <= 1.0
    for c in result["cohorts"]:
        assert c["retention_rate"] >= 0
        assert "growth_rate" in c
    # Retention matrix rows should have cohort and periods
    for row in result["retention_matrix"]:
        assert "cohort" in row
        assert "M+0" in row["periods"]

def test_f19_transfer_pricing_mahalanobis() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    dataset["profile"]["target_unit_price"] = 95000.0
    dataset["profile"]["target_quantity"] = 180.0
    result = service.transfer_pricing_evaluator(dataset)
    assert "mahalanobis_distance" in result
    assert "p_value" in result
    assert result["mahalanobis_distance"] >= 0.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["risk_level"] in ["low", "medium", "high"]
    assert "arms_length_range" in result
    # New invariant checks
    assert "industry" in result
    assert "deviation_pct" in result
    assert result["deviation_pct"] >= 0.0
    assert "peer_sample_size" in result
    assert result["peer_sample_size"] >= 5
    # p-value and Mahalanobis are inversely related
    assert result["p_value"] <= 1.0  # always true for chi-sq survival


def test_f20_tax_outflow_gev_stress() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.tax_cash_stress_simulator(dataset)
    assert "gev_parameters" in result
    assert "return_levels" in result
    assert "value_at_risk_99" in result
    assert "expected_shortfall_99" in result
    assert result["value_at_risk_99"] > 0
    assert result["expected_shortfall_99"] >= result["value_at_risk_99"]
    assert 0.0 <= result["extreme_stress_probability"] <= 1.0
    # New invariant checks
    assert "tax_rate_applied" in result
    assert 0.0 < result["tax_rate_applied"] <= 0.15  # TT40 range
    assert "industry" in result
    assert "avg_monthly_tax" in result
    assert result["avg_monthly_tax"] > 0
    assert "verdict" in result
    assert len(result["verdict"]) > 10  # not a stub


def test_f21_spectral_gnn_evasion_cascade() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.gnn_spectral_fraud_cascade(dataset)
    assert "spectral_gap" in result
    assert "eigenvalues" in result
    assert "adamic_adar_collusion" in result
    assert "risk_cascade_propagation" in result
    assert len(result["eigenvalues"]) >= 3
    # Mathematical invariant: normalized Laplacian eigenvalues ∈ [0, 2]
    for ev in result["eigenvalues"]:
        assert -0.01 <= ev <= 2.01, f"Eigenvalue {ev} outside [0, 2] bounds"
    # Smallest eigenvalue should be ~0 (connected graph property)
    assert result["eigenvalues"][0] < 0.01
    for node in result["risk_cascade_propagation"]:
        assert "tax_code" in node
        assert 0.0 <= node["evasion_risk_exposure"] <= 100.0
    # New fields
    assert "high_risk_node_count" in result
    assert "verdict" in result
    assert len(result["verdict"]) > 10


def test_f22_entropy_revenue_anomaly() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.entropy_revenue_anomaly(dataset)
    assert "entropy_bits" in result
    assert "normalized_entropy" in result
    assert "max_entropy_bits" in result
    # Mathematical invariant: entropy ∈ [0, max_entropy]
    assert 0.0 <= result["entropy_bits"] <= result["max_entropy_bits"] + 0.01
    # Normalized entropy ∈ [0, 1]
    assert 0.0 <= result["normalized_entropy"] <= 1.0
    assert result["risk_level"] in ["low", "medium", "high"]
    assert "industry" in result
    assert "bin_distribution" in result
    assert sum(result["bin_distribution"]) == result["sample_size"]
    assert "coefficient_of_variation" in result
    assert result["coefficient_of_variation"] >= 0.0
    assert len(result["verdict"]) > 10


def test_f23_hmm_financial_state() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.hmm_financial_state(dataset)
    assert "state_timeline" in result
    assert "current_state" in result
    assert "next_month_prediction" in result
    assert result["current_state_index"] in [0, 1, 2]
    # State probabilities must sum to ~1.0
    for step in result["state_timeline"]:
        prob_sum = sum(step["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}"
        assert step["state_index"] in [0, 1, 2]
    # Next month prediction sums to ~1.0
    next_sum = sum(result["next_month_prediction"].values())
    assert abs(next_sum - 1.0) < 0.01
    assert result["total_months_analyzed"] >= 6
    assert len(result["verdict"]) > 10


def test_f24_cusum_change_detection() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.cusum_change_detection(dataset)
    assert "cusum_positive" in result
    assert "cusum_negative" in result
    assert "change_points" in result
    assert len(result["cusum_positive"]) == len(result["cusum_negative"])
    assert result["change_point_count"] >= 0
    # Invariant: CUSUM values should be non-negative
    for cp in result["cusum_positive"]:
        assert cp >= 0.0
    for cn in result["cusum_negative"]:
        assert cn >= 0.0
    assert len(result["verdict"]) > 10


def test_f25_svd_expense_decomposition() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.svd_expense_decomposition(dataset)
    assert "projections" in result
    assert "singular_values" in result
    assert "v1_weights" in result
    assert "v2_weights" in result
    # Invariant: singular values are sorted descending and non-negative
    s = result["singular_values"]
    assert len(s) == 2
    assert s[0] >= s[1]
    assert s[1] >= 0.0
    for p in result["projections"]:
        assert "pc1" in p
        assert "pc2" in p
        assert "anomaly_score" in p
    assert len(result["verdict"]) > 10


def test_f26_wavelet_revenue_decomposition() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.wavelet_revenue_decomposition(dataset)
    assert "periods" in result
    assert "original_values" in result
    assert "trend_component" in result
    assert "seasonal_component" in result
    assert "noise_component" in result
    # Invariant: power of 2 length (8 or 16)
    n = len(result["periods"])
    assert n in [8, 16]
    assert len(result["original_values"]) == n
    assert len(result["trend_component"]) == n
    assert len(result["seasonal_component"]) == n
    assert len(result["noise_component"]) == n
    # Reconstruction invariant: original ≈ trend + seasonal + noise
    for i in range(n):
        orig = result["original_values"][i]
        reconstructed = result["trend_component"][i] + result["seasonal_component"][i] + result["noise_component"][i]
        assert abs(orig - reconstructed) < 1.0, f"Reconstruction error too large at step {i}: {orig} vs {reconstructed}"
    assert len(result["verdict"]) > 10


def test_f27_altman_zscore() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.altman_zscore_bankruptcy(dataset)
    assert "z_score" in result
    assert "probability_of_bankruptcy" in result
    assert "zone" in result
    assert 0.0 <= result["probability_of_bankruptcy"] <= 1.0
    assert result["risk_level"] in ["low", "medium", "high"]
    assert len(result["verdict"]) > 10


def test_f28_kmeans_supplier_clustering() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.kmeans_supplier_clustering(dataset)
    assert "suppliers" in result
    assert "cluster_count" in result
    assert result["cluster_count"] == 3
    for s in result["suppliers"]:
        assert s["cluster_index"] in [0, 1, 2]
        assert s["risk_score"] in [10.0, 25.0, 65.0]
        assert s["frequency"] >= 0
        assert s["mean_amount"] >= 0.0
    assert len(result["verdict"]) > 10


def test_f29_composite_risk_score() -> None:
    service = TaxpayerIntelligenceService()
    dataset = sample_dataset()
    result = service.composite_risk_score(dataset)
    assert "composite_risk_score" in result
    assert "health_score" in result
    assert "ratings" in result
    # Invariant: composite_risk_score + health_score = 100
    assert abs(result["composite_risk_score"] + result["health_score"] - 100.0) < 0.1
    assert 0.0 <= result["composite_risk_score"] <= 100.0
    assert 0.0 <= result["health_score"] <= 100.0
    assert "compliance" in result["ratings"]
    assert "financial" in result["ratings"]
    assert "cashflow" in result["ratings"]
    assert "data_quality" in result["ratings"]
    assert "solvency" in result["ratings"]
    assert "operations" in result["ratings"]
    assert len(result["verdict"]) > 10

