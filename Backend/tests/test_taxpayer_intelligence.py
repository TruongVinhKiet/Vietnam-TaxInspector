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

    reconcile = service.reconcile_4way({}, dataset)
    assert reconcile["summary"]["open_case_count"] >= 1
    assert reconcile["model_name"]
    assert "needs_human_confirmation" in reconcile

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
