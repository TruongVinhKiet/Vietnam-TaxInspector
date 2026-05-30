from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "Frontend"


PAGES_TO_SCRIPTS = {
    "business_dashboard.html": "business_dashboard.js",
    "business_registration.html": "business_registration.js",
    "business_calculator.html": "business_calculator.js",
    "business_calendar.html": "business_calendar.js",
    "business_invoices.html": "business_invoices.js",
    "business_filing.html": "business_filing.js",
    "business_debts.html": "business_debts.js",
    "business_legal.html": "business_legal.js",
    "business_growth.html": "business_growth.js",
    "business_accounting.html": "business_accounting.js",
    "business_expenses.html": "business_expenses.js",
    "business_claims.html": "business_claims.js",
    "business_profile.html": "business_profile.js",
}


def test_business_pages_load_common_and_page_specific_taxpayer_js() -> None:
    for page, script in PAGES_TO_SCRIPTS.items():
        html = (FRONTEND / "pages" / page).read_text(encoding="utf-8")
        assert "business_taxpayer_common.js" in html
        assert "business_intelligence.js" in html
        assert script in html


def test_stub_functions_are_bound_by_page_scripts() -> None:
    contracts = {
        "business_calendar.js": ["syncCalendar", "saveNotificationSettings"],
        "business_invoices.js": ["issueInvoice", "scanInvoice"],
        "business_filing.js": ["exportXml", "signFiling", "generateQr", "confirmPayment"],
        "business_debts.js": ["downloadReceipt", "checkPassportBan"],
        "business_legal.js": ["sendChatMessage", "handleKeyPress"],
        "business_growth.js": ["requestUpgrade", "requestClosure"],
        "business_accounting.js": ["exportBook"],
        "business_expenses.js": ["addExpense"],
        "business_claims.js": ["recalculateRisk", "submitAppeal"],
    }
    for script, symbols in contracts.items():
        source = (FRONTEND / "js" / script).read_text(encoding="utf-8")
        for symbol in symbols:
            assert f"window.{symbol}" in source


def test_business_intelligence_frontend_contract() -> None:
    source = (FRONTEND / "js" / "business_intelligence.js").read_text(encoding="utf-8")
    for endpoint in [
        "/intelligence/overview",
        "/intelligence/forecast",
        "/intelligence/peer-benchmark",
        "/intelligence/charts",
        "/intelligence/anomalies",
        "/intelligence/model-catalog",
        "/intelligence/optimize-tax",
        "/intelligence/claim-assist",
        "/intelligence/scenario-dashboard",
        "/intelligence/ocr-reconcile",
        "/intelligence/cashflow-risk",
        "/intelligence/supplier-risk-graph",
        "/intelligence/auto-bookkeeping",
        "/intelligence/tax-return-precheck",
        "/intelligence/policy-impact",
        "/intelligence/business-upgrade-readiness",
        "/intelligence/copilot",
        "/intelligence/advanced-dashboard",
        "/intelligence/document-ai/extract",
        "/intelligence/document-ai/reconcile",
        "/intelligence/forecast/probabilistic",
        "/intelligence/digital-twin/simulate",
        "/intelligence/graph/risk",
        "/intelligence/ledger/autopost",
        "/intelligence/filing/precheck-advanced",
        "/intelligence/cashflow/delinquency",
        "/intelligence/legal/graphrag",
        "/intelligence/next-best-action",
        "/intelligence/model-governance",
        "/connectors/bank/import",
        "/connectors/einvoice/import",
        "/connectors/ecommerce/import",
        "/intelligence/reconcile/4way",
        "/intelligence/reconciliation-cases",
        "/intelligence/channel-attribution",
        "/intelligence/tax-reserve/optimize",
        "/intelligence/supplier-account-risk",
        "/intelligence/inventory/analyze",
        "/intelligence/evidence-bundle",
        "/intelligence/legal/change-impact",
        "/intelligence/model-governance/production",
        "/intelligence/legal-chat",
        "/intelligence/feedback",
    ]:
        assert endpoint in source
    assert "taxpayer-ai-overview-panel" in source
    assert "window.askTaxpayerLegalAI" in source
    assert "window.runTaxpayerOptimization" in source
    assert "window.assistTaxpayerClaim" in source
    assert "window.precheckTaxReturnAI" in source
    assert "window.autoBookkeepingAI" in source
    assert "window.reconcileTaxpayerDocumentAI" in source
    assert "window.advancedDocumentExtractAI" in source
    assert "window.digitalTwinAI" in source
    assert "window.advancedPrecheckTaxReturnAI" in source
    assert "window.ledgerAutopostAI" in source
    assert "window.legalGraphRAGAI" in source
    assert "window.importBankConnectorAI" in source
    assert "window.reconcile4WayAI" in source
    assert "window.taxReserveOptimizerAI" in source
    assert "window.evidenceBundleAI" in source
