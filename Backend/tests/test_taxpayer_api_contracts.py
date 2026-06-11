import inspect
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routers import taxpayer


def test_taxpayer_router_exposes_required_group_endpoints() -> None:
    paths = {route.path.removeprefix("/api/taxpayer") for route in taxpayer.router.routes}
    required = {
        "/calendar/deadlines",
        "/calendar/settings",
        "/calendar/export.ics",
        "/invoices/issue",
        "/invoices/scan",
        "/filings/draft",
        "/filings/{filing_id}/submit",
        "/debts/summary",
        "/debts/impersonation-check",
        "/legal/chat",
        "/legal/rates",
        "/growth/event",
        "/accounting/revenue",
        "/accounting/expense",
        "/expenses/check",
        "/claims/appeal",
        "/claims/appointment",
        "/connectors/bank/import",
        "/connectors/einvoice/import",
        "/connectors/ecommerce/import",
        "/intelligence/capabilities",
        "/intelligence/overview",
        "/intelligence/forecast",
        "/intelligence/what-if",
        "/intelligence/expense-classify",
        "/intelligence/document-ocr",
        "/intelligence/invoice-risk",
        "/intelligence/recommendations",
        "/intelligence/peer-benchmark",
        "/intelligence/charts",
        "/intelligence/anomalies",
        "/intelligence/optimize-tax",
        "/intelligence/claim-assist",
        "/intelligence/model-catalog",
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
        "/intelligence/benford-analysis",
        "/intelligence/seasonal-decomposition",
        "/intelligence/monte-carlo-simulation",
        "/intelligence/survival-analysis",
        "/intelligence/breakeven-analysis",
        "/intelligence/bayesian-forecast",
        "/intelligence/isolation-forest-expenses",
        "/intelligence/markov-chain-prediction",
        "/intelligence/explainability",
        "/intelligence/pagerank-supplier-trust",
        "/intelligence/autoencoder-bank-anomaly",
        "/intelligence/rfm-customer-segmentation",
        "/intelligence/working-capital",
        "/intelligence/price-elasticity",
        "/intelligence/regulatory-change-diff",
        "/intelligence/compliance-risk-heatmap",
        "/intelligence/tax-calendar-optimization",
        "/intelligence/cohort-analysis",
        "/intelligence/transfer-pricing",
        "/intelligence/outflow-stress",
        "/intelligence/spectral-cascade",
    }
    assert required.issubset(paths)


def test_taxpayer_routes_keep_taxpayer_auth_dependency_in_signature() -> None:
    for route in taxpayer.router.routes:
        signature = inspect.signature(route.endpoint)
        assert "current_user" in signature.parameters, f"{route.path} must require taxpayer identity"
