from datetime import date, timedelta
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.ensemble import HeuristicRuleScorer


def test_age_under_180_days_gets_higher_penalty():
    scorer = HeuristicRuleScorer()
    company = {
        "tax_code": "TEST-001",
        "registration_date": date.today() - timedelta(days=100),
    }

    score = scorer.score_node(company, cycle_nodes=set(), in_amount=0.0, out_amount=0.0, degree=0)
    assert score == 0.35


def test_age_between_180_and_365_days_gets_mid_penalty():
    scorer = HeuristicRuleScorer()
    company = {
        "tax_code": "TEST-002",
        "registration_date": date.today() - timedelta(days=240),
    }

    score = scorer.score_node(company, cycle_nodes=set(), in_amount=0.0, out_amount=0.0, degree=0)
    assert score == 0.25


def test_age_above_365_days_gets_no_age_penalty():
    scorer = HeuristicRuleScorer()
    company = {
        "tax_code": "TEST-003",
        "registration_date": date.today() - timedelta(days=500),
    }

    score = scorer.score_node(company, cycle_nodes=set(), in_amount=0.0, out_amount=0.0, degree=0)
    assert score == 0.0
