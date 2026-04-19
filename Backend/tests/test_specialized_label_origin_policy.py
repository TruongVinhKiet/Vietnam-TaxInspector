import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine import train_audit_value, train_vat_refund


def test_audit_label_origin_policy_normalization_defaults_to_exclude_synthetic():
    assert train_audit_value._normalize_label_origin_policy(None) == "exclude_synthetic"
    assert train_audit_value._normalize_label_origin_policy("unknown") == "exclude_synthetic"


def test_vat_label_origin_policy_normalization_defaults_to_exclude_synthetic():
    assert train_vat_refund._normalize_label_origin_policy(None) == "exclude_synthetic"
    assert train_vat_refund._normalize_label_origin_policy("invalid") == "exclude_synthetic"


def test_audit_label_origin_filter_sql_has_expected_semantics():
    all_sql = train_audit_value._build_label_origin_filter_sql("all")
    real_sql = train_audit_value._build_label_origin_filter_sql("real_only")
    exclude_sql = train_audit_value._build_label_origin_filter_sql("exclude_synthetic")

    assert all_sql == "TRUE"
    assert "IN" in real_sql and "manual_inspector" in real_sql and "imported_casework" in real_sql
    assert "NOT IN" in exclude_sql and "bootstrap_generated" in exclude_sql and "auto_seed" in exclude_sql


def test_vat_label_origin_filter_sql_has_expected_semantics():
    all_sql = train_vat_refund._build_label_origin_filter_sql("all")
    real_sql = train_vat_refund._build_label_origin_filter_sql("real_only")
    exclude_sql = train_vat_refund._build_label_origin_filter_sql("exclude_synthetic")

    assert all_sql == "TRUE"
    assert "IN" in real_sql and "field_verified" in real_sql
    assert "NOT IN" in exclude_sql and "bootstrap_generated" in exclude_sql


def test_origin_filter_sql_accepts_custom_origin_expression():
    audit_sql = train_audit_value._build_label_origin_filter_sql("real_only", "'manual_inspector'")
    vat_sql = train_vat_refund._build_label_origin_filter_sql("exclude_synthetic", "'manual_inspector'")

    assert "l.label_origin" not in audit_sql
    assert "l.label_origin" not in vat_sql
    assert "'manual_inspector'" in audit_sql
    assert "'manual_inspector'" in vat_sql
