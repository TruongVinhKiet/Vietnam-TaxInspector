from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, Numeric, ForeignKey, Text, JSON
from sqlalchemy.sql import func

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    badge_id = Column(String(50), unique=True, index=True)
    full_name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True)
    phone = Column(String(20))
    password_hash = Column(String(255))
    role = Column(String(20), default="viewer")
    face_data = Column(Text, nullable=True)           # JSON string of 128-d face descriptor vector
    face_verified = Column(Boolean, default=False)
    cccd_data = Column(Text, nullable=True)            # AES-256 encrypted CCCD number
    cccd_verified = Column(Boolean, default=False)
    signature_data = Column(Text, nullable=True)        # AES-256 encrypted Base64 signature image
    avatar_data = Column(Text, nullable=True)           # Base64 normalized avatar image (data URL)
    signature_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(128), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=False), nullable=False, index=True)
    used = Column(Boolean, default=False, nullable=False)
    used_at = Column(DateTime(timezone=False), nullable=True)
    created_at = Column(DateTime(timezone=False), server_default=func.now())


class Company(Base):
    __tablename__ = "companies"

    tax_code = Column(String(20), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    industry = Column(String(100))
    province = Column(String(100))                      # Geographic slice for fairness metrics
    registration_date = Column(Date)
    risk_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    country_inferred = Column(String(100), nullable=True)
    confidence_country = Column(Float, default=0.0)
    is_within_vietnam = Column(Boolean, nullable=True)
    geocoding_method = Column(String(50), nullable=True)
    geocoded_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TaxReturn(Base):
    __tablename__ = "tax_returns"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), index=True)
    quarter = Column(String(10), nullable=False)
    revenue = Column(Numeric(15, 2), default=0.0)
    expenses = Column(Numeric(15, 2), default=0.0)
    tax_paid = Column(Numeric(15, 2), default=0.0)
    status = Column(String(50), default="submitted")
    filing_date = Column(Date, nullable=False)
    due_date = Column(Date, nullable=True)              # Original due date for temporal features
    tax_type = Column(String(50), default="VAT")        # VAT, CIT, PIT
    amendment_number = Column(Integer, default=0)       # 0 = original, 1+ = amendment
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    seller_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), index=True)
    buyer_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), index=True)
    amount = Column(Numeric(15, 2), nullable=False)
    vat_rate = Column(Numeric(5, 2), default=10.0)
    date = Column(Date, nullable=False)
    invoice_number = Column(String(50), unique=True)
    payment_status = Column(String(30), default="unknown")   # For graph enrichment
    goods_category = Column(String(100))                      # Product category for motif analysis
    is_adjustment = Column(Boolean, default=False)            # Credit note / adjustment
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """
    Nhật ký kiểm toán bất biến (Immutable Audit Log).
    Chỉ INSERT – nghiêm cấm UPDATE hoặc DELETE.
    Ghi lại mọi hành vi liên quan đến xác thực và bảo mật.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)       # NULL cho anonymous events (failed login)
    badge_id = Column(String(50), nullable=True, index=True)   # Badge ID of the actor
    action = Column(String(100), nullable=False, index=True)   # e.g. LOGIN_SUCCESS, LOGIN_FAILED, FACE_SETUP, etc.
    detail = Column(Text, nullable=True)                       # Additional context/reason
    ip_address = Column(String(45), nullable=True)             # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)            # Browser user agent string
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AIAnalysisBatch(Base):
    """
    Quản lý tiến độ xử lý file CSV (Batch AI Analysis).
    status: pending -> processing -> done / failed
    """
    __tablename__ = "ai_analysis_batches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=True)
    total_rows = Column(Integer, default=0)
    processed_rows = Column(Integer, default=0)
    status = Column(String(20), nullable=False, default="pending", index=True)
    error_message = Column(Text, nullable=True)
    result_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class AIRiskAssessment(Base):
    """
    Kết quả chấm điểm rủi ro cho từng doanh nghiệp.
    Lưu trữ Features (F1-F4), anomaly_score, risk_score và giải thích SHAP dưới dạng JSON.
    """
    __tablename__ = "ai_risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("ai_analysis_batches.id", ondelete="CASCADE"), nullable=True, index=True)
    tax_code = Column(String(20), nullable=False, index=True)
    company_name = Column(String(255), nullable=True)
    industry = Column(String(100), nullable=True)
    year = Column(Integer, nullable=True)
    revenue = Column(Numeric(18, 2), nullable=True)
    total_expenses = Column(Numeric(18, 2), nullable=True)
    f1_divergence = Column(Float, nullable=True)
    f2_ratio_limit = Column(Float, nullable=True)
    f3_vat_structure = Column(Float, nullable=True)
    f4_peer_comparison = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=False, default=0.0, index=True)
    risk_level = Column(String(20), default="low")
    model_version = Column(String(80), nullable=True, index=True)
    model_confidence = Column(Float, nullable=True)  # Real confidence from pipeline: max(prob, 1-prob)*100
    red_flags = Column(JSON, nullable=True)
    shap_explanation = Column(JSON, nullable=True)
    yearly_history = Column(JSON, nullable=True)  # [{year, revenue, total_expenses}, ...]
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ════════════════════════════════════════════════════════════════
#  NEW: Flagship Model Tables (Phase 0 data expansion)
# ════════════════════════════════════════════════════════════════

class TaxPayment(Base):
    """
    Actual tax payment records – tracks real payment dates vs due dates.
    Core data source for the Temporal Compliance Intelligence model (Program A).
    """
    __tablename__ = "tax_payments"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    tax_period = Column(String(20), nullable=False)      # e.g. '2025-Q1', '2025-M03'
    tax_type = Column(String(50), nullable=False, default="VAT")
    amount_due = Column(Numeric(18, 2), nullable=False, default=0.0)
    amount_paid = Column(Numeric(18, 2), nullable=False, default=0.0)
    due_date = Column(Date, nullable=False)
    actual_payment_date = Column(Date, nullable=True)     # NULL if not yet paid
    # days_overdue is computed column in PG – not mapped in ORM (read-only via raw SQL)
    penalty_amount = Column(Numeric(18, 2), default=0.0)
    payment_method = Column(String(50), nullable=True)
    status = Column(String(30), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InvoiceLineItem(Base):
    """
    Dense invoice detail for graph enrichment.
    Enables motif detection and link prediction at granular item level (Program B).
    """
    __tablename__ = "invoice_line_items"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True)
    item_description = Column(String(500), nullable=True)
    item_code = Column(String(100), nullable=True)        # HS code or internal product code
    quantity = Column(Numeric(12, 3), default=1.0)
    unit_price = Column(Numeric(18, 2), nullable=False, default=0.0)
    line_amount = Column(Numeric(18, 2), nullable=False, default=0.0)
    vat_amount = Column(Numeric(18, 2), default=0.0)
    unit = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InspectorLabel(Base):
    """
    Ground-truth labels from tax inspectors.
    Used for supervised model retraining, false positive reduction, and audit trail.
    """
    __tablename__ = "inspector_labels"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), nullable=False, index=True)
    inspector_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    label_type = Column(String(50), nullable=False)       # 'fraud_confirmed', 'fraud_rejected', etc.
    confidence = Column(String(20), default="medium")     # low, medium, high
    label_origin = Column(String(40), nullable=False, default="manual_inspector")
    assessment_id = Column(Integer, ForeignKey("ai_risk_assessments.id", ondelete="SET NULL"), nullable=True)
    model_version = Column(String(80), nullable=True, index=True)
    evidence_summary = Column(Text, nullable=True)
    decision = Column(String(50), nullable=True)          # 'investigate', 'dismiss', 'escalate', 'penalize'
    decision_date = Column(Date, nullable=True)
    tax_period = Column(String(20), nullable=True)
    amount_recovered = Column(Numeric(18, 2), nullable=True)
    intervention_action = Column(String(50), nullable=True)  # monitor/auto_reminder/structured_outreach/field_audit/escalated_enforcement
    intervention_attempted = Column(Boolean, nullable=False, default=False)
    outcome_status = Column(String(30), nullable=True)       # pending/in_progress/recovered/partial_recovered/unrecoverable/dismissed
    predicted_collection_uplift = Column(Numeric(18, 2), nullable=True)
    expected_recovery = Column(Numeric(18, 2), nullable=True)
    expected_net_recovery = Column(Numeric(18, 2), nullable=True)
    estimated_audit_cost = Column(Numeric(18, 2), nullable=True)
    actual_audit_cost = Column(Numeric(18, 2), nullable=True)
    actual_audit_hours = Column(Float, nullable=True)
    outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)
    kpi_window_days = Column(Integer, nullable=False, default=90)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KPITriggerPolicy(Base):
    """
    Governance rules for split-trigger readiness decisions.
    Each row defines one metric gate for one model track.
    """
    __tablename__ = "kpi_trigger_policies"

    id = Column(Integer, primary_key=True, index=True)
    track_name = Column(String(50), nullable=False, index=True)      # audit_value | vat_refund | intervention
    metric_name = Column(String(80), nullable=False, index=True)      # precision_top_50 | roi_positive_rate | fn_high_risk
    comparator = Column(String(8), nullable=False, default=">=")     # >= | > | <= | < | ==
    threshold = Column(Float, nullable=False)
    min_sample = Column(Integer, nullable=False, default=50)
    window_days = Column(Integer, nullable=False, default=28)
    cooldown_days = Column(Integer, nullable=False, default=14)
    enabled = Column(Boolean, nullable=False, default=True)
    rationale = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ════════════════════════════════════════════════════════════════
#  Feature Store (point-in-time, leakage-safe)
# ════════════════════════════════════════════════════════════════

class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(80), nullable=False, index=True)
    version = Column(String(40), nullable=False, index=True)
    owner = Column(String(80), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String(20), nullable=False, index=True)  # company|invoice|edge|case
    entity_id = Column(String(120), nullable=False, index=True)   # tax_code / invoice_number / composite key
    as_of_date = Column(Date, nullable=False, index=True)
    feature_set_id = Column(Integer, ForeignKey("feature_sets.id", ondelete="CASCADE"), nullable=False, index=True)
    features_json = Column(JSON, nullable=False, default=dict)
    source_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ════════════════════════════════════════════════════════════════
#  Model Registry + Inference Audit Trail
# ════════════════════════════════════════════════════════════════

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(80), nullable=False, index=True)
    model_version = Column(String(80), nullable=False, index=True)
    artifact_path = Column(String(400), nullable=True)
    feature_set_id = Column(Integer, ForeignKey("feature_sets.id", ondelete="SET NULL"), nullable=True, index=True)
    train_data_hash = Column(String(64), nullable=True)
    code_hash = Column(String(64), nullable=True)
    metrics_json = Column(JSON, nullable=True)
    gates_json = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="staging")  # staging|prod|archived
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InferenceAuditLog(Base):
    __tablename__ = "inference_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(80), nullable=False, index=True)
    model_version = Column(String(80), nullable=False, index=True)
    request_id = Column(String(64), nullable=True, index=True)
    actor_badge_id = Column(String(50), nullable=True)
    actor_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    entity_type = Column(String(20), nullable=False, index=True)
    entity_id = Column(String(120), nullable=False, index=True)
    as_of_date = Column(Date, nullable=True, index=True)
    input_feature_hash = Column(String(64), nullable=True)
    output_hash = Column(String(64), nullable=True)
    outputs_json = Column(JSON, nullable=True)
    explanation_ref = Column(String(200), nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelQualitySnapshot(Base):
    __tablename__ = "model_quality_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(80), nullable=False, index=True)
    model_version = Column(String(80), nullable=True, index=True)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    quality_json = Column(JSON, nullable=False, default=dict)
    status = Column(String(20), nullable=True, default="unknown")
    status_reason = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FeatureDriftStat(Base):
    __tablename__ = "feature_drift_stats"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(80), nullable=False, index=True)
    model_version = Column(String(80), nullable=True, index=True)
    feature_name = Column(String(120), nullable=False, index=True)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    psi = Column(Float, nullable=True)
    ks = Column(Float, nullable=True)
    missing_rate = Column(Float, nullable=True)
    mean = Column(Float, nullable=True)
    std = Column(Float, nullable=True)
    baseline_mean = Column(Float, nullable=True)
    baseline_std = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InvoiceEvent(Base):
    __tablename__ = "invoice_events"

    id = Column(Integer, primary_key=True, index=True)
    invoice_number = Column(String(50), ForeignKey("invoices.invoice_number", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(30), nullable=False, index=True)
    event_time = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    reason = Column(String(200), nullable=True)
    replaced_invoice_number = Column(String(50), nullable=True)
    payload_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InvoiceFingerprint(Base):
    __tablename__ = "invoice_fingerprints"

    id = Column(Integer, primary_key=True, index=True)
    invoice_number = Column(String(50), ForeignKey("invoices.invoice_number", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    hash_near_dup = Column(String(64), nullable=True, index=True)
    hash_line_items = Column(String(64), nullable=True)
    hash_counterparty = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InvoiceRiskPrediction(Base):
    __tablename__ = "invoice_risk_predictions"

    id = Column(Integer, primary_key=True, index=True)
    invoice_number = Column(String(50), ForeignKey("invoices.invoice_number", ondelete="CASCADE"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    risk_score = Column(Float, nullable=False, default=0.0, index=True)
    risk_level = Column(String(20), nullable=False, default="low")
    reason_codes = Column(JSON, nullable=True)
    explanations = Column(JSON, nullable=True)
    linked_invoice_ids = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VatRefundCase(Base):
    __tablename__ = "vat_refund_cases"

    case_id = Column(String(40), primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    period = Column(String(20), nullable=False, index=True)
    requested_amount = Column(Numeric(18, 2), nullable=False, default=0.0)
    submitted_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    status = Column(String(30), nullable=False, default="submitted")
    channel = Column(String(30), nullable=True)
    documents_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VatRefundCaseLink(Base):
    __tablename__ = "vat_refund_case_links"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(40), ForeignKey("vat_refund_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    invoice_number = Column(String(50), ForeignKey("invoices.invoice_number", ondelete="CASCADE"), nullable=False, index=True)
    link_type = Column(String(20), nullable=False, default="supporting")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VatRefundPrediction(Base):
    __tablename__ = "vat_refund_predictions"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(40), ForeignKey("vat_refund_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    risk_score = Column(Float, nullable=False, default=0.0)
    expected_loss = Column(Numeric(18, 2), nullable=False, default=0.0)
    reason_codes = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EntityIdentity(Base):
    __tablename__ = "entity_identities"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    legal_name = Column(String(255), nullable=True)
    normalized_name = Column(String(255), nullable=True)
    address = Column(Text, nullable=True)
    phone = Column(String(30), nullable=True)
    email = Column(String(120), nullable=True)
    representative_name = Column(String(255), nullable=True)
    representative_id = Column(String(50), nullable=True, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EntityAliasEdge(Base):
    __tablename__ = "entity_alias_edges"

    id = Column(Integer, primary_key=True, index=True)
    src_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    dst_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    edge_type = Column(String(30), nullable=False, index=True)
    score = Column(Float, nullable=False, default=0.0, index=True)
    evidence_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PhoenixCandidate(Base):
    __tablename__ = "phoenix_candidates"

    id = Column(Integer, primary_key=True, index=True)
    old_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    new_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    score = Column(Float, nullable=False, default=0.0, index=True)
    signals_json = Column(JSON, nullable=True)
    as_of_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TradeRecord(Base):
    __tablename__ = "trade_records"

    record_id = Column(String(40), primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    counterparty_country = Column(String(100), nullable=True)
    goods_category = Column(String(100), nullable=True)
    hs_code = Column(String(20), nullable=True)
    quantity = Column(Numeric(18, 3), nullable=True)
    unit_price = Column(Numeric(18, 2), nullable=True)
    trade_date = Column(Date, nullable=False, index=True)
    channel = Column(String(30), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PricingReferenceCurve(Base):
    __tablename__ = "pricing_reference_curves"

    curve_id = Column(String(60), primary_key=True, index=True)
    goods_key = Column(String(120), nullable=False, index=True)
    country_pair = Column(String(120), nullable=True)
    time_bucket = Column(String(20), nullable=False, index=True)
    p10 = Column(Numeric(18, 2), nullable=True)
    p50 = Column(Numeric(18, 2), nullable=True)
    p90 = Column(Numeric(18, 2), nullable=True)
    n_samples = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MispricingPrediction(Base):
    __tablename__ = "mispricing_predictions"

    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(String(40), ForeignKey("trade_records.record_id", ondelete="CASCADE"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    z_score = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=False, default=0.0)
    reason_codes = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditCase(Base):
    __tablename__ = "audit_cases"

    case_id = Column(String(40), primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    case_type = Column(String(40), nullable=True)
    status = Column(String(30), nullable=True)
    auditor_team = Column(String(80), nullable=True)
    effort_hours = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditOutcome(Base):
    __tablename__ = "audit_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(40), ForeignKey("audit_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    recovered_amount = Column(Numeric(18, 2), nullable=True)
    penalty_amount = Column(Numeric(18, 2), nullable=True)
    dispute_flag = Column(Boolean, default=False)
    final_amount = Column(Numeric(18, 2), nullable=True)
    closing_reason = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditSelectionPrediction(Base):
    __tablename__ = "audit_selection_predictions"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    prob_recovery = Column(Float, nullable=True)
    expected_recovery = Column(Numeric(18, 2), nullable=True)
    expected_effort = Column(Float, nullable=True)
    priority_score = Column(Float, nullable=True)
    reason_codes = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CollectionAction(Base):
    __tablename__ = "collection_actions"

    action_id = Column(String(40), primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    action_type = Column(String(40), nullable=False)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(String(40), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CollectionOutcome(Base):
    __tablename__ = "collection_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    tax_period = Column(String(20), nullable=True)
    amount_collected = Column(Numeric(18, 2), nullable=True)
    collected_at = Column(DateTime(timezone=True), nullable=True)
    action_id = Column(String(40), ForeignKey("collection_actions.action_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NBAPrediction(Base):
    __tablename__ = "nba_predictions"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    recommended_action = Column(String(40), nullable=True)
    uplift_pp = Column(Float, nullable=True)
    expected_collection = Column(Numeric(18, 2), nullable=True)
    confidence = Column(String(20), nullable=True)
    reason_codes = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CaseQueue(Base):
    __tablename__ = "case_queue"

    case_id = Column(String(40), primary_key=True, index=True)
    case_type = Column(String(30), nullable=False, index=True)
    entity_id = Column(String(120), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sla_due_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(30), nullable=True)
    priority = Column(String(20), nullable=True)


class CaseEvent(Base):
    __tablename__ = "case_events"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(40), ForeignKey("case_queue.case_id", ondelete="CASCADE"), nullable=False, index=True)
    event_time = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    event_type = Column(String(60), nullable=True)
    actor = Column(String(80), nullable=True)
    payload_json = Column(JSON, nullable=True)


class CaseTriagePrediction(Base):
    __tablename__ = "case_triage_predictions"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(40), ForeignKey("case_queue.case_id", ondelete="CASCADE"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    priority_score = Column(Float, nullable=True)
    urgency_level = Column(String(20), nullable=True)
    next_steps = Column(JSON, nullable=True)
    routing_team = Column(String(80), nullable=True)
    reason_codes = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class KPIMetricSnapshot(Base):
    """
    Point-in-time KPI values used to track split-trigger readiness over time.
    """
    __tablename__ = "kpi_metric_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    track_name = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(80), nullable=False, index=True)
    metric_value = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=False, default=0)
    comparator = Column(String(8), nullable=True)
    threshold = Column(Float, nullable=True)
    status = Column(String(30), nullable=False, default="no_metric")
    window_days = Column(Integer, nullable=False, default=28)
    source = Column(String(80), nullable=False, default="split_trigger_status")
    details = Column(JSON, nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class OwnershipLink(Base):
    """
    Company ownership relationships.
    Enables shell company / "sân sau" detection via ownership graph (Program B).
    """
    __tablename__ = "ownership_links"

    id = Column(Integer, primary_key=True, index=True)
    parent_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    child_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    ownership_percent = Column(Numeric(5, 2), nullable=False, default=0.0)
    relationship_type = Column(String(50), nullable=False, default="shareholder")
    person_name = Column(String(255), nullable=True)
    person_id = Column(String(50), nullable=True, index=True)
    effective_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)                # NULL if still active
    data_source = Column(String(100), nullable=True)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class OffshoreEntity(Base):
    """
    Offshore entity master data.
    - entity_code: external/alphanumeric offshore identity
    - proxy_tax_code: internal numeric proxy used for relational joins
    """
    __tablename__ = "offshore_entities"

    id = Column(Integer, primary_key=True, index=True)
    entity_code = Column(String(30), unique=True, nullable=False, index=True)
    proxy_tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="SET NULL"), unique=True, nullable=True, index=True)
    name = Column(String(255), nullable=False)
    country = Column(String(100), nullable=False)
    jurisdiction_risk_weight = Column(Float, default=0.5)
    risk_score = Column(Float, default=50.0)
    entity_type = Column(String(50), default="shell_company")
    registration_date = Column(Date, nullable=True)
    status = Column(String(30), default="active")
    data_source = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DelinquencyPrediction(Base):
    """
    Cached predictions from the temporal delinquency model (Program A).
    One row per company per prediction run.
    """
    __tablename__ = "delinquency_predictions"

    id = Column(Integer, primary_key=True, index=True)
    tax_code = Column(String(20), ForeignKey("companies.tax_code", ondelete="CASCADE"), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False, server_default=func.current_date())
    prob_30d = Column(Float, nullable=False, default=0.0)
    prob_60d = Column(Float, nullable=False, default=0.0)
    prob_90d = Column(Float, nullable=False, default=0.0)
    risk_cluster = Column(String(100), nullable=True)
    top_reasons = Column(JSON, nullable=True)             # [{reason, weight}, ...]
    model_version = Column(String(80), nullable=True)
    model_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


