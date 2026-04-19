from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import date, datetime


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(protected_namespaces=())


# --- Auth/Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    badge_id: Optional[str] = None


class LoginRequest(BaseModel):
    badge_id: str
    password: str = Field(min_length=1, max_length=128)


class GenericMessage(BaseModel):
    success: bool
    message: str


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=8, max_length=100)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=20, max_length=256)
    new_password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)

# --- User Schemas ---
class UserBase(BaseModel):
    badge_id: str
    full_name: str
    department: str
    email: str
    phone: Optional[str] = None
    role: Literal["viewer", "analyst", "inspector", "admin"] = "viewer"

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    avatar_data: Optional[str] = None
    face_verified: bool = False
    cccd_verified: bool = False
    signature_verified: bool = False
    created_at: datetime


# --- Biometric Auth Schemas ---
class BiometricSetupRequest(BaseModel):
    descriptor: List[float] = Field(..., description="128-d face descriptor vector or CCCD data as list")

class CccdSetupRequest(BaseModel):
    cccd_number: str = Field(..., min_length=9, max_length=12, description="Số CCCD 12 chữ số")

class BiometricLoginRequest(BaseModel):
    descriptor: List[float] = Field(..., description="128-d face descriptor vector for matching")

class CccdLoginRequest(BaseModel):
    cccd_number: str = Field(..., min_length=9, max_length=12)


# --- Signature Auth Schemas ---
class SignatureSetupRequest(BaseModel):
    signature_image: str = Field(..., description="Base64 PNG image of the signature drawn on canvas")

class SignatureLoginRequest(BaseModel):
    temp_token: str = Field(..., description="Temporary token from Step 1 (face/cccd)")
    signature_image: str = Field(..., description="Base64 PNG image of the signature for verification")


# --- Phone Update Schema ---
class UpdatePhoneRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15, description="Số điện thoại mới")


class UpdateAvatarRequest(BaseModel):
    avatar_image: str = Field(
        ...,
        min_length=64,
        max_length=8_000_000,
        description="Data URL hoặc raw Base64 PNG/JPEG (toi da 5MB).",
    )



# --- Company Schemas ---
class CompanyBase(BaseModel):
    tax_code: str
    name: str
    industry: Optional[str] = None
    registration_date: Optional[date] = None


class CompanyCreate(CompanyBase):
    pass


class CompanyResponse(CompanyBase):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    risk_score: float
    is_active: bool


# --- ML Prediction Schemas ---
class FraudRiskPrediction(BaseModel):
    tax_code: str
    risk_score: float
    red_flags: List[str]


class VATGraphNode(BaseModel):
    id: str
    label: str
    group: Optional[str] = "default"


class VATGraphEdge(BaseModel):
    from_node: str
    to_node: str
    value: float


class GraphResponse(BaseModel):
    nodes: List[VATGraphNode]
    edges: List[VATGraphEdge]
    suspicious_clusters: List[List[str]]


class DelinquencyPrediction(BaseModel):
    tax_code: str
    company_name: str
    probability: float
    cluster: str


# --- AI Batch Analysis Schemas ---
class BatchUploadResponse(BaseModel):
    batch_id: int
    filename: str
    file_size_mb: float
    status: str
    message: str


class BatchStatusResponse(BaseModel):
    batch_id: int
    filename: str
    status: str
    total_rows: Optional[int] = 0
    processed_rows: Optional[int] = 0
    progress_percent: float = 0.0
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RiskYearlyHistoryPoint(BaseModel):
    year: int
    revenue: float = 0.0
    total_expenses: float = 0.0


class RiskYearlyFeaturePoint(BaseModel):
    year: int
    risk_score: float = 0.0
    f1_divergence: float = 0.0
    f2_ratio_limit: float = 0.0
    f3_vat_structure: float = 0.0
    f4_peer_comparison: float = 0.0


class SingleRiskTierSankeyNode(BaseModel):
    name: str
    year: int
    tier: Literal["low", "medium", "high", "critical"]
    risk_score: float = 0.0


class SingleRiskTierSankeyLink(BaseModel):
    source: str
    target: str
    value: int = 0


class SingleRiskTierSankeyPayload(BaseModel):
    nodes: List[SingleRiskTierSankeyNode] = Field(default_factory=list)
    links: List[SingleRiskTierSankeyLink] = Field(default_factory=list)


class SingleCumulativeRiskPoint(BaseModel):
    year: int
    period_count: int = 0
    percent_periods: float = 0.0
    percent_risk: float = 0.0


class SingleCumulativeRiskCurvePayload(BaseModel):
    points: List[SingleCumulativeRiskPoint] = Field(default_factory=list)
    total_periods: int = 0
    total_risk: float = 0.0
    top_10pct_risk_share: float = 0.0
    top_20pct_risk_share: float = 0.0


class SingleMarginDistributionBin(BaseModel):
    start: float
    end: float
    label: str
    count: int = 0


class SingleMarginDistributionPayload(BaseModel):
    available: bool = False
    industry: Optional[str] = None
    sample_size: int = 0
    company_margin: Optional[float] = None
    percentile: Optional[float] = None
    mean_margin: Optional[float] = None
    median_margin: Optional[float] = None
    company_bin_index: Optional[int] = None
    bins: List[SingleMarginDistributionBin] = Field(default_factory=list)


class SingleRedFlagsTimelinePoint(BaseModel):
    year: int
    flag_count: int = 0
    flag_ids: List[str] = Field(default_factory=list)


class SingleRedFlagsTimelineFlag(BaseModel):
    flag_id: str
    label: str
    severity: Literal["low", "medium", "high"] = "medium"
    first_year: int
    last_year: int
    trigger_count: int = 0
    years: List[int] = Field(default_factory=list)


class SingleRedFlagsTimelinePayload(BaseModel):
    year_points: List[SingleRedFlagsTimelinePoint] = Field(default_factory=list)
    flags: List[SingleRedFlagsTimelineFlag] = Field(default_factory=list)


class DecisionIntelligenceSignal(BaseModel):
    key: str
    label: str
    value: Optional[float] = None
    severity: Literal["low", "medium", "high"] = "medium"
    summary: str = ""


class DecisionIntelligencePayload(BaseModel):
    recommended_action: Literal[
        "periodic_monitoring",
        "enhanced_monitoring",
        "targeted_review",
        "urgent_audit",
    ] = "periodic_monitoring"
    action_label: str = "Theo doi dinh ky"
    action_deadline_days: int = 30
    priority_score: int = 0
    rationale: str = ""
    next_steps: List[str] = Field(default_factory=list)
    top_signals: List[DecisionIntelligenceSignal] = Field(default_factory=list)
    should_escalate: bool = False


class InterventionUpliftPayload(BaseModel):
    recommended_action: Literal[
        "monitor",
        "auto_reminder",
        "structured_outreach",
        "field_audit",
        "escalated_enforcement",
    ] = "monitor"
    priority_score: int = 0
    expected_risk_reduction_pp: float = 0.0
    expected_penalty_saving: float = 0.0
    expected_collection_uplift: float = 0.0
    confidence: Literal["low", "medium", "high"] = "medium"
    rationale: str = ""
    next_steps: List[str] = Field(default_factory=list)


class VATRefundSignalIndicator(BaseModel):
    key: str
    label: str
    value: Optional[float] = None
    severity: Literal["low", "medium", "high"] = "medium"
    summary: str = ""


class VATRefundSignalsPayload(BaseModel):
    has_signal: bool = False
    queue: Literal["monitor", "refund_watchlist", "priority_refund_audit"] = "monitor"
    level: Literal["low", "medium", "high", "critical"] = "low"
    score: int = 0
    rationale: str = ""
    vat_input_output_ratio: Optional[float] = None
    estimated_refund_gap: Optional[float] = None
    recommended_checks: List[str] = Field(default_factory=list)
    indicators: List[VATRefundSignalIndicator] = Field(default_factory=list)


class AuditValueDriver(BaseModel):
    key: str
    label: str
    value: Optional[float] = None
    impact: Literal["low", "medium", "high"] = "medium"
    summary: str = ""


class AuditValuePayload(BaseModel):
    estimated_recovery: float = 0.0
    expected_net_recovery: float = 0.0
    recoverability_ratio: float = 0.0
    audit_hours_estimate: float = 0.0
    estimated_audit_cost: float = 0.0
    priority_score: int = 0
    recommended_lane: Literal["monitor", "desk_review", "targeted_audit", "priority_audit"] = "monitor"
    confidence: Literal["low", "medium", "high"] = "medium"
    rationale: str = ""
    drivers: List[AuditValueDriver] = Field(default_factory=list)


class RiskAssessmentDetail(BaseModel):
    tax_code: str
    company_name: Optional[str] = None
    industry: Optional[str] = None
    year: Optional[int] = None
    revenue: float = 0.0
    total_expenses: float = 0.0
    f1_divergence: Optional[float] = None
    f2_ratio_limit: Optional[float] = None
    f3_vat_structure: Optional[float] = None
    f4_peer_comparison: Optional[float] = None
    anomaly_score: Optional[float] = None
    model_confidence: Optional[float] = None
    model_version: Optional[str] = None
    risk_score: float = 0.0
    risk_level: str = "low"
    red_flags: List[dict] = Field(default_factory=list)
    shap_explanation: List[dict] = Field(default_factory=list)
    yearly_history: List[RiskYearlyHistoryPoint] = Field(default_factory=list)
    yearly_feature_scores: List[RiskYearlyFeaturePoint] = Field(default_factory=list)
    previous_year_features: Optional[RiskYearlyFeaturePoint] = None
    feature_deltas: Dict[str, float] = Field(default_factory=dict)
    single_risk_tier_sankey: SingleRiskTierSankeyPayload = Field(default_factory=SingleRiskTierSankeyPayload)
    single_cumulative_risk_curve: SingleCumulativeRiskCurvePayload = Field(default_factory=SingleCumulativeRiskCurvePayload)
    single_margin_distribution: SingleMarginDistributionPayload = Field(default_factory=SingleMarginDistributionPayload)
    single_red_flags_timeline: SingleRedFlagsTimelinePayload = Field(default_factory=SingleRedFlagsTimelinePayload)
    decision_intelligence: DecisionIntelligencePayload = Field(default_factory=DecisionIntelligencePayload)
    intervention_uplift: InterventionUpliftPayload = Field(default_factory=InterventionUpliftPayload)
    vat_refund_signals: VATRefundSignalsPayload = Field(default_factory=VATRefundSignalsPayload)
    audit_value: AuditValuePayload = Field(default_factory=AuditValuePayload)
    split_trigger_status: Dict[str, Any] = Field(default_factory=dict)
    source: Literal["cached", "realtime"] = "realtime"
    history_source: str = "unavailable"
    history_year_count: int = 0


class RiskCompanyListItem(BaseModel):
    tax_code: str
    name: str
    industry: Optional[str] = None
    is_active: bool = True
    risk_score: float = 0.0
    latest_risk_score: Optional[float] = None
    intervention_action: Optional[Literal[
        "monitor",
        "auto_reminder",
        "structured_outreach",
        "field_audit",
        "escalated_enforcement",
    ]] = None
    intervention_priority: Optional[int] = None
    assessment_count: int = 0
    latest_assessment_at: Optional[str] = None
    assessed: bool = False


class RiskCompanyListResponse(BaseModel):
    mode: Literal["all", "assessed"]
    page: int
    page_size: int
    total: int
    total_pages: int
    results: List[RiskCompanyListItem] = Field(default_factory=list)


class FraudMetricSnapshot(BaseModel):
    auc_roc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None


class FraudPerformanceQuality(BaseModel):
    available: bool = False
    sample_size: int = 0
    metrics: FraudMetricSnapshot = Field(default_factory=FraudMetricSnapshot)
    criteria: Dict[str, Any] = Field(default_factory=dict)


class FraudCalibrationQuality(BaseModel):
    available: bool = False
    method: Optional[str] = None
    trained_at: Optional[str] = None
    brier_improvement: Optional[float] = None
    raw_metrics: FraudMetricSnapshot = Field(default_factory=FraudMetricSnapshot)
    calibrated_metrics: FraudMetricSnapshot = Field(default_factory=FraudMetricSnapshot)
    criteria: Dict[str, Any] = Field(default_factory=dict)


class FraudModelInfoSummary(BaseModel):
    model_version: Optional[str] = None
    updated_at: Optional[str] = None
    calibrator_available: bool = False
    quality_report_available: bool = False


class FraudGateSummary(BaseModel):
    performance_pass: Optional[bool] = None
    calibration_pass: Optional[bool] = None
    soft_gate_pass: Optional[bool] = None
    soft_gate_warnings: List[str] = Field(default_factory=list)
    all_pass: bool = False


class FraudDriftQuality(BaseModel):
    detected: bool = False
    severity: str = "insufficient_data"
    drifted_features: List[str] = Field(default_factory=list)
    recommendation: str = ""


class FraudQualityResponse(BaseModel):
    status: Literal["healthy", "warning", "degraded", "unknown"] = "unknown"
    generated_at: str
    model_info: FraudModelInfoSummary
    gate_summary: FraudGateSummary
    performance: FraudPerformanceQuality
    calibration: FraudCalibrationQuality
    drift: FraudDriftQuality


# --- Graph Model Quality Schemas ---
class GraphServingQuality(BaseModel):
    available: bool = False
    overall_pass: Optional[bool] = None
    node_f1: Optional[float] = None
    edge_f1: Optional[float] = None
    node_pr_auc_delta: Optional[float] = None
    edge_pr_auc_delta: Optional[float] = None
    criteria: Dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[str] = None


class GraphStressQuality(BaseModel):
    available: bool = False
    overall_pass: Optional[bool] = None
    worst_node_f1_delta: Optional[float] = None
    unseen_node_generalization_gap: Optional[float] = None
    temporal_plus3m_edge_f1_drop: Optional[float] = None
    temporal_plus3m_edge_prauc_drop: Optional[float] = None
    criteria: Dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[str] = None


class GraphDriftQuality(BaseModel):
    detected: bool = False
    severity: str = "insufficient_data"
    drifted_features: List[str] = Field(default_factory=list)
    recommendation: str = ""


class GraphModelInfoSummary(BaseModel):
    model_version: Optional[str] = None
    amount_feature_mode: Optional[str] = None
    updated_at: Optional[str] = None


class GraphGateSummary(BaseModel):
    serving_pass: Optional[bool] = None
    stress_pass: Optional[bool] = None
    all_pass: bool = False


class GraphQualityResponse(BaseModel):
    status: Literal["healthy", "warning", "degraded", "unknown"] = "unknown"
    generated_at: str
    model_info: GraphModelInfoSummary
    gate_summary: GraphGateSummary
    serving: GraphServingQuality
    stress: GraphStressQuality
    drift: GraphDriftQuality


# ════════════════════════════════════════════════════════════════
#  NEW: Flagship Model Schemas (Phase 0 API contract expansion)
# ════════════════════════════════════════════════════════════════

# --- Enhanced Delinquency Prediction (replaces mock) ---
class DelinquencyTopReason(BaseModel):
    reason: str
    weight: float = 0.0


class DelinquencyEarlyWarning(BaseModel):
    has_warning: bool = False
    queue: Literal["monitor", "watchlist", "priority_review"] = "monitor"
    level: Literal["low", "medium", "high", "critical"] = "low"
    tags: List[str] = Field(default_factory=list)
    reason: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DelinquencyInterventionUplift(BaseModel):
    recommended_action: Literal[
        "monitor",
        "auto_reminder",
        "structured_outreach",
        "field_audit",
        "escalated_enforcement",
    ] = "monitor"
    priority_score: int = 0
    expected_risk_reduction_pp: float = 0.0
    expected_penalty_saving: float = 0.0
    expected_collection_uplift: float = 0.0
    confidence: Literal["low", "medium", "high"] = "medium"
    rationale: str = ""
    next_steps: List[str] = Field(default_factory=list)

class DelinquencyPredictionItem(BaseModel):
    """Single company delinquency prediction – replaces old mock DelinquencyPrediction."""
    tax_code: str
    company_name: str = ""
    probability: float = 0.0        # Overall delinquency probability (backward-compat)
    prob_30d: float = 0.0           # P(overdue within 30 days)
    prob_60d: float = 0.0           # P(overdue within 60 days)
    prob_90d: float = 0.0           # P(overdue within 90 days)
    cluster: str = ""               # Risk cluster label
    top_reasons: List[DelinquencyTopReason] = Field(default_factory=list)
    model_version: Optional[str] = None
    model_confidence: Optional[float] = None
    prediction_date: Optional[str] = None
    score_source: Literal["ml_model", "statistical_baseline", "no_data", "inference_error"] = "statistical_baseline"
    prediction_age_days: Optional[int] = None
    freshness: Literal["fresh", "aging", "stale", "unknown"] = "unknown"
    monotonic_adjusted: bool = False
    early_warning: DelinquencyEarlyWarning = Field(default_factory=DelinquencyEarlyWarning)
    intervention_uplift: DelinquencyInterventionUplift = Field(default_factory=DelinquencyInterventionUplift)
    split_trigger_status: Dict[str, Any] = Field(default_factory=dict)
    payment_history_summary: Optional[Dict[str, Any]] = None

class DelinquencyListResponse(BaseModel):
    total: int = 0
    page: int = 1
    page_size: int = 20
    predictions: List[DelinquencyPredictionItem] = Field(default_factory=list)
    model_info: Optional[Dict[str, Any]] = None


class DelinquencyBatchPredictRequest(BaseModel):
    tax_codes: Optional[List[str]] = None
    limit: int = Field(default=200, ge=1, le=2000)
    refresh_existing: bool = False


class DelinquencyBatchPredictItem(BaseModel):
    tax_code: str
    status: Literal["created", "updated", "skipped", "failed"]
    score_source: Literal["ml_model", "statistical_baseline", "no_data", "inference_error"]
    prob_90d: Optional[float] = None
    model_version: Optional[str] = None
    message: Optional[str] = None


class DelinquencyBatchPredictResponse(BaseModel):
    total_candidates: int = 0
    processed: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    items: List[DelinquencyBatchPredictItem] = Field(default_factory=list)


# --- Enhanced Fraud Risk Scoring (replaces mock) ---
class FraudRiskPredictionEnhanced(BaseModel):
    """Enhanced single-company scoring – backward compatible with old FraudRiskPrediction."""
    tax_code: str
    company_name: str = ""
    risk_score: float = 0.0
    risk_level: str = "low"
    red_flags: List[str] = Field(default_factory=list)
    model_confidence: Optional[float] = None
    model_version: Optional[str] = None
    f1_divergence: Optional[float] = None
    f2_ratio_limit: Optional[float] = None
    f3_vat_structure: Optional[float] = None
    f4_peer_comparison: Optional[float] = None
    anomaly_score: Optional[float] = None
    top_features: List[Dict[str, Any]] = Field(default_factory=list)


# --- Inspector Labels (ground-truth feedback) ---
class InspectorLabelCreate(BaseModel):
    tax_code: str
    label_type: str = Field(..., description="fraud_confirmed, fraud_rejected, delinquency_confirmed, etc.")
    confidence: Literal["low", "medium", "high"] = "medium"
    label_origin: Optional[Literal[
        "manual_inspector",
        "field_verified",
        "imported_casework",
        "bootstrap_generated",
        "auto_seed",
    ]] = None
    assessment_id: Optional[int] = None
    model_version: Optional[str] = None
    evidence_summary: Optional[str] = None
    decision: Optional[str] = None
    decision_date: Optional[date] = None
    tax_period: Optional[str] = None
    amount_recovered: Optional[float] = None
    intervention_action: Optional[Literal[
        "monitor",
        "auto_reminder",
        "structured_outreach",
        "field_audit",
        "escalated_enforcement",
    ]] = None
    intervention_attempted: bool = False
    outcome_status: Optional[Literal[
        "pending",
        "in_progress",
        "recovered",
        "partial_recovered",
        "unrecoverable",
        "dismissed",
    ]] = None
    predicted_collection_uplift: Optional[float] = None
    expected_recovery: Optional[float] = None
    expected_net_recovery: Optional[float] = None
    estimated_audit_cost: Optional[float] = None
    actual_audit_cost: Optional[float] = None
    actual_audit_hours: Optional[float] = None
    outcome_recorded_at: Optional[datetime] = None
    kpi_window_days: int = Field(default=90, ge=7, le=365)

class InspectorLabelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    tax_code: str
    inspector_id: Optional[int] = None
    assessment_id: Optional[int] = None
    label_type: str
    confidence: str = "medium"
    label_origin: str = "manual_inspector"
    model_version: Optional[str] = None
    evidence_summary: Optional[str] = None
    decision: Optional[str] = None
    decision_date: Optional[date] = None
    tax_period: Optional[str] = None
    amount_recovered: Optional[float] = None
    intervention_action: Optional[str] = None
    intervention_attempted: bool = False
    outcome_status: Optional[str] = None
    predicted_collection_uplift: Optional[float] = None
    expected_recovery: Optional[float] = None
    expected_net_recovery: Optional[float] = None
    estimated_audit_cost: Optional[float] = None
    actual_audit_cost: Optional[float] = None
    actual_audit_hours: Optional[float] = None
    outcome_recorded_at: Optional[datetime] = None
    kpi_window_days: int = 90
    created_at: Optional[datetime] = None


class InspectorLabelBulkCreate(BaseModel):
    labels: List[InspectorLabelCreate] = Field(..., min_length=1, max_length=500)
    strict_mode: bool = True


class InspectorLabelBulkResult(BaseModel):
    inserted: int = 0
    rejected: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    created_ids: List[int] = Field(default_factory=list)


# --- Multi-Scenario What-If Comparison (Program C) ---
class ScenarioDefinition(BaseModel):
    name: str = "Scenario"
    adjustments: Dict[str, float] = Field(default_factory=dict)

class ScenarioResult(BaseModel):
    name: str
    adjustments: Dict[str, float] = Field(default_factory=dict)
    simulated_risk_score: float = 0.0
    risk_level: str = "low"
    delta_risk: float = 0.0
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    simulated_features: Dict[str, float] = Field(default_factory=dict)
    recommended_action: Optional[str] = None

class MultiScenarioRequest(BaseModel):
    scenarios: List[ScenarioDefinition] = Field(..., min_length=1, max_length=10)

class MultiScenarioResponse(BaseModel):
    tax_code: str
    company_name: str = ""
    baseline_risk_score: float = 0.0
    baseline_risk_level: str = "low"
    scenarios: List[ScenarioResult] = Field(default_factory=list)
    best_scenario: Optional[str] = None
    worst_scenario: Optional[str] = None


# --- Ownership / Company Network ---
class OwnershipLinkItem(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    parent_tax_code: str
    child_tax_code: str
    ownership_percent: float = 0.0
    relationship_type: str = "shareholder"
    person_name: Optional[str] = None
    effective_date: Optional[date] = None
    end_date: Optional[date] = None
    verified: bool = False


