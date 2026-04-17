from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import date, datetime


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
    model_config = ConfigDict(from_attributes=True)
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
    model_config = ConfigDict(from_attributes=True)

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

