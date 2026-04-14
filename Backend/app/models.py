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
    registration_date = Column(Date)
    risk_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
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
    model_confidence = Column(Float, nullable=True)  # Real confidence from pipeline: max(prob, 1-prob)*100
    red_flags = Column(JSON, nullable=True)
    shap_explanation = Column(JSON, nullable=True)
    yearly_history = Column(JSON, nullable=True)  # [{year, revenue, total_expenses}, ...]
    created_at = Column(DateTime(timezone=True), server_default=func.now())

