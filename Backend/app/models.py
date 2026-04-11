from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, Numeric, ForeignKey
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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
