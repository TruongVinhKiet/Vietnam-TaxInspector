# -*- coding: utf-8 -*-
"""
Taxpayer APIs for household-business groups 3-11.

The router is intentionally hybrid: all workflows run in sandbox mode by default
and use adapter boundaries for integrations that require real government/provider
credentials.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..auth import get_current_taxpayer
from ..database import get_db
from .. import models
from ..services.taxpayer_adapters import CalendarGateway, ExternalTaxGateway, NotificationGateway, PaymentGateway
from ..services.taxpayer_intelligence import TaxpayerIntelligenceService
from ..services.taxpayer_rules import (
    BASELINE_SOURCES,
    INDUSTRY_TAX_RATES,
    NO_INVOICE_ALLOWED_CASES,
    build_deadlines,
    build_ics,
    calculate_tax_by_industry,
    classify_household_group,
    debt_days_overdue,
    depreciation_schedule,
    e_invoice_requirement,
    evaluate_expense,
    hkd_vs_llc_comparison,
    installment_plan,
    late_payment_penalty,
    legal_answer,
    passport_ban_risk,
    revenue_threshold_summary,
    search_industry_rates,
)


router = APIRouter(prefix="/api/taxpayer", tags=["Taxpayer Portal"])
DOC_UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "data" / "taxpayer_documents"
INTELLIGENCE = TaxpayerIntelligenceService()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _serialize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _row(row: Any) -> dict[str, Any] | None:
    if not row:
        return None
    return _serialize(dict(row._mapping))


def _rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [_serialize(dict(item._mapping)) for item in rows]


def _taxpayer_id(user: models.User) -> str:
    return str(user.badge_id or user.id)


def ensure_taxpayer_schema(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_profiles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL,
            tax_code VARCHAR(20) NOT NULL,
            full_name VARCHAR(200),
            business_name VARCHAR(255),
            household_group INTEGER DEFAULT 2,
            annual_revenue NUMERIC(18,2) DEFAULT 650000000,
            industry VARCHAR(80) DEFAULT 'commerce',
            address TEXT,
            email VARCHAR(200),
            phone VARCHAR(40),
            settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_notification_settings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL,
            sms_enabled BOOLEAN DEFAULT TRUE,
            email_enabled BOOLEAN DEFAULT TRUE,
            zns_enabled BOOLEAN DEFAULT FALSE,
            days_before INTEGER[] DEFAULT ARRAY[7,3,0],
            phone VARCHAR(40),
            email VARCHAR(200),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_deadlines (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            deadline_code VARCHAR(80) NOT NULL,
            title VARCHAR(300) NOT NULL,
            due_date DATE NOT NULL,
            status VARCHAR(30) DEFAULT 'upcoming',
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, deadline_code)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_einvoices (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            invoice_number VARCHAR(80) UNIQUE NOT NULL,
            direction VARCHAR(20) DEFAULT 'out',
            status VARCHAR(40) DEFAULT 'draft',
            seller_tax_code VARCHAR(20),
            buyer_tax_code VARCHAR(20),
            partner_name VARCHAR(255),
            issue_date DATE DEFAULT CURRENT_DATE,
            amount NUMERIC(18,2) DEFAULT 0,
            vat_rate NUMERIC(5,2) DEFAULT 0,
            vat_amount NUMERIC(18,2) DEFAULT 0,
            total_amount NUMERIC(18,2) DEFAULT 0,
            item_description TEXT,
            external_ref VARCHAR(120),
            source_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            risk_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_filings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            form_code VARCHAR(40) NOT NULL,
            period VARCHAR(30) NOT NULL,
            filing_type VARCHAR(40) DEFAULT 'original',
            status VARCHAR(40) DEFAULT 'draft',
            revenue NUMERIC(18,2) DEFAULT 0,
            expenses NUMERIC(18,2) DEFAULT 0,
            gtgt_tax NUMERIC(18,2) DEFAULT 0,
            tncn_tax NUMERIC(18,2) DEFAULT 0,
            total_tax NUMERIC(18,2) DEFAULT 0,
            xml_payload TEXT,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            external_ref VARCHAR(120),
            idempotency_key VARCHAR(160) UNIQUE NOT NULL,
            submitted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_payments (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            filing_id INTEGER,
            period VARCHAR(30),
            tax_type VARCHAR(50) DEFAULT 'GTGT_TNCN',
            amount_due NUMERIC(18,2) DEFAULT 0,
            amount_paid NUMERIC(18,2) DEFAULT 0,
            due_date DATE,
            status VARCHAR(40) DEFAULT 'pending',
            payment_ref VARCHAR(120),
            qr_payload TEXT,
            paid_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_debt_items (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            period VARCHAR(30),
            tax_type VARCHAR(50),
            amount_due NUMERIC(18,2) DEFAULT 0,
            amount_paid NUMERIC(18,2) DEFAULT 0,
            due_date DATE,
            status VARCHAR(40) DEFAULT 'overdue',
            source_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS business_revenue_entries (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            entry_date DATE NOT NULL DEFAULT CURRENT_DATE,
            channel VARCHAR(80) DEFAULT 'direct',
            amount NUMERIC(18,2) DEFAULT 0,
            description TEXT,
            source_ref VARCHAR(120),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS business_expense_entries (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expense_date DATE NOT NULL DEFAULT CURRENT_DATE,
            category VARCHAR(80) DEFAULT 'other',
            amount NUMERIC(18,2) DEFAULT 0,
            payment_method VARCHAR(40) DEFAULT 'bank_transfer',
            has_invoice BOOLEAN DEFAULT FALSE,
            supplier_type VARCHAR(80),
            description TEXT,
            deductible_status VARCHAR(40),
            evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS business_assets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            asset_name VARCHAR(255) NOT NULL,
            purchase_date DATE NOT NULL,
            cost NUMERIC(18,2) DEFAULT 0,
            useful_life_months INTEGER DEFAULT 12,
            monthly_depreciation NUMERIC(18,2) DEFAULT 0,
            status VARCHAR(40) DEFAULT 'active',
            schedule_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS business_documents (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            doc_type VARCHAR(80) DEFAULT 'evidence',
            filename VARCHAR(500) NOT NULL,
            content_type VARCHAR(160),
            file_size INTEGER DEFAULT 0,
            sha256 VARCHAR(64),
            storage_path TEXT,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_business_events (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            event_type VARCHAR(80) NOT NULL,
            start_date DATE,
            end_date DATE,
            status VARCHAR(40) DEFAULT 'draft',
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            external_ref VARCHAR(120),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_claims (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            claim_type VARCHAR(80) NOT NULL,
            decision_no VARCHAR(120),
            description TEXT,
            status VARCHAR(40) DEFAULT 'draft',
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            external_ref VARCHAR(120),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_policy_rules (
            id SERIAL PRIMARY KEY,
            rule_key VARCHAR(120) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            category VARCHAR(120),
            source_url TEXT,
            article_ref VARCHAR(200),
            effective_from DATE,
            effective_to DATE,
            confidence NUMERIC(5,4) DEFAULT 0.8,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_outbox (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            channel VARCHAR(40) NOT NULL,
            subject VARCHAR(255),
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status VARCHAR(40) DEFAULT 'queued',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_feature_snapshots (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
            period VARCHAR(30),
            feature_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            input_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, snapshot_date, period)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_model_predictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            prediction_type VARCHAR(80) NOT NULL,
            model_name VARCHAR(160) NOT NULL,
            model_version VARCHAR(80),
            confidence VARCHAR(30) DEFAULT 'low',
            confidence_score NUMERIC(8,4),
            input_hash VARCHAR(64),
            score NUMERIC(10,4),
            label VARCHAR(120),
            explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_recommendations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            recommendation_key VARCHAR(120) NOT NULL,
            title VARCHAR(300) NOT NULL,
            priority VARCHAR(30) DEFAULT 'medium',
            action_label VARCHAR(200),
            status VARCHAR(40) DEFAULT 'open',
            source_model VARCHAR(160),
            explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, recommendation_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_document_extractions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            document_id INTEGER,
            doc_type VARCHAR(80) DEFAULT 'evidence',
            input_filename VARCHAR(500),
            extraction_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            model_name VARCHAR(160) NOT NULL,
            model_version VARCHAR(80),
            confidence VARCHAR(30) DEFAULT 'low',
            input_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_ai_feedback (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            target_type VARCHAR(80) NOT NULL,
            target_id VARCHAR(120),
            signal VARCHAR(40) NOT NULL,
            comment TEXT,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_bank_transactions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            transaction_date DATE NOT NULL DEFAULT CURRENT_DATE,
            bank_account VARCHAR(80),
            counterparty_name VARCHAR(255),
            counterparty_tax_code VARCHAR(20),
            direction VARCHAR(20) DEFAULT 'in',
            amount NUMERIC(18,2) DEFAULT 0,
            channel VARCHAR(80) DEFAULT 'bank_transfer',
            description TEXT,
            matched_entity_type VARCHAR(80),
            matched_entity_id INTEGER,
            anomaly_score NUMERIC(10,4) DEFAULT 0,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_ledger_entries (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            entry_date DATE NOT NULL DEFAULT CURRENT_DATE,
            book_code VARCHAR(40) DEFAULT 'S1a-HKD',
            entry_type VARCHAR(40) DEFAULT 'revenue',
            account_code VARCHAR(40),
            amount NUMERIC(18,2) DEFAULT 0,
            description TEXT,
            source_type VARCHAR(80),
            source_id INTEGER,
            confidence NUMERIC(8,4) DEFAULT 0,
            status VARCHAR(40) DEFAULT 'draft',
            explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_entity_graph_nodes (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            node_key VARCHAR(160) NOT NULL,
            node_type VARCHAR(80) NOT NULL,
            label VARCHAR(255),
            risk_score NUMERIC(10,4) DEFAULT 0,
            embedding_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, node_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_entity_graph_edges (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            source_key VARCHAR(160) NOT NULL,
            target_key VARCHAR(160) NOT NULL,
            edge_type VARCHAR(80) NOT NULL,
            amount NUMERIC(18,2) DEFAULT 0,
            event_date DATE,
            risk_score NUMERIC(10,4) DEFAULT 0,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_scenarios (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            scenario_key VARCHAR(120) NOT NULL,
            title VARCHAR(255),
            input_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            model_name VARCHAR(160),
            model_version VARCHAR(80),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_forecast_intervals (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            target_metric VARCHAR(80) NOT NULL,
            period VARCHAR(30) NOT NULL,
            p10 NUMERIC(18,2) DEFAULT 0,
            p50 NUMERIC(18,2) DEFAULT 0,
            p90 NUMERIC(18,2) DEFAULT 0,
            model_name VARCHAR(160),
            model_version VARCHAR(80),
            input_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, target_metric, period, input_hash)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_document_fields (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            extraction_id INTEGER,
            field_name VARCHAR(120) NOT NULL,
            field_value TEXT,
            confidence NUMERIC(8,4) DEFAULT 0,
            source_span_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_model_explanations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            prediction_id INTEGER,
            explainer VARCHAR(80) DEFAULT 'rule_shap_counterfactual_baseline',
            reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
            counterfactual_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_privacy_consents (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            consent_key VARCHAR(120) NOT NULL,
            status VARCHAR(40) DEFAULT 'not_granted',
            scope_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            granted_at TIMESTAMP,
            revoked_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, consent_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_copilot_sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            session_key VARCHAR(120) NOT NULL,
            page VARCHAR(160),
            messages_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            status VARCHAR(40) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_feature_store_daily (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            feature_date DATE NOT NULL DEFAULT CURRENT_DATE,
            feature_namespace VARCHAR(120) DEFAULT 'taxpayer_intelligence',
            features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            input_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, feature_date, feature_namespace)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_bank_accounts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            bank_name VARCHAR(160),
            account_number VARCHAR(80),
            account_name VARCHAR(255),
            currency VARCHAR(12) DEFAULT 'VND',
            status VARCHAR(40) DEFAULT 'active',
            source VARCHAR(80) DEFAULT 'sandbox_import',
            external_id VARCHAR(160),
            hash VARCHAR(64),
            idempotency_key VARCHAR(180) NOT NULL,
            consent_scope VARCHAR(160) DEFAULT 'bank_statement_import',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, idempotency_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_einvoice_line_items (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            invoice_id INTEGER,
            invoice_number VARCHAR(80),
            line_no INTEGER DEFAULT 1,
            item_name TEXT,
            unit VARCHAR(60),
            quantity NUMERIC(18,4) DEFAULT 1,
            unit_price NUMERIC(18,2) DEFAULT 0,
            amount NUMERIC(18,2) DEFAULT 0,
            vat_rate NUMERIC(5,2) DEFAULT 0,
            vat_amount NUMERIC(18,2) DEFAULT 0,
            source VARCHAR(80) DEFAULT 'sandbox_import',
            external_id VARCHAR(160),
            hash VARCHAR(64),
            idempotency_key VARCHAR(180) NOT NULL,
            consent_scope VARCHAR(160) DEFAULT 'einvoice_import',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, idempotency_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_platform_orders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            platform VARCHAR(80) DEFAULT 'manual',
            order_code VARCHAR(160),
            order_date DATE NOT NULL DEFAULT CURRENT_DATE,
            settlement_date DATE,
            gross_amount NUMERIC(18,2) DEFAULT 0,
            fees NUMERIC(18,2) DEFAULT 0,
            cod_amount NUMERIC(18,2) DEFAULT 0,
            net_amount NUMERIC(18,2) DEFAULT 0,
            withholding_tax NUMERIC(18,2) DEFAULT 0,
            payment_channel VARCHAR(80),
            status VARCHAR(40) DEFAULT 'completed',
            matched_revenue_id INTEGER,
            source VARCHAR(80) DEFAULT 'sandbox_import',
            external_id VARCHAR(160),
            hash VARCHAR(64),
            idempotency_key VARCHAR(180) NOT NULL,
            consent_scope VARCHAR(160) DEFAULT 'ecommerce_import',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, idempotency_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_inventory_movements (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            movement_date DATE NOT NULL DEFAULT CURRENT_DATE,
            sku VARCHAR(160),
            item_name TEXT,
            movement_type VARCHAR(40) DEFAULT 'in',
            quantity NUMERIC(18,4) DEFAULT 0,
            unit_cost NUMERIC(18,2) DEFAULT 0,
            total_cost NUMERIC(18,2) DEFAULT 0,
            source_document_type VARCHAR(80),
            source_document_id INTEGER,
            source VARCHAR(80) DEFAULT 'sandbox_import',
            external_id VARCHAR(160),
            hash VARCHAR(64),
            idempotency_key VARCHAR(180) NOT NULL,
            consent_scope VARCHAR(160) DEFAULT 'inventory_import',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, idempotency_key)
        );
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_reconciliation_cases (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            case_key VARCHAR(180) NOT NULL,
            case_type VARCHAR(80) DEFAULT '4way_reconciliation',
            severity VARCHAR(40) DEFAULT 'medium',
            status VARCHAR(40) DEFAULT 'open',
            title VARCHAR(300),
            description TEXT,
            entity_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
            suggested_actions JSONB NOT NULL DEFAULT '[]'::jsonb,
            score NUMERIC(10,4) DEFAULT 0,
            model_name VARCHAR(160),
            model_version VARCHAR(80),
            explanation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            source VARCHAR(80) DEFAULT 'taxpayer_intelligence',
            external_id VARCHAR(160),
            hash VARCHAR(64),
            idempotency_key VARCHAR(180) NOT NULL,
            consent_scope VARCHAR(160) DEFAULT 'taxpayer_reconciliation',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, case_key)
        );
    """))
    for ddl in [
        "ALTER TABLE taxpayer_bank_transactions ADD COLUMN IF NOT EXISTS source VARCHAR(80) DEFAULT 'sandbox_import'",
        "ALTER TABLE taxpayer_bank_transactions ADD COLUMN IF NOT EXISTS external_id VARCHAR(160)",
        "ALTER TABLE taxpayer_bank_transactions ADD COLUMN IF NOT EXISTS hash VARCHAR(64)",
        "ALTER TABLE taxpayer_bank_transactions ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(180)",
        "ALTER TABLE taxpayer_bank_transactions ADD COLUMN IF NOT EXISTS consent_scope VARCHAR(160) DEFAULT 'bank_statement_import'",
        "ALTER TABLE taxpayer_bank_transactions ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ]:
        conn.execute(text(ddl))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_revenue_user_date ON business_revenue_entries(user_id, entry_date DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_expense_user_date ON business_expense_entries(user_id, expense_date DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_invoice_user_date ON taxpayer_einvoices(user_id, issue_date DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_predictions_user_type ON taxpayer_model_predictions(user_id, prediction_type, created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_predictions_cache ON taxpayer_model_predictions(user_id, prediction_type, input_hash, created_at DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_recommendations_user_status ON taxpayer_recommendations(user_id, status, priority);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_bank_user_date ON taxpayer_bank_transactions(user_id, transaction_date DESC);"))
    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_taxpayer_bank_idempotency ON taxpayer_bank_transactions(user_id, idempotency_key);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_bank_hash ON taxpayer_bank_transactions(user_id, hash);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_ledger_user_date ON taxpayer_ledger_entries(user_id, entry_date DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_graph_edges_user ON taxpayer_entity_graph_edges(user_id, source_key, target_key);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_model_explanations_user ON taxpayer_model_explanations(user_id, prediction_id);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_einvoice_lines_invoice ON taxpayer_einvoice_line_items(user_id, invoice_number);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_platform_orders_user_date ON taxpayer_platform_orders(user_id, order_date DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_inventory_user_sku ON taxpayer_inventory_movements(user_id, sku, movement_date DESC);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_taxpayer_reconciliation_user_status ON taxpayer_reconciliation_cases(user_id, status, severity);"))
    seed_policy_rules(conn)


def seed_policy_rules(conn) -> None:
    count = conn.execute(text("SELECT COUNT(*) FROM taxpayer_policy_rules")).scalar()
    if count and count > 0:
        return
    for source in BASELINE_SOURCES:
        conn.execute(
            text("""
                INSERT INTO taxpayer_policy_rules
                (rule_key, title, category, source_url, article_ref, effective_from, confidence, payload_json)
                VALUES (:key, :title, :category, :source_url, :article_ref, :effective_from, :confidence, CAST(:payload AS JSONB))
                ON CONFLICT (rule_key) DO NOTHING
            """),
            {
                "key": source["key"],
                "title": source["title"],
                "category": source["category"],
                "source_url": source["source_url"],
                "article_ref": source["article_ref"],
                "effective_from": source["effective_from"],
                "confidence": source["confidence"],
                "payload": _json(source),
            },
        )


def _get_or_create_profile(db: Session, user: models.User) -> dict[str, Any]:
    ensure_taxpayer_schema(db.connection())
    row = db.execute(
        text("SELECT * FROM taxpayer_profiles WHERE user_id = :user_id"),
        {"user_id": user.id},
    ).first()
    if not row:
        annual_revenue = 650_000_000.0
        group = classify_household_group(annual_revenue)["group"]
        db.execute(
            text("""
                INSERT INTO taxpayer_profiles
                (user_id, tax_code, full_name, business_name, household_group, annual_revenue, industry, email, phone)
                VALUES (:user_id, :tax_code, :full_name, :business_name, :group, :annual_revenue, 'commerce', :email, :phone)
            """),
            {
                "user_id": user.id,
                "tax_code": _taxpayer_id(user),
                "full_name": user.full_name,
                "business_name": user.full_name,
                "group": group,
                "annual_revenue": annual_revenue,
                "email": user.email,
                "phone": user.phone,
            },
        )
        db.execute(
            text("""
                INSERT INTO taxpayer_notification_settings (user_id, phone, email)
                VALUES (:user_id, :phone, :email)
                ON CONFLICT (user_id) DO NOTHING
            """),
            {"user_id": user.id, "phone": user.phone, "email": user.email},
        )
        db.commit()
        row = db.execute(
            text("SELECT * FROM taxpayer_profiles WHERE user_id = :user_id"),
            {"user_id": user.id},
        ).first()
    return _row(row) or {}


def _profile_with_dynamic_summary(db: Session, user: models.User) -> dict[str, Any]:
    profile = _get_or_create_profile(db, user)
    cumulative = db.execute(
        text("""
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM business_revenue_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM entry_date) = :year
        """),
        {"user_id": user.id, "year": date.today().year},
    ).scalar()
    annual = _to_float(profile.get("annual_revenue"), 650_000_000.0)
    dynamic = revenue_threshold_summary(_to_float(cumulative), annual)
    return {**profile, "dynamic": dynamic, "group_info": classify_household_group(annual)}


def _load_intelligence_dataset(db: Session, user: models.User, year: int | None = None) -> dict[str, Any]:
    ensure_taxpayer_schema(db.connection())
    target_year = year or date.today().year
    profile = _profile_with_dynamic_summary(db, user)
    params = {"user_id": user.id, "year": target_year}
    revenue_rows = db.execute(
        text("""
            SELECT *
            FROM business_revenue_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM entry_date) = :year
            ORDER BY entry_date ASC, id ASC
        """),
        params,
    ).all()
    expense_rows = db.execute(
        text("""
            SELECT *
            FROM business_expense_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM expense_date) = :year
            ORDER BY expense_date ASC, id ASC
        """),
        params,
    ).all()
    invoice_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_einvoices
            WHERE user_id = :user_id
            ORDER BY issue_date DESC, id DESC
            LIMIT 300
        """),
        {"user_id": user.id},
    ).all()
    filing_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_filings
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 120
        """),
        {"user_id": user.id},
    ).all()
    payment_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_payments
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 120
        """),
        {"user_id": user.id},
    ).all()
    debt_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_debt_items
            WHERE user_id = :user_id
            ORDER BY due_date DESC NULLS LAST, id DESC
            LIMIT 120
        """),
        {"user_id": user.id},
    ).all()
    document_rows = db.execute(
        text("""
            SELECT *
            FROM business_documents
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 120
        """),
        {"user_id": user.id},
    ).all()
    claim_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_claims
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 80
        """),
        {"user_id": user.id},
    ).all()
    bank_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_bank_transactions
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM transaction_date) = :year
            ORDER BY transaction_date ASC, id ASC
            LIMIT 500
        """),
        params,
    ).all()
    bank_account_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_bank_accounts
            WHERE user_id = :user_id
            ORDER BY ingested_at DESC
            LIMIT 80
        """),
        {"user_id": user.id},
    ).all()
    einvoice_line_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_einvoice_line_items
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 500
        """),
        {"user_id": user.id},
    ).all()
    platform_order_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_platform_orders
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM order_date) = :year
            ORDER BY order_date ASC, id ASC
            LIMIT 500
        """),
        params,
    ).all()
    inventory_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_inventory_movements
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM movement_date) = :year
            ORDER BY movement_date ASC, id ASC
            LIMIT 500
        """),
        params,
    ).all()
    reconciliation_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_reconciliation_cases
            WHERE user_id = :user_id
            ORDER BY updated_at DESC, id DESC
            LIMIT 200
        """),
        {"user_id": user.id},
    ).all()
    ledger_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_ledger_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM entry_date) = :year
            ORDER BY entry_date ASC, id ASC
            LIMIT 500
        """),
        params,
    ).all()
    graph_node_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_entity_graph_nodes
            WHERE user_id = :user_id
            ORDER BY risk_score DESC NULLS LAST, updated_at DESC
            LIMIT 300
        """),
        {"user_id": user.id},
    ).all()
    graph_edge_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_entity_graph_edges
            WHERE user_id = :user_id
            ORDER BY event_date DESC NULLS LAST, created_at DESC
            LIMIT 500
        """),
        {"user_id": user.id},
    ).all()
    scenario_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_scenarios
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 80
        """),
        {"user_id": user.id},
    ).all()
    consent_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_privacy_consents
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
        """),
        {"user_id": user.id},
    ).all()
    feedback_rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_ai_feedback
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 120
        """),
        {"user_id": user.id},
    ).all()
    deadlines = build_deadlines(target_year, int(profile.get("household_group") or 2))
    return {
        "year": target_year,
        "today": date.today().isoformat(),
        "profile": profile,
        "revenue_entries": _rows(revenue_rows),
        "expense_entries": _rows(expense_rows),
        "invoices": _rows(invoice_rows),
        "filings": _rows(filing_rows),
        "payments": _rows(payment_rows),
        "debts": _rows(debt_rows),
        "documents": _rows(document_rows),
        "claims": _rows(claim_rows),
        "bank_transactions": _rows(bank_rows),
        "bank_accounts": _rows(bank_account_rows),
        "einvoice_line_items": _rows(einvoice_line_rows),
        "platform_orders": _rows(platform_order_rows),
        "inventory_movements": _rows(inventory_rows),
        "reconciliation_cases": _rows(reconciliation_rows),
        "ledger_entries": _rows(ledger_rows),
        "graph_nodes": _rows(graph_node_rows),
        "graph_edges": _rows(graph_edge_rows),
        "scenarios": _rows(scenario_rows),
        "privacy_consents": _rows(consent_rows),
        "ai_feedback": _rows(feedback_rows),
        "deadlines": deadlines,
    }


def _save_feature_snapshot(db: Session, user: models.User, snapshot: dict[str, Any]) -> None:
    period = str(snapshot.get("year") or date.today().year)
    db.execute(
        text("""
            INSERT INTO taxpayer_feature_snapshots
            (user_id, snapshot_date, period, feature_json, input_hash)
            VALUES (:user_id, CURRENT_DATE, :period, CAST(:feature AS JSONB), :input_hash)
            ON CONFLICT (user_id, snapshot_date, period)
            DO UPDATE SET feature_json = EXCLUDED.feature_json,
                          input_hash = EXCLUDED.input_hash,
                          created_at = CURRENT_TIMESTAMP
        """),
        {
            "user_id": user.id,
            "period": period,
            "feature": _json(snapshot),
            "input_hash": snapshot.get("input_hash") or INTELLIGENCE.input_hash(snapshot),
        },
    )
    db.execute(
        text("""
            INSERT INTO taxpayer_feature_store_daily
            (user_id, feature_date, feature_namespace, features_json, quality_json, input_hash)
            VALUES (:user_id, CURRENT_DATE, 'taxpayer_intelligence',
                    CAST(:features AS JSONB), CAST(:quality AS JSONB), :input_hash)
            ON CONFLICT (user_id, feature_date, feature_namespace)
            DO UPDATE SET features_json = EXCLUDED.features_json,
                          quality_json = EXCLUDED.quality_json,
                          input_hash = EXCLUDED.input_hash,
                          created_at = CURRENT_TIMESTAMP
        """),
        {
            "user_id": user.id,
            "features": _json(snapshot),
            "quality": _json({
                "data_quality_score": snapshot.get("data_quality_score"),
                "sample_size": snapshot.get("sample_size"),
            }),
            "input_hash": snapshot.get("input_hash") or INTELLIGENCE.input_hash(snapshot),
        },
    )


def _save_prediction(db: Session, user: models.User, prediction_type: str, result: dict[str, Any]) -> int | None:
    model = result.get("model") or {}
    label = (
        result.get("label")
        or result.get("risk_level")
        or result.get("status")
        or result.get("signals", {}).get("margin_position")
    )
    score = (
        result.get("risk_score")
        or result.get("score")
        or result.get("projected_year_end_revenue")
        or result.get("annualized_revenue")
    )
    row = db.execute(
        text("""
            INSERT INTO taxpayer_model_predictions
            (user_id, prediction_type, model_name, model_version, confidence, confidence_score,
             input_hash, score, label, explanation_json, output_json)
            VALUES (:user_id, :prediction_type, :model_name, :model_version, :confidence,
                    :confidence_score, :input_hash, :score, :label,
                    CAST(:explanation AS JSONB), CAST(:output AS JSONB))
            RETURNING id
        """),
        {
            "user_id": user.id,
            "prediction_type": prediction_type,
            "model_name": model.get("model_name") or INTELLIGENCE.model_name,
            "model_version": model.get("model_version") or INTELLIGENCE.model_version,
            "confidence": model.get("confidence") or "low",
            "confidence_score": _to_float(model.get("confidence_score"), 0.0),
            "input_hash": model.get("input_hash") or INTELLIGENCE.input_hash(result),
            "score": _to_float(score, 0.0),
            "label": str(label or prediction_type)[:120],
            "explanation": _json({"model": model, "prediction_type": prediction_type}),
            "output": _json(result),
        },
    ).first()
    prediction_id = _to_int(row[0]) if row else None
    if prediction_id:
        reason_codes = result.get("reason_codes") or result.get("explanation", {}).get("reason_codes") or []
        counterfactual = result.get("counterfactual") or result.get("explanation", {}).get("counterfactual") or {}
        db.execute(
            text("""
                INSERT INTO taxpayer_model_explanations
                (user_id, prediction_id, reason_codes, counterfactual_json, explanation_json)
                VALUES (:user_id, :prediction_id, CAST(:reason_codes AS JSONB),
                        CAST(:counterfactual AS JSONB), CAST(:explanation AS JSONB))
            """),
            {
                "user_id": user.id,
                "prediction_id": prediction_id,
                "reason_codes": _json(reason_codes),
                "counterfactual": _json(counterfactual),
                "explanation": _json(result.get("explanation") or {"model": model, "prediction_type": prediction_type}),
            },
        )
    return prediction_id


def _prediction_fingerprint(prediction_type: str, dataset: dict[str, Any] | None = None, payload: dict[str, Any] | None = None) -> str:
    dataset = dataset or {}
    payload = payload or {}
    profile = dataset.get("profile") or {}
    counts = {
        key: len(dataset.get(key) or [])
        for key in [
            "revenue_entries",
            "expense_entries",
            "invoices",
            "filings",
            "payments",
            "debts",
            "documents",
            "claims",
            "bank_transactions",
            "platform_orders",
            "inventory_movements",
            "reconciliation_cases",
            "ledger_entries",
        ]
    }
    fingerprint_payload = {
        "prediction_type": prediction_type,
        "model_version": INTELLIGENCE.model_version,
        "year": dataset.get("year"),
        "today": dataset.get("today"),
        "profile": {
            "tax_code": profile.get("tax_code"),
            "household_group": profile.get("household_group"),
            "annual_revenue": profile.get("annual_revenue"),
            "industry": profile.get("industry"),
        },
        "counts": counts,
        "payload": payload,
    }
    return INTELLIGENCE.input_hash(fingerprint_payload)


def _cached_prediction(db: Session, user: models.User, prediction_type: str, input_hash: str, ttl_seconds: int = 600) -> tuple[int | None, dict[str, Any] | None]:
    ensure_taxpayer_schema(db.connection())
    cutoff = datetime.utcnow() - timedelta(seconds=max(0, ttl_seconds))
    row = db.execute(
        text("""
            SELECT id, output_json
            FROM taxpayer_model_predictions
            WHERE user_id = :user_id
              AND prediction_type = :prediction_type
              AND input_hash = :input_hash
              AND created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {
            "user_id": user.id,
            "prediction_type": prediction_type,
            "input_hash": input_hash,
            "cutoff": cutoff,
        },
    ).first()
    if not row:
        return None, None
    output = row._mapping["output_json"]
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            output = {}
    if not isinstance(output, dict):
        output = {}
    return _to_int(row._mapping["id"], None), _serialize(output)


def _stamp_prediction_contract(result: dict[str, Any], input_hash: str, cache_status: str) -> dict[str, Any]:
    model = result.setdefault("model", {})
    model["input_hash"] = input_hash
    result["input_hash"] = input_hash
    result["cache_status"] = cache_status
    result.setdefault("model_name", model.get("model_name") or INTELLIGENCE.model_name)
    result.setdefault("model_version", model.get("model_version") or INTELLIGENCE.model_version)
    result.setdefault("confidence", model.get("confidence") or "low")
    result.setdefault("confidence_score", model.get("confidence_score"))
    if "data_sufficiency" not in result:
        sufficiency = INTELLIGENCE.data_sufficiency(result.get("snapshot") or result)
        result["data_sufficiency"] = sufficiency
        result["data_sufficiency_score"] = sufficiency["score"]
    return result


def _upsert_recommendations(db: Session, user: models.User, recommendations: list[dict[str, Any]]) -> None:
    for item in recommendations:
        key = item.get("key")
        if not key:
            continue
        db.execute(
            text("""
                INSERT INTO taxpayer_recommendations
                (user_id, recommendation_key, title, priority, action_label, status, source_model,
                 explanation_json, payload_json, updated_at)
                VALUES (:user_id, :key, :title, :priority, :action_label, 'open', :source_model,
                        CAST(:explanation AS JSONB), CAST(:payload AS JSONB), CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, recommendation_key)
                DO UPDATE SET title = EXCLUDED.title,
                              priority = EXCLUDED.priority,
                              action_label = EXCLUDED.action_label,
                              source_model = EXCLUDED.source_model,
                              explanation_json = EXCLUDED.explanation_json,
                              payload_json = EXCLUDED.payload_json,
                              status = 'open',
                              updated_at = CURRENT_TIMESTAMP
            """),
            {
                "user_id": user.id,
                "key": key,
                "title": item.get("title") or key,
                "priority": item.get("priority") or "medium",
                "action_label": item.get("action_label"),
                "source_model": item.get("source_model") or INTELLIGENCE.model_name,
                "explanation": _json({"reason": item.get("reason"), "confidence": item.get("confidence")}),
                "payload": _json(item),
            },
        )


def _filing_xml(payload: dict[str, Any]) -> str:
    escaped = {key: str(value).replace("&", "&amp;").replace("<", "&lt;") for key, value in payload.items()}
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<TaxInspectorFiling form=\"{form_code}\">\n"
        "  <Period>{period}</Period>\n"
        "  <Revenue>{revenue}</Revenue>\n"
        "  <Expenses>{expenses}</Expenses>\n"
        "  <GTGTTax>{gtgt_tax}</GTGTTax>\n"
        "  <TNCNTax>{tncn_tax}</TNCNTax>\n"
        "  <TotalTax>{total_tax}</TotalTax>\n"
        "</TaxInspectorFiling>\n"
    ).format(**escaped)


def _parse_invoice_xml(xml_bytes: bytes) -> dict[str, Any]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise HTTPException(status_code=400, detail=f"File XML hoa don khong hop le: {exc}") from exc

    def by_local_name(name: str) -> str | None:
        for elem in root.iter():
            local = elem.tag.split("}")[-1].lower()
            if local == name.lower() and elem.text:
                return elem.text.strip()
        return None

    seller = by_local_name("MST") or by_local_name("SellerTaxCode") or by_local_name("NBmst")
    buyer = by_local_name("BuyerTaxCode") or by_local_name("NMmst")
    number = by_local_name("InvoiceNumber") or by_local_name("SHDon") or by_local_name("soHoaDon")
    amount = by_local_name("TotalAmount") or by_local_name("TgTCThue") or "0"
    return {
        "seller_tax_code": seller,
        "buyer_tax_code": buyer,
        "invoice_number": number,
        "amount": _to_float(amount),
        "sha256": hashlib.sha256(xml_bytes).hexdigest(),
    }


def _connector_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _connector_key(user_id: int, scope: str, payload: Any) -> str:
    return f"{user_id}:{scope}:{_connector_hash(payload)[:40]}"


def _date_or_today(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if value:
        try:
            return datetime.fromisoformat(str(value)[:10]).date().isoformat()
        except Exception:
            return date.today().isoformat()
    return date.today().isoformat()


def _payload_list(payload: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload] if payload else []


def _invoice_line_items_from_xml(xml_bytes: bytes, invoice_number: str | None) -> list[dict[str, Any]]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return []

    rows = []
    likely_line_names = {"hhdvu", "invoiceitem", "item", "line", "ctiet"}
    for elem in root.iter():
        local = elem.tag.split("}")[-1].lower()
        if local not in likely_line_names:
            continue
        values: dict[str, str] = {}
        for child in list(elem):
            key = child.tag.split("}")[-1]
            if child.text:
                values[key] = child.text.strip()
        if not values:
            continue
        amount = values.get("ThTien") or values.get("Amount") or values.get("THTTien") or values.get("amount")
        name = values.get("THHDVu") or values.get("ItemName") or values.get("Name") or values.get("item_name")
        rows.append(
            {
                "invoice_number": invoice_number,
                "item_name": name or "Hang hoa/dich vu",
                "unit": values.get("DVTinh") or values.get("Unit"),
                "quantity": _to_float(values.get("SLuong") or values.get("Quantity"), 1.0),
                "unit_price": _to_float(values.get("DGia") or values.get("UnitPrice")),
                "amount": _to_float(amount),
                "vat_rate": _to_float(values.get("TSuat") or values.get("VatRate")),
                "metadata_json": values,
            }
        )
    return rows


@router.get("/init")
def init_taxpayer_schema(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    _get_or_create_profile(db, current_user)
    db.commit()
    return {"status": "success", "message": "Taxpayer schema initialized."}


@router.get("/profile/summary")
def get_profile_summary(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _profile_with_dynamic_summary(db, current_user)
    deadlines = build_deadlines(date.today().year, int(profile.get("household_group") or 2))
    debt_rows = db.execute(
        text("""
            SELECT COALESCE(SUM(amount_due - amount_paid), 0) AS total
            FROM taxpayer_debt_items
            WHERE user_id = :user_id AND status IN ('overdue', 'pending')
        """),
        {"user_id": current_user.id},
    ).scalar()
    return {
        "profile": profile,
        "deadlines_count": len(deadlines),
        "next_deadlines": deadlines[:5],
        "debt_total": _to_float(debt_rows),
        "policy_sources": BASELINE_SOURCES,
    }


@router.patch("/profile")
def update_profile(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    _get_or_create_profile(db, current_user)
    annual_revenue = _to_float(payload.get("annual_revenue"), 650_000_000.0)
    group = _to_int(payload.get("household_group"), classify_household_group(annual_revenue)["group"])
    db.execute(
        text("""
            UPDATE taxpayer_profiles
            SET business_name = COALESCE(:business_name, business_name),
                annual_revenue = :annual_revenue,
                household_group = :household_group,
                industry = COALESCE(:industry, industry),
                address = COALESCE(:address, address),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = :user_id
        """),
        {
            "user_id": current_user.id,
            "business_name": payload.get("business_name"),
            "annual_revenue": annual_revenue,
            "household_group": group,
            "industry": payload.get("industry"),
            "address": payload.get("address"),
        },
    )
    db.commit()
    return {"status": "success", "profile": _profile_with_dynamic_summary(db, current_user)}


# ---------------------------------------------------------------------------
# Production connector imports
# ---------------------------------------------------------------------------


@router.post("/connectors/bank/import")
def import_bank_connector(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """Import bank accounts/transactions with idempotent sandbox-first semantics."""
    ensure_taxpayer_schema(db.connection())
    source = str(payload.get("source") or payload.get("provider") or "bank_csv_sandbox")[:80]
    consent_scope = str(payload.get("consent_scope") or "bank_statement_import")[:160]
    accounts = _payload_list(payload, "accounts", "bank_accounts") if any(k in payload for k in ("accounts", "bank_accounts")) else []
    transactions = (
        _payload_list(payload, "transactions", "items", "rows")
        if any(k in payload for k in ("transactions", "items", "rows"))
        else ([] if accounts else [payload])
    )
    if not accounts and not transactions:
        raise HTTPException(status_code=400, detail="Can co accounts hoac transactions de import sao ke.")

    inserted_accounts = 0
    inserted_transactions = 0
    duplicate_transactions = 0
    for account in accounts:
        account_payload = {**account, "source": source}
        key = str(account.get("idempotency_key") or _connector_key(current_user.id, "bank_account", account_payload))[:180]
        h = _connector_hash(account_payload)
        row = db.execute(
            text("""
                INSERT INTO taxpayer_bank_accounts
                (user_id, bank_name, account_number, account_name, currency, status, source,
                 external_id, hash, idempotency_key, consent_scope, metadata_json)
                VALUES (:user_id, :bank_name, :account_number, :account_name, :currency, :status, :source,
                        :external_id, :hash, :idempotency_key, :consent_scope, CAST(:metadata AS JSONB))
                ON CONFLICT (user_id, idempotency_key)
                DO UPDATE SET bank_name = EXCLUDED.bank_name,
                              account_number = EXCLUDED.account_number,
                              account_name = EXCLUDED.account_name,
                              status = EXCLUDED.status,
                              metadata_json = EXCLUDED.metadata_json,
                              ingested_at = CURRENT_TIMESTAMP
                RETURNING id
            """),
            {
                "user_id": current_user.id,
                "bank_name": account.get("bank_name") or account.get("bank"),
                "account_number": account.get("account_number") or account.get("account_no"),
                "account_name": account.get("account_name") or account.get("holder_name"),
                "currency": account.get("currency") or "VND",
                "status": account.get("status") or "active",
                "source": source,
                "external_id": account.get("external_id") or account.get("account_id"),
                "hash": h,
                "idempotency_key": key,
                "consent_scope": consent_scope,
                "metadata": _json(account_payload),
            },
        ).first()
        inserted_accounts += 1 if row else 0

    for item in transactions:
        amount = _to_float(item.get("amount") or item.get("credit") or item.get("debit"))
        if item.get("debit") and not item.get("credit"):
            amount = -abs(_to_float(item.get("debit")))
        direction = str(item.get("direction") or ("out" if amount < 0 else "in")).lower()
        tx_payload = {**item, "source": source, "amount": amount, "direction": direction}
        key = str(item.get("idempotency_key") or _connector_key(current_user.id, "bank_transaction", tx_payload))[:180]
        h = _connector_hash(tx_payload)
        existing = db.execute(
            text("SELECT id FROM taxpayer_bank_transactions WHERE user_id = :user_id AND idempotency_key = :key"),
            {"user_id": current_user.id, "key": key},
        ).first()
        if existing:
            duplicate_transactions += 1
            continue
        db.execute(
            text("""
                INSERT INTO taxpayer_bank_transactions
                (user_id, transaction_date, bank_account, counterparty_name, counterparty_tax_code,
                 direction, amount, channel, description, metadata_json, source, external_id,
                 hash, idempotency_key, consent_scope)
                VALUES (:user_id, :transaction_date, :bank_account, :counterparty_name, :counterparty_tax_code,
                        :direction, :amount, :channel, :description, CAST(:metadata AS JSONB), :source,
                        :external_id, :hash, :idempotency_key, :consent_scope)
            """),
            {
                "user_id": current_user.id,
                "transaction_date": _date_or_today(item.get("transaction_date") or item.get("date") or item.get("booking_date")),
                "bank_account": item.get("bank_account") or item.get("account_number"),
                "counterparty_name": item.get("counterparty_name") or item.get("partner_name"),
                "counterparty_tax_code": item.get("counterparty_tax_code") or item.get("tax_code"),
                "direction": direction,
                "amount": abs(amount),
                "channel": item.get("channel") or "bank_transfer",
                "description": item.get("description") or item.get("memo") or item.get("content"),
                "metadata": _json(tx_payload),
                "source": source,
                "external_id": item.get("external_id") or item.get("transaction_id") or item.get("ref"),
                "hash": h,
                "idempotency_key": key,
                "consent_scope": consent_scope,
            },
        )
        inserted_transactions += 1
    db.commit()
    model = INTELLIGENCE.model_meta(
        {"source": source, "accounts": len(accounts), "transactions": len(transactions)},
        confidence="high" if inserted_transactions or inserted_accounts else "medium",
    )
    return {
        "status": "success",
        "connector": "bank",
        "inserted_accounts": inserted_accounts,
        "inserted_transactions": inserted_transactions,
        "duplicate_transactions": duplicate_transactions,
        "model": model,
    }


@router.post("/connectors/einvoice/import")
def import_einvoice_connector(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """Import HĐĐT XML/PDF/QR metadata and normalize line items."""
    ensure_taxpayer_schema(db.connection())
    source = str(payload.get("source") or "einvoice_xml_sandbox")[:80]
    consent_scope = str(payload.get("consent_scope") or "einvoice_import")[:160]
    raw_xml = payload.get("xml") or payload.get("xml_text")
    parsed_xml: dict[str, Any] = {}
    xml_line_items: list[dict[str, Any]] = []
    if raw_xml:
        xml_bytes = str(raw_xml).encode("utf-8")
        parsed_xml = _parse_invoice_xml(xml_bytes)
        xml_line_items = _invoice_line_items_from_xml(xml_bytes, parsed_xml.get("invoice_number"))
    invoices = _payload_list(payload, "invoices", "items") if not raw_xml else [{**payload, **parsed_xml}]
    if not invoices:
        raise HTTPException(status_code=400, detail="Can co invoice/XML de import HĐĐT.")

    imported_invoices = 0
    imported_lines = 0
    invoice_ids: list[int] = []
    for invoice in invoices:
        invoice_number = str(invoice.get("invoice_number") or invoice.get("number") or parsed_xml.get("invoice_number") or f"IMP-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")[:80]
        direction = str(invoice.get("direction") or "in").lower()
        amount = _to_float(invoice.get("amount") or invoice.get("subtotal") or parsed_xml.get("amount"))
        total = _to_float(invoice.get("total_amount") or invoice.get("total") or amount)
        vat_rate = _to_float(invoice.get("vat_rate"))
        vat_amount = _to_float(invoice.get("vat_amount"), max(0.0, total - amount))
        invoice_payload = {**invoice, "source": source, "parsed_xml": parsed_xml}
        h = _connector_hash(invoice_payload)
        row = db.execute(
            text("""
                INSERT INTO taxpayer_einvoices
                (user_id, invoice_number, direction, status, seller_tax_code, buyer_tax_code,
                 partner_name, issue_date, amount, vat_rate, vat_amount, total_amount,
                 item_description, external_ref, source_payload, risk_json)
                VALUES (:user_id, :invoice_number, :direction, :status, :seller_tax_code, :buyer_tax_code,
                        :partner_name, :issue_date, :amount, :vat_rate, :vat_amount, :total_amount,
                        :item_description, :external_ref, CAST(:payload AS JSONB), CAST(:risk AS JSONB))
                ON CONFLICT (invoice_number)
                DO UPDATE SET status = EXCLUDED.status,
                              amount = EXCLUDED.amount,
                              total_amount = EXCLUDED.total_amount,
                              source_payload = EXCLUDED.source_payload
                RETURNING id
            """),
            {
                "user_id": current_user.id,
                "invoice_number": invoice_number,
                "direction": direction,
                "status": invoice.get("status") or "imported",
                "seller_tax_code": invoice.get("seller_tax_code") or parsed_xml.get("seller_tax_code"),
                "buyer_tax_code": invoice.get("buyer_tax_code") or parsed_xml.get("buyer_tax_code"),
                "partner_name": invoice.get("partner_name") or invoice.get("seller_name") or invoice.get("buyer_name"),
                "issue_date": _date_or_today(invoice.get("issue_date") or invoice.get("date")),
                "amount": amount,
                "vat_rate": vat_rate,
                "vat_amount": vat_amount,
                "total_amount": total,
                "item_description": invoice.get("item_description") or invoice.get("description"),
                "external_ref": invoice.get("external_ref") or invoice.get("external_id") or h[:24],
                "payload": _json(invoice_payload),
                "risk": _json({"source": source, "hash": h, "connector": "einvoice"}),
            },
        ).first()
        invoice_id = _to_int(row[0]) if row else None
        if invoice_id:
            imported_invoices += 1
            invoice_ids.append(invoice_id)
        line_items = invoice.get("line_items") or invoice.get("lines") or xml_line_items or [
            {
                "invoice_number": invoice_number,
                "item_name": invoice.get("item_description") or invoice.get("description") or "Hang hoa/dich vu",
                "quantity": invoice.get("quantity") or 1,
                "unit_price": amount,
                "amount": amount,
                "vat_rate": vat_rate,
                "vat_amount": vat_amount,
            }
        ]
        for idx, line in enumerate(line_items, start=1):
            line_payload = {**line, "invoice_number": invoice_number, "source": source}
            key = str(line.get("idempotency_key") or _connector_key(current_user.id, "einvoice_line", {**line_payload, "idx": idx}))[:180]
            line_hash = _connector_hash(line_payload)
            db.execute(
                text("""
                    INSERT INTO taxpayer_einvoice_line_items
                    (user_id, invoice_id, invoice_number, line_no, item_name, unit, quantity,
                     unit_price, amount, vat_rate, vat_amount, source, external_id, hash,
                     idempotency_key, consent_scope, metadata_json)
                    VALUES (:user_id, :invoice_id, :invoice_number, :line_no, :item_name, :unit,
                            :quantity, :unit_price, :amount, :vat_rate, :vat_amount, :source,
                            :external_id, :hash, :idempotency_key, :consent_scope, CAST(:metadata AS JSONB))
                    ON CONFLICT (user_id, idempotency_key) DO NOTHING
                """),
                {
                    "user_id": current_user.id,
                    "invoice_id": invoice_id,
                    "invoice_number": invoice_number,
                    "line_no": _to_int(line.get("line_no"), idx),
                    "item_name": line.get("item_name") or line.get("name") or "Hang hoa/dich vu",
                    "unit": line.get("unit"),
                    "quantity": _to_float(line.get("quantity"), 1.0),
                    "unit_price": _to_float(line.get("unit_price") or line.get("price")),
                    "amount": _to_float(line.get("amount")),
                    "vat_rate": _to_float(line.get("vat_rate")),
                    "vat_amount": _to_float(line.get("vat_amount")),
                    "source": source,
                    "external_id": line.get("external_id"),
                    "hash": line_hash,
                    "idempotency_key": key,
                    "consent_scope": consent_scope,
                    "metadata": _json(line_payload),
                },
            )
            imported_lines += 1
    db.commit()
    return {
        "status": "success",
        "connector": "einvoice",
        "imported_invoices": imported_invoices,
        "imported_line_items": imported_lines,
        "invoice_ids": invoice_ids,
        "model": INTELLIGENCE.model_meta({"source": source, "invoice_count": imported_invoices, "line_count": imported_lines}, confidence="high"),
    }


@router.post("/connectors/ecommerce/import")
def import_ecommerce_connector(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """Import marketplace/POS/COD orders without requiring live platform credentials."""
    ensure_taxpayer_schema(db.connection())
    source = str(payload.get("source") or payload.get("platform") or "ecommerce_sandbox")[:80]
    consent_scope = str(payload.get("consent_scope") or "ecommerce_import")[:160]
    orders = _payload_list(payload, "orders", "items", "rows")
    if not orders:
        raise HTTPException(status_code=400, detail="Can co orders/items de import giao dich TMĐT/POS/COD.")

    imported = 0
    duplicates = 0
    for item in orders:
        gross = _to_float(item.get("gross_amount") or item.get("amount") or item.get("total_amount"))
        fees = _to_float(item.get("fees") or item.get("platform_fee"))
        withholding = _to_float(item.get("withholding_tax") or item.get("tax_withheld"))
        net = _to_float(item.get("net_amount"), max(0.0, gross - fees - withholding))
        order_payload = {**item, "source": source, "gross_amount": gross, "net_amount": net}
        key = str(item.get("idempotency_key") or _connector_key(current_user.id, "platform_order", order_payload))[:180]
        h = _connector_hash(order_payload)
        existing = db.execute(
            text("SELECT id FROM taxpayer_platform_orders WHERE user_id = :user_id AND idempotency_key = :key"),
            {"user_id": current_user.id, "key": key},
        ).first()
        if existing:
            duplicates += 1
            continue
        db.execute(
            text("""
                INSERT INTO taxpayer_platform_orders
                (user_id, platform, order_code, order_date, settlement_date, gross_amount, fees,
                 cod_amount, net_amount, withholding_tax, payment_channel, status, source,
                 external_id, hash, idempotency_key, consent_scope, metadata_json)
                VALUES (:user_id, :platform, :order_code, :order_date, :settlement_date, :gross_amount,
                        :fees, :cod_amount, :net_amount, :withholding_tax, :payment_channel, :status,
                        :source, :external_id, :hash, :idempotency_key, :consent_scope, CAST(:metadata AS JSONB))
            """),
            {
                "user_id": current_user.id,
                "platform": item.get("platform") or source,
                "order_code": item.get("order_code") or item.get("order_id"),
                "order_date": _date_or_today(item.get("order_date") or item.get("date")),
                "settlement_date": _date_or_today(item.get("settlement_date")) if item.get("settlement_date") else None,
                "gross_amount": gross,
                "fees": fees,
                "cod_amount": _to_float(item.get("cod_amount")),
                "net_amount": net,
                "withholding_tax": withholding,
                "payment_channel": item.get("payment_channel") or item.get("channel"),
                "status": item.get("status") or "completed",
                "source": source,
                "external_id": item.get("external_id") or item.get("order_id"),
                "hash": h,
                "idempotency_key": key,
                "consent_scope": consent_scope,
                "metadata": _json(order_payload),
            },
        )
        imported += 1
    db.commit()
    return {
        "status": "success",
        "connector": "ecommerce",
        "imported_orders": imported,
        "duplicates": duplicates,
        "model": INTELLIGENCE.model_meta({"source": source, "orders": len(orders)}, confidence="high" if imported else "medium"),
    }


# ---------------------------------------------------------------------------
# Calendar and deadline APIs
# ---------------------------------------------------------------------------


@router.get("/calendar/deadlines")
def calendar_deadlines(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    target_year = year or date.today().year
    group = int(profile.get("household_group") or 2)
    deadlines = build_deadlines(target_year, group)
    for item in deadlines:
        db.execute(
            text("""
                INSERT INTO taxpayer_deadlines (user_id, deadline_code, title, due_date, status, payload_json)
                VALUES (:user_id, :code, :title, :due_date, :status, CAST(:payload AS JSONB))
                ON CONFLICT (user_id, deadline_code)
                DO UPDATE SET title = EXCLUDED.title, due_date = EXCLUDED.due_date,
                              status = EXCLUDED.status, payload_json = EXCLUDED.payload_json
            """),
            {
                "user_id": current_user.id,
                "code": item["code"],
                "title": item["title"],
                "due_date": item["due_date"],
                "status": item["status"],
                "payload": _json(item),
            },
        )
    db.commit()
    return {"status": "success", "year": target_year, "group": group, "deadlines": deadlines}


@router.get("/calendar/settings")
def calendar_settings(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    _get_or_create_profile(db, current_user)
    row = db.execute(
        text("SELECT * FROM taxpayer_notification_settings WHERE user_id = :user_id"),
        {"user_id": current_user.id},
    ).first()
    return {"status": "success", "settings": _row(row)}


@router.put("/calendar/settings")
def save_calendar_settings(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    days_before = payload.get("days_before")
    if isinstance(days_before, int):
        days_before = [days_before]
    if not days_before:
        days_before = [7, 3, 0]
    db.execute(
        text("""
            INSERT INTO taxpayer_notification_settings
            (user_id, sms_enabled, email_enabled, zns_enabled, days_before, phone, email, updated_at)
            VALUES (:user_id, :sms, :email_enabled, :zns, :days_before, :phone, :email, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id)
            DO UPDATE SET sms_enabled = EXCLUDED.sms_enabled,
                          email_enabled = EXCLUDED.email_enabled,
                          zns_enabled = EXCLUDED.zns_enabled,
                          days_before = EXCLUDED.days_before,
                          phone = EXCLUDED.phone,
                          email = EXCLUDED.email,
                          updated_at = CURRENT_TIMESTAMP
        """),
        {
            "user_id": current_user.id,
            "sms": bool(payload.get("sms_enabled", True)),
            "email_enabled": bool(payload.get("email_enabled", True)),
            "zns": bool(payload.get("zns_enabled", False)),
            "days_before": [int(item) for item in days_before],
            "phone": payload.get("phone") or current_user.phone,
            "email": payload.get("email") or current_user.email,
        },
    )
    gateway_result = NotificationGateway().schedule(
        {
            "channels": [name for name, enabled in {"sms": payload.get("sms_enabled", True), "email": payload.get("email_enabled", True)}.items() if enabled],
            "days_before": days_before,
        }
    )
    db.execute(
        text("""
            INSERT INTO taxpayer_outbox (user_id, channel, subject, payload_json, status)
            VALUES (:user_id, 'in_app', :subject, CAST(:payload AS JSONB), 'queued')
        """),
        {"user_id": current_user.id, "subject": "Cap nhat nhac nho thue", "payload": _json(gateway_result)},
    )
    db.commit()
    return {"status": "success", "gateway": gateway_result}


@router.post("/calendar/sync")
def sync_calendar(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    deadlines = build_deadlines(_to_int((payload or {}).get("year"), date.today().year), int(profile.get("household_group") or 2))
    result = CalendarGateway().sync(len(deadlines))
    return {"status": "success", **result}


@router.get("/calendar/export.ics")
def export_calendar_ics(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    deadlines = build_deadlines(year or date.today().year, int(profile.get("household_group") or 2))
    ics = build_ics(deadlines)
    return Response(
        ics,
        media_type="text/calendar",
        headers={"Content-Disposition": "attachment; filename=taxpayer-deadlines.ics"},
    )


@router.get("/calendar/revenue-threshold")
def get_revenue_threshold(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    target_year = year or date.today().year
    cumulative = db.execute(
        text("""
            SELECT COALESCE(SUM(amount), 0)
            FROM business_revenue_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM entry_date) = :year
        """),
        {"user_id": current_user.id, "year": target_year},
    ).scalar()
    return {"status": "success", **revenue_threshold_summary(_to_float(cumulative), _to_float(profile.get("annual_revenue")))}


# ---------------------------------------------------------------------------
# Invoice APIs
# ---------------------------------------------------------------------------


@router.get("/invoices/einvoice-requirement")
def get_einvoice_requirement(
    annual_revenue: float | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    revenue = annual_revenue if annual_revenue is not None else _to_float(profile.get("annual_revenue"))
    return {"status": "success", "requirement": e_invoice_requirement(revenue)}


@router.post("/invoices/issue")
def issue_invoice(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    unit_price = _to_float(payload.get("unit_price") or payload.get("price"))
    quantity = _to_float(payload.get("quantity") or payload.get("qty"), 1.0)
    vat_rate = _to_float(payload.get("vat_rate"), 8.0)
    amount = round(unit_price * quantity, 2)
    vat_amount = round(amount * vat_rate / 100.0, 2)
    total = round(amount + vat_amount, 2)
    seq = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    invoice_number = payload.get("invoice_number") or f"HD-{date.today().year}-{seq[-6:]}"
    gateway_payload = {
        **payload,
        "invoice_number": invoice_number,
        "seller_tax_code": profile.get("tax_code"),
        "amount": amount,
        "vat_rate": vat_rate,
        "total_amount": total,
    }
    gateway = ExternalTaxGateway().issue_invoice(gateway_payload)
    db.execute(
        text("""
            INSERT INTO taxpayer_einvoices
            (user_id, invoice_number, direction, status, seller_tax_code, buyer_tax_code, partner_name,
             issue_date, amount, vat_rate, vat_amount, total_amount, item_description, external_ref,
             source_payload, risk_json)
            VALUES (:user_id, :invoice_number, 'out', :status, :seller_tax_code, :buyer_tax_code, :partner_name,
                    CURRENT_DATE, :amount, :vat_rate, :vat_amount, :total_amount, :item_description,
                    :external_ref, CAST(:payload AS JSONB), CAST(:risk AS JSONB))
        """),
        {
            "user_id": current_user.id,
            "invoice_number": invoice_number,
            "status": gateway["status"],
            "seller_tax_code": profile.get("tax_code"),
            "buyer_tax_code": payload.get("buyer_tax_code") or payload.get("buyer_mst"),
            "partner_name": payload.get("buyer_name"),
            "amount": amount,
            "vat_rate": vat_rate,
            "vat_amount": vat_amount,
            "total_amount": total,
            "item_description": payload.get("item_description") or payload.get("description"),
            "external_ref": gateway["external_ref"],
            "payload": _json(gateway_payload),
            "risk": _json(gateway),
        },
    )
    db.commit()
    return {
        "status": "success",
        "invoice": {
            "invoice_number": invoice_number,
            "amount": amount,
            "vat_amount": vat_amount,
            "total_amount": total,
            "gateway": gateway,
        },
    }


@router.get("/invoices/log")
def invoice_log(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_einvoices
            WHERE user_id = :user_id
            ORDER BY issue_date DESC, id DESC
            LIMIT :limit
        """),
        {"user_id": current_user.id, "limit": max(1, min(int(limit), 100))},
    ).all()
    return {"status": "success", "invoices": _rows(rows)}


@router.post("/invoices/scan")
def scan_invoice(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    result = ExternalTaxGateway().check_invoice(payload)
    seller = payload.get("seller_tax_code") or payload.get("tax_code")
    if seller:
        invoice_number = payload.get("invoice_number") or f"SCAN-{datetime.utcnow().strftime('%H%M%S')}"
        db.execute(
            text("""
                INSERT INTO taxpayer_einvoices
                (user_id, invoice_number, direction, status, seller_tax_code, partner_name,
                 amount, source_payload, risk_json)
                VALUES (:user_id, :invoice_number, 'in', :status, :seller, :partner,
                        :amount, CAST(:payload AS JSONB), CAST(:risk AS JSONB))
                ON CONFLICT (invoice_number) DO NOTHING
            """),
            {
                "user_id": current_user.id,
                "invoice_number": invoice_number,
                "status": result["status"],
                "seller": seller,
                "partner": payload.get("partner_name"),
                "amount": _to_float(payload.get("amount")),
                "payload": _json(payload),
                "risk": _json(result),
            },
        )
        db.commit()
    return {"status": "success", "scan": result}


@router.post("/invoices/scan-xml")
async def scan_invoice_xml(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    if not file.filename.lower().endswith(".xml"):
        raise HTTPException(status_code=400, detail="Chi ho tro file XML hoa don.")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File vuot qua 5MB.")
    parsed = _parse_invoice_xml(content)
    parsed["filename"] = file.filename
    result = ExternalTaxGateway().check_invoice(parsed)
    return {"status": "success", "parsed": parsed, "scan": result}


@router.post("/invoices/single-issue-request")
def request_single_invoice(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    ref = f"HDDT-ONCE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    db.execute(
        text("""
            INSERT INTO taxpayer_business_events (user_id, event_type, status, payload_json, external_ref)
            VALUES (:user_id, 'single_invoice_request', 'submitted_sandbox', CAST(:payload AS JSONB), :ref)
        """),
        {"user_id": current_user.id, "payload": _json(payload), "ref": ref},
    )
    db.commit()
    return {"status": "success", "external_ref": ref, "message": "Da lap yeu cau cap hoa don tung lan o sandbox."}


# ---------------------------------------------------------------------------
# Filing and payment APIs
# ---------------------------------------------------------------------------


@router.post("/filings/draft")
def create_filing_draft(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    revenue = _to_float(payload.get("revenue") or payload.get("gtgt_revenue"))
    expenses = _to_float(payload.get("expenses"))
    industry = payload.get("industry") or profile.get("industry") or "commerce"
    taxes = calculate_tax_by_industry(revenue, industry)
    form_code = payload.get("form_code") or ("01/TKN-CNKD" if int(profile.get("household_group") or 2) == 1 else "01/CNKD")
    period = payload.get("period") or f"{date.today().year}-Q{((date.today().month - 1) // 3) + 1}"
    filing_type = payload.get("filing_type") or "original"
    fingerprint = hashlib.sha256(
        _json(
            {
                "form_code": form_code,
                "period": period,
                "filing_type": filing_type,
                "revenue": revenue,
                "expenses": expenses,
                "industry": industry,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    idempotency_key = payload.get("idempotency_key") or f"{current_user.id}:{form_code}:{period}:{filing_type}:{fingerprint}"
    existing = db.execute(
        text("SELECT * FROM taxpayer_filings WHERE idempotency_key = :key"),
        {"key": idempotency_key},
    ).first()
    if existing:
        return {"status": "success", "filing": _row(existing), "idempotent": True}
    xml_payload = _filing_xml(
        {
            "form_code": form_code,
            "period": period,
            "revenue": revenue,
            "expenses": expenses,
            "gtgt_tax": taxes["gtgt_tax"],
            "tncn_tax": taxes["tncn_tax"],
            "total_tax": taxes["total_tax"],
        }
    )
    row = db.execute(
        text("""
            INSERT INTO taxpayer_filings
            (user_id, form_code, period, filing_type, status, revenue, expenses, gtgt_tax, tncn_tax,
             total_tax, xml_payload, payload_json, idempotency_key)
            VALUES (:user_id, :form_code, :period, :filing_type, 'draft', :revenue, :expenses, :gtgt_tax,
                    :tncn_tax, :total_tax, :xml_payload, CAST(:payload AS JSONB), :idempotency_key)
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "form_code": form_code,
            "period": period,
            "filing_type": filing_type,
            "revenue": revenue,
            "expenses": expenses,
            "gtgt_tax": taxes["gtgt_tax"],
            "tncn_tax": taxes["tncn_tax"],
            "total_tax": taxes["total_tax"],
            "xml_payload": xml_payload,
            "payload": _json({**payload, "taxes": taxes}),
            "idempotency_key": idempotency_key,
        },
    ).first()
    db.commit()
    return {"status": "success", "filing": _row(row), "idempotent": False}


@router.post("/filings/{filing_id}/submit")
def submit_filing(
    filing_id: int,
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    filing = db.execute(
        text("SELECT * FROM taxpayer_filings WHERE id = :id AND user_id = :user_id"),
        {"id": filing_id, "user_id": current_user.id},
    ).first()
    if not filing:
        raise HTTPException(status_code=404, detail="Khong tim thay to khai.")
    filing_dict = _row(filing) or {}
    if filing_dict.get("status") == "submitted":
        return {"status": "success", "filing": filing_dict, "idempotent": True}
    gateway = ExternalTaxGateway().submit_filing({**filing_dict, "signature": (payload or {}).get("signature")})
    db.execute(
        text("""
            UPDATE taxpayer_filings
            SET status = 'submitted', submitted_at = CURRENT_TIMESTAMP,
                external_ref = :external_ref, updated_at = CURRENT_TIMESTAMP
            WHERE id = :id AND user_id = :user_id
        """),
        {"id": filing_id, "user_id": current_user.id, "external_ref": gateway["external_ref"]},
    )
    db.execute(
        text("""
            INSERT INTO taxpayer_payments (user_id, filing_id, period, amount_due, status)
            VALUES (:user_id, :filing_id, :period, :amount_due, 'pending')
        """),
        {
            "user_id": current_user.id,
            "filing_id": filing_id,
            "period": filing_dict.get("period"),
            "amount_due": _to_float(filing_dict.get("total_tax")),
        },
    )
    db.commit()
    updated = db.execute(text("SELECT * FROM taxpayer_filings WHERE id = :id"), {"id": filing_id}).first()
    return {"status": "success", "filing": _row(updated), "gateway": gateway}


@router.post("/filings/{filing_id}/validate-proof")
def validate_filing_proof(
    filing_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    filing = db.execute(
        text("SELECT * FROM taxpayer_filings WHERE id = :id AND user_id = :user_id"),
        {"id": filing_id, "user_id": current_user.id},
    ).first()
    if not filing:
        raise HTTPException(status_code=404, detail="Không tìm thấy tờ khai.")
    filing_dict = _row(filing)

    docs = db.execute(
        text("SELECT * FROM business_documents WHERE user_id = :user_id ORDER BY created_at DESC"),
        {"user_id": current_user.id},
    ).all()
    docs_list = _rows(docs)

    declared_rev = float(filing_dict.get("revenue") or 0)
    total_proof_amount = 0.0
    matched_docs = []
    issues = []

    for d in docs_list:
        meta = d.get("metadata_json") or {}
        amount = float(meta.get("amount") or 0)
        if d.get("doc_type") in ["evidence", "invoice", "bank_proof"]:
            total_proof_amount += amount
            matched_docs.append({
                "id": d["id"],
                "filename": d["filename"],
                "doc_type": d["doc_type"],
                "amount": amount,
                "billing_id": meta.get("billing_id") or ("HD-" + d["sha256"][:8].upper() if d.get("sha256") else f"HD-{d['id']}")
            })

    if total_proof_amount < declared_rev:
        diff = declared_rev - total_proof_amount
        issues.append({
            "code": "UNDER_PROOFED",
            "severity": "high",
            "title": "Chênh lệch minh chứng doanh thu",
            "message": f"Tổng số tiền trên các minh chứng đính kèm ({total_proof_amount:,.0f} VND) thấp hơn doanh thu kê khai ({declared_rev:,.0f} VND) là {diff:,.0f} VND.",
            "suggestion": "Vui lòng đính kèm thêm hóa đơn hoặc sao kê ngân hàng."
        })
    elif total_proof_amount > declared_rev * 1.05:
        issues.append({
            "code": "OVER_PROOFED",
            "severity": "medium",
            "title": "Minh chứng vượt doanh thu kê khai",
            "message": f"Tổng số tiền trên các hóa đơn/chứng từ ({total_proof_amount:,.0f} VND) vượt quá doanh thu kê khai.",
            "suggestion": "Rà soát lại các hóa đơn nháp hoặc trùng lặp mã thanh toán."
        })

    for md in matched_docs:
        if not md["billing_id"] or len(md["billing_id"]) < 5:
            issues.append({
                "code": "INVALID_BILLING_ID",
                "severity": "medium",
                "title": "Mã hóa đơn không hợp lệ",
                "message": f"Tài liệu {md['filename']} có mã hóa đơn/chứng từ không hợp lệ.",
                "suggestion": "Cập nhật mã hóa đơn theo định dạng chuẩn của Tổng cục Thuế."
            })

    status = "valid" if not issues else "warning"
    if any(i["severity"] == "high" for i in issues):
        status = "invalid"

    return {
        "status": "success",
        "validation_status": status,
        "declared_revenue": declared_rev,
        "total_proof_amount": total_proof_amount,
        "issues": issues,
        "matched_documents": matched_docs
    }


@router.post("/filings/{filing_id}/amend")
def amend_filing(
    filing_id: int,
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    original = db.execute(
        text("SELECT * FROM taxpayer_filings WHERE id = :id AND user_id = :user_id"),
        {"id": filing_id, "user_id": current_user.id},
    ).first()
    if not original:
        raise HTTPException(status_code=404, detail="Khong tim thay to khai goc.")
    original_dict = _row(original) or {}
    payload = {**original_dict, **payload, "filing_type": "amendment", "idempotency_key": f"{current_user.id}:amend:{filing_id}:{datetime.utcnow().timestamp()}"}
    return create_filing_draft(payload, db, current_user)


@router.get("/filings/status")
def filing_status(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_filings
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        {"user_id": current_user.id, "limit": max(1, min(int(limit), 100))},
    ).all()
    return {"status": "success", "filings": _rows(rows)}


@router.get("/filings/{filing_id}/xml")
def download_filing_xml(
    filing_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    row = db.execute(
        text("SELECT xml_payload, period FROM taxpayer_filings WHERE id = :id AND user_id = :user_id"),
        {"id": filing_id, "user_id": current_user.id},
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Khong tim thay to khai.")
    return Response(
        row[0],
        media_type="application/xml",
        headers={"Content-Disposition": f"attachment; filename=filing-{row[1]}.xml"},
    )


@router.post("/filings/payment-qr")
def create_payment_qr(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    amount = _to_float(payload.get("amount") or payload.get("amount_due"))
    qr = PaymentGateway().create_qr({"tax_code": profile.get("tax_code"), "amount": amount, **payload})
    row = db.execute(
        text("""
            INSERT INTO taxpayer_payments (user_id, filing_id, period, amount_due, status, payment_ref, qr_payload)
            VALUES (:user_id, :filing_id, :period, :amount, 'pending', :payment_ref, :qr_payload)
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "filing_id": payload.get("filing_id"),
            "period": payload.get("period"),
            "amount": amount,
            "payment_ref": qr["payment_ref"],
            "qr_payload": qr["qr_payload"],
        },
    ).first()
    db.commit()
    return {"status": "success", "payment": _row(row), "qr": qr}


@router.post("/filings/payment-confirm")
def confirm_payment(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    payment_ref = payload.get("payment_ref")
    if not payment_ref:
        row = db.execute(
            text("""
                SELECT *
                FROM taxpayer_payments
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"user_id": current_user.id},
        ).first()
    else:
        row = db.execute(
            text("SELECT * FROM taxpayer_payments WHERE user_id = :user_id AND payment_ref = :payment_ref"),
            {"user_id": current_user.id, "payment_ref": payment_ref},
        ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Khong tim thay lenh thanh toan.")
    data = _row(row) or {}
    if data.get("status") == "paid":
        return {"status": "success", "payment": data, "idempotent": True}
    db.execute(
        text("""
            UPDATE taxpayer_payments
            SET status = 'paid', amount_paid = amount_due, paid_at = CURRENT_TIMESTAMP
            WHERE id = :id AND user_id = :user_id
        """),
        {"id": data["id"], "user_id": current_user.id},
    )
    db.commit()
    updated = db.execute(text("SELECT * FROM taxpayer_payments WHERE id = :id"), {"id": data["id"]}).first()
    return {"status": "success", "payment": _row(updated), "idempotent": False}


# ---------------------------------------------------------------------------
# Debts and obligations
# ---------------------------------------------------------------------------


@router.get("/debts/summary")
def debt_summary(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_debt_items
            WHERE user_id = :user_id
            ORDER BY due_date ASC NULLS LAST, id DESC
        """),
        {"user_id": current_user.id},
    ).all()
    debts = _rows(rows)
    total = sum(_to_float(item.get("amount_due")) - _to_float(item.get("amount_paid")) for item in debts)
    max_days = max([debt_days_overdue(item["due_date"]) for item in debts if item.get("due_date")] or [0])
    return {
        "status": "success",
        "debts": debts,
        "total_debt": total,
        "passport_ban_risk": passport_ban_risk(total, max_days),
    }


@router.get("/debts/history")
def payment_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(
        text("""
            SELECT *
            FROM taxpayer_payments
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """),
        {"user_id": current_user.id},
    ).all()
    return {"status": "success", "payments": _rows(rows)}


@router.get("/debts/passport-ban-risk")
def get_passport_ban_risk(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    return debt_summary(db, current_user)["passport_ban_risk"]


@router.get("/debts/impersonation-check")
def impersonation_check(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _get_or_create_profile(db, current_user)
    return {"status": "success", "result": ExternalTaxGateway().check_impersonation(str(profile.get("tax_code")))}


@router.post("/debts/refund-offset")
def request_refund_offset(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    ref = f"REFUND-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    db.execute(
        text("""
            INSERT INTO taxpayer_business_events (user_id, event_type, status, payload_json, external_ref)
            VALUES (:user_id, 'refund_offset_request', 'submitted_sandbox', CAST(:payload AS JSONB), :ref)
        """),
        {"user_id": current_user.id, "payload": _json(payload), "ref": ref},
    )
    db.commit()
    return {"status": "success", "external_ref": ref, "message": "Da lap yeu cau hoan/bu tru sandbox."}


@router.post("/debts/installment")
def request_installment(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    plan = installment_plan(_to_float(payload.get("amount")), _to_int(payload.get("months"), 3))
    db.execute(
        text("""
            INSERT INTO taxpayer_business_events (user_id, event_type, status, payload_json)
            VALUES (:user_id, 'installment_request', 'draft', CAST(:payload AS JSONB))
        """),
        {"user_id": current_user.id, "payload": _json({**payload, "plan": plan})},
    )
    db.commit()
    return {"status": "success", "plan": plan}


@router.get("/debts/late-penalty")
def debt_late_penalty(
    amount: float,
    days: int,
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", "penalty": late_payment_penalty(amount, days)}


# ---------------------------------------------------------------------------
# Legal lookup and chat
# ---------------------------------------------------------------------------


@router.post("/legal/chat")
def taxpayer_legal_chat(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    question = payload.get("message") or payload.get("question") or ""
    rows = db.execute(text("SELECT * FROM taxpayer_policy_rules ORDER BY updated_at DESC")).all()
    answer = legal_answer(question, _rows(rows))
    return {"status": "success", "session_id": payload.get("session_id") or f"taxpayer-{current_user.id}", **answer}


@router.get("/legal/rates")
def legal_rates(
    query: str | None = None,
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", "rates": search_industry_rates(query)}


@router.get("/legal/documents")
def legal_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(text("SELECT * FROM taxpayer_policy_rules ORDER BY effective_from DESC NULLS LAST")).all()
    return {"status": "success", "documents": _rows(rows)}


@router.get("/legal/updates")
def legal_updates(
    industry: str | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    docs = legal_documents(db, current_user)["documents"]
    return {
        "status": "success",
        "updates": [
            {
                "title": item["title"],
                "category": item.get("category"),
                "source_url": item.get("source_url"),
                "impact": "Can doi chieu neu nganh nghe/nhom doanh thu cua ban bi anh huong.",
            }
            for item in docs
        ],
    }


@router.get("/legal/hkd-vs-llc")
def compare_hkd_llc(
    revenue: float = 1_200_000_000,
    expenses: float = 800_000_000,
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", "comparison": hkd_vs_llc_comparison(revenue, expenses)}


# ---------------------------------------------------------------------------
# Growth and business-change APIs
# ---------------------------------------------------------------------------


@router.post("/growth/event")
def create_growth_event(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    event_type = payload.get("event_type") or payload.get("type") or "change"
    ref = f"EVT-{event_type.upper()}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    row = db.execute(
        text("""
            INSERT INTO taxpayer_business_events
            (user_id, event_type, start_date, end_date, status, payload_json, external_ref)
            VALUES (:user_id, :event_type, :start_date, :end_date, 'submitted_sandbox',
                    CAST(:payload AS JSONB), :ref)
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "event_type": event_type,
            "start_date": payload.get("start_date"),
            "end_date": payload.get("end_date"),
            "payload": _json(payload),
            "ref": ref,
        },
    ).first()
    db.commit()
    return {"status": "success", "event": _row(row)}


@router.get("/growth/readiness")
def growth_readiness(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    profile = _profile_with_dynamic_summary(db, current_user)
    annual = _to_float(profile.get("annual_revenue"))
    return {
        "status": "success",
        "household_group": classify_household_group(annual),
        "einvoice": e_invoice_requirement(annual),
        "llc_comparison": hkd_vs_llc_comparison(annual, annual * 0.65),
        "inventory_opening_required": annual > 3_000_000_000,
    }


# ---------------------------------------------------------------------------
# Accounting and document APIs
# ---------------------------------------------------------------------------


@router.post("/accounting/revenue")
def add_revenue_entry(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    row = db.execute(
        text("""
            INSERT INTO business_revenue_entries (user_id, entry_date, channel, amount, description, source_ref)
            VALUES (:user_id, :entry_date, :channel, :amount, :description, :source_ref)
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "entry_date": payload.get("entry_date") or date.today().isoformat(),
            "channel": payload.get("channel") or "direct",
            "amount": _to_float(payload.get("amount")),
            "description": payload.get("description"),
            "source_ref": payload.get("source_ref"),
        },
    ).first()
    db.commit()
    return {"status": "success", "entry": _row(row), "threshold": get_revenue_threshold(date.today().year, db, current_user)}


@router.get("/accounting/revenue")
def list_revenue_entries(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    target_year = year or date.today().year
    rows = db.execute(
        text("""
            SELECT *
            FROM business_revenue_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM entry_date) = :year
            ORDER BY entry_date DESC, id DESC
        """),
        {"user_id": current_user.id, "year": target_year},
    ).all()
    return {"status": "success", "entries": _rows(rows)}


@router.post("/accounting/expense")
def add_expense_entry(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    evaluation = evaluate_expense(payload)
    row = db.execute(
        text("""
            INSERT INTO business_expense_entries
            (user_id, expense_date, category, amount, payment_method, has_invoice, supplier_type,
             description, deductible_status, evidence_json)
            VALUES (:user_id, :expense_date, :category, :amount, :payment_method, :has_invoice,
                    :supplier_type, :description, :status, CAST(:evidence AS JSONB))
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "expense_date": payload.get("expense_date") or date.today().isoformat(),
            "category": evaluation["category"],
            "amount": evaluation["amount"],
            "payment_method": payload.get("payment_method") or "bank_transfer",
            "has_invoice": bool(payload.get("has_invoice", False)),
            "supplier_type": payload.get("supplier_type"),
            "description": payload.get("description"),
            "status": evaluation["status"],
            "evidence": _json(evaluation),
        },
    ).first()
    db.commit()
    return {"status": "success", "entry": _row(row), "evaluation": evaluation}


@router.post("/accounting/assets")
def add_asset(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    schedule = depreciation_schedule(
        _to_float(payload.get("cost")),
        payload.get("purchase_date") or date.today().isoformat(),
        _to_int(payload.get("useful_life_months"), 36),
    )
    row = db.execute(
        text("""
            INSERT INTO business_assets
            (user_id, asset_name, purchase_date, cost, useful_life_months, monthly_depreciation, schedule_json)
            VALUES (:user_id, :asset_name, :purchase_date, :cost, :months, :monthly, CAST(:schedule AS JSONB))
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "asset_name": payload.get("asset_name") or payload.get("name") or "Tai san co dinh",
            "purchase_date": payload.get("purchase_date") or date.today().isoformat(),
            "cost": schedule["cost"],
            "months": schedule["useful_life_months"],
            "monthly": schedule["monthly_depreciation"],
            "schedule": _json(schedule["schedule"]),
        },
    ).first()
    db.commit()
    return {"status": "success", "asset": _row(row), "depreciation": schedule}


@router.get("/accounting/books/{book_code}")
def get_accounting_book(
    book_code: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    code = book_code.upper()
    if code.startswith("S1"):
        rows = db.execute(
            text("SELECT entry_date AS date, channel, description, amount FROM business_revenue_entries WHERE user_id = :user_id ORDER BY entry_date DESC"),
            {"user_id": current_user.id},
        ).all()
    elif code.startswith("S2"):
        rows = db.execute(
            text("SELECT expense_date AS date, category, description, amount, deductible_status FROM business_expense_entries WHERE user_id = :user_id ORDER BY expense_date DESC"),
            {"user_id": current_user.id},
        ).all()
    else:
        rows = db.execute(
            text("SELECT asset_name, purchase_date, cost, monthly_depreciation, status FROM business_assets WHERE user_id = :user_id ORDER BY purchase_date DESC"),
            {"user_id": current_user.id},
        ).all()
    return {"status": "success", "book_code": code, "rows": _rows(rows)}


@router.get("/accounting/report.xlsx")
def export_accounting_excel(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    try:
        from openpyxl import Workbook
    except Exception as exc:
        raise HTTPException(status_code=500, detail="openpyxl chua san sang de xuat Excel.") from exc
    revenue_rows = get_accounting_book("S1a", db, current_user)["rows"]
    expense_rows = get_accounting_book("S2a", db, current_user)["rows"]
    wb = Workbook()
    ws = wb.active
    ws.title = "S1a Doanh thu"
    ws.append(["Ngay", "Kenh", "Mo ta", "So tien"])
    for item in revenue_rows:
        ws.append([item.get("date"), item.get("channel"), item.get("description"), item.get("amount")])
    ws2 = wb.create_sheet("S2a Chi phi")
    ws2.append(["Ngay", "Loai", "Mo ta", "So tien", "Trang thai"])
    for item in expense_rows:
        ws2.append([item.get("date"), item.get("category"), item.get("description"), item.get("amount"), item.get("deductible_status")])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=taxpayer-accounting.xlsx"},
    )


@router.get("/accounting/report.pdf")
def export_accounting_pdf(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise HTTPException(status_code=500, detail="reportlab chua san sang de xuat PDF.") from exc
    summary = get_profile_summary(db, current_user)
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    c.drawString(40, 800, "TaxInspector - Bao cao tom tat HKD")
    c.drawString(40, 775, f"Nguoi nop thue: {current_user.full_name}")
    c.drawString(40, 750, f"Doanh thu luy ke: {summary['profile']['dynamic']['cumulative_revenue']:.0f} VND")
    c.drawString(40, 725, f"No thue: {summary['debt_total']:.0f} VND")
    c.showPage()
    c.save()
    out.seek(0)
    return StreamingResponse(out, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=taxpayer-summary.pdf"})


@router.post("/accounting/documents")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = "evidence",
    billing_id: str | None = Form(default=None),
    amount: float | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File vuot qua gioi han 10MB.")
    user_dir = DOC_UPLOAD_ROOT / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    safe_name = f"{digest[:16]}-{Path(file.filename).name}"
    target = user_dir / safe_name
    target.write_bytes(content)
    
    metadata = {
        "stored_name": safe_name,
        "billing_id": billing_id,
        "amount": amount
    }
    
    row = db.execute(
        text("""
            INSERT INTO business_documents
            (user_id, doc_type, filename, content_type, file_size, sha256, storage_path, metadata_json)
            VALUES (:user_id, :doc_type, :filename, :content_type, :size, :sha256, :path, CAST(:metadata AS JSONB))
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "doc_type": doc_type,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "sha256": digest,
            "path": str(target),
            "metadata": _json(metadata),
        },
    ).first()
    db.commit()
    return {"status": "success", "document": _row(row)}


@router.get("/accounting/documents")
def list_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(
        text("SELECT * FROM business_documents WHERE user_id = :user_id ORDER BY created_at DESC"),
        {"user_id": current_user.id},
    ).all()
    return {"status": "success", "documents": _rows(rows)}


# ---------------------------------------------------------------------------
# Expense-specific APIs
# ---------------------------------------------------------------------------


@router.post("/expenses/check")
def check_expense(
    payload: dict[str, Any] = Body(...),
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", "evaluation": evaluate_expense(payload)}


@router.get("/expenses/summary")
def expense_summary(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    target_year = year or date.today().year
    rows = db.execute(
        text("""
            SELECT deductible_status, COALESCE(SUM(amount),0) AS total, COUNT(*) AS count
            FROM business_expense_entries
            WHERE user_id = :user_id AND EXTRACT(YEAR FROM expense_date) = :year
            GROUP BY deductible_status
        """),
        {"user_id": current_user.id, "year": target_year},
    ).all()
    return {"status": "success", "year": target_year, "summary": _rows(rows), "no_invoice_cases": NO_INVOICE_ALLOWED_CASES}


@router.get("/expenses/no-invoice-cases")
def get_no_invoice_cases(
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", "cases": NO_INVOICE_ALLOWED_CASES}


@router.post("/expenses/bhxh")
def calculate_bhxh(
    payload: dict[str, Any] = Body(...),
    current_user: models.User = Depends(get_current_taxpayer),
):
    salary_base = _to_float(payload.get("salary_base"))
    employees = max(0, _to_int(payload.get("employees")))
    # Sandbox estimate only; exact rates should come from a dedicated social-insurance rule table.
    employer_rate = 0.215
    deductible = salary_base * employees * employer_rate
    voluntary_owner = min(_to_float(payload.get("owner_voluntary")), 1_000_000.0)
    return {
        "status": "success",
        "employees": employees,
        "salary_base": salary_base,
        "employer_bhxh_estimate": round(deductible, 2),
        "owner_voluntary_deduction_cap": voluntary_owner,
        "note": "Uoc tinh sandbox, can doi chieu quy dinh BHXH tai thoi diem ke khai.",
    }


# ---------------------------------------------------------------------------
# Claims and taxpayer rights
# ---------------------------------------------------------------------------


@router.get("/claims/rights")
def taxpayer_rights(
    current_user: models.User = Depends(get_current_taxpayer),
):
    rights = [
        "Duoc huong dan, giai thich chinh sach va thu tuc thue.",
        "Duoc tra cuu ket qua xu ly ho so thue.",
        "Duoc nhan bien ban, ket luan kiem tra va quyet dinh xu ly.",
        "Duoc khieu nai/quyet dinh an dinh, truy thu thue trong thoi han phap luat quy dinh.",
        "Duoc boi thuong thiet hai neu co quan thue gay thiet hai trai phap luat.",
    ]
    return {"status": "success", "rights": rights, "hotline": "1900 558 138"}


@router.post("/claims/appeal")
def submit_appeal(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    ref = f"APPEAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    row = db.execute(
        text("""
            INSERT INTO taxpayer_claims
            (user_id, claim_type, decision_no, description, status, payload_json, external_ref)
            VALUES (:user_id, :claim_type, :decision_no, :description, 'submitted_sandbox',
                    CAST(:payload AS JSONB), :ref)
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "claim_type": payload.get("claim_type") or "appeal",
            "decision_no": payload.get("decision_no") or payload.get("appeal-code"),
            "description": payload.get("description") or payload.get("appeal-reason"),
            "payload": _json(payload),
            "ref": ref,
        },
    ).first()
    db.commit()
    return {"status": "success", "claim": _row(row), "message": "Da ghi nhan khieu nai sandbox."}


@router.post("/claims/complaint")
def submit_complaint(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = {**payload, "claim_type": "officer_complaint"}
    return submit_appeal(payload, db, current_user)


@router.post("/claims/appointment")
def book_tax_office_appointment(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    ref = f"APT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    db.execute(
        text("""
            INSERT INTO taxpayer_business_events (user_id, event_type, start_date, status, payload_json, external_ref)
            VALUES (:user_id, 'tax_office_appointment', :start_date, 'booked_sandbox', CAST(:payload AS JSONB), :ref)
        """),
        {
            "user_id": current_user.id,
            "start_date": payload.get("appointment_date") or date.today().isoformat(),
            "payload": _json(payload),
            "ref": ref,
        },
    )
    db.commit()
    return {"status": "success", "appointment_ref": ref, "hotline": "1900 558 138"}


@router.get("/claims/timeline")
def claims_timeline(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    rows = db.execute(
        text("SELECT * FROM taxpayer_claims WHERE user_id = :user_id ORDER BY created_at DESC"),
        {"user_id": current_user.id},
    ).all()
    return {"status": "success", "claims": _rows(rows)}


# ---------------------------------------------------------------------------
# Taxpayer ML / data science intelligence APIs
# ---------------------------------------------------------------------------


@router.get("/intelligence/capabilities")
def intelligence_capabilities(
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", **INTELLIGENCE.capability_registry()}


@router.get("/intelligence/overview")
def intelligence_overview(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    cache_hash = _prediction_fingerprint("overview", dataset)
    cached_id, cached = _cached_prediction(db, current_user, "overview", cache_hash, ttl_seconds=600)
    if cached:
        return {"status": "success", "prediction_id": cached_id, **_stamp_prediction_contract(cached, cache_hash, "hit")}
    result = INTELLIGENCE.overview(dataset)
    result = _stamp_prediction_contract(result, cache_hash, "miss")
    _save_feature_snapshot(db, current_user, result["snapshot"])
    prediction_id = _save_prediction(db, current_user, "overview", result)
    _upsert_recommendations(db, current_user, result.get("top_recommendations") or [])
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/forecast")
def intelligence_forecast(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    cache_hash = _prediction_fingerprint("forecast", dataset)
    cached_id, cached = _cached_prediction(db, current_user, "forecast", cache_hash, ttl_seconds=600)
    if cached:
        return {"status": "success", "prediction_id": cached_id, **_stamp_prediction_contract(cached, cache_hash, "hit")}
    result = INTELLIGENCE.forecast(dataset)
    result = _stamp_prediction_contract(result, cache_hash, "miss")
    prediction_id = _save_prediction(db, current_user, "forecast", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/what-if")
def intelligence_what_if(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.what_if(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "what_if", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/expense-classify")
def intelligence_expense_classify(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    result = INTELLIGENCE.classify_expense(payload)
    prediction_id = _save_prediction(db, current_user, "expense_classify", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/document-ocr")
async def intelligence_document_ocr(
    file: UploadFile | None = File(default=None),
    doc_type: str = Form(default="evidence"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    content: bytes | None = None
    filename: str | None = None
    if file is not None:
        filename = file.filename
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File vuot qua gioi han 10MB.")
    result = INTELLIGENCE.extract_document(content, filename, doc_type)
    row = db.execute(
        text("""
            INSERT INTO taxpayer_document_extractions
            (user_id, doc_type, input_filename, extraction_json, model_name, model_version,
             confidence, input_hash)
            VALUES (:user_id, :doc_type, :filename, CAST(:extraction AS JSONB), :model_name,
                    :model_version, :confidence, :input_hash)
            RETURNING id
        """),
        {
            "user_id": current_user.id,
            "doc_type": doc_type,
            "filename": filename,
            "extraction": _json(result),
            "model_name": result["model"]["model_name"],
            "model_version": result["model"]["model_version"],
            "confidence": result["model"]["confidence"],
            "input_hash": result["model"]["input_hash"],
        },
    ).first()
    prediction_id = _save_prediction(db, current_user, "document_ocr", result)
    db.commit()
    return {"status": "success", "extraction_id": _to_int(row[0]) if row else None, "prediction_id": prediction_id, **result}


@router.post("/intelligence/invoice-risk")
def intelligence_invoice_risk(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.invoice_risk(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "invoice_risk", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/recommendations")
def intelligence_recommendations(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.recommendations(dataset)
    _upsert_recommendations(db, current_user, result.get("recommendations") or [])
    prediction_id = _save_prediction(db, current_user, "recommendations", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/peer-benchmark")
def intelligence_peer_benchmark(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.peer_benchmark(dataset)
    prediction_id = _save_prediction(db, current_user, "peer_benchmark", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/charts")
def intelligence_charts(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.chart_analytics(dataset)
    prediction_id = _save_prediction(db, current_user, "chart_analytics", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/anomalies")
def intelligence_anomalies(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.anomaly_insights(dataset)
    prediction_id = _save_prediction(db, current_user, "anomaly_insights", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/optimize-tax")
def intelligence_optimize_tax(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.optimize_tax(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "tax_optimization", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/claim-assist")
def intelligence_claim_assist(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.claim_assist(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "claim_assist", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/model-catalog")
def intelligence_model_catalog(
    current_user: models.User = Depends(get_current_taxpayer),
):
    return {"status": "success", **INTELLIGENCE.model_catalog()}


@router.get("/intelligence/scenario-dashboard")
def intelligence_scenario_dashboard(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.scenario_dashboard(dataset)
    _upsert_recommendations(db, current_user, result.get("next_best_actions") or [])
    prediction_id = _save_prediction(db, current_user, "scenario_dashboard", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/ocr-reconcile")
async def intelligence_ocr_reconcile(
    file: UploadFile | None = File(default=None),
    doc_type: str = Form(default="evidence"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user)
    content: bytes | None = None
    filename: str | None = None
    if file is not None:
        filename = file.filename
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File vuot qua gioi han 10MB.")
    result = INTELLIGENCE.ocr_reconcile(content, filename, doc_type, dataset)
    row = db.execute(
        text("""
            INSERT INTO taxpayer_document_extractions
            (user_id, doc_type, input_filename, extraction_json, model_name, model_version,
             confidence, input_hash)
            VALUES (:user_id, :doc_type, :filename, CAST(:extraction AS JSONB), :model_name,
                    :model_version, :confidence, :input_hash)
            RETURNING id
        """),
        {
            "user_id": current_user.id,
            "doc_type": doc_type,
            "filename": filename,
            "extraction": _json(result),
            "model_name": result["model"]["model_name"],
            "model_version": result["model"]["model_version"],
            "confidence": result["model"]["confidence"],
            "input_hash": result["model"]["input_hash"],
        },
    ).first()
    prediction_id = _save_prediction(db, current_user, "ocr_reconcile", result)
    db.commit()
    return {"status": "success", "extraction_id": _to_int(row[0]) if row else None, "prediction_id": prediction_id, **result}


@router.get("/intelligence/cashflow-risk")
def intelligence_cashflow_risk(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    cache_hash = _prediction_fingerprint("cashflow_risk", dataset)
    cached_id, cached = _cached_prediction(db, current_user, "cashflow_risk", cache_hash, ttl_seconds=600)
    if cached:
        return {"status": "success", "prediction_id": cached_id, **_stamp_prediction_contract(cached, cache_hash, "hit")}
    result = INTELLIGENCE.cashflow_risk(dataset)
    result = _stamp_prediction_contract(result, cache_hash, "miss")
    prediction_id = _save_prediction(db, current_user, "cashflow_risk", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/supplier-risk-graph")
def intelligence_supplier_risk_graph(
    tax_code: str | None = None,
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.supplier_risk_graph(dataset, tax_code=tax_code)
    prediction_id = _save_prediction(db, current_user, "supplier_risk_graph", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/auto-bookkeeping")
def intelligence_auto_bookkeeping(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.auto_bookkeeping(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "auto_bookkeeping", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/tax-return-precheck")
def intelligence_tax_return_precheck(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.tax_return_precheck(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "tax_return_precheck", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/policy-impact")
def intelligence_policy_impact(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.policy_impact(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "policy_impact", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/business-upgrade-readiness")
def intelligence_business_upgrade_readiness(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.business_upgrade_readiness(dataset)
    prediction_id = _save_prediction(db, current_user, "business_upgrade_readiness", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/copilot")
def intelligence_copilot(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.copilot(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "copilot", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/advanced-dashboard")
def intelligence_advanced_dashboard(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    cache_hash = _prediction_fingerprint("advanced_dashboard", dataset)
    cached_id, cached = _cached_prediction(db, current_user, "advanced_dashboard", cache_hash, ttl_seconds=600)
    if cached:
        return {"status": "success", "prediction_id": cached_id, **_stamp_prediction_contract(cached, cache_hash, "hit")}
    result = INTELLIGENCE.advanced_dashboard(dataset)
    result = _stamp_prediction_contract(result, cache_hash, "miss")
    _save_feature_snapshot(db, current_user, result.get("probabilistic_forecast", {}))
    _upsert_recommendations(db, current_user, result.get("top_actions") or [])
    prediction_id = _save_prediction(db, current_user, "advanced_dashboard", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/document-ai/extract")
async def intelligence_document_ai_extract(
    file: UploadFile | None = File(default=None),
    doc_type: str = Form(default="evidence"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user)
    content: bytes | None = None
    filename: str | None = None
    if file is not None:
        filename = file.filename
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File vuot qua gioi han 10MB.")
    result = INTELLIGENCE.document_ai_extract(content, filename, doc_type, dataset)
    row = db.execute(
        text("""
            INSERT INTO taxpayer_document_extractions
            (user_id, doc_type, input_filename, extraction_json, model_name, model_version,
             confidence, input_hash)
            VALUES (:user_id, :doc_type, :filename, CAST(:extraction AS JSONB), :model_name,
                    :model_version, :confidence, :input_hash)
            RETURNING id
        """),
        {
            "user_id": current_user.id,
            "doc_type": doc_type,
            "filename": filename,
            "extraction": _json(result),
            "model_name": result["model"]["model_name"],
            "model_version": result["model"]["model_version"],
            "confidence": result["model"]["confidence"],
            "input_hash": result["model"]["input_hash"],
        },
    ).first()
    extraction_id = _to_int(row[0]) if row else None
    for field_name, field_value in (result.get("extracted_fields") or {}).items():
        db.execute(
            text("""
                INSERT INTO taxpayer_document_fields
                (user_id, extraction_id, field_name, field_value, confidence, source_span_json)
                VALUES (:user_id, :extraction_id, :field_name, :field_value, :confidence,
                        CAST(:source_span AS JSONB))
            """),
            {
                "user_id": current_user.id,
                "extraction_id": extraction_id,
                "field_name": str(field_name)[:120],
                "field_value": str(field_value),
                "confidence": _to_float((result.get("field_confidence") or {}).get(field_name), 0.0),
                "source_span": _json({"source": "document_ai_extract"}),
            },
        )
    prediction_id = _save_prediction(db, current_user, "document_ai_extract", result)
    db.commit()
    return {"status": "success", "extraction_id": extraction_id, "prediction_id": prediction_id, **result}


@router.post("/intelligence/document-ai/reconcile")
async def intelligence_document_ai_reconcile(
    file: UploadFile | None = File(default=None),
    doc_type: str = Form(default="evidence"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user)
    content: bytes | None = None
    filename: str | None = None
    if file is not None:
        filename = file.filename
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File vuot qua gioi han 10MB.")
    result = INTELLIGENCE.document_ai_reconcile(content, filename, doc_type, dataset)
    row = db.execute(
        text("""
            INSERT INTO taxpayer_document_extractions
            (user_id, doc_type, input_filename, extraction_json, model_name, model_version,
             confidence, input_hash)
            VALUES (:user_id, :doc_type, :filename, CAST(:extraction AS JSONB), :model_name,
                    :model_version, :confidence, :input_hash)
            RETURNING id
        """),
        {
            "user_id": current_user.id,
            "doc_type": doc_type,
            "filename": filename,
            "extraction": _json(result),
            "model_name": result["model"]["model_name"],
            "model_version": result["model"]["model_version"],
            "confidence": result["model"]["confidence"],
            "input_hash": result["model"]["input_hash"],
        },
    ).first()
    prediction_id = _save_prediction(db, current_user, "document_ai_reconcile", result)
    db.commit()
    return {"status": "success", "extraction_id": _to_int(row[0]) if row else None, "prediction_id": prediction_id, **result}


@router.get("/intelligence/forecast/probabilistic")
def intelligence_forecast_probabilistic(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.probabilistic_forecast(dataset)
    prediction_id = _save_prediction(db, current_user, "probabilistic_forecast", result)
    for item in result.get("intervals") or []:
        db.execute(
            text("""
                INSERT INTO taxpayer_forecast_intervals
                (user_id, target_metric, period, p10, p50, p90, model_name, model_version, input_hash)
                VALUES (:user_id, 'revenue', :period, :p10, :p50, :p90,
                        :model_name, :model_version, :input_hash)
                ON CONFLICT (user_id, target_metric, period, input_hash) DO NOTHING
            """),
            {
                "user_id": current_user.id,
                "period": item.get("period"),
                "p10": _to_float(item.get("p10")),
                "p50": _to_float(item.get("p50")),
                "p90": _to_float(item.get("p90")),
                "model_name": result["model"]["model_name"],
                "model_version": result["model"]["model_version"],
                "input_hash": result["model"]["input_hash"],
            },
        )
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/digital-twin/simulate")
def intelligence_digital_twin_simulate(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.digital_twin_simulate(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "digital_twin_simulate", result)
    db.execute(
        text("""
            INSERT INTO taxpayer_scenarios
            (user_id, scenario_key, title, input_json, output_json, model_name, model_version)
            VALUES (:user_id, :scenario_key, :title, CAST(:input AS JSONB), CAST(:output AS JSONB),
                    :model_name, :model_version)
        """),
        {
            "user_id": current_user.id,
            "scenario_key": str(payload.get("scenario_key") or "digital_twin")[:120],
            "title": payload.get("title") or "Mo phong HKD/TNHH",
            "input": _json(payload),
            "output": _json(result),
            "model_name": result["model"]["model_name"],
            "model_version": result["model"]["model_version"],
        },
    )
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/graph/risk")
def intelligence_graph_risk(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.graph_risk(dataset)
    prediction_id = _save_prediction(db, current_user, "graph_risk", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/ledger/autopost")
def intelligence_ledger_autopost(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.ledger_autopost(payload, dataset)
    for item in result.get("ledger_entries") or []:
        db.execute(
            text("""
                INSERT INTO taxpayer_ledger_entries
                (user_id, entry_date, book_code, entry_type, account_code, amount, description,
                 source_type, confidence, status, explanation_json)
                VALUES (:user_id, CURRENT_DATE, :book_code, :entry_type, :account_code, :amount,
                        :description, 'ai_autopost', :confidence, 'draft_ai_suggested',
                        CAST(:explanation AS JSONB))
            """),
            {
                "user_id": current_user.id,
                "book_code": item.get("book_code"),
                "entry_type": item.get("entry_type"),
                "account_code": item.get("account_code"),
                "amount": _to_float(item.get("amount")),
                "description": item.get("description"),
                "confidence": _to_float(item.get("confidence_score")),
                "explanation": _json(result.get("explanation")),
            },
        )
    prediction_id = _save_prediction(db, current_user, "ledger_autopost", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/filing/precheck-advanced")
def intelligence_filing_precheck_advanced(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.filing_precheck_advanced(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "filing_precheck_advanced", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/cashflow/delinquency")
def intelligence_cashflow_delinquency(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    cache_hash = _prediction_fingerprint("cashflow_delinquency", dataset)
    cached_id, cached = _cached_prediction(db, current_user, "cashflow_delinquency", cache_hash, ttl_seconds=600)
    if cached:
        return {"status": "success", "prediction_id": cached_id, **_stamp_prediction_contract(cached, cache_hash, "hit")}
    result = INTELLIGENCE.cashflow_delinquency(dataset)
    result = _stamp_prediction_contract(result, cache_hash, "miss")
    prediction_id = _save_prediction(db, current_user, "cashflow_delinquency", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/explainability")
def intelligence_explainability(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F9: SHAP Explainability Engine for Compliance Risk Scoring"""
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.explainability(dataset)
    prediction_id = _save_prediction(db, current_user, "explainability_shap", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}



@router.post("/intelligence/legal/graphrag")
def intelligence_legal_graphrag(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.legal_graphrag(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "legal_graphrag", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/next-best-action")
def intelligence_next_best_action(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.next_best_action(dataset)
    _upsert_recommendations(db, current_user, result.get("actions") or [])
    prediction_id = _save_prediction(db, current_user, "next_best_action", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/model-governance")
def intelligence_model_governance(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.model_governance(dataset)
    prediction_id = _save_prediction(db, current_user, "model_governance", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/reconcile/4way")
def intelligence_reconcile_4way(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.reconcile_4way(payload, dataset)
    for case in result.get("cases") or []:
        case_hash = _connector_hash(case)
        db.execute(
            text("""
                INSERT INTO taxpayer_reconciliation_cases
                (user_id, case_key, case_type, severity, status, title, description, entity_refs,
                 suggested_actions, score, model_name, model_version, explanation_json, source,
                 external_id, hash, idempotency_key, consent_scope, updated_at)
                VALUES (:user_id, :case_key, :case_type, :severity, :status, :title, :description,
                        CAST(:entity_refs AS JSONB), CAST(:suggested_actions AS JSONB), :score,
                        :model_name, :model_version, CAST(:explanation AS JSONB), 'taxpayer_intelligence',
                        :external_id, :hash, :idempotency_key, 'taxpayer_reconciliation', CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, case_key)
                DO UPDATE SET severity = EXCLUDED.severity,
                              status = EXCLUDED.status,
                              description = EXCLUDED.description,
                              entity_refs = EXCLUDED.entity_refs,
                              suggested_actions = EXCLUDED.suggested_actions,
                              score = EXCLUDED.score,
                              model_name = EXCLUDED.model_name,
                              model_version = EXCLUDED.model_version,
                              explanation_json = EXCLUDED.explanation_json,
                              hash = EXCLUDED.hash,
                              updated_at = CURRENT_TIMESTAMP
            """),
            {
                "user_id": current_user.id,
                "case_key": str(case.get("case_key") or case_hash[:32])[:180],
                "case_type": case.get("case_type") or "4way_reconciliation",
                "severity": case.get("severity") or "medium",
                "status": case.get("status") or "open",
                "title": case.get("title"),
                "description": case.get("description"),
                "entity_refs": _json(case.get("entity_refs") or []),
                "suggested_actions": _json(case.get("suggested_actions") or []),
                "score": _to_float(case.get("score")),
                "model_name": result["model"]["model_name"],
                "model_version": result["model"]["model_version"],
                "explanation": _json(result.get("explanation") or {}),
                "external_id": case.get("external_id") or case_hash[:24],
                "hash": case_hash,
                "idempotency_key": _connector_key(current_user.id, "reconciliation_case", case),
            },
        )
    prediction_id = _save_prediction(db, current_user, "reconcile_4way", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/reconciliation-cases")
def intelligence_reconciliation_cases(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    cache_hash = _prediction_fingerprint("reconciliation_cases", dataset)
    cached_id, cached = _cached_prediction(db, current_user, "reconciliation_cases", cache_hash, ttl_seconds=300)
    if cached:
        return {"status": "success", "prediction_id": cached_id, **_stamp_prediction_contract(cached, cache_hash, "hit")}
    result = INTELLIGENCE.reconciliation_cases(dataset)
    result = _stamp_prediction_contract(result, cache_hash, "miss")
    prediction_id = _save_prediction(db, current_user, "reconciliation_cases", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/channel-attribution")
def intelligence_channel_attribution(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.channel_attribution(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "channel_attribution", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/tax-reserve/optimize")
def intelligence_tax_reserve_optimize(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.tax_reserve_optimize(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "tax_reserve_optimize", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/price-elasticity")
def intelligence_price_elasticity(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    result = INTELLIGENCE.price_elasticity(payload)
    prediction_id = _save_prediction(db, current_user, "price_elasticity", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/ecommerce/reconcile")
def intelligence_ecommerce_reconcile(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.ecommerce_reconcile(dataset)
    prediction_id = _save_prediction(db, current_user, "ecommerce_reconcile", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/debate")
def intelligence_debate(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    result = INTELLIGENCE.debate_agents(payload)
    prediction_id = _save_prediction(db, current_user, "debate_agents", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/supplier-account-risk")
def intelligence_supplier_account_risk(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.supplier_account_risk(dataset)
    prediction_id = _save_prediction(db, current_user, "supplier_account_risk", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/inventory/analyze")
def intelligence_inventory_analyze(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    ensure_taxpayer_schema(db.connection())
    for idx, item in enumerate(payload.get("movements") or []):
        if not isinstance(item, dict):
            continue
        movement_payload = {**item, "source": payload.get("source") or "inventory_manual"}
        key = str(item.get("idempotency_key") or _connector_key(current_user.id, "inventory_movement", {**movement_payload, "idx": idx}))[:180]
        h = _connector_hash(movement_payload)
        db.execute(
            text("""
                INSERT INTO taxpayer_inventory_movements
                (user_id, movement_date, sku, item_name, movement_type, quantity, unit_cost,
                 total_cost, source_document_type, source_document_id, source, external_id,
                 hash, idempotency_key, consent_scope, metadata_json)
                VALUES (:user_id, :movement_date, :sku, :item_name, :movement_type, :quantity,
                        :unit_cost, :total_cost, :source_document_type, :source_document_id,
                        :source, :external_id, :hash, :idempotency_key, :consent_scope,
                        CAST(:metadata AS JSONB))
                ON CONFLICT (user_id, idempotency_key) DO NOTHING
            """),
            {
                "user_id": current_user.id,
                "movement_date": _date_or_today(item.get("movement_date") or item.get("date")),
                "sku": item.get("sku"),
                "item_name": item.get("item_name") or item.get("name"),
                "movement_type": item.get("movement_type") or item.get("type") or "in",
                "quantity": _to_float(item.get("quantity")),
                "unit_cost": _to_float(item.get("unit_cost") or item.get("price")),
                "total_cost": _to_float(item.get("total_cost") or _to_float(item.get("quantity")) * _to_float(item.get("unit_cost") or item.get("price"))),
                "source_document_type": item.get("source_document_type"),
                "source_document_id": _to_int(item.get("source_document_id"), None),
                "source": movement_payload["source"],
                "external_id": item.get("external_id"),
                "hash": h,
                "idempotency_key": key,
                "consent_scope": payload.get("consent_scope") or "inventory_import",
                "metadata": _json(movement_payload),
            },
        )
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.inventory_analyze(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "inventory_analyze", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/evidence-bundle")
def intelligence_evidence_bundle(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.evidence_bundle(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "evidence_bundle", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/legal/change-impact")
def intelligence_legal_change_impact(
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    payload = payload or {}
    dataset = _load_intelligence_dataset(db, current_user, _to_int(payload.get("year"), date.today().year))
    result = INTELLIGENCE.legal_change_impact(payload, dataset)
    prediction_id = _save_prediction(db, current_user, "legal_change_impact", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/model-governance/production")
def intelligence_model_governance_production(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.model_governance_production(dataset)
    prediction_id = _save_prediction(db, current_user, "model_governance_production", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/legal-chat")
def intelligence_legal_chat(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Vui long nhap cau hoi.")
    rows = db.execute(text("SELECT * FROM taxpayer_policy_rules ORDER BY updated_at DESC")).all()
    answer = legal_answer(question, _rows(rows))
    result = {
        "session_id": payload.get("session_id") or f"taxpayer-ai-{current_user.id}",
        "answer": answer,
        "model": INTELLIGENCE.model_meta(
            {"question": question, "citations": answer.get("citations")},
            confidence="medium" if answer.get("citations") else "low",
        ),
    }
    prediction_id = _save_prediction(db, current_user, "legal_chat", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/benford-analysis")
def intelligence_benford_analysis(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F1: Benford's Law first-digit fraud scanner — chi-square GoF test."""
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.benford_analysis(dataset)
    prediction_id = _save_prediction(db, current_user, "benford_analysis", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/seasonal-decomposition")
def intelligence_seasonal_decomposition(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F2: Seasonal decomposition (Trend / Seasonal / Residual)."""
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.seasonal_decomposition(dataset)
    prediction_id = _save_prediction(db, current_user, "seasonal_decomposition", result)
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/monte-carlo-simulation")
def intelligence_monte_carlo_simulation(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F3: Monte Carlo Tax Simulation — stochastically simulates future tax outcomes."""
    rev_mean = float(payload.get("revenue_mean") or 0.0)
    vol = float(payload.get("volatility_pct") or 15.0)
    exp = float(payload.get("expense_ratio_pct") or 50.0)
    tax_rate = float(payload.get("tax_rate_pct") or 1.5)
    iters = int(payload.get("iterations") or 10000)

    if rev_mean <= 0:
        raise HTTPException(status_code=400, detail="Doanh thu phai lon hon 0.")

    result = INTELLIGENCE.monte_carlo_simulation(
        revenue_mean=rev_mean,
        volatility_pct=vol,
        expense_ratio_pct=exp,
        tax_rate_pct=tax_rate,
        iterations=iters
    )
    prediction_id = _save_prediction(db, current_user, "monte_carlo_simulation", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/survival-analysis")
def intelligence_survival_analysis(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F4: Survival Analysis — Delinquency hazard model."""
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.survival_analysis(dataset)
    prediction_id = _save_prediction(db, current_user, "survival_analysis", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/breakeven-analysis")
def intelligence_breakeven_analysis(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F4: Breakeven Analysis (CVP) Engine"""
    fixed_costs = float(payload.get("fixed_costs") or 0.0)
    var_ratio = float(payload.get("variable_cost_ratio_pct") or 50.0)
    current_rev = float(payload.get("current_revenue") or 0.0)
    target_prof = float(payload.get("target_profit") or 0.0)

    if fixed_costs <= 0:
        raise HTTPException(status_code=400, detail="Chi phi co dinh phai lon hon 0.")

    result = INTELLIGENCE.breakeven_analysis(
        fixed_costs=fixed_costs,
        variable_cost_ratio=var_ratio,
        current_revenue=current_rev,
        target_profit=target_prof
    )
    prediction_id = _save_prediction(db, current_user, "breakeven_analysis", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/bayesian-forecast")
def intelligence_bayesian_forecast(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F6: Bayesian Revenue Forecasting with Uncertainty Engine."""
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.bayesian_forecast(dataset)
    prediction_id = _save_prediction(db, current_user, "bayesian_forecast", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/feedback")
def intelligence_feedback(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    ensure_taxpayer_schema(db.connection())
    target_type = str(payload.get("target_type") or "prediction")[:80]
    signal = str(payload.get("signal") or "").strip()[:40]
    if not signal:
        raise HTTPException(status_code=400, detail="Thieu tin hieu phan hoi.")
    row = db.execute(
        text("""
            INSERT INTO taxpayer_ai_feedback
            (user_id, target_type, target_id, signal, comment, payload_json)
            VALUES (:user_id, :target_type, :target_id, :signal, :comment, CAST(:payload AS JSONB))
            RETURNING *
        """),
        {
            "user_id": current_user.id,
            "target_type": target_type,
            "target_id": str(payload.get("target_id") or payload.get("prediction_id") or "")[:120],
            "signal": signal,
            "comment": payload.get("comment"),
            "payload": _json(payload),
        },
    ).first()
    db.commit()
    return {"status": "success", "feedback": _row(row)}


@router.get("/intelligence/isolation-forest-expenses")
def intelligence_isolation_forest_expenses(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.isolation_forest_expenses(dataset)
    prediction_id = _save_prediction(db, current_user, "isolation_forest_expenses", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/markov-chain-prediction")
def intelligence_markov_chain_prediction(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.markov_chain_prediction(dataset)
    prediction_id = _save_prediction(db, current_user, "markov_chain_prediction", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/pagerank-supplier-trust")
def intelligence_pagerank_supplier_trust(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.pagerank_supplier_trust(dataset)
    prediction_id = _save_prediction(db, current_user, "pagerank_supplier_trust", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/autoencoder-bank-anomaly")
def intelligence_autoencoder_bank_anomaly(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.autoencoder_bank_anomaly(dataset)
    prediction_id = _save_prediction(db, current_user, "autoencoder_bank_anomaly", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/rfm-customer-segmentation")
def intelligence_rfm_customer_segmentation(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.rfm_customer_segmentation(dataset)
    prediction_id = _save_prediction(db, current_user, "rfm_customer_segmentation", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/working-capital")
def intelligence_working_capital(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.working_capital_optimization(dataset)
    prediction_id = _save_prediction(db, current_user, "working_capital_optimization", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/regulatory-change-diff")
def intelligence_regulatory_change_diff(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.regulatory_change_diff(dataset)
    prediction_id = _save_prediction(db, current_user, "regulatory_change_diff", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/compliance-risk-heatmap")
def intelligence_compliance_risk_heatmap(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.compliance_risk_heatmap(dataset)
    prediction_id = _save_prediction(db, current_user, "compliance_risk_heatmap", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/tax-calendar-optimization")
def intelligence_tax_calendar_optimization(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.tax_calendar_optimization(dataset)
    prediction_id = _save_prediction(db, current_user, "tax_calendar_optimization", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/cohort-analysis")
def intelligence_cohort_analysis(
    year: int | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    dataset = _load_intelligence_dataset(db, current_user, year)
    result = INTELLIGENCE.cohort_analysis(dataset)
    prediction_id = _save_prediction(db, current_user, "cohort_analysis", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/transfer-pricing")
def intelligence_transfer_pricing(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F19: Transfer Pricing Risk Evaluator (Arm's Length Deviation) using Mahalanobis Distance."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    if "profile" not in dataset or not dataset["profile"]:
        dataset["profile"] = {}
    dataset["profile"]["target_unit_price"] = float(payload.get("target_unit_price") or 95000.0)
    dataset["profile"]["target_quantity"] = float(payload.get("target_quantity") or 180.0)
    
    result = INTELLIGENCE.transfer_pricing_evaluator(dataset)
    prediction_id = _save_prediction(db, current_user, "transfer_pricing_evaluator", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.post("/intelligence/outflow-stress")
def intelligence_outflow_stress(
    payload: dict[str, Any] = Body(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F20: Tax Outflow GEV Stress Simulator (Extreme Value Theory)."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.tax_cash_stress_simulator(dataset)
    prediction_id = _save_prediction(db, current_user, "tax_cash_stress_simulator", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/spectral-cascade")
def intelligence_spectral_cascade(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F21: GNN-Simulated Spectral Evasion Cascade and Collusion Analysis."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.gnn_spectral_fraud_cascade(dataset)
    prediction_id = _save_prediction(db, current_user, "gnn_spectral_fraud_cascade", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/entropy-revenue")
def intelligence_entropy_revenue(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F22: Shannon Entropy Revenue Anomaly — kiểm tra chất lượng dữ liệu doanh thu."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.entropy_revenue_anomaly(dataset)
    prediction_id = _save_prediction(db, current_user, "entropy_revenue_anomaly", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/hmm-financial-state")
def intelligence_hmm_financial_state(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F23: Hidden Markov Model — cảnh báo sớm trạng thái tài chính."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.hmm_financial_state(dataset)
    prediction_id = _save_prediction(db, current_user, "hmm_financial_state", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/cusum-change-detection")
def intelligence_cusum_change_detection(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F24: CUSUM Change-Point Detection — phát hiện điểm chuyển đổi doanh thu."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.cusum_change_detection(dataset)
    prediction_id = _save_prediction(db, current_user, "cusum_change_detection", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/svd-expense-decomposition")
def intelligence_svd_expense_decomposition(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F25: Singular Value Decomposition — phân tích cấu trúc chi phí."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.svd_expense_decomposition(dataset)
    prediction_id = _save_prediction(db, current_user, "svd_expense_decomposition", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/wavelet-revenue")
def intelligence_wavelet_revenue(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F26: Haar Wavelet Multi-Resolution — tách xu hướng và biến động mùa vụ."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.wavelet_revenue_decomposition(dataset)
    prediction_id = _save_prediction(db, current_user, "wavelet_revenue_decomposition", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/altman-zscore")
def intelligence_altman_zscore(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F27: Altman Z-Score Bankruptcy Prediction — đánh giá sức khỏe tài chính & khả năng phá sản."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.altman_zscore_bankruptcy(dataset)
    prediction_id = _save_prediction(db, current_user, "altman_zscore_bankruptcy", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/kmeans-supplier-clustering")
def intelligence_kmeans_supplier_clustering(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F28: K-Means++ Supplier Clustering — phân nhóm rủi ro đối tác/nhà cung cấp."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.kmeans_supplier_clustering(dataset)
    prediction_id = _save_prediction(db, current_user, "kmeans_supplier_clustering", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}


@router.get("/intelligence/composite-risk-score")
def intelligence_composite_risk_score(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_taxpayer),
):
    """F29: Gradient Boosting Composite Risk — điểm sức khỏe thuế tổng hợp."""
    dataset = _load_intelligence_dataset(db, current_user, None)
    result = INTELLIGENCE.composite_risk_score(dataset)
    prediction_id = _save_prediction(db, current_user, "composite_risk_score", result)
    db.commit()
    return {"status": "success", "prediction_id": prediction_id, **result}
