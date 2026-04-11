-- Drop existing tables if they exist to avoid conflict during initialization
DROP TABLE IF EXISTS invoices CASCADE;
DROP TABLE IF EXISTS tax_returns CASCADE;
DROP TABLE IF EXISTS companies CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. Table for System Users (Admin/Data Scientists/Officers)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    badge_id VARCHAR(50) UNIQUE NOT NULL,       -- Mã số cán bộ (VTI-XXXX)
    full_name VARCHAR(100) NOT NULL,            -- Họ và tên
    department VARCHAR(100) NOT NULL,           -- Đơn vị công tác
    email VARCHAR(100) UNIQUE NOT NULL,         -- Email công vụ (@gdt.gov.vn)
    phone VARCHAR(20),                          -- Số điện thoại (Optional)
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'viewer', -- Vai trò quyền hạn (admin, inspector, analyst, viewer)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Base Entity: Companies (Nodes for Graph and Main ML Target)
CREATE TABLE companies (
    tax_code VARCHAR(20) PRIMARY KEY, -- Using tax_code as the main identifier
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    registration_date DATE,
    risk_score FLOAT DEFAULT 0.0,     -- Updated dynamically by Fraud Scoring Model
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Historical Data: Tax Returns (Used for Risk Scoring & Delinquency Prediction)
CREATE TABLE tax_returns (
    id SERIAL PRIMARY KEY,
    tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
    quarter VARCHAR(10) NOT NULL,     -- Format: Q1-2023, Q2-2023
    revenue NUMERIC(15, 2) NOT NULL DEFAULT 0.0,
    expenses NUMERIC(15, 2) NOT NULL DEFAULT 0.0,
    tax_paid NUMERIC(15, 2) NOT NULL DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'submitted', -- 'submitted', 'late', 'pending_approval'
    filing_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Transactions Data: VAT Invoices (Edges for VAT Graph Analysis)
CREATE TABLE invoices (
    id SERIAL PRIMARY KEY,
    seller_tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
    buyer_tax_code VARCHAR(20) NOT NULL REFERENCES companies(tax_code) ON DELETE CASCADE,
    amount NUMERIC(15, 2) NOT NULL,
    vat_rate NUMERIC(5, 2) DEFAULT 10.0,
    date DATE NOT NULL,
    invoice_number VARCHAR(50) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization: Indexes for fast querying (crucial for Graph NetworkX parsing)
CREATE INDEX idx_seller on invoices (seller_tax_code);
CREATE INDEX idx_buyer on invoices (buyer_tax_code);
CREATE INDEX idx_tax_returns on tax_returns (tax_code);
