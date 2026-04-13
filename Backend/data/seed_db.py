"""
seed_db.py – Nạp dữ liệu từ tax_data_mock.csv vào PostgreSQL
================================================================
Đọc file CSV 15,000 bản ghi, tự động:
  1. Tạo bảng nếu chưa tồn tại (thông qua SQLAlchemy models)
  2. INSERT 5,000 doanh nghiệp vào bảng `companies`
  3. INSERT dữ liệu tài chính 3 năm vào bảng `tax_returns`

Cách chạy:
    cd e:\\TaxInspector\\Backend
    python data/seed_db.py
"""

import os
import sys
from pathlib import Path
from datetime import date

import pandas as pd

# Add Backend root to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.database import engine, SessionLocal, Base
from app.models import Company, TaxReturn


def main():
    print("=" * 60)
    print("  TaxInspector – Database Seeder")
    print("=" * 60)

    # ---- 1. Kiểm tra và tạo bảng ----
    print("\n[1/4] Kiểm tra và tạo bảng trong PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    print("       Tất cả bảng đã sẵn sàng.")

    # ---- 2. Đọc CSV ----
    csv_path = Path(__file__).resolve().parent / "tax_data_mock.csv"
    if not csv_path.exists():
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        print("       Hãy chạy generate_mock_data.py trước!")
        sys.exit(1)

    print(f"\n[2/4] Đọc dữ liệu từ {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"       Tổng bản ghi: {len(df):,}")
    print(f"       Số DN: {df['tax_code'].nunique():,}")

    # ---- 3. Nạp Companies ----
    print("\n[3/4] Nạp doanh nghiệp vào bảng 'companies'...")
    db = SessionLocal()

    try:
        # Lấy danh sách unique companies (dùng năm mới nhất)
        latest = df.sort_values("year", ascending=False).groupby("tax_code").first().reset_index()

        inserted_companies = 0
        skipped_companies = 0

        for _, row in latest.iterrows():
            tax_code = str(row["tax_code"])

            # Kiểm tra xem đã tồn tại chưa
            existing = db.query(Company).filter(Company.tax_code == tax_code).first()
            if existing:
                skipped_companies += 1
                continue

            company = Company(
                tax_code=tax_code,
                name=str(row.get("company_name", "")),
                industry=str(row.get("industry", "")),
                registration_date=date(2020, 1, 1),  # Ngày đăng ký giả lập
                risk_score=0.0,
                is_active=True,
            )
            db.add(company)
            inserted_companies += 1

            # Commit every 500 records to avoid memory issues
            if inserted_companies % 500 == 0:
                db.commit()
                print(f"       ...đã nạp {inserted_companies:,} doanh nghiệp")

        db.commit()
        print(f"       Hoàn tất: {inserted_companies:,} DN mới, {skipped_companies:,} DN đã tồn tại.")

        # ---- 4. Nạp Tax Returns ----
        print("\n[4/4] Nạp dữ liệu tài chính vào bảng 'tax_returns'...")

        inserted_returns = 0
        skipped_returns = 0

        for _, row in df.iterrows():
            tax_code = str(row["tax_code"])
            year = int(row["year"])

            # Kiểm tra trùng lặp (cùng tax_code + cùng quarter)
            quarter_label = f"Q4-{year}"
            existing_return = (
                db.query(TaxReturn)
                .filter(TaxReturn.tax_code == tax_code, TaxReturn.quarter == quarter_label)
                .first()
            )
            if existing_return:
                skipped_returns += 1
                continue

            tax_return = TaxReturn(
                tax_code=tax_code,
                quarter=quarter_label,
                revenue=float(row.get("revenue", 0)),
                expenses=float(row.get("total_expenses", 0)),
                tax_paid=float(row.get("tax_paid", 0)),
                status="submitted",
                filing_date=date(year, 12, 31),
            )
            db.add(tax_return)
            inserted_returns += 1

            if inserted_returns % 1000 == 0:
                db.commit()
                print(f"       ...đã nạp {inserted_returns:,} bản ghi tài chính")

        db.commit()
        print(f"       Hoàn tất: {inserted_returns:,} bản ghi mới, {skipped_returns:,} bản ghi đã tồn tại.")

    except Exception as e:
        db.rollback()
        print(f"\n[LỖI] {e}")
        raise
    finally:
        db.close()

    print(f"\n{'=' * 60}")
    print(f"  Nạp dữ liệu thành công!")
    print(f"  - {inserted_companies:,} doanh nghiệp")
    print(f"  - {inserted_returns:,} bản ghi tài chính")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
