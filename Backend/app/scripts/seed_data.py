"""
Synthetic Data Generator for TaxInspector.
Run from Backend root:  python -m app.scripts.seed_data
"""
import random
import uuid
from datetime import date, timedelta

from ..database import SessionLocal, engine, Base
from ..models import Company, TaxReturn, Invoice


def generate_synthetic_data(num_companies: int = 50, num_invoices: int = 200):
    # Recreate tables (WARNING: drops all existing data)
    print("[WARN] Dropping existing tables and recreating schema...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        # --- Companies ---
        print(f"[INFO] Generating {num_companies} companies...")
        companies = []
        industries = ["Ban le", "Xay dung", "Van tai", "Cong nghe", "Bat dong san", "Thuc pham"]
        for i in range(num_companies):
            tax_code = f"{random.randint(10000000, 99999999):010d}"
            c = Company(
                tax_code=tax_code,
                name=f"Cong ty Co phan Mo phong {i + 1}",
                industry=random.choice(industries),
                registration_date=date(2020, 1, 1) + timedelta(days=random.randint(0, 1000)),
                risk_score=round(random.uniform(0.0, 40.0), 2),
            )
            db.add(c)
            companies.append(c)
        db.commit()

        # --- Tax Returns ---
        print("[INFO] Generating tax returns...")
        quarters = ["Q1-2023", "Q2-2023", "Q3-2023", "Q4-2023"]
        for c in companies:
            for q in quarters:
                rev = round(random.uniform(1_000_000, 50_000_000), 2)
                exp = round(rev * random.uniform(0.6, 0.95), 2)
                tax_return = TaxReturn(
                    tax_code=c.tax_code,
                    quarter=q,
                    revenue=rev,
                    expenses=exp,
                    tax_paid=round((rev - exp) * 0.2, 2),
                    filing_date=date(2024, 1, 31),
                )
                db.add(tax_return)
        db.commit()

        # --- Invoices (normal + circular fraud ring) ---
        print(f"[INFO] Generating {num_invoices} invoices (including fraud ring)...")
        for _ in range(num_invoices):
            seller = random.choice(companies)
            buyer = random.choice(companies)
            if seller.tax_code != buyer.tax_code:
                inv = Invoice(
                    seller_tax_code=seller.tax_code,
                    buyer_tax_code=buyer.tax_code,
                    amount=round(random.uniform(10_000, 500_000), 2),
                    date=date.today() - timedelta(days=random.randint(0, 90)),
                    invoice_number=uuid.uuid4().hex[:12],
                )
                db.add(inv)

        # Inject a circular trading ring (3 companies)
        fraud_ring = companies[:3]
        for i in range(len(fraud_ring)):
            seller = fraud_ring[i]
            buyer = fraud_ring[(i + 1) % len(fraud_ring)]
            inv = Invoice(
                seller_tax_code=seller.tax_code,
                buyer_tax_code=buyer.tax_code,
                amount=99_000_000.0,
                date=date.today(),
                invoice_number=uuid.uuid4().hex[:12],
            )
            db.add(inv)

        db.commit()
        print("[OK] Database seeding completed successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    generate_synthetic_data()
