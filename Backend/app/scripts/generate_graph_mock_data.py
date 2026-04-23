"""
generate_graph_mock_data.py – Sinh dữ liệu giả lập Mạng lưới VAT
===================================================================
Tạo ra:
  - 80 công ty hợp pháp (Legit) + 20 công ty ma (Shell Corp)
  - ~1200 hoá đơn bình thường (1 chiều)
  - ~120 hoá đơn xoay vòng (A->B->C->A trong < 72h)
  - Tọa độ PostGIS ngẫu nhiên tại Việt Nam (Hà Nội / TP.HCM / Đà Nẵng)

Sử dụng: python -m app.scripts.generate_graph_mock_data
"""

import os, sys, random, math
from pathlib import Path
from datetime import date, timedelta, datetime

# ── Setup paths ──
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import psycopg2

# ── DB Connection ──
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "TaxInspector")

# ── Configuration ──
NUM_LEGIT_COMPANIES = 8500
NUM_SHELL_COMPANIES = 1500
NUM_NORMAL_INVOICES = 50000
NUM_HARD_NEGATIVE_PAIRS = 500    # High-volume legit reciprocal trade pairs
INVOICES_PER_HARD_PAIR = 10      # Invoices per direction
NUM_CIRCULAR_RINGS = 300         # Each ring: 3-5 companies forming a loop
INVOICES_PER_RING_LEG = 5        # Invoices per direction in each ring

# Timeline: Jan 2024 – Dec 2024 (12 months for temporal train/val/test split)
DATA_YEAR = 2024
DATA_START = date(DATA_YEAR, 1, 1)
DATA_END = date(DATA_YEAR, 12, 31)

random.seed(42)

# ── Geographic clusters (Vietnam + Offshore for Forensic Simulation) ──
GEO_CLUSTERS = {
    # Vietnam Domestic (90%)
    "Hà Nội":   {"lat": 21.0285, "lng": 105.8542, "radius_km": 15, "country": "Vietnam"},
    "TP.HCM":   {"lat": 10.8231, "lng": 106.6297, "radius_km": 20, "country": "Vietnam"},
    "Đà Nẵng":  {"lat": 16.0544, "lng": 108.2022, "radius_km": 10, "country": "Vietnam"},
    "Hải Phòng": {"lat": 20.8449, "lng": 106.6881, "radius_km": 8, "country": "Vietnam"},
    "Cần Thơ":  {"lat": 10.0452, "lng": 105.7469, "radius_km": 7, "country": "Vietnam"},
    
    # Offshore Tax-Havens / Holding Jurisdictions (10%)
    "Singapore": {"lat": 1.3521, "lng": 103.8198, "radius_km": 5, "country": "Singapore"},
    "Cayman Islands": {"lat": 19.3133, "lng": -81.2546, "radius_km": 5, "country": "Cayman Islands"},
    "British Virgin Islands": {"lat": 18.4207, "lng": -64.6400, "radius_km": 5, "country": "British Virgin Islands"},
}

INDUSTRIES = [
    "Sản xuất", "Thương mại", "Xây dựng", "Vận tải & Logistics",
    "Công nghệ thông tin", "Thực phẩm & Đồ uống", "Bất động sản",
    "Dệt may", "Nông nghiệp", "Dịch vụ tài chính",
]

LEGIT_COMPANY_PREFIXES = [
    "TNHH", "Công ty CP", "CTY TNHH", "DN Tư Nhân", "Tập đoàn",
]

SHELL_COMPANY_PREFIXES = [
    "CTY TNHH MTV", "DNTN", "Hộ KD", "CTY TNHH",
]

LEGIT_NAMES = [
    "Minh Phát", "Hưng Thịnh", "Việt Tiến", "Đại Nam", "Phúc Lộc",
    "An Khang", "Thái Bình", "Hoàng Gia", "Tiến Đạt", "Nam Á",
    "Quang Huy", "Đông Á", "Tân Long", "Bình Minh", "Phương Đông",
    "Kỹ thuật Sài Gòn", "Thiên Phúc", "Minh Quang", "Đức Thắng", "Trường Phát",
    "Đại Việt", "Toàn Phát", "Phú Quý", "Solar Tech", "Green Farm",
    "Smart Logistics", "VinaTrade", "NovaBuild", "Golden Star", "Blue Ocean",
    "TechVina", "AgriPro", "MekongFish", "CityMart", "QuickShip",
    "SteelMax", "TopGear", "EcoPlast", "PowerGrid", "FreshFood",
    "HanoiTex", "SaigonPrint", "DeltaChem", "AlphaSteel", "BetaCon",
    "GammaTrans", "OmegaTech", "SigmaFin", "ThetaAgri", "KappaMart",
    "DigiServe", "CyberTrade", "NeoFarm", "StarShip", "PrimeBuild",
    "SwiftCargo", "IronWorks", "ClearView", "SunGrow", "AquaPure",
    "UniTex", "MegaStore", "ProFab", "CoreSteel", "NextGen",
    "VietBuild", "HanoiParts", "SaigonAuto", "MekongRice", "CoastalFish",
    "Highland Tea", "CentralWood", "SouthSteel", "NorthFarm", "EastTrade",
    "WestLog", "PeakPower", "BaseChem", "RiverMill", "LakeView",
]

SHELL_NAMES = [
    "An Phú Holdings", "Thành Đạt TM", "Vạn Lợi DV", "Kim Ngọc Invest",
    "ĐL Phát Tài", "Huy Hoàng Corp", "TM Quốc Bảo", "Golden Lotus TM",
    "Tín Phát DV", "Minh Nguyên TM", "Bách Thắng DV", "Thịnh Vượng Corp",
    "Lucky Star TM", "Đức Tài DV", "Vĩnh Phúc Holdings", "Gia Bảo Invest",
    "Tân Thịnh TM", "Long Phú DV", "Kim Thành TM", "An Lộc Corp",
]


def random_point_near(lat: float, lng: float, radius_km: float):
    """Generate a random point within radius_km of (lat, lng)."""
    r = radius_km / 111.32  # rough degrees
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0, r)
    return (
        round(lat + dist * math.cos(angle), 6),
        round(lng + dist * math.sin(angle), 6),
    )


def generate_tax_code(index: int, is_shell: bool = False) -> str:
    if is_shell:
        prefix = "99"
    else:
        prefix = random.choice(["01", "03", "04", "06", "08", "31", "79", "48"])
    mid = f"{index + 100000:06d}"
    suffix = str(random.randint(10, 99))
    return f"{prefix}{mid}{suffix}"


def generate_companies(conn):
    """Insert legit + shell companies with PostGIS geometry."""
    cur = conn.cursor()
    companies = []

    # ── Legit companies ──
    for i in range(NUM_LEGIT_COMPANIES):
        tax_code = generate_tax_code(i + 1)
        cluster_name = random.choice(list(GEO_CLUSTERS.keys()))
        cluster = GEO_CLUSTERS[cluster_name]
        lat, lng = random_point_near(cluster["lat"], cluster["lng"], cluster["radius_km"])
        name = f"{random.choice(LEGIT_COMPANY_PREFIXES)} {LEGIT_NAMES[i % len(LEGIT_NAMES)]}"
        industry = random.choice(INDUSTRIES)
        # Registration date: 2-15 years ago
        reg_date = date.today() - timedelta(days=random.randint(730, 5475))
        companies.append({
            "tax_code": tax_code, "name": name, "industry": industry,
            "registration_date": reg_date, "risk_score": round(random.uniform(0, 35), 1),
            "is_active": True, "lat": lat, "lng": lng,
            "is_shell": False,
        })

    # ── Shell companies (suspicious) ──
    for i in range(NUM_SHELL_COMPANIES):
        tax_code = generate_tax_code(900 + i, is_shell=True)
        # Shell companies often cluster near each other or share addresses
        cluster_name = random.choice(["Hà Nội", "TP.HCM"])
        cluster = GEO_CLUSTERS[cluster_name]
        lat, lng = random_point_near(cluster["lat"], cluster["lng"], 2)  # very tight cluster
        name = f"{random.choice(SHELL_COMPANY_PREFIXES)} {SHELL_NAMES[i % len(SHELL_NAMES)]}"
        industry = random.choice(["Thương mại", "Dịch vụ tài chính", "Xây dựng"])
        # Recently registered (< 1 year)
        reg_date = date.today() - timedelta(days=random.randint(30, 365))
        companies.append({
            "tax_code": tax_code, "name": name, "industry": industry,
            "registration_date": reg_date, "risk_score": round(random.uniform(60, 95), 1),
            "is_active": True, "lat": lat, "lng": lng,
            "is_shell": True,
        })

    # Insert into DB using executemany with PostGIS ST_MakePoint
    cur.executemany("""
        INSERT INTO companies (tax_code, name, industry, registration_date, risk_score, is_active, geom)
        VALUES (%(tax_code)s, %(name)s, %(industry)s, %(registration_date)s, %(risk_score)s, %(is_active)s,
                ST_SetSRID(ST_MakePoint(%(lng)s, %(lat)s), 4326))
        ON CONFLICT (tax_code) DO UPDATE SET
            name = EXCLUDED.name,
            industry = EXCLUDED.industry,
            registration_date = EXCLUDED.registration_date,
            risk_score = EXCLUDED.risk_score,
            geom = EXCLUDED.geom
    """, companies)

    conn.commit()
    print(f"[OK] Inserted {len(companies)} companies ({NUM_LEGIT_COMPANIES} legit + {NUM_SHELL_COMPANIES} shell)")
    return companies


def generate_normal_invoices(conn, companies):
    """Generate one-way normal invoices between legit companies."""
    cur = conn.cursor()
    legit = [c for c in companies if not c["is_shell"]]
    invoices = []

    # Spread normal invoices across full 12-month timeline (Jan–Dec 2024)
    total_days = (DATA_END - DATA_START).days
    for i in range(NUM_NORMAL_INVOICES):
        seller = random.choice(legit)
        buyer = random.choice(legit)
        while buyer["tax_code"] == seller["tax_code"]:
            buyer = random.choice(legit)

        amount = round(random.uniform(5_000_000, 2_000_000_000), 2)  # 5M - 2B VND
        vat_rate = random.choice([8.0, 10.0])
        inv_date = DATA_START + timedelta(days=random.randint(0, total_days))
        inv_number = f"INV-{inv_date.strftime('%Y%m')}-{i+1:05d}"

        invoices.append({
            "seller_tax_code": seller["tax_code"],
            "buyer_tax_code": buyer["tax_code"],
            "amount": amount,
            "vat_rate": vat_rate,
            "date": inv_date,
            "invoice_number": inv_number,
        })

    cur.executemany("""
        INSERT INTO invoices (seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number)
        VALUES (%(seller_tax_code)s, %(buyer_tax_code)s, %(amount)s, %(vat_rate)s, %(date)s, %(invoice_number)s)
        ON CONFLICT (invoice_number) DO NOTHING
    """, invoices)
    conn.commit()
    print(f"[OK] Inserted {len(invoices)} normal invoices (Jan–Dec {DATA_YEAR})")
    return invoices


def generate_circular_invoices(conn, companies):
    """
    Generate circular invoice rings: A -> B -> C -> ... -> A
    Each ring uses a mix of shell + legit companies.
    Invoices in a ring are dated within 48-72 hours of each other.
    """
    cur = conn.cursor()
    shells = [c for c in companies if c["is_shell"]]
    legit = [c for c in companies if not c["is_shell"]]
    circular_invoices = []
    ring_info = []

    for ring_idx in range(NUM_CIRCULAR_RINGS):
        ring_size = random.randint(3, 5)
        # Pick 1-2 legit + rest shell
        num_legit_in_ring = random.randint(1, min(2, ring_size - 1))
        ring_members = (
            random.sample(shells, min(ring_size - num_legit_in_ring, len(shells)))
            + random.sample(legit, num_legit_in_ring)
        )
        random.shuffle(ring_members)

        # Base date for this ring: T7–T12 2024 (Jul–Dec) so they fall in val/test period
        base_date = date(DATA_YEAR, 7, 1) + timedelta(days=random.randint(0, 180))
        # Base amount for the ring (large, suspicious amounts)
        base_amount = round(random.uniform(500_000_000, 5_000_000_000), 2)

        ring_tax_codes = [m["tax_code"] for m in ring_members]
        ring_info.append(ring_tax_codes)

        for leg in range(ring_size):
            seller = ring_members[leg]
            buyer = ring_members[(leg + 1) % ring_size]  # circular

            for inv_idx in range(INVOICES_PER_RING_LEG):
                # Slightly vary amount to look realistic
                amount = round(base_amount * random.uniform(0.85, 1.15), 2)
                # Invoices within 48h of each other
                hours_offset = leg * random.randint(12, 36) + inv_idx * random.randint(1, 6)
                inv_date = base_date + timedelta(hours=hours_offset)
                inv_number = f"CIRC-R{ring_idx+1:02d}-L{leg+1}-{inv_idx+1:03d}"

                circular_invoices.append({
                    "seller_tax_code": seller["tax_code"],
                    "buyer_tax_code": buyer["tax_code"],
                    "amount": amount,
                    "vat_rate": 10.0,
                    "date": inv_date.date() if isinstance(inv_date, datetime) else inv_date,
                    "invoice_number": inv_number,
                })

    cur.executemany("""
        INSERT INTO invoices (seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number)
        VALUES (%(seller_tax_code)s, %(buyer_tax_code)s, %(amount)s, %(vat_rate)s, %(date)s, %(invoice_number)s)
        ON CONFLICT (invoice_number) DO NOTHING
    """, circular_invoices)
    conn.commit()
    print(f"[OK] Inserted {len(circular_invoices)} circular invoices across {NUM_CIRCULAR_RINGS} rings")
    print(f"    Rings: {ring_info}")
    return circular_invoices, ring_info


def generate_hard_negatives(conn, companies):
    """
    Generate hard negative examples: large, reciprocal trade pairs between
    LEGIT companies that look suspicious (high volume, short intervals)
    but are entirely legitimate.

    This forces the model to distinguish between:
    - Real circular fraud (shell companies, value wash)
    - Legitimate high-frequency B2B trading (supply chain partners)
    """
    cur = conn.cursor()
    legit = [c for c in companies if not c["is_shell"]]
    hard_neg_invoices = []

    # Select diverse industry pairs for realism
    industry_pairs = [
        ("Sản xuất", "Thương mại"),       # manufacturer <-> distributor
        ("Nông nghiệp", "Thực phẩm & Đồ uống"),  # farmer <-> food processor
        ("Xây dựng", "Sản xuất"),         # builder <-> materials supplier
        ("Vận tải & Logistics", "Thương mại"),  # shipper <-> trader
        ("Công nghệ thông tin", "Dịch vụ tài chính"),  # IT provider <-> bank
    ]

    pairs_created = 0
    for pair_idx in range(NUM_HARD_NEGATIVE_PAIRS):
        # Pick two legit companies from matching industries
        ind_a, ind_b = industry_pairs[pair_idx % len(industry_pairs)]
        candidates_a = [c for c in legit if c["industry"] == ind_a]
        candidates_b = [c for c in legit if c["industry"] == ind_b]

        if not candidates_a or not candidates_b:
            candidates_a = legit
            candidates_b = legit

        comp_a = random.choice(candidates_a)
        comp_b = random.choice(candidates_b)
        while comp_b["tax_code"] == comp_a["tax_code"]:
            comp_b = random.choice(candidates_b if candidates_b else legit)

        # Generate bidirectional invoices (A->B and B->A) over entire timeline
        for direction in range(2):
            seller = comp_a if direction == 0 else comp_b
            buyer = comp_b if direction == 0 else comp_a

            for inv_idx in range(INVOICES_PER_HARD_PAIR):
                # Large amounts (similar to fraud rings) but legitimate
                amount = round(random.uniform(200_000_000, 3_000_000_000), 2)
                # Spread across full year, some bursts within 48h like fraud
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                inv_date = date(DATA_YEAR, month, day)
                inv_number = f"HNEG-P{pair_idx+1:02d}-D{direction}-{inv_idx+1:03d}"

                hard_neg_invoices.append({
                    "seller_tax_code": seller["tax_code"],
                    "buyer_tax_code": buyer["tax_code"],
                    "amount": amount,
                    "vat_rate": random.choice([8.0, 10.0]),
                    "date": inv_date,
                    "invoice_number": inv_number,
                })

        pairs_created += 1

    if hard_neg_invoices:
        cur.executemany("""
            INSERT INTO invoices (seller_tax_code, buyer_tax_code, amount, vat_rate, date, invoice_number)
            VALUES (%(seller_tax_code)s, %(buyer_tax_code)s, %(amount)s, %(vat_rate)s, %(date)s, %(invoice_number)s)
            ON CONFLICT (invoice_number) DO NOTHING
        """, hard_neg_invoices)
        conn.commit()

    print(f"[OK] Inserted {len(hard_neg_invoices)} hard negative invoices ({pairs_created} reciprocal pairs)")
    return len(hard_neg_invoices)


def main():
    print("=" * 60)
    print("  TaxInspector – Graph Mock Data Generator")
    print("=" * 60)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        dbname=DB_NAME
    )
    try:
        # Enable PostGIS if not already
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.commit()
        print("[OK] PostGIS extension enabled")

        # Add geom column if missing
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'companies' AND column_name = 'geom'
                ) THEN
                    ALTER TABLE companies ADD COLUMN geom geometry(Point, 4326);
                END IF;
            END $$;
        """)
        conn.commit()
        print("[OK] Geom column verified")

        companies = generate_companies(conn)
        generate_normal_invoices(conn, companies)
        circular_invs, rings = generate_circular_invoices(conn, companies)
        hard_neg_count = generate_hard_negatives(conn, companies)

        # Summary
        cur.execute("SELECT COUNT(*) FROM companies;")
        total_companies = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM invoices;")
        total_invoices = cur.fetchone()[0]

        print(f"\n{'='*60}")
        print(f"  TỔNG KẾT: {total_companies} công ty | {total_invoices} hoá đơn")
        print(f"  Fraud Rings: {len(rings)}")
        print(f"  Hard Negatives: {hard_neg_count} invoices")
        print(f"  Timeline: {DATA_START} → {DATA_END}")
        print(f"{'='*60}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
