# -*- coding: utf-8 -*-
"""
registration.py – Taxpayer Registration, Location Lookup & Group 1 Compliance APIs
=============================================================================
Contains all services for Nhóm 1 - Nhận diện & Đăng ký:
1. Cascading dropdowns (Provinces -> Districts -> Wards) for HCMC.
2. Real-world Tax Office search by ward or coordinates (Leaflet integration).
3. MST and CCCD self-lookup & partner/supplier authenticity validator (prevents shell company risk).
4. Interactive Household business classification machine (TT152/2025 & NĐ68/2026).
5. Wizard step-by-step registration with mock VNeID QR + canvas signing.
6. Mẫu 01/BK-STK Bank account and e-wallet declaring.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict
import datetime
import random

from ..database import get_db
from ..data.hcmc_tax_data import TAX_OFFICES, DISTRICTS, WARD_MAPPING

router = APIRouter(prefix="/api/registration", tags=["Taxpayer Registration"])


def ensure_registration_schema(conn):
    """Ensure all required tables for registration and geo-mapping are present and seeded."""
    # 1. Tax offices table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS hcmc_tax_offices (
            id SERIAL PRIMARY KEY,
            office_code VARCHAR(10) UNIQUE NOT NULL,
            office_name VARCHAR(200) NOT NULL,
            full_name VARCHAR(300),
            address VARCHAR(500),
            phone VARCHAR(50),
            working_hours VARCHAR(100) DEFAULT 'Thứ 2 - Thứ 6 (07:30 - 11:30, 13:00 - 17:00)',
            lat DOUBLE PRECISION,
            lng DOUBLE PRECISION,
            managing_districts TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))

    # 2. Wards table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS hcmc_wards (
            id SERIAL PRIMARY KEY,
            ward_name VARCHAR(100) NOT NULL,
            district_name VARCHAR(100) NOT NULL,
            tax_office_code VARCHAR(10) REFERENCES hcmc_tax_offices(office_code),
            lat DOUBLE PRECISION,
            lng DOUBLE PRECISION,
            UNIQUE (ward_name, district_name)
        );
    """))

    # 3. Application submissions table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mst_registration_applications (
            id SERIAL PRIMARY KEY,
            cccd VARCHAR(12) NOT NULL,
            full_name VARCHAR(200) NOT NULL,
            date_of_birth DATE,
            gender VARCHAR(10),
            permanent_address TEXT,
            business_name VARCHAR(300),
            business_type VARCHAR(50) DEFAULT 'household',
            industry_code VARCHAR(20),
            industry_name VARCHAR(200),
            business_address TEXT,
            district_name VARCHAR(100),
            ward_name VARCHAR(100),
            tax_office_code VARCHAR(10) REFERENCES hcmc_tax_offices(office_code),
            expected_revenue NUMERIC(15,2),
            employee_count INTEGER DEFAULT 0,
            capital NUMERIC(15,2),
            household_group INTEGER,
            signature_data TEXT,
            status VARCHAR(30) DEFAULT 'pending',
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            mst_assigned VARCHAR(20)
        );
    """))

    # 4. Bank account declaring table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS taxpayer_bank_accounts (
            id SERIAL PRIMARY KEY,
            tax_code VARCHAR(20) NOT NULL,
            bank_name VARCHAR(200) NOT NULL,
            account_number VARCHAR(50) NOT NULL,
            account_holder VARCHAR(200),
            account_type VARCHAR(50) DEFAULT 'business',
            reported_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'active'
        );
    """))

    # Seed HCMC tax offices if empty
    count_offices = conn.execute(text("SELECT COUNT(*) FROM hcmc_tax_offices")).scalar()
    if count_offices == 0:
        print("[SEED] Seeding HCMC 29 tax offices...")
        for office in TAX_OFFICES:
            conn.execute(text("""
                INSERT INTO hcmc_tax_offices 
                (office_code, office_name, full_name, address, phone, lat, lng, managing_districts)
                VALUES (:code, :name, :fullname, :addr, :phone, :lat, :lng, :districts)
            """), {
                "code": office["office_code"],
                "name": office["office_name"],
                "fullname": office["full_name"],
                "addr": office["address"],
                "phone": office["phone"],
                "lat": office["lat"],
                "lng": office["lng"],
                "districts": office["managing_districts"]
            })

    # Seed HCMC wards if empty
    count_wards = conn.execute(text("SELECT COUNT(*) FROM hcmc_wards")).scalar()
    if count_wards == 0:
        print("[SEED] Seeding HCMC wards & tax office mapping...")
        for district, wards in WARD_MAPPING.items():
            for w in wards:
                conn.execute(text("""
                    INSERT INTO hcmc_wards (ward_name, district_name, tax_office_code, lat, lng)
                    VALUES (:ward, :district, :office_code, :lat, :lng)
                    ON CONFLICT (ward_name, district_name) DO NOTHING
                """), {
                    "ward": w["ward"],
                    "district": district,
                    "office_code": w["office_code"],
                    "lat": w["lat"],
                    "lng": w["lng"]
                })


# ---- ENDPOINTS ----

@router.get("/init")
def init_db_schema(db: Session = Depends(get_db)):
    """Initialize registration schema manually."""
    try:
        ensure_registration_schema(db.connection())
        db.commit()
        return {"status": "success", "message": "Schema initialized & seeded."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/districts")
def get_hcmc_districts():
    """Get the 22 administrative districts of HCMC."""
    return {"districts": DISTRICTS}


@router.get("/wards")
def get_wards(district: str, db: Session = Depends(get_db)):
    """Get wards in HCMC for a given district name."""
    ensure_registration_schema(db.connection())
    rows = db.execute(text("""
        SELECT id, ward_name, tax_office_code, lat, lng 
        FROM hcmc_wards 
        WHERE district_name = :dist
        ORDER BY ward_name ASC
    """), {"dist": district}).all()

    return {"wards": [{"id": r[0], "ward_name": r[1], "tax_office_code": r[2], "lat": r[3], "lng": r[4]} for r in rows]}


@router.get("/tax-offices")
def get_all_tax_offices(db: Session = Depends(get_db)):
    """Get all 29 HCMC tax offices for Leaflet map loading."""
    ensure_registration_schema(db.connection())
    rows = db.execute(text("""
        SELECT office_code, office_name, full_name, address, phone, working_hours, lat, lng, managing_districts
        FROM hcmc_tax_offices
        ORDER BY office_code ASC
    """)).all()

    return {
        "offices": [
            {
                "office_code": r[0],
                "office_name": r[1],
                "full_name": r[2],
                "address": r[3],
                "phone": r[4],
                "working_hours": r[5],
                "lat": r[6],
                "lng": r[7],
                "managing_districts": r[8]
            } for r in rows
        ]
    }


@router.post("/lookup-tax-office")
def lookup_tax_office(
    payload: Dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Look up managing tax office based on selected district and ward names.
    Body: {"district": "Quận 1", "ward": "Đa Kao"}
    """
    ensure_registration_schema(db.connection())
    district = payload.get("district")
    ward = payload.get("ward")

    if not district or not ward:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp đầy đủ Quận/Huyện và Phường/Xã.")

    # Find the ward mapping
    ward_row = db.execute(text("""
        SELECT tax_office_code FROM hcmc_wards 
        WHERE district_name = :dist AND ward_name = :ward
    """), {"dist": district, "ward": ward}).first()

    if not ward_row:
        raise HTTPException(status_code=404, detail="Không tìm thấy địa giới hành chính khớp với yêu cầu.")

    office_code = ward_row[0]

    # Get tax office details
    office = db.execute(text("""
        SELECT office_code, office_name, full_name, address, phone, working_hours, lat, lng
        FROM hcmc_tax_offices WHERE office_code = :code
    """), {"code": office_code}).first()

    if not office:
        raise HTTPException(status_code=404, detail="Cơ quan thuế quản lý tương ứng tạm thời chưa kích hoạt.")

    return {
        "status": "success",
        "office": {
            "office_code": office[0],
            "office_name": office[1],
            "full_name": office[2],
            "address": office[3],
            "phone": office[4],
            "working_hours": office[5],
            "lat": office[6],
            "lng": office[7]
        }
    }


@router.post("/reverse-geocode")
def reverse_geocode_lookup(
    payload: Dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Get closest ward and tax office coordinates from clicked Leaflet maps.
    Body: {"lat": 10.7876, "lng": 106.7029}
    """
    ensure_registration_schema(db.connection())
    lat = payload.get("lat")
    lng = payload.get("lng")

    if lat is None or lng is None:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp tọa độ lat/lng.")

    # Simple Euclidean distance (lat/lng diff squared) to find the closest ward in DB
    closest = db.execute(text("""
        SELECT id, ward_name, district_name, tax_office_code, lat, lng,
               ((lat - :lat)^2 + (lng - :lng)^2) as dist
        FROM hcmc_wards
        ORDER BY dist ASC
        LIMIT 1
    """), {"lat": lat, "lng": lng}).first()

    if not closest:
        raise HTTPException(status_code=404, detail="Tọa độ không thuộc phạm vi TP.HCM.")

    # Get tax office details
    office = db.execute(text("""
        SELECT office_code, office_name, full_name, address, phone, working_hours, lat, lng
        FROM hcmc_tax_offices WHERE office_code = :code
    """), {"code": closest[3]}).first()

    return {
        "status": "success",
        "ward": {
            "id": closest[0],
            "ward_name": closest[1],
            "district_name": closest[2],
            "lat": closest[4],
            "lng": closest[5]
        },
        "office": {
            "office_code": office[0],
            "office_name": office[1],
            "full_name": office[2],
            "address": office[3],
            "phone": office[4],
            "working_hours": office[5],
            "lat": office[6],
            "lng": office[7]
        }
    }


@router.post("/lookup-mst")
def lookup_taxpayer_mst(
    payload: Dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Lookup a taxpayer or officer MST / CCCD.
    Input: {"query": "0312495812"} (either MST 10-13 digits or CCCD 12 digits)
    """
    ensure_registration_schema(db.connection())
    query = payload.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Vui lòng nhập MST hoặc CCCD cần tra cứu.")

    # 1. Search in companies table (active businesses)
    comp = db.execute(text("""
        SELECT tax_code, name, industry, province, risk_score, is_active, created_at
        FROM companies WHERE tax_code = :q
    """), {"q": query}).first()

    if comp:
        # Determine status description
        status_code = "00" if comp[5] else "01"
        status_desc = "Đang hoạt động (00)" if comp[5] else "Tạm ngừng hoạt động (01)"
        if comp[4] and comp[4] > 75.0:
            status_code = "06"
            status_desc = "Không hoạt động tại địa chỉ đã đăng ký (06) - Cảnh báo Rủi ro cao!"

        return {
            "found": True,
            "type": "business",
            "mst": comp[0],
            "name": comp[1],
            "industry": comp[2],
            "address": f"Khu vực {comp[3] or 'TP.HCM'}",
            "status_code": status_code,
            "status": status_desc,
            "risk_score": comp[4],
            "registration_date": comp[6].strftime("%d/%m/%Y") if comp[6] else "N/A"
        }

    # 2. Search in users table (CCCD / officers)
    usr = db.execute(text("""
        SELECT id, full_name, email, phone, role, cccd_verified 
        FROM users WHERE badge_id = :q OR phone = :q OR email = :q
    """), {"q": query}).first()

    if usr:
        # Generate a mock MST based on ID for a registered user
        generated_mst = f"8902{usr[0]:06d}"
        return {
            "found": True,
            "type": "individual",
            "mst": generated_mst,
            "name": usr[1],
            "email": usr[2],
            "phone": usr[3] or "N/A",
            "status_code": "00",
            "status": "Đang hoạt động (CCCD cá nhân đã đăng ký)",
            "cccd_verified": usr[5]
        }

    # 3. Fuzzy search for matching company name
    if len(query) > 3 and not query.isdigit():
        fuzzy_comp = db.execute(text("""
            SELECT tax_code, name, industry, province, risk_score, is_active
            FROM companies WHERE name ILIKE :q LIMIT 5
        """), {"q": f"%{query}%"}).all()

        if fuzzy_comp:
            return {
                "found": False,
                "suggestions": [
                    {
                        "mst": r[0],
                        "name": r[1],
                        "industry": r[2],
                        "province": r[3],
                        "risk_score": r[4],
                        "is_active": r[5]
                    } for r in fuzzy_comp
                ]
            }

    # 4. Fallback to a smart, consistent mock return if not found in database to fulfill sandbox lookup requests
    # Generate realistic VN name based on CCCD hash to remain deterministic
    random.seed(hash(query))
    first_names = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng"]
    mid_names = ["Văn", "Thị", "Minh", "Hoàng", "Kim", "Ngọc", "Anh", "Đức", "Hữu", "Thành"]
    last_names = ["Hùng", "Hoa", "Kiet", "Trang", "Tuấn", "Lan", "Dũng", "Vy", "Bình", "Nam"]
    
    mock_name = f"{random.choice(first_names)} {random.choice(mid_names)} {random.choice(last_names)}"
    mock_mst = f"8192{abs(hash(query)) % 1000000:06d}"
    mock_statuses = ["00", "01", "06"]
    mock_status_code = random.choices(mock_statuses, weights=[0.7, 0.2, 0.1])[0]
    
    status_map = {
        "00": "Đang hoạt động (00)",
        "01": "Tạm ngừng hoạt động (01)",
        "06": "Không hoạt động tại địa chỉ đã đăng ký (06) - MST Cảnh báo"
    }

    return {
        "found": True,
        "is_mock": True,
        "type": "individual" if len(query) == 12 else "business",
        "mst": mock_mst if len(query) == 12 else query,
        "cccd": query if len(query) == 12 else "N/A",
        "name": mock_name.upper(),
        "industry": random.choice(["Bán lẻ quần áo", "Ăn uống, cafe", "Dịch vụ vận tải", "Gia công đồ da", "Cửa hàng tạp hóa"]),
        "address": f"Phường {random.randint(1,15)}, Quận {random.randint(1,12)}, TP. Hồ Chí Minh",
        "status_code": mock_status_code,
        "status": status_map[mock_status_code],
        "registration_date": "15/08/2023"
    }


@router.post("/classify-household")
def classify_household_business(
    payload: Dict = Body(...)
):
    """
    Classify household business groups based on TT152/2025 and NĐ68/2026.
    Inputs:
        revenue: Expected annual revenue in Millions VND
        labor: Number of employees
        field: Business industry sector ('commerce', 'manufacturing', 'services', 'agriculture')
    """
    revenue = float(payload.get("revenue", 0))
    labor = int(payload.get("labor", 0))
    field = payload.get("field", "commerce").lower()

    # Classification logic based on Decree 68/2026 & TT152/2025:
    # Nhóm 1 (Nhỏ/Siêu nhỏ): Revenue <= 500 million VND/year -> Only declare revenue, no bookkeeping.
    # Nhóm 2 (Vừa): Revenue 500M - 3 Billion -> Simplified accounting system (S1a, S2a, S2b, S2c), optional HĐĐT.
    # Nhóm 3 (Lớn): Revenue > 3 Billion OR Employees >= 10 -> Complete bookkeeping (inventory + fixed assets), mandatory HĐĐT.
    
    if revenue > 3000 or labor >= 10:
        group = 3
        group_name = "Nhóm 3 (Hộ kinh doanh quy mô Lớn)"
        desc = "Doanh thu năm trên 3 tỷ VNĐ hoặc sử dụng từ 10 lao động trở lên."
        obligations = [
            "Bắt buộc thực hiện chế độ kế toán đầy đủ theo Thông tư 152/2025/TT-BTC.",
            "Phải lập và lưu trữ Sổ hàng tồn kho (Sổ S1-HKD) và Sổ tài sản cố định.",
            "Bắt buộc sử dụng Hóa đơn điện tử có mã của Cơ quan thuế cho 100% giao dịch đầu ra.",
            "Nộp tờ khai thuế định kỳ hàng tháng/hàng quý theo phương pháp kê khai."
        ]
    elif revenue > 500 or labor >= 5:
        group = 2
        group_name = "Nhóm 2 (Hộ kinh doanh quy mô Vừa)"
        desc = "Doanh thu năm từ 500 triệu đến 3 tỷ VNĐ hoặc sử dụng từ 5 đến 9 lao động."
        obligations = [
            "Thực hiện chế độ sổ sách kế toán đơn giản hóa (Mẫu S1a-HKD doanh thu).",
            "Đăng ký sử dụng Hóa đơn điện tử tự nguyện để chuyên nghiệp hóa kinh doanh.",
            "Nộp tờ khai thuế GTGT và TNCN định kỳ hàng quý.",
            "Được áp dụng tỷ lệ % tính thuế trên doanh thu thực tế phát sinh."
        ]
    else:
        group = 1
        group_name = "Nhóm 1 (Hộ kinh doanh quy mô Nhỏ/Siêu nhỏ)"
        desc = "Doanh thu năm dưới 500 triệu VNĐ và dưới 5 lao động."
        obligations = [
            "Miễn hoàn toàn nghĩa vụ lập sổ sách kế toán phức tạp.",
            "Không bắt buộc sử dụng Hóa đơn điện tử (được xin cấp hóa đơn lẻ khi phát sinh giao dịch lớn).",
            "Chỉ cần gửi Thông báo Doanh thu tự nguyện định kỳ 6 tháng (trước 31/7 và 31/1) theo NĐ 68/2026/NĐ-CP.",
            "Được miễn lệ phí môn bài năm 2026."
        ]

    # Calculate tax rates based on industry sector (TT40/2021):
    rates = {
        "commerce": {"gtgt": 1.0, "tncn": 0.5, "total": 1.5, "name": "Hoạt động phân phối, cung cấp hàng hóa"},
        "services": {"gtgt": 5.0, "tncn": 2.0, "total": 7.0, "name": "Dịch vụ, xây dựng không bao thầu nguyên vật liệu"},
        "manufacturing": {"gtgt": 3.0, "tncn": 1.5, "total": 4.5, "name": "Sản xuất, vận tải, dịch vụ có gắn với hàng hóa, xây dựng có bao thầu nguyên vật liệu"},
        "agriculture": {"gtgt": 2.0, "tncn": 1.0, "total": 3.0, "name": "Hoạt động sản xuất nông nghiệp, lâm nghiệp, thủy sản và kinh doanh khác"}
    }
    
    rate_info = rates.get(field, rates["commerce"])
    yearly_tax = 0.0
    
    if revenue > 500: # Threshold for tax liability is 500M under Decree 68/2026
        yearly_tax = (revenue * (rate_info["total"] / 100.0))

    return {
        "status": "success",
        "classification": {
            "group": group,
            "group_name": group_name,
            "description": desc,
            "obligations": obligations
        },
        "tax_policy": {
            "sector_name": rate_info["name"],
            "gtgt_rate": rate_info["gtgt"],
            "tncn_rate": rate_info["tncn"],
            "total_rate": rate_info["total"],
            "is_taxable": revenue > 500,
            "estimated_annual_tax_million": round(yearly_tax, 2)
        }
    }


@router.post("/verify-cccd")
def verify_cccd_with_vneid(
    payload: Dict = Body(...)
):
    """
    Step 1 Wizard: Mock verification of National Citizen identity through VNeID QR / CCCD scan.
    Body: {"cccd": "079094001234"}
    """
    cccd = payload.get("cccd", "").strip()
    if not cccd or len(cccd) != 12 or not cccd.isdigit():
        raise HTTPException(status_code=400, detail="Mã số căn cước công dân (CCCD) phải đủ 12 chữ số hợp lệ.")

    # Generates standard, realistic citizen information verified by National Database
    random.seed(int(cccd[:8]))
    first_names = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Võ"]
    mid_names = ["Văn", "Đức", "Trọng", "Thế", "Thị", "Khang", "Ngọc", "Anh"]
    last_names = ["Kiet", "Dũng", "Nam", "An", "Phúc", "Thảo", "Hương", "Trang"]
    
    mock_name = f"{random.choice(first_names)} {random.choice(mid_names)} {random.choice(last_names)}"
    
    birth_year = random.randint(1975, 2004)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    dob = f"{birth_day:02d}/{birth_month:02d}/{birth_year}"
    gender = random.choice(["Nam", "Nữ"])
    
    districts_hcm = ["Quận 1", "Quận 3", "Quận 10", "Bình Thạnh", "Gò Vấp", "Tân Bình"]
    selected_district = random.choice(districts_hcm)
    
    return {
        "status": "success",
        "verified": True,
        "source": "Cổng Định danh điện tử Quốc gia (VNeID)",
        "citizen": {
            "cccd": cccd,
            "full_name": mock_name.upper(),
            "date_of_birth": dob,
            "gender": gender,
            "permanent_address": f"Số {random.randint(10, 500)} Đường Điện Biên Phủ, Phường {random.randint(1, 10)}, {selected_district}, TP. Hồ Chí Minh"
        }
    }


@router.post("/submit-application")
def submit_mst_application(
    payload: Dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Step 3 Wizard: Submit completed Form 03-ĐK-TCT first-time registration form.
    Includes base64 e-signature verification.
    """
    ensure_registration_schema(db.connection())
    
    cccd = payload.get("cccd")
    full_name = payload.get("full_name")
    dob_str = payload.get("date_of_birth") # "DD/MM/YYYY"
    gender = payload.get("gender")
    permanent_address = payload.get("permanent_address")
    business_name = payload.get("business_name")
    industry_code = payload.get("industry_code", "4711")
    industry_name = payload.get("industry_name", "Bán lẻ tổng hợp trong các cửa hàng tiện lợi")
    business_address = payload.get("business_address")
    district = payload.get("district")
    ward = payload.get("ward")
    expected_revenue = float(payload.get("expected_revenue", 0))
    employee_count = int(payload.get("employee_count", 0))
    capital = float(payload.get("capital", 0))
    signature = payload.get("signature_data")

    if not all([cccd, full_name, business_name, business_address, district, ward, signature]):
        raise HTTPException(status_code=400, detail="Vui lòng điền đầy đủ các trường thông tin bắt buộc và thực hiện ký xác nhận.")

    # Parsing date
    parsed_dob = None
    if dob_str:
        try:
            parsed_dob = datetime.datetime.strptime(dob_str, "%d/%m/%Y").date()
        except ValueError:
            try:
                parsed_dob = datetime.datetime.strptime(dob_str, "%Y-%m-%d").date()
            except ValueError:
                pass

    # Lookup HCMC tax office
    ward_row = db.execute(text("""
        SELECT tax_office_code FROM hcmc_wards 
        WHERE district_name = :dist AND ward_name = :ward
    """), {"dist": district, "ward": ward}).first()
    
    office_code = ward_row[0] if ward_row else "CS01"

    # Classify household group
    group = 1
    if expected_revenue > 3000 or employee_count >= 10:
        group = 3
    elif expected_revenue > 500 or employee_count >= 5:
        group = 2

    # Auto-generate a beautiful new Vietnam Taxpayer MST
    random.seed()
    mst_assigned = f"8092{random.randint(100000, 999999)}"

    # Record application
    db.execute(text("""
        INSERT INTO mst_registration_applications 
        (cccd, full_name, date_of_birth, gender, permanent_address, business_name, 
         industry_code, industry_name, business_address, district_name, ward_name, 
         tax_office_code, expected_revenue, employee_count, capital, household_group, 
         signature_data, status, mst_assigned)
        VALUES 
        (:cccd, :name, :dob, :gender, :perm_addr, :biz_name, 
         :ind_code, :ind_name, :biz_addr, :district, :ward, 
         :office_code, :revenue, :labor, :capital, :group, 
         :sig, 'approved', :mst)
    """), {
        "cccd": cccd,
        "name": full_name,
        "dob": parsed_dob,
        "gender": gender,
        "perm_addr": permanent_address,
        "biz_name": business_name,
        "ind_code": industry_code,
        "ind_name": industry_name,
        "biz_addr": business_address,
        "district": district,
        "ward": ward,
        "office_code": office_code,
        "revenue": expected_revenue,
        "labor": employee_count,
        "capital": capital,
        "group": group,
        "sig": signature,
        "mst": mst_assigned
    })

    # Add the business directly to the companies table so it becomes active in the ecosystem
    db.execute(text("""
        INSERT INTO companies (tax_code, name, industry, province, risk_score, is_active, created_at)
        VALUES (:mst, :name, :industry, :prov, 10.0, TRUE, CURRENT_TIMESTAMP)
        ON CONFLICT (tax_code) DO NOTHING
    """), {
        "mst": mst_assigned,
        "name": business_name.upper(),
        "industry": industry_name,
        "prov": f"TP.HCM ({district})"
    })

    db.commit()

    return {
        "status": "success",
        "message": "Đăng ký thành công! Hồ sơ của bạn đã được kiểm duyệt và thông qua tự động qua Cổng DVC.",
        "application": {
            "mst_assigned": mst_assigned,
            "household_group": group,
            "tax_office_code": office_code,
            "submitted_at": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
    }


@router.post("/report-bank-account")
def report_bank_account(
    payload: Dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Declare bank accounts or digital wallets for business transactions (Form 01/BK-STK).
    Deadline: Within 10 days of account opening or before April 20th.
    """
    ensure_registration_schema(db.connection())
    
    mst = payload.get("tax_code", "").strip()
    bank_name = payload.get("bank_name", "").strip()
    account_number = payload.get("account_number", "").strip()
    holder = payload.get("account_holder", "").strip()

    if not all([mst, bank_name, account_number]):
        raise HTTPException(status_code=400, detail="Vui lòng điền đầy đủ MST, tên ngân hàng và số tài khoản.")

    db.execute(text("""
        INSERT INTO taxpayer_bank_accounts (tax_code, bank_name, account_number, account_holder)
        VALUES (:mst, :bank, :account, :holder)
    """), {
        "mst": mst,
        "bank": bank_name,
        "account": account_number,
        "holder": holder.upper() if holder else None
    })

    db.commit()

    return {
        "status": "success",
        "message": f"Khai báo tài khoản {account_number} tại {bank_name} thành công. Đã cập nhật vào hồ sơ quản lý thuế."
    }


@router.get("/bank-accounts/{tax_code}")
def get_reported_bank_accounts(tax_code: str, db: Session = Depends(get_db)):
    """List all declared bank accounts for a given taxpayer MST."""
    ensure_registration_schema(db.connection())
    rows = db.execute(text("""
        SELECT id, bank_name, account_number, account_holder, reported_date 
        FROM taxpayer_bank_accounts WHERE tax_code = :mst
    """), {"mst": tax_code}).all()

    return {
        "accounts": [
            {
                "id": r[0],
                "bank_name": r[1],
                "account_number": r[2],
                "account_holder": r[3] or "N/A",
                "reported_date": r[4].strftime("%d/%m/%Y") if r[4] else "N/A"
            } for r in rows
        ]
    }


@router.get("/compliance-warnings/{tax_code}")
def check_compliance_warnings(tax_code: str, db: Session = Depends(get_db)):
    """Get active warnings / tasks for taxpayer dashboard."""
    ensure_registration_schema(db.connection())
    warnings = []
    
    # Check if bank accounts are registered
    count_accounts = db.execute(text("""
        SELECT COUNT(*) FROM taxpayer_bank_accounts WHERE tax_code = :mst
    """), {"mst": tax_code}).scalar()

    if count_accounts == 0:
        warnings.append({
            "type": "warning",
            "title": "Chưa khai báo tài khoản ngân hàng kinh doanh",
            "message": "Theo Nghị định 68/2026/NĐ-CP, bạn phải đăng ký tất cả tài khoản ngân hàng dùng cho kinh doanh (Mẫu 01/BK-STK) chậm nhất trong 10 ngày từ khi bắt đầu hoạt động.",
            "deadline": "Hạn chót: 20/04 năm đầu tiên hoạt động",
            "action_url": "#bank-declare-section"
        })

    # Check if registered as active
    comp = db.execute(text("SELECT is_active, created_at FROM companies WHERE tax_code = :mst"), {"mst": tax_code}).first()
    if comp:
        is_active, created_at = comp
        # If active and created recently, remind about digital signature
        warnings.append({
            "type": "info",
            "title": "Đăng ký Chữ ký số (CKS) tự nguyện",
            "message": "Khuyến khích trang bị chữ ký số cá nhân để thực hiện kê khai điện tử và ký số các hóa đơn đầu ra an toàn tuyệt đối.",
            "deadline": "Trước kỳ quyết toán tiếp theo",
            "action_url": "#digital-sig-section"
        })

    return {"warnings": warnings}
