import pandas as pd
import random
import uuid
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
COMPANIES_FILE = os.path.join(DATA_DIR, 'risk_data_5000_companies.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'nlp_invoices_15000.csv')

# Suspicious keywords
SUSPICIOUS_KEYWORDS = [
    "Chi phí tư vấn quản lý", "Chi phí dịch vụ khác", "Hoa hồng môi giới",
    "Phí dịch vụ tư vấn rủi ro", "Hỗ trợ maketing", "Chi phí quảng cáo ảo",
    "Tư vấn đầu tư", "Phí dịch vụ chuyên gia", "Chi phí nghiên cứu thị trường"
]

# Normal items by industry
NORMAL_ITEMS = {
    "Xây dựng": ["Sắt thép Pomina", "Xi măng Hà Tiên", "Gạch ống", "Cát xây", "Bê tông tươi", "Đá 1x2", "Sơn nước ngoại thất", "Ống nhựa Bình Minh", "Gỗ cốp pha"],
    "Sản xuất phần mềm": ["Bản quyền Visual Studio", "Server AWS M5", "License Jira", "Macbook Pro M2", "Bàn phím cơ", "Gói dịch vụ Cloud", "Thiết bị mạng Cisco", "RAM DDR5"],
    "Bán buôn thực phẩm": ["Gạo ST25", "Nước mắm Nam Ngư", "Thịt bò Kobe", "Rau sạch Đà Lạt", "Bia Heineken", "Mỳ Hảo Hảo", "Đường tinh luyện", "Dầu ăn Tường An"],
    "Sản xuất công nghiệp": ["Máy nén khí", "Phụ tùng máy dập", "Dầu nhờn công nghiệp", "Dây chuyền băng tải", "Motor điện", "Vòng bi SKF", "Tấm inox 304", "Van bi inox"],
    "Thương mại dịch vụ": ["Máy in Canon", "Văn phòng phẩm", "Bàn ghế Hòa Phát", "Máy lạnh Daikin", "Nước tinh khiết", "Mực in", "Giấy in A4", "Quạt trần"],
    "Hoạt động tư vấn quản lý": ["Sách quản trị", "Giấy in A4", "Mực in", "Laptop Dell", "Chữ ký số", "Phần mềm kế toán", "Gói lưu trữ hồ sơ"]
}

def generate_description(industry, is_fraud, fraud_type):
    items = []
    num_items = random.randint(1, 4)
    
    # 1. Normal items
    available_normal = NORMAL_ITEMS.get(industry, NORMAL_ITEMS["Thương mại dịch vụ"])
    
    if is_fraud == 0:
        items = random.sample(available_normal, k=min(num_items, len(available_normal)))
    else:
        if fraud_type == "suspicious_keyword":
            # Add normal items
            if num_items > 1:
                items = random.sample(available_normal, k=num_items-1)
            # Add one suspicious keyword
            items.append(random.choice(SUSPICIOUS_KEYWORDS))
        
        elif fraud_type == "industry_mismatch":
            # Pick items from a totally different industry
            wrong_industries = [ind for ind in NORMAL_ITEMS.keys() if ind != industry]
            wrong_industry = random.choice(wrong_industries)
            wrong_items = NORMAL_ITEMS[wrong_industry]
            items = random.sample(wrong_items, k=min(num_items, len(wrong_items)))
            
    # Add random quantities and prices
    descriptions = []
    for item in items:
        qty = random.randint(1, 100)
        if fraud_type == "suspicious_keyword" and item in SUSPICIOUS_KEYWORDS:
            qty = 1  # Services usually qty 1
        descriptions.append(f"{item}")
        
    return "\n".join(descriptions)

def generate_nlp_data():
    print(f"Reading companies from {COMPANIES_FILE}...")
    try:
        df_companies = pd.read_csv(COMPANIES_FILE)
    except FileNotFoundError:
        print("Could not find companies file. Generating mock ones.")
        df_companies = pd.DataFrame({
            'tax_code': [f"010{str(i).zfill(7)}" for i in range(5000)],
            'industry': random.choices(list(NORMAL_ITEMS.keys()), k=5000)
        })

    records = []
    
    print("Generating 15,000 invoices...")
    for idx, row in df_companies.iterrows():
        tax_code = str(row['tax_code'])
        industry = row.get('industry', 'Thương mại dịch vụ')
        if pd.isna(industry):
            industry = 'Thương mại dịch vụ'
            
        # Generate 3 invoices per company
        for i in range(3):
            # 80% safe, 20% fraud
            is_fraud = 1 if random.random() < 0.2 else 0
            
            fraud_type = "none"
            if is_fraud == 1:
                fraud_type = random.choice(["suspicious_keyword", "industry_mismatch"])
                
            desc = generate_description(industry, is_fraud, fraud_type)
            
            records.append({
                'invoice_id': f"INV-{uuid.uuid4().hex[:8].upper()}",
                'tax_code': tax_code,
                'industry': industry,
                'description': desc,
                'is_fraud': is_fraud,
                'fraud_type': fraud_type
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Generated {len(df_out)} invoices and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_nlp_data()
