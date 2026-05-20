import pandas as pd
import json
import urllib.request
from bs4 import BeautifulSoup
import re
from pathlib import Path

# Paths
DATA_DIR = Path(r"e:\TaxInspector\Backend\data\data")
PROVINCES_JSON = DATA_DIR / "vietnam_provinces.json"

def clean_number(x):
    if pd.isna(x): return 0
    s = str(x).replace(",", "").replace(".", "").strip()
    match = re.search(r"(\d+)", s)
    return int(match.group(1)) if match else 0

def clean_float(x):
    if pd.isna(x): return 0.0
    s = str(x).replace(",", ".").strip()
    match = re.search(r"(\d+\.?\d*)", s)
    return float(match.group(1)) if match else 0.0

def clean_province_name(name):
    # Remove citations like [1]
    name = re.sub(r'\[.*?\]', '', str(name)).strip()
    # Handle "Thừa Thiên Huế", "Thừa Thiên – Huế", "Bà Rịa - Vũng Tàu"
    name = name.replace("–", "-").replace(" - ", "-")
    return name

print("Fetching province data from Wikipedia...")
url = "https://en.wikipedia.org/wiki/Provinces_of_Vietnam"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read()

# Parse tables
tables = pd.read_html(html)

# The correct table has headers: Province/City, Area, Population, Density
target_table = None
for t in tables:
    if any(c for c in t.columns if "Population" in str(c)) and any(c for c in t.columns if "Area" in str(c)):
        target_table = t
        break

if target_table is None:
    raise ValueError("Could not find provinces table on Wikipedia")

# Flatten multi-level columns if necessary
if isinstance(target_table.columns, pd.MultiIndex):
    target_table.columns = ['_'.join(col).strip() for col in target_table.columns.values]

print(f"Found table with shape {target_table.shape}")

# Create a mapping dictionary for update
wiki_data = {}
for _, row in target_table.iterrows():
    # Find columns dynamically based on keywords
    name_col = next((c for c in target_table.columns if "Province" in str(c) or "City" in str(c) or "Name" in str(c)), None)
    pop_col = next((c for c in target_table.columns if "Population" in str(c)), None)
    
    if not name_col or not pop_col: continue
    
    name = clean_province_name(row[name_col])
    if "Region" in name or name == "Total": continue # Skip aggregates
    
    pop = clean_number(row[pop_col])
    if pop > 0:
        wiki_data[name.lower()] = {
            "population": pop
        }

print(f"Extracted data for {len(wiki_data)} provinces.")

# Update the local JSON
with open(PROVINCES_JSON, "r", encoding="utf-8") as f:
    provinces = json.load(f)

updated_count = 0
for p in provinces:
    # Match name
    p_name = clean_province_name(p["province_name"]).lower()
    match = None
    if p_name in wiki_data:
        match = wiki_data[p_name]
    else:
        # Try fuzzy match (e.g. Ho Chi Minh City vs TP Ho Chi Minh)
        for w_name, w_data in wiki_data.items():
            if p_name in w_name or w_name in p_name or (p_name == "tp. hồ chí minh" and "ho chi minh" in w_name) or (p_name == "bà rịa-vũng tàu" and "vung tau" in w_name):
                match = w_data
                break
    
    if match:
        p["population"] = match["population"]
        updated_count += 1
        
# For GDP, let's fetch from another page "List of Vietnamese provinces by GDP"
print("Fetching GDP data from Wikipedia...")
gdp_url = "https://en.wikipedia.org/wiki/List_of_Vietnamese_provinces_by_GDP"
try:
    gdp_req = urllib.request.Request(gdp_url, headers={'User-Agent': 'Mozilla/5.0'})
    gdp_html = urllib.request.urlopen(gdp_req).read()
    gdp_tables = pd.read_html(gdp_html)
    
    gdp_target = None
    for t in gdp_tables:
        if any(c for c in t.columns if "GDP" in str(c)) and any(c for c in t.columns if "Province" in str(c)):
            gdp_target = t
            break
            
    if gdp_target is not None:
        if isinstance(gdp_target.columns, pd.MultiIndex):
            gdp_target.columns = ['_'.join(col).strip() for col in gdp_target.columns.values]
            
        gdp_mapping = {}
        for _, row in gdp_target.iterrows():
            name_col = next((c for c in gdp_target.columns if "Province" in str(c)), None)
            gdp_col = next((c for c in gdp_target.columns if "VND" in str(c) and "billion" in str(c).lower()), None)
            
            if name_col and gdp_col:
                name = clean_province_name(row[name_col]).lower()
                gdp_val = clean_number(row[gdp_col])
                if gdp_val > 0:
                    gdp_mapping[name] = gdp_val
        
        gdp_updated = 0
        for p in provinces:
            p_name = clean_province_name(p["province_name"]).lower()
            match = None
            if p_name in gdp_mapping:
                match = gdp_mapping[p_name]
            else:
                for w_name, w_val in gdp_mapping.items():
                    if p_name in w_name or w_name in p_name:
                        match = w_val
                        break
            if match:
                p["gdp_billion_vnd"] = match
                gdp_updated += 1
        print(f"Updated GDP for {gdp_updated} provinces.")
except Exception as e:
    print(f"GDP fetch failed: {e}")

with open(PROVINCES_JSON, "w", encoding="utf-8") as f:
    json.dump(provinces, f, ensure_ascii=False, indent=2)

print(f"Successfully updated population for {updated_count} provinces in {PROVINCES_JSON.name}")
