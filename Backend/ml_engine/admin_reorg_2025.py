"""2025 Vietnam provincial-level reorganisation reference data.

The merger list is sourced from public summaries of the National Assembly
Resolution on the 2025 provincial-level reorganisation. Geometry is still
loaded from reviewed GeoJSON files; this module only provides grouping and
metadata for deriving 34-unit analytical views from the legacy 63-unit baseline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional


VN34_SOURCE_REFS = [
    {
        "name": "VTV/VNA - Vietnam now has 34 provincial-level administrative units",
        "url": "https://english.vtv.vn/news/vietnam-now-has-34-provincial-level-administrative-units-20250612122748074.htm",
        "observed_claim": "National Assembly Resolution passed on June 12, 2025; 34 provincial-level administrative units, operations from July 1, 2025.",
    },
    {
        "name": "Vietnam National Authority of Tourism - Vietnam's New Provincial System",
        "url": "https://vietnam.travel/things-to-do/vietnam%E2%80%99s-new-provincial-system-34-names-you-should-know",
        "observed_claim": "Lists 23 newly formed units and 11 unchanged administrative units with merged predecessors.",
    },
]


VN34_UNITS: List[Dict[str, Any]] = [
    {"code": "VN34-HN", "name": "Hà Nội", "type": "city", "members": ["Hà Nội"], "unchanged": True},
    {"code": "VN34-HUE", "name": "Huế", "type": "city", "members": ["Thừa Thiên Huế"], "unchanged": True},
    {"code": "VN34-CB", "name": "Cao Bằng", "type": "province", "members": ["Cao Bằng"], "unchanged": True},
    {"code": "VN34-DB", "name": "Điện Biên", "type": "province", "members": ["Điện Biên"], "unchanged": True},
    {"code": "VN34-HT", "name": "Hà Tĩnh", "type": "province", "members": ["Hà Tĩnh"], "unchanged": True},
    {"code": "VN34-LC2", "name": "Lai Châu", "type": "province", "members": ["Lai Châu"], "unchanged": True},
    {"code": "VN34-LS", "name": "Lạng Sơn", "type": "province", "members": ["Lạng Sơn"], "unchanged": True},
    {"code": "VN34-NA", "name": "Nghệ An", "type": "province", "members": ["Nghệ An"], "unchanged": True},
    {"code": "VN34-QN", "name": "Quảng Ninh", "type": "province", "members": ["Quảng Ninh"], "unchanged": True},
    {"code": "VN34-SL", "name": "Sơn La", "type": "province", "members": ["Sơn La"], "unchanged": True},
    {"code": "VN34-TH", "name": "Thanh Hóa", "type": "province", "members": ["Thanh Hóa"], "unchanged": True},
    {"code": "VN34-LC", "name": "Lào Cai", "type": "province", "members": ["Lào Cai", "Yên Bái"], "area_km2": 13256.92, "official_population": 1778785},
    {"code": "VN34-TN", "name": "Thái Nguyên", "type": "province", "members": ["Bắc Kạn", "Thái Nguyên"], "area_km2": 8375.21, "official_population": 1799489},
    {"code": "VN34-PT", "name": "Phú Thọ", "type": "province", "members": ["Vĩnh Phúc", "Hòa Bình", "Phú Thọ"], "area_km2": 9361.38, "official_population": 4022638},
    {"code": "VN34-BN", "name": "Bắc Ninh", "type": "province", "members": ["Bắc Giang", "Bắc Ninh"], "area_km2": 4718.6, "official_population": 3619433},
    {"code": "VN34-HY", "name": "Hưng Yên", "type": "province", "members": ["Thái Bình", "Hưng Yên"], "area_km2": 2514.81, "official_population": 3567943},
    {"code": "VN34-HP", "name": "Hải Phòng", "type": "city", "members": ["Hải Phòng", "Hải Dương"], "area_km2": 3194.72, "official_population": 4664124},
    {"code": "VN34-NB", "name": "Ninh Bình", "type": "province", "members": ["Hà Nam", "Nam Định", "Ninh Bình"], "area_km2": 3942.62, "official_population": 4412264},
    {"code": "VN34-QT", "name": "Quảng Trị", "type": "province", "members": ["Quảng Bình", "Quảng Trị"], "area_km2": 12700.0, "official_population": 1870845},
    {"code": "VN34-DN", "name": "Đà Nẵng", "type": "city", "members": ["Đà Nẵng", "Quảng Nam"], "area_km2": 11859.59, "official_population": 3065628},
    {"code": "VN34-QNG", "name": "Quảng Ngãi", "type": "province", "members": ["Kon Tum", "Quảng Ngãi"], "area_km2": 14832.55, "official_population": 2161755},
    {"code": "VN34-GL", "name": "Gia Lai", "type": "province", "members": ["Bình Định", "Gia Lai"], "area_km2": 21576.53, "official_population": 3583693},
    {"code": "VN34-KH", "name": "Khánh Hòa", "type": "province", "members": ["Ninh Thuận", "Khánh Hòa"], "area_km2": 8555.86, "official_population": 2243554},
    {"code": "VN34-LD", "name": "Lâm Đồng", "type": "province", "members": ["Đắk Nông", "Bình Thuận", "Lâm Đồng"], "area_km2": 24233.07, "official_population": 3872999},
    {"code": "VN34-DL", "name": "Đắk Lắk", "type": "province", "members": ["Phú Yên", "Đắk Lắk"], "area_km2": 18096.40, "official_population": 3346853},
    {"code": "VN34-HCM", "name": "Hồ Chí Minh", "type": "city", "members": ["Hồ Chí Minh", "Bà Rịa - Vũng Tàu", "Bình Dương"], "area_km2": 6772.59, "official_population": 14002598},
    {"code": "VN34-DNAI", "name": "Đồng Nai", "type": "province", "members": ["Bình Phước", "Đồng Nai"], "area_km2": 12737.18, "official_population": 4491408},
    {"code": "VN34-TNI", "name": "Tây Ninh", "type": "province", "members": ["Long An", "Tây Ninh"], "area_km2": 8536.44, "official_population": 3254170},
    {"code": "VN34-CT", "name": "Cần Thơ", "type": "city", "members": ["Cần Thơ", "Sóc Trăng", "Hậu Giang"], "area_km2": 6360.83, "official_population": 4199824},
    {"code": "VN34-VL", "name": "Vĩnh Long", "type": "province", "members": ["Bến Tre", "Trà Vinh", "Vĩnh Long"], "area_km2": 6296.20, "official_population": 4257581},
    {"code": "VN34-DT", "name": "Đồng Tháp", "type": "province", "members": ["Tiền Giang", "Đồng Tháp"], "area_km2": 5938.64, "official_population": 4370046},
    {"code": "VN34-CM", "name": "Cà Mau", "type": "province", "members": ["Bạc Liêu", "Cà Mau"], "area_km2": 7942.39, "official_population": 2606672},
    {"code": "VN34-AG", "name": "An Giang", "type": "province", "members": ["Kiên Giang", "An Giang"], "area_km2": 9888.91, "official_population": 4952238},
    {"code": "VN34-TQ", "name": "Tuyên Quang", "type": "province", "members": ["Hà Giang", "Tuyên Quang"], "area_km2": 13795.5, "official_population": 1865270},
]


def normalize_admin_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("ę", "ê")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    text = re.sub(r"^(tinh|thanh pho|tp\.?|city|province)\s+", "", text)
    text = re.sub(r"\s+(city|province)$", "", text)
    text = text.replace("đ", "d")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unit_for_legacy_name(name: Any) -> Optional[Dict[str, Any]]:
    normalized = normalize_admin_name(name)
    for unit in VN34_UNITS:
        if normalized in {normalize_admin_name(member) for member in unit["members"]}:
            return unit
    return None


def units_by_code() -> Dict[str, Dict[str, Any]]:
    return {unit["code"]: unit for unit in VN34_UNITS}


def flatten_member_names() -> List[str]:
    names: List[str] = []
    for unit in VN34_UNITS:
        names.extend(str(member) for member in unit["members"])
    return names
