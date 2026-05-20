"""
news_crawler.py - Real-Time Economic News Crawler for Vietnam Digital Twin
==========================================================================
Fetches RSS feeds, filters economic relevance, and classifies articles
using Gemini API or rule-based fallback.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from ml_engine.macro_event_ingest import MacroEventCandidate

__all__ = ["crawl_all_feeds", "RSS_FEEDS"]

RSS_FEEDS = [
    {
        "name": "VnExpress Kinh doanh",
        "url": "https://vnexpress.net/rss/kinh-doanh.rss",
        "source_name": "VnExpress",
        "language": "vi",
    },
    {
        "name": "VnExpress Thời sự",
        "url": "https://vnexpress.net/rss/thoi-su.rss",
        "source_name": "VnExpress",
        "language": "vi",
    },
    {
        "name": "Tuổi Trẻ Kinh tế",
        "url": "https://tuoitre.vn/rss/kinh-doanh.rss",
        "source_name": "Tuoi Tre",
        "language": "vi",
    },
    {
        "name": "Thanh Niên Kinh tế",
        "url": "https://thanhnien.vn/rss/kinh-te.rss",
        "source_name": "Thanh Nien",
        "language": "vi",
    },
    {
        "name": "TTXVN Kinh tế",
        "url": "https://www.vietnamplus.vn/rss/kinhte.rss",
        "source_name": "TTXVN/VietnamPlus",
        "language": "vi",
    },
    {
        "name": "CafeF Vĩ mô",
        "url": "https://cafef.vn/rss/vi-mo-dau-tu.rss",
        "source_name": "CafeF",
        "language": "vi",
    },
    {
        "name": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "source_name": "Reuters",
        "language": "en",
    },
    {
        "name": "Bloomberg Asia",
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "source_name": "Bloomberg",
        "language": "en",
    },
]

ECON_KEYWORDS_VI = [
    "gdp", "kinh tế", "thuế", "ngân sách", "fdi", "đầu tư", "xuất khẩu",
    "nhập khẩu", "lạm phát", "cpi", "lãi suất", "tỷ giá", "chứng khoán",
    "bất động sản", "doanh nghiệp", "công nghiệp", "nông nghiệp",
    "thương mại", "hiệp định", "wto", "cptpp", "evfta", "rcep",
    "ngân hàng", "tín dụng", "nợ xấu", "trái phiếu", "cổ phiếu",
    "thất nghiệp", "việc làm", "lương", "giá xăng", "giá điện",
    "bão", "lũ", "hạn hán", "thiên tai", "dịch bệnh",
    "samsung", "foxconn", "intel", "vinfast", "formosa",
    "khu công nghiệp", "cảng biển", "sân bay", "cao tốc", "metro",
    "chính sách", "nghị định", "luật", "nghị quyết", "quyết định",
    "mỹ", "trung quốc", "asean", "nhật bản", "hàn quốc",
]

ECON_KEYWORDS_EN = [
    "gdp", "economy", "tax", "fdi", "investment", "export", "import",
    "inflation", "interest rate", "trade war", "tariff", "vietnam",
    "supply chain", "semiconductor", "manufacturing",
]

PROVINCE_NAME_TO_CODE = {
    "hà nội": "01", "hà giang": "02", "cao bằng": "04",
    "bắc kạn": "06", "tuyên quang": "08", "lào cai": "10",
    "điện biên": "11", "lai châu": "12", "sơn la": "14",
    "yên bái": "15", "hòa bình": "17", "thái nguyên": "19",
    "lạng sơn": "20", "quảng ninh": "22", "bắc giang": "24",
    "phú thọ": "25", "vĩnh phúc": "26", "bắc ninh": "27",
    "hải dương": "30", "hải phòng": "31", "hưng yên": "33",
    "thái bình": "34", "hà nam": "35", "nam định": "36",
    "ninh bình": "37", "thanh hóa": "38", "nghệ an": "40",
    "hà tĩnh": "42", "quảng bình": "44", "quảng trị": "45",
    "thừa thiên huế": "46", "đà nẵng": "48", "quảng nam": "49",
    "quảng ngãi": "51", "bình định": "52", "phú yên": "54",
    "khánh hòa": "56", "ninh thuận": "58", "bình thuận": "60",
    "kon tum": "62", "gia lai": "64", "đắk lắk": "66",
    "đắk nông": "67", "lâm đồng": "68", "bình phước": "70",
    "tây ninh": "72", "bình dương": "74", "đồng nai": "75",
    "bà rịa - vũng tàu": "77", "bà rịa vũng tàu": "77",
    "tp. hồ chí minh": "79", "hồ chí minh": "79", "tp.hcm": "79", "tphcm": "79",
    "long an": "80", "tiền giang": "82", "bến tre": "83",
    "trà vinh": "84", "vĩnh long": "86", "đồng tháp": "87",
    "an giang": "89", "kiên giang": "91", "cần thơ": "92",
    "hậu giang": "93", "sóc trăng": "94", "bạc liêu": "95", "cà mau": "96",
}

def extract_province_codes(text: str) -> List[str]:
    """Extract GSO province codes mentioned in Vietnamese text."""
    text_lower = text.lower()
    codes = set()
    for name, code in sorted(PROVINCE_NAME_TO_CODE.items(), key=lambda x: len(x[0]), reverse=True):
        if name in text_lower:
            codes.add(code)
    return sorted(codes)

def fetch_rss_articles(feed_config: dict, timeout: int = 15) -> List[dict]:
    url = feed_config["url"]
    source_name = feed_config["source_name"]
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    articles = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pubDate = item.findtext("pubDate", default="").strip()
            description = item.findtext("description", default="").strip()
            
            # Strip HTML tags
            description = re.sub(r'<[^>]+>', '', description).strip()
            
            # Convert pubDate
            iso_date = None
            if pubDate:
                try:
                    dt = parsedate_to_datetime(pubDate)
                    iso_date = dt.date().isoformat()
                except Exception:
                    pass

            articles.append({
                "title": title,
                "link": link,
                "pubDate": iso_date,
                "description": description,
                "source_name": source_name,
                "language": feed_config["language"]
            })
    except Exception as e:
        print(f"Error fetching RSS from {url}: {e}")
    return articles

def filter_economic_relevance(articles: List[dict], language: str = "vi") -> List[dict]:
    filtered = []
    keywords = ECON_KEYWORDS_VI if language == "vi" else ECON_KEYWORDS_EN
    for article in articles:
        text = (article.get("title", "") + " " + article.get("description", "")).lower()
        match_count = sum(1 for kw in keywords if kw in text)
        if match_count >= 2:
            filtered.append(article)
    return filtered

# ── Sector keyword map for fallback classifier ──
_SECTOR_KEYWORDS = {
    "Dệt may": ["dệt may", "giày da", "textile", "garment"],
    "Bất động sản": ["bất động sản", "nhà đất", "real estate", "chung cư"],
    "Du lịch & Khách sạn": ["du lịch", "khách sạn", "tourism", "hotel"],
    "Nông nghiệp": ["nông nghiệp", "lúa", "cà phê", "thủy sản", "agriculture"],
    "Công nghệ thông tin": ["cntt", "phần mềm", "công nghệ", "semiconductor", "chip"],
    "Vận tải & Logistics": ["vận tải", "logistics", "cảng", "hàng không", "shipping"],
    "FDI": ["fdi", "đầu tư nước ngoài", "foreign investment"],
    "Tài chính & Ngân hàng": ["ngân hàng", "tín dụng", "chứng khoán", "trái phiếu"],
    "Năng lượng": ["điện", "xăng dầu", "năng lượng", "solar", "energy"],
    "Xây dựng": ["xây dựng", "cao tốc", "metro", "hạ tầng", "infrastructure"],
}


def _extract_sectors(text: str) -> List[str]:
    """Extract affected economic sectors from article text."""
    text_lower = text.lower()
    return [sector for sector, keywords in _SECTOR_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)]


def classify_article_fallback(article: dict) -> dict:
    """Keyword-based classification when API is not available."""
    text = (article.get("title", "") + " " + article.get("description", "")).lower()

    classification: Dict[str, Any] = {
        "is_macro_economic_event": True,
        "event_type": "unknown",
        "severity": "low",
        "affected_provinces": extract_province_codes(text),
        "affected_sectors": _extract_sectors(text),
        "impact_hints": {},
        "summary_vi": article.get("title", "")
    }

    if any(kw in text for kw in ["bão", "lũ", "hạn hán", "sạt lở", "thiên tai"]):
        classification.update({"event_type": "natural_disaster", "severity": "high"})
    elif any(kw in text for kw in ["dịch", "covid", "cúm", "h5n1"]):
        classification.update({"event_type": "pandemic", "severity": "high"})
    elif any(kw in text for kw in ["thuế", "nghị định", "luật", "chính sách", "nghị quyết"]):
        classification.update({"event_type": "policy", "severity": "medium"})
    elif any(kw in text for kw in ["fdi", "đầu tư", "samsung", "foxconn", "nhà máy"]):
        classification.update({"event_type": "growth", "severity": "medium"})
    elif any(kw in text for kw in ["hiệp định", "fta", "cptpp", "evfta", "rcep"]):
        classification.update({"event_type": "trade_agreement", "severity": "medium"})
    elif any(kw in text for kw in ["nợ xấu", "ngân hàng", "chứng khoán", "trái phiếu"]):
        classification.update({"event_type": "financial_crisis", "severity": "medium"})
    elif any(kw in text for kw in ["tariff", "trade war", "thuế quan", "cấm vận"]):
        classification.update({"event_type": "trade_war", "severity": "high"})

    return classification

CLASSIFICATION_PROMPT = """Bạn là chuyên gia phân tích kinh tế vĩ mô Việt Nam.

Phân loại bài báo kinh tế sau và trả về JSON:

Tiêu đề: {title}
Nội dung: {description}
Nguồn: {source_name}
Ngày: {published_at}

Trả về JSON với cấu trúc:
{{
  "is_macro_economic_event": true/false,
  "event_type": "policy|trade_war|natural_disaster|pandemic|financial_crisis|trade_agreement|growth|geopolitics|infrastructure_shock|unknown",
  "severity": "low|medium|high|extreme",
  "affected_provinces": ["01", "79"],
  "affected_sectors": ["Dệt may", "FDI"],
  "impact_hints": {{
    "impact_gdp_pct": -2.0,
    "impact_tax_revenue_pct": -3.5,
    "impact_unemployment_pct": 0.5,
    "impact_fdi_pct": -5.0
  }},
  "summary_vi": "Tóm tắt 1-2 câu bằng tiếng Việt"
}}

Quy tắc:
- affected_provinces dùng mã GSO: 01=Hà Nội, 79=TP.HCM, 48=Đà Nẵng, 31=Hải Phòng, 74=Bình Dương, 75=Đồng Nai, 27=Bắc Ninh, 24=Bắc Giang, 22=Quảng Ninh, 40=Nghệ An, 56=Khánh Hòa, 77=Bà Rịa-Vũng Tàu, 66=Đắk Lắk, 92=Cần Thơ, 96=Cà Mau, 51=Quảng Ngãi
- Nếu ảnh hưởng toàn quốc: affected_provinces = []
- Nếu bài không liên quan kinh tế vĩ mô: is_macro_economic_event = false
- impact_hints chỉ cần ước lượng sơ bộ, không cần chính xác
- Chỉ trả về JSON, không thêm giải thích
"""

# ── Singleton Gemini model handle ──
_gemini_model = None


def _get_gemini_model(api_key: str):
    """Lazy-init Gemini model once per process to avoid redundant configure calls."""
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    return _gemini_model


def _extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from Gemini response, stripping markdown code fences if present."""
    cleaned = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def classify_article_with_gemini(article: dict, api_key: str) -> dict:
    """Classify an article using Gemini API with fallback on failure."""
    prompt = CLASSIFICATION_PROMPT.format(
        title=article.get("title", ""),
        description=article.get("description", ""),
        source_name=article.get("source_name", ""),
        published_at=article.get("pubDate", ""),
    )

    try:
        model = _get_gemini_model(api_key)
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            parsed = _extract_json_from_response(response.text)
            if parsed and isinstance(parsed, dict):
                return parsed
    except Exception as e:
        print(f"[NewsCrawler] Gemini API error for '{article.get('title', '')[:60]}': {e}")

    return classify_article_fallback(article)

def crawl_all_feeds(
    api_key: str,
    feeds: Optional[List[dict]] = None,
    max_per_feed: int = 15,
) -> List[MacroEventCandidate]:
    """Crawl all configured RSS feeds and return classified MacroEventCandidates."""
    if feeds is None:
        feeds = RSS_FEEDS

    candidates: List[MacroEventCandidate] = []
    use_ai = bool(api_key)

    for feed in feeds:
        feed_name = feed.get("name", feed.get("source_name", "unknown"))
        print(f"  📡 Fetching {feed_name}...")

        articles = fetch_rss_articles(feed)
        relevant = filter_economic_relevance(articles, language=feed["language"])
        print(f"     {len(articles)} articles fetched, {len(relevant)} economically relevant")

        for article in relevant[:max_per_feed]:
            if use_ai:
                classification = classify_article_with_gemini(article, api_key)
                time.sleep(0.5)  # Rate limiting for Gemini API
            else:
                classification = classify_article_fallback(article)

            if not classification.get("is_macro_economic_event"):
                continue

            candidates.append(MacroEventCandidate(
                title=article.get("title", ""),
                description=classification.get("summary_vi") or article.get("description", ""),
                source_name=article.get("source_name", ""),
                source_url=article.get("link", ""),
                published_at=article.get("pubDate"),
                event_type=classification.get("event_type", "unknown"),
                affected_provinces=classification.get("affected_provinces", []),
                affected_sectors=classification.get("affected_sectors", []),
                impact_hints=classification.get("impact_hints", {}),
                raw_payload={"article": article, "classification": classification},
            ))

    return candidates
