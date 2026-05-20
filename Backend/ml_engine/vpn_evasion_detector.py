"""
vpn_evasion_detector.py – Network Forensic Layer for VAT Invoice Tracing
=========================================================================
5-Layer defense system to detect VPN/proxy evasion during VAT invoice submission.

Layers:
    L1 – IP Intelligence: datacenter/VPN ASN classification
    L2 – Geo-Velocity: impossible travel detection
    L3 – Device Fingerprint Consistency: fingerprint change tracking
    L4 – Timezone Mismatch: browser TZ vs IP geolocation TZ
    L5 – Graph Consistency: cluster-level VPN co-usage detection

Novel contribution: First application of multi-layer network forensics
specifically designed for tax fraud evasion detection in VAT invoice systems.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────
#  Data Structures
# ────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    """Input data for VPN analysis – one per HTTP request."""
    ip_address: str
    user_agent: str = ""
    accept_language: str = ""
    screen_resolution: str = ""
    timezone_offset: int = 420          # minutes from UTC (default: GMT+7 Vietnam)
    browser_timezone: str = "Asia/Ho_Chi_Minh"
    tax_code: str = ""
    session_id: str = ""
    request_path: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class VPNAnalysisResult:
    """Output of the 5-layer VPN analysis."""
    composite_score: float = 0.0        # 0.0 - 1.0
    is_vpn_detected: bool = False       # composite > threshold
    layer_scores: Dict[str, float] = field(default_factory=dict)
    risk_boost: float = 0.0             # how much to add to fraud risk_score
    explanation: str = ""               # human-readable
    evidence: Dict[str, Any] = field(default_factory=dict)
    ip_type: str = "unknown"            # datacenter|residential|mobile|vpn|tor
    is_tor: bool = False
    is_known_vpn: bool = False
    geo_country: str = ""
    geo_city: str = ""
    asn_org: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ────────────────────────────────────────────────────────────
#  IP Range Database
# ────────────────────────────────────────────────────────────

# Preloaded datacenter/VPN IP ranges for fast lookup
_IP_RANGES_DB: Optional[List[Dict]] = None
_IP_RANGES_PATH = Path(__file__).resolve().parent.parent / "data" / "data" / "ip_ranges_datacenter.json"


def _load_ip_ranges() -> List[Dict]:
    """Load IP ranges from JSON database file."""
    global _IP_RANGES_DB
    if _IP_RANGES_DB is not None:
        return _IP_RANGES_DB

    if _IP_RANGES_PATH.exists():
        try:
            with open(_IP_RANGES_PATH, "r", encoding="utf-8") as f:
                _IP_RANGES_DB = json.load(f)
            return _IP_RANGES_DB
        except Exception as e:
            print(f"[VPN] Warning: failed to load IP ranges: {e}")

    _IP_RANGES_DB = []
    return _IP_RANGES_DB


def _classify_ip(ip_str: str) -> Tuple[str, str, float]:
    """
    Classify IP address against known datacenter/VPN ranges.
    Returns: (ip_type, provider_name, confidence)
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return ("unknown", "", 0.0)

    # Private IP → definitely not VPN
    if ip.is_private or ip.is_loopback:
        return ("private", "local_network", 1.0)

    ranges = _load_ip_ranges()
    for entry in ranges:
        try:
            network = ipaddress.ip_network(entry["cidr"], strict=False)
            if ip in network:
                return (
                    entry.get("ip_type", "datacenter"),
                    entry.get("provider", "unknown"),
                    entry.get("confidence", 0.9)
                )
        except (ValueError, KeyError):
            continue

    return ("residential", "", 0.1)


# ────────────────────────────────────────────────────────────
#  Device Fingerprinting
# ────────────────────────────────────────────────────────────

def compute_device_fingerprint(session: SessionData) -> str:
    """
    Compute a SHA-256 hash from device characteristics.
    Used to track if the same tax_code suddenly changes device fingerprint.
    """
    raw = f"{session.user_agent}|{session.screen_resolution}|{session.accept_language}|{session.timezone_offset}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ────────────────────────────────────────────────────────────
#  Timezone → Approximate Geolocation Mapping
# ────────────────────────────────────────────────────────────

# Maps timezone names to expected UTC offsets (minutes)
_TZ_OFFSET_MAP: Dict[str, int] = {
    "Asia/Ho_Chi_Minh": 420,
    "Asia/Bangkok": 420,
    "Asia/Jakarta": 420,
    "Asia/Shanghai": 480,
    "Asia/Tokyo": 540,
    "Asia/Seoul": 540,
    "Asia/Singapore": 480,
    "Asia/Kolkata": 330,
    "Europe/London": 0,
    "Europe/Paris": 60,
    "Europe/Berlin": 60,
    "America/New_York": -300,
    "America/Los_Angeles": -480,
    "America/Chicago": -360,
    "Australia/Sydney": 600,
    "Pacific/Auckland": 720,
}

# IP geolocation: country → expected UTC offset range (min, max in minutes)
_COUNTRY_TZ_RANGES: Dict[str, Tuple[int, int]] = {
    "VN": (420, 420),
    "TH": (420, 420),
    "SG": (480, 480),
    "CN": (480, 480),
    "JP": (540, 540),
    "KR": (540, 540),
    "US": (-600, -240),
    "GB": (0, 60),
    "DE": (60, 120),
    "FR": (60, 120),
    "AU": (480, 660),
    "IN": (330, 330),
}


# ────────────────────────────────────────────────────────────
#  Layer 1: IP Intelligence
# ────────────────────────────────────────────────────────────

def _layer1_ip_intelligence(session: SessionData) -> Tuple[float, Dict[str, Any]]:
    """
    Score: 0.0 = clean residential IP, 1.0 = known datacenter/VPN IP.
    """
    ip_type, provider, confidence = _classify_ip(session.ip_address)

    evidence = {
        "ip_address": session.ip_address,
        "ip_type": ip_type,
        "provider": provider,
        "classification_confidence": confidence,
    }

    if ip_type in ("datacenter", "vpn", "tor", "proxy"):
        score = min(1.0, confidence * 1.1)  # slightly boost high-confidence matches
    elif ip_type == "private":
        score = 0.0
    else:
        score = max(0.0, 0.05)  # small base score for any public IP

    return (round(score, 4), evidence)


# ────────────────────────────────────────────────────────────
#  Layer 2: Geo-Velocity (Impossible Travel)
# ────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _layer2_geo_velocity(
    current_session: SessionData,
    previous_sessions: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Detect impossible travel: if distance/time > 800 km/h between logins → VPN flag.
    """
    MAX_SPEED_KMH = 800.0  # typical max commercial flight speed

    if not previous_sessions:
        return (0.0, {"reason": "no_previous_sessions"})

    # Get most recent previous session with geo data
    prev = None
    for ps in sorted(previous_sessions, key=lambda x: x.get("created_at", ""), reverse=True):
        if ps.get("geo_lat") and ps.get("geo_lng"):
            prev = ps
            break

    if not prev:
        return (0.0, {"reason": "no_geo_data_in_history"})

    # For the current session, we need geo data from IP lookup
    # If not available, skip
    curr_lat = getattr(current_session, '_geo_lat', None)
    curr_lng = getattr(current_session, '_geo_lng', None)
    if curr_lat is None or curr_lng is None:
        return (0.0, {"reason": "no_current_geo_data"})

    prev_lat = float(prev.get("geo_lat", 0))
    prev_lng = float(prev.get("geo_lng", 0))
    prev_time_str = prev.get("created_at", "")

    try:
        if isinstance(prev_time_str, datetime):
            prev_time = prev_time_str
        else:
            prev_time = datetime.fromisoformat(str(prev_time_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return (0.0, {"reason": "invalid_previous_timestamp"})

    distance_km = _haversine_km(prev_lat, prev_lng, curr_lat, curr_lng)
    time_diff = current_session.timestamp - prev_time
    hours = max(time_diff.total_seconds() / 3600.0, 0.001)
    speed_kmh = distance_km / hours

    evidence = {
        "distance_km": round(distance_km, 1),
        "time_hours": round(hours, 2),
        "speed_kmh": round(speed_kmh, 1),
        "max_allowed_kmh": MAX_SPEED_KMH,
        "previous_ip": prev.get("ip_address", ""),
        "previous_city": prev.get("geo_city", ""),
    }

    if speed_kmh > MAX_SPEED_KMH:
        # Impossible travel detected
        ratio = min(speed_kmh / MAX_SPEED_KMH, 3.0)
        score = min(1.0, 0.5 + ratio * 0.15)
        evidence["verdict"] = "impossible_travel"
    elif speed_kmh > 400:
        score = 0.3  # suspicious but possible (air travel)
        evidence["verdict"] = "fast_travel"
    else:
        score = 0.0
        evidence["verdict"] = "normal"

    return (round(score, 4), evidence)


# ────────────────────────────────────────────────────────────
#  Layer 3: Device Fingerprint Consistency
# ────────────────────────────────────────────────────────────

def _layer3_fingerprint_consistency(
    session: SessionData,
    previous_fingerprints: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Track how many distinct device fingerprints a single tax_code has used.
    Frequent changes suggest VPN/device spoofing.
    """
    current_fp = compute_device_fingerprint(session)
    unique_fps = set(previous_fingerprints)
    unique_fps.add(current_fp)

    evidence = {
        "current_fingerprint": current_fp[:16] + "...",
        "total_unique_fingerprints": len(unique_fps),
        "is_new_fingerprint": current_fp not in set(previous_fingerprints),
    }

    # Scoring: 1-2 fingerprints = normal, 3-4 = suspicious, 5+ = high risk
    n = len(unique_fps)
    if n <= 2:
        score = 0.0
    elif n <= 4:
        score = 0.2 + (n - 2) * 0.15
    else:
        score = min(1.0, 0.5 + (n - 4) * 0.1)

    return (round(score, 4), evidence)


# ────────────────────────────────────────────────────────────
#  Layer 4: Timezone Mismatch
# ────────────────────────────────────────────────────────────

def _layer4_timezone_mismatch(
    session: SessionData,
    ip_country_code: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """
    Compare browser-reported timezone with IP-derived country timezone.
    Significant mismatch indicates VPN.
    """
    browser_tz = session.browser_timezone
    browser_offset = session.timezone_offset  # minutes from UTC

    # Get expected offset from browser timezone name
    expected_from_browser_tz = _TZ_OFFSET_MAP.get(browser_tz, None)

    # Get expected offset range from IP country
    country_range = _COUNTRY_TZ_RANGES.get(ip_country_code.upper(), None)

    evidence = {
        "browser_timezone": browser_tz,
        "browser_offset_min": browser_offset,
        "ip_country": ip_country_code,
    }

    # If we have country data from IP, compare
    if country_range:
        expected_min, expected_max = country_range
        evidence["expected_offset_range"] = [expected_min, expected_max]

        # Check if browser offset falls within expected range for IP country
        if browser_offset < expected_min - 30 or browser_offset > expected_max + 30:
            # Significant mismatch — browser says one timezone, IP says another
            diff_minutes = min(
                abs(browser_offset - expected_min),
                abs(browser_offset - expected_max)
            )
            if diff_minutes > 180:  # > 3 hours
                score = 0.9
                evidence["verdict"] = "severe_mismatch"
            elif diff_minutes > 90:  # > 1.5 hours
                score = 0.6
                evidence["verdict"] = "moderate_mismatch"
            else:
                score = 0.3
                evidence["verdict"] = "minor_mismatch"
            evidence["mismatch_minutes"] = diff_minutes
        else:
            score = 0.0
            evidence["verdict"] = "consistent"
    elif expected_from_browser_tz is not None:
        # Can only check browser timezone name vs reported offset
        diff = abs(browser_offset - expected_from_browser_tz)
        if diff > 60:
            score = 0.5  # suspicious
            evidence["verdict"] = "tz_name_offset_mismatch"
        else:
            score = 0.0
            evidence["verdict"] = "consistent_name_only"
    else:
        score = 0.0
        evidence["verdict"] = "insufficient_data"

    return (round(score, 4), evidence)


# ────────────────────────────────────────────────────────────
#  Layer 5: Graph Consistency
# ────────────────────────────────────────────────────────────

def _layer5_graph_consistency(
    session: SessionData,
    cluster_ip_map: Dict[str, List[str]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if multiple companies in the same invoice cluster share the same
    datacenter IP range — suggests they use a common VPN for coordinated fraud.
    """
    tax_code = session.tax_code
    if not tax_code or not cluster_ip_map:
        return (0.0, {"reason": "no_cluster_data"})

    # Get IPs used by companies in the same cluster as this tax_code
    cluster_ips = cluster_ip_map.get(tax_code, [])
    if len(cluster_ips) < 2:
        return (0.0, {"reason": "insufficient_cluster_data"})

    # Check how many cluster members share the same /24 subnet
    subnets: Dict[str, int] = {}
    for ip_str in cluster_ips:
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.version == 4:
                subnet = str(ipaddress.ip_network(f"{ip_str}/24", strict=False))
            else:
                subnet = str(ipaddress.ip_network(f"{ip_str}/48", strict=False))
            subnets[subnet] = subnets.get(subnet, 0) + 1
        except ValueError:
            continue

    # Find the most shared subnet
    if not subnets:
        return (0.0, {"reason": "no_valid_ips"})

    max_sharing = max(subnets.values())
    total_ips = len(cluster_ips)

    evidence = {
        "cluster_size": total_ips,
        "max_subnet_sharing": max_sharing,
        "shared_subnets": {k: v for k, v in subnets.items() if v >= 2},
    }

    # If >50% of cluster shares same subnet → very suspicious
    sharing_ratio = max_sharing / max(total_ips, 1)
    if sharing_ratio > 0.5 and max_sharing >= 3:
        score = 0.9
        evidence["verdict"] = "coordinated_vpn_cluster"
    elif sharing_ratio > 0.3 and max_sharing >= 2:
        score = 0.5
        evidence["verdict"] = "suspicious_cluster"
    else:
        score = 0.0
        evidence["verdict"] = "normal_distribution"

    return (round(score, 4), evidence)


# ────────────────────────────────────────────────────────────
#  Main Analysis Engine
# ────────────────────────────────────────────────────────────

# Layer weights for composite score calculation
LAYER_WEIGHTS = {
    "L1": 0.30,   # IP Intelligence
    "L2": 0.25,   # Geo-Velocity
    "L3": 0.15,   # Fingerprint
    "L4": 0.15,   # Timezone
    "L5": 0.15,   # Graph Consistency
}

VPN_DETECTION_THRESHOLD = 0.55
RISK_BOOST_MAX = 0.25


def analyze_session(
    session: SessionData,
    previous_sessions: Optional[List[Dict[str, Any]]] = None,
    previous_fingerprints: Optional[List[str]] = None,
    cluster_ip_map: Optional[Dict[str, List[str]]] = None,
    ip_country_code: str = "",
) -> VPNAnalysisResult:
    """
    Run all 5 layers of VPN evasion detection on a single session.

    Args:
        session: Current session data
        previous_sessions: List of recent sessions for the same tax_code (for geo-velocity)
        previous_fingerprints: List of device fingerprint hashes for this tax_code
        cluster_ip_map: {tax_code: [ip1, ip2, ...]} for graph consistency check
        ip_country_code: 2-letter country code derived from IP geolocation

    Returns:
        VPNAnalysisResult with composite score and per-layer breakdown
    """
    if previous_sessions is None:
        previous_sessions = []
    if previous_fingerprints is None:
        previous_fingerprints = []
    if cluster_ip_map is None:
        cluster_ip_map = {}

    # L1: IP Intelligence
    l1_score, l1_evidence = _layer1_ip_intelligence(session)

    # Determine IP type for result
    ip_type = l1_evidence.get("ip_type", "unknown")
    asn_org = l1_evidence.get("provider", "")
    is_tor = ip_type == "tor"
    is_known_vpn = ip_type in ("vpn", "datacenter", "proxy")

    # L2: Geo-Velocity
    l2_score, l2_evidence = _layer2_geo_velocity(session, previous_sessions)

    # L3: Fingerprint Consistency
    l3_score, l3_evidence = _layer3_fingerprint_consistency(session, previous_fingerprints)

    # L4: Timezone Mismatch
    l4_score, l4_evidence = _layer4_timezone_mismatch(session, ip_country_code)

    # L5: Graph Consistency
    l5_score, l5_evidence = _layer5_graph_consistency(session, cluster_ip_map)

    # Compute weighted composite score
    layer_scores = {
        "L1_ip_intel": l1_score,
        "L2_geo_velocity": l2_score,
        "L3_fingerprint": l3_score,
        "L4_timezone": l4_score,
        "L5_graph_consistency": l5_score,
    }

    composite = (
        l1_score * LAYER_WEIGHTS["L1"]
        + l2_score * LAYER_WEIGHTS["L2"]
        + l3_score * LAYER_WEIGHTS["L3"]
        + l4_score * LAYER_WEIGHTS["L4"]
        + l5_score * LAYER_WEIGHTS["L5"]
    )
    composite = round(min(1.0, max(0.0, composite)), 4)

    is_vpn = composite >= VPN_DETECTION_THRESHOLD
    risk_boost = round(min(RISK_BOOST_MAX, composite * 0.4), 4) if is_vpn else 0.0

    # Build explanation
    triggered_layers = []
    if l1_score >= 0.5:
        triggered_layers.append(f"L1-IP thuộc {ip_type} ({asn_org})")
    if l2_score >= 0.3:
        speed = l2_evidence.get("speed_kmh", 0)
        triggered_layers.append(f"L2-Di chuyển bất khả thi ({speed:.0f} km/h)")
    if l3_score >= 0.2:
        n_fps = l3_evidence.get("total_unique_fingerprints", 0)
        triggered_layers.append(f"L3-{n_fps} thiết bị khác nhau")
    if l4_score >= 0.3:
        triggered_layers.append(f"L4-Timezone không khớp ({l4_evidence.get('verdict', '')})")
    if l5_score >= 0.3:
        triggered_layers.append(f"L5-Cluster chia sẻ IP ({l5_evidence.get('verdict', '')})")

    if is_vpn:
        explanation = f"⚠️ VPN/Proxy phát hiện (điểm: {composite:.2f}). Tầng cảnh báo: {'; '.join(triggered_layers) or 'Tổng hợp nhiều chỉ số'}"
    elif composite > 0.3:
        explanation = f"🟡 Đáng ngờ (điểm: {composite:.2f}). Dấu hiệu: {'; '.join(triggered_layers) or 'Nhẹ'}"
    else:
        explanation = f"🟢 Kết nối sạch (điểm: {composite:.2f})"

    return VPNAnalysisResult(
        composite_score=composite,
        is_vpn_detected=is_vpn,
        layer_scores=layer_scores,
        risk_boost=risk_boost,
        explanation=explanation,
        evidence={
            "L1": l1_evidence,
            "L2": l2_evidence,
            "L3": l3_evidence,
            "L4": l4_evidence,
            "L5": l5_evidence,
        },
        ip_type=ip_type,
        is_tor=is_tor,
        is_known_vpn=is_known_vpn,
        geo_country=ip_country_code,
        asn_org=asn_org,
    )


# ────────────────────────────────────────────────────────────
#  Batch Analysis (for forensic investigation)
# ────────────────────────────────────────────────────────────

def analyze_batch(
    sessions: List[SessionData],
    history_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    fingerprint_map: Optional[Dict[str, List[str]]] = None,
    cluster_ip_map: Optional[Dict[str, List[str]]] = None,
) -> List[VPNAnalysisResult]:
    """
    Analyze multiple sessions at once. Useful for retrospective forensic sweeps.
    """
    if history_map is None:
        history_map = {}
    if fingerprint_map is None:
        fingerprint_map = {}

    results = []
    for session in sessions:
        prev_sessions = history_map.get(session.tax_code, [])
        prev_fps = fingerprint_map.get(session.tax_code, [])

        result = analyze_session(
            session=session,
            previous_sessions=prev_sessions,
            previous_fingerprints=prev_fps,
            cluster_ip_map=cluster_ip_map or {},
        )
        results.append(result)

    return results


# ────────────────────────────────────────────────────────────
#  Statistics / Reporting
# ────────────────────────────────────────────────────────────

def compute_vpn_statistics(results: List[VPNAnalysisResult]) -> Dict[str, Any]:
    """Aggregate statistics from a batch of VPN analysis results."""
    if not results:
        return {"total": 0}

    total = len(results)
    vpn_detected = sum(1 for r in results if r.is_vpn_detected)
    suspicious = sum(1 for r in results if 0.3 <= r.composite_score < VPN_DETECTION_THRESHOLD)
    clean = total - vpn_detected - suspicious

    scores = [r.composite_score for r in results]
    avg_score = sum(scores) / total

    # Per-layer average
    layer_avgs = {}
    for key in ["L1_ip_intel", "L2_geo_velocity", "L3_fingerprint", "L4_timezone", "L5_graph_consistency"]:
        vals = [r.layer_scores.get(key, 0.0) for r in results]
        layer_avgs[key] = round(sum(vals) / total, 4)

    return {
        "total": total,
        "vpn_detected": vpn_detected,
        "suspicious": suspicious,
        "clean": clean,
        "vpn_rate_pct": round(vpn_detected / total * 100, 2),
        "avg_composite_score": round(avg_score, 4),
        "layer_averages": layer_avgs,
        "ip_type_distribution": _count_distribution([r.ip_type for r in results]),
    }


def _count_distribution(values: List[str]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for v in values:
        dist[v] = dist.get(v, 0) + 1
    return dist
