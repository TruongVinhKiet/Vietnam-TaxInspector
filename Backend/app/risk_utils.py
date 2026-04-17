"""Shared risk classification helpers used across routers and ML modules."""


def classify_delinquency_cluster(prob_30d: float, prob_60d: float, prob_90d: float) -> str:
    """Assign a human-readable delinquency risk cluster from 30/60/90 probabilities."""
    if prob_30d >= 0.7:
        return "Nhóm rủi ro cao"
    if prob_30d >= 0.5:
        return "Rủi ro trung bình-cao"
    if prob_60d >= 0.5:
        return "Vấn đề Dòng tiền"
    if prob_90d >= 0.5:
        return "Suy giảm Theo mùa"
    if prob_90d >= 0.3:
        return "Theo dõi"
    return "Ổn định"