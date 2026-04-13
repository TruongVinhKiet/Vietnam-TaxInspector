"""
feature_engineering.py – Tính toán Đặc trưng Nghiệp vụ Thuế (Tax Features)
=============================================================================
Biến đổi dữ liệu thô (Doanh thu, Chi phí, VAT...) thành các Features
mang tính phân tích nghiệp vụ cao phục vụ cho mô hình AI.

Features:
    F1 – Lệch pha Tăng trưởng (Growth Divergence)
    F2 – Tỷ lệ Chi phí/Doanh thu (Expense Ratio)
    F3 – Cấu trúc VAT (VAT Structure Ratio)
    F4 – So sánh Ngành (Peer Comparison)
"""

import pandas as pd
import numpy as np
from typing import Optional


class TaxFeatureEngineer:
    """
    Transform raw financial data into ML-ready features.
    Expects a DataFrame with columns:
        tax_code, year, revenue, cost_of_goods, operating_expenses,
        total_expenses, net_profit, vat_input, vat_output,
        industry, industry_avg_profit_margin
    """

    # All engineered feature column names
    FEATURE_COLS = [
        "f1_divergence",
        "f2_ratio_limit",
        "f3_vat_structure",
        "f4_peer_comparison",
        "revenue_log",
        "expense_log",
        "profit_margin",
        "revenue_growth_rate",
        "expense_growth_rate",
        "vat_net_ratio",
    ]

    def __init__(self):
        pass

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method: compute all features for the full DataFrame.
        Returns a new DataFrame with original columns + engineered features.
        """
        df = df.copy()

        # Sort by company & year for correct lag computations
        df = df.sort_values(["tax_code", "year"]).reset_index(drop=True)

        # --- Basic derived columns ---
        df["profit_margin"] = np.where(
            df["revenue"] > 0,
            df["net_profit"] / df["revenue"],
            0.0
        )

        df["revenue_log"] = np.log1p(df["revenue"].clip(lower=0))
        df["expense_log"] = np.log1p(df["total_expenses"].clip(lower=0))

        # --- Lag columns (previous year values per company) ---
        df["prev_revenue"] = df.groupby("tax_code")["revenue"].shift(1)
        df["prev_expenses"] = df.groupby("tax_code")["total_expenses"].shift(1)

        # --- Growth rates ---
        df["revenue_growth_rate"] = np.where(
            (df["prev_revenue"].notna()) & (df["prev_revenue"] > 0),
            df["revenue"] / df["prev_revenue"],
            1.0
        )
        df["expense_growth_rate"] = np.where(
            (df["prev_expenses"].notna()) & (df["prev_expenses"] > 0),
            df["total_expenses"] / df["prev_expenses"],
            1.0
        )

        # ============================================================
        # F1: Lệch pha Tăng trưởng (Growth Divergence)
        # F1 = Revenue_Growth - Expense_Growth
        # F1 << 0 → Chi phí tăng nhanh hơn doanh thu rất nhiều
        # ============================================================
        df["f1_divergence"] = df["revenue_growth_rate"] - df["expense_growth_rate"]

        # ============================================================
        # F2: Tỷ lệ Chi phí/Doanh thu (Expense Ratio Limit)
        # F2 = total_expenses / revenue
        # F2 > 0.98 liên tục → Dấu hiệu chuyển giá
        # ============================================================
        df["f2_ratio_limit"] = np.where(
            df["revenue"] > 0,
            df["total_expenses"] / df["revenue"],
            0.0
        )

        # ============================================================
        # F3: Cấu trúc VAT (VAT Input/Output Ratio)
        # F3 = vat_input / vat_output
        # F3 ≈ 1 hoặc > 1 → Dấu hiệu mua bán hóa đơn lòng vòng
        # ============================================================
        df["f3_vat_structure"] = np.where(
            df["vat_output"] > 0,
            df["vat_input"] / df["vat_output"],
            0.0
        )

        # ============================================================
        # F4: So sánh Ngành (Peer Comparison)
        # F4 = profit_margin(DN) - industry_avg_profit_margin
        # F4 << 0 → DN có biên lợi nhuận thấp bất thường so với ngành
        # ============================================================
        df["f4_peer_comparison"] = df["profit_margin"] - df["industry_avg_profit_margin"]

        # --- VAT net ratio (supplementary feature) ---
        df["vat_net_ratio"] = np.where(
            df["vat_output"] > 0,
            (df["vat_output"] - df["vat_input"]) / df["vat_output"],
            0.0
        )

        # --- Clean up temporary columns ---
        df.drop(columns=["prev_revenue", "prev_expenses"], inplace=True, errors="ignore")

        # --- Fill NaN for first-year (no lag) ---
        for col in self.FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the feature matrix (X) for ML model input.
        Should be called AFTER compute_features().
        """
        return df[self.FEATURE_COLS].values.astype(np.float64)

    def generate_red_flags(self, row: pd.Series) -> list[dict]:
        """
        Sinh cảnh báo dấu hiệu bất thường (Red Flags) bằng tiếng Việt có dấu.
        Trả về list of {icon, title, description} dicts.
        """
        flags = []

        # F1: Chi phí tăng nhanh hơn doanh thu
        if row.get("f1_divergence", 0) < -0.3:
            rev_g = row.get("revenue_growth_rate", 1)
            exp_g = row.get("expense_growth_rate", 1)
            flags.append({
                "icon": "warning",
                "title": f"Chi phí tăng {exp_g:.0%} nhưng doanh thu chỉ tăng {rev_g:.0%}",
                "description": "Lệch pha tăng trưởng bất thường: chi phí tăng nhanh hơn doanh thu rất nhiều, dấu hiệu đội chi phí đầu vào."
            })

        # F2: Tỷ lệ chi phí/doanh thu gần 100%
        if row.get("f2_ratio_limit", 0) > 0.95:
            ratio_pct = row.get("f2_ratio_limit", 0) * 100
            flags.append({
                "icon": "trending_up",
                "title": f"Tỷ lệ chi phí/doanh thu đạt {ratio_pct:.1f}%",
                "description": "Đẩy cao chi phí đầu vào để triệt tiêu lợi nhuận tính thuế thu nhập doanh nghiệp."
            })

        # F3: Nghi vấn mua bán hoá đơn vòng lặp
        if row.get("f3_vat_structure", 0) > 0.90:
            vat_ratio = row.get("f3_vat_structure", 0) * 100
            flags.append({
                "icon": "link_off",
                "title": f"VAT đầu vào chiếm {vat_ratio:.1f}% VAT đầu ra",
                "description": "Dấu hiệu mua bán hoá đơn vòng lặp (carousel fraud), số thuế VAT phải nộp gần bằng 0."
            })

        # F4: Biên lợi nhuận thấp hơn ngành
        if row.get("f4_peer_comparison", 0) < -0.08:
            margin_pct = row.get("profit_margin", 0) * 100
            avg_pct = row.get("industry_avg_profit_margin", 0) * 100
            flags.append({
                "icon": "analytics",
                "title": f"Biên lợi nhuận {margin_pct:.1f}% – thấp hơn ngành ({avg_pct:.1f}%)",
                "description": "Tỷ suất lợi nhuận thấp bất thường so với trung bình ngành, nghi vấn chuyển giá hoặc khai giảm doanh thu."
            })

        # Anomaly score cao
        if row.get("anomaly_score", 0) > 0.6:
            anomaly_pct = row.get("anomaly_score", 0) * 100
            flags.append({
                "icon": "psychology",
                "title": f"AI phát hiện hình thái tài chính dị biệt ({anomaly_pct:.0f}%)",
                "description": "Mô hình Isolation Forest xác định cấu trúc tài chính của doanh nghiệp này khác biệt vượt trội so với đa số DN trong cùng ngành."
            })

        return flags
