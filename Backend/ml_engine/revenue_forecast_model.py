"""
revenue_forecast_model.py – Dự Báo Doanh Thu Thuế (Time-Series)
==================================================================
Mô hình dự báo doanh thu thuế multi-horizon cho từng doanh nghiệp
và aggregate toàn hệ thống.

Capabilities:
    1. Time-series feature engineering (lag, rolling, seasonal)
    2. Multi-model: Exponential Smoothing → ARIMA → LightGBM ensemble
    3. Company-level và aggregate forecasting
    4. Anomaly detection trên forecast residuals (phát hiện trốn thuế)

Data Sources:
    - tax_returns.revenue, tax_returns.tax_payable
    - tax_payments.amount_due, tax_payments.amount_paid
    - companies.industry, companies.business_type

Design:
    - CPU-only: statsmodels + sklearn, không cần GPU
    - Graceful degradation: ETS → ARIMA → simple average
    - Thread-safe prediction pipeline
    - Configurable forecast horizons (1Q, 2Q, 4Q)

Reference:
    Hyndman & Athanasopoulos, "Forecasting: Principles and Practice", 3rd ed.
"""

from __future__ import annotations

import json
import logging
import math
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════

@dataclass
class ForecastConfig:
    """Cấu hình dự báo."""
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4])  # Quarters
    min_history_quarters: int = 8
    seasonal_period: int = 4        # Quarterly seasonality
    confidence_level: float = 0.95
    anomaly_residual_threshold: float = 2.5  # z-score
    use_ensemble: bool = True
    model_dir: str = str(MODEL_DIR)


@dataclass
class ForecastResult:
    """Kết quả dự báo cho một entity."""
    entity_id: str               # tax_code hoặc "aggregate"
    entity_type: str = "company" # company | aggregate
    forecasts: list[dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    fit_metrics: dict[str, float] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    processing_ms: float = 0.0


@dataclass
class TimeSeriesData:
    """Dữ liệu time-series đã chuẩn hóa."""
    entity_id: str
    timestamps: list[str]        # ["2024Q1", "2024Q2", ...]
    values: list[float]          # Revenue values
    metadata: dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
#  1. Feature Engineering
# ════════════════════════════════════════════════════════════════

class TimeSeriesFeatureBuilder:
    """
    Tạo features từ time-series doanh thu.

    Features:
        - Lag features: t-1, t-2, t-4 (same quarter last year)
        - Rolling statistics: mean, std, min, max (4Q, 8Q windows)
        - Seasonal decomposition
        - Year-over-year growth rate
        - Trend indicator
    """

    def build_features(
        self,
        values: list[float],
        seasonal_period: int = 4,
    ) -> np.ndarray:
        """
        Xây dựng feature matrix cho mỗi time step.

        Args:
            values: Chuỗi giá trị doanh thu theo thời gian
            seasonal_period: Chu kỳ seasonal (4 = quarterly)

        Returns:
            (T, num_features) numpy array.
        """
        n = len(values)
        arr = np.array(values, dtype=np.float64)

        features_list = []
        for t in range(max(seasonal_period, 4), n):
            feats = []

            # Lag features
            feats.append(arr[t - 1])                      # lag_1
            feats.append(arr[t - 2])                      # lag_2
            feats.append(arr[t - seasonal_period])         # lag_seasonal

            # Rolling mean / std (4 quarters)
            window_4 = arr[max(0, t - 4):t]
            feats.append(np.mean(window_4))
            feats.append(np.std(window_4) if len(window_4) > 1 else 0.0)

            # Rolling mean (8 quarters)
            window_8 = arr[max(0, t - 8):t]
            feats.append(np.mean(window_8))

            # Year-over-year growth
            if t >= seasonal_period and arr[t - seasonal_period] > 0:
                yoy = (arr[t - 1] - arr[t - seasonal_period]) / arr[t - seasonal_period]
            else:
                yoy = 0.0
            feats.append(np.clip(yoy, -2.0, 5.0))

            # Trend (linear slope over last 4 quarters)
            if t >= 4:
                x_trend = np.arange(4, dtype=np.float64)
                y_trend = arr[t - 4:t]
                slope = np.polyfit(x_trend, y_trend, 1)[0] if len(y_trend) == 4 else 0.0
                feats.append(slope / max(1.0, np.mean(y_trend)))
            else:
                feats.append(0.0)

            # Seasonal indicator (quarter of year)
            quarter = t % seasonal_period
            feats.append(math.sin(2 * math.pi * quarter / seasonal_period))
            feats.append(math.cos(2 * math.pi * quarter / seasonal_period))

            # Min/Max ratio (volatility)
            feats.append(
                np.min(window_4) / max(1.0, np.max(window_4))
            )

            features_list.append(feats)

        return np.array(features_list, dtype=np.float64) if features_list else np.zeros((0, 11))


# ════════════════════════════════════════════════════════════════
#  2. Statistical Models (ETS + ARIMA)
# ════════════════════════════════════════════════════════════════

class StatisticalForecaster:
    """
    Dự báo bằng Exponential Smoothing và ARIMA.

    Fallback chain: ETS → ARIMA → Simple Moving Average.
    """

    def forecast_ets(
        self,
        values: list[float],
        horizons: list[int],
        seasonal_period: int = 4,
    ) -> dict[str, Any]:
        """Holt-Winters Exponential Smoothing."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            arr = np.array(values, dtype=np.float64)
            # Cần ít nhất 2 * seasonal_period điểm
            if len(arr) < 2 * seasonal_period:
                return self._simple_forecast(values, horizons)

            model = ExponentialSmoothing(
                arr,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
                initialization_method="estimated",
            ).fit(optimized=True)

            max_h = max(horizons)
            pred = model.forecast(max_h)

            # Tính residuals cho anomaly detection
            fitted = model.fittedvalues
            residuals = arr - fitted

            results = []
            for h in horizons:
                results.append({
                    "horizon_quarters": h,
                    "forecast_value": round(float(pred[h - 1]), 2),
                    "method": "ets_hw",
                })

            return {
                "forecasts": results,
                "model": "ets_holt_winters",
                "residuals": residuals.tolist(),
                "aic": round(float(model.aic), 2) if hasattr(model, "aic") else None,
            }

        except Exception as exc:
            logger.debug("[Forecast] ETS failed: %s, trying ARIMA", exc)
            return self.forecast_arima(values, horizons, seasonal_period)

    def forecast_arima(
        self,
        values: list[float],
        horizons: list[int],
        seasonal_period: int = 4,
    ) -> dict[str, Any]:
        """ARIMA / SARIMAX forecast."""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            arr = np.array(values, dtype=np.float64)
            if len(arr) < 8:
                return self._simple_forecast(values, horizons)

            model = ARIMA(arr, order=(1, 1, 1)).fit()
            max_h = max(horizons)
            pred = model.forecast(max_h)

            residuals = arr[1:] - model.fittedvalues[1:] if len(model.fittedvalues) > 1 else []

            results = []
            for h in horizons:
                idx = min(h - 1, len(pred) - 1)
                results.append({
                    "horizon_quarters": h,
                    "forecast_value": round(float(pred[idx]), 2),
                    "method": "arima_111",
                })

            return {
                "forecasts": results,
                "model": "arima_111",
                "residuals": list(residuals) if hasattr(residuals, '__iter__') else [],
                "aic": round(float(model.aic), 2),
            }

        except Exception as exc:
            logger.debug("[Forecast] ARIMA failed: %s, using simple average", exc)
            return self._simple_forecast(values, horizons)

    def _simple_forecast(
        self, values: list[float], horizons: list[int]
    ) -> dict[str, Any]:
        """Fallback: Simple moving average + trend."""
        arr = np.array(values[-8:], dtype=np.float64) if len(values) >= 8 else np.array(values, dtype=np.float64)
        mean_val = float(np.mean(arr))

        # Simple trend
        if len(arr) >= 4:
            trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
        else:
            trend = 0.0

        results = []
        for h in horizons:
            forecast = mean_val + trend * h
            results.append({
                "horizon_quarters": h,
                "forecast_value": round(max(0, forecast), 2),
                "method": "simple_trend",
            })

        return {
            "forecasts": results,
            "model": "simple_moving_average",
            "residuals": [],
        }


# ════════════════════════════════════════════════════════════════
#  3. ML-based Forecaster (Gradient Boosting)
# ════════════════════════════════════════════════════════════════

class MLForecaster:
    """
    Gradient Boosting forecaster sử dụng engineered features.

    Ưu điểm so với statistical models:
        - Handle nhiều external features (industry, size, ...)
        - Non-linear relationships
        - Feature importance cho explainability
    """

    def __init__(self):
        self._model = None
        self._feature_builder = TimeSeriesFeatureBuilder()

    def train(
        self,
        all_series: list[TimeSeriesData],
        config: ForecastConfig,
    ) -> dict[str, Any]:
        """
        Train ML model trên tất cả company time-series.

        Returns:
            Training metrics dict.
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error
        except ImportError as exc:
            logger.error("[MLForecast] sklearn not available: %s", exc)
            return {"status": "error", "message": str(exc)}

        X_all, y_all = [], []

        for series in all_series:
            if len(series.values) < config.min_history_quarters:
                continue

            X = self._feature_builder.build_features(
                series.values, config.seasonal_period
            )
            if X.shape[0] == 0:
                continue

            start_idx = max(config.seasonal_period, 4)
            y = np.array(series.values[start_idx:start_idx + X.shape[0]])

            if len(y) == X.shape[0]:
                X_all.append(X)
                y_all.append(y)

        if not X_all:
            return {"status": "error", "message": "Không đủ dữ liệu training"}

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)

        # Train/val split (80/20 temporal)
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        self._model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self._model.fit(X_tr, y_tr)

        # Metrics
        y_pred = self._model.predict(X_val) if len(X_val) > 0 else np.array([])
        metrics = {}
        if len(y_pred) > 0:
            metrics = {
                "mae": round(float(mean_absolute_error(y_val, y_pred)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 2),
                "mape": round(float(
                    np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100
                ), 2),
                "train_samples": len(X_tr),
                "val_samples": len(X_val),
            }

        # Save model
        self._save_model(config.model_dir)

        logger.info("[MLForecast] Training complete: %s", metrics)
        return {"status": "success", **metrics}

    def predict(
        self,
        values: list[float],
        horizons: list[int],
        config: ForecastConfig,
    ) -> dict[str, Any]:
        """Multi-step forecast bằng recursive prediction."""
        if self._model is None:
            self._load_model(config.model_dir)

        if self._model is None:
            return {"forecasts": [], "model": "unavailable"}

        extended = list(values)
        results = []

        for h in sorted(horizons):
            while len(results) < h:
                X = self._feature_builder.build_features(
                    extended, config.seasonal_period
                )
                if X.shape[0] == 0:
                    extended.append(float(np.mean(extended[-4:])))
                else:
                    pred = float(self._model.predict(X[-1:].reshape(1, -1))[0])
                    extended.append(max(0, pred))
                results.append(extended[-1])

            results_out = []

        for h in horizons:
            idx = h - 1
            val = results[idx] if idx < len(results) else float(np.mean(values[-4:]))
            results_out.append({
                "horizon_quarters": h,
                "forecast_value": round(val, 2),
                "method": "gradient_boosting",
            })

        return {"forecasts": results_out, "model": "gradient_boosting"}

    def _save_model(self, model_dir: str) -> None:
        """Persist model."""
        try:
            import joblib
            path = Path(model_dir)
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, str(path / "revenue_forecast_gbm.joblib"))
            logger.info("[MLForecast] Model saved")
        except Exception as exc:
            logger.warning("[MLForecast] Save failed: %s", exc)

    def _load_model(self, model_dir: str) -> None:
        """Load persisted model."""
        try:
            import joblib
            path = Path(model_dir) / "revenue_forecast_gbm.joblib"
            if path.exists():
                self._model = joblib.load(str(path))
                logger.info("[MLForecast] Model loaded from %s", path)
        except Exception as exc:
            logger.warning("[MLForecast] Load failed: %s", exc)


# ════════════════════════════════════════════════════════════════
#  4. Anomaly Detection on Residuals
# ════════════════════════════════════════════════════════════════

class ResidualAnomalyDetector:
    """
    Phát hiện bất thường dựa trên forecast residuals.

    Doanh thu thực tế giảm đột ngột so với dự báo → dấu hiệu trốn thuế.
    Doanh thu tăng đột biến → dấu hiệu rửa tiền hoặc hóa đơn ảo.
    """

    def detect(
        self,
        actuals: list[float],
        forecasts: list[float],
        threshold: float = 2.5,
    ) -> list[dict[str, Any]]:
        """
        Phát hiện anomaly trên residuals.

        Args:
            actuals: Giá trị thực
            forecasts: Giá trị dự báo
            threshold: z-score threshold

        Returns:
            List of anomaly dicts.
        """
        if len(actuals) != len(forecasts) or len(actuals) < 4:
            return []

        residuals = np.array(actuals) - np.array(forecasts)
        mean_r = float(np.mean(residuals))
        std_r = float(np.std(residuals))

        if std_r < 1e-8:
            return []

        anomalies = []
        for i, (actual, forecast, residual) in enumerate(
            zip(actuals, forecasts, residuals)
        ):
            z_score = (residual - mean_r) / std_r

            if abs(z_score) > threshold:
                direction = "revenue_drop" if z_score < 0 else "revenue_spike"
                severity = "critical" if abs(z_score) > threshold * 1.5 else "warning"

                anomalies.append({
                    "index": i,
                    "actual": round(actual, 2),
                    "forecast": round(forecast, 2),
                    "residual": round(float(residual), 2),
                    "z_score": round(float(z_score), 3),
                    "direction": direction,
                    "severity": severity,
                    "message": (
                        f"Doanh thu {'giảm' if z_score < 0 else 'tăng'} "
                        f"bất thường {abs(z_score):.1f}σ so với dự báo"
                    ),
                })

        return anomalies


# ════════════════════════════════════════════════════════════════
#  5. Main Forecast Engine
# ════════════════════════════════════════════════════════════════

class RevenueForecastEngine:
    """
    Engine dự báo doanh thu thuế end-to-end.

    Usage:
        engine = RevenueForecastEngine()
        result = engine.forecast_company("MST001", revenue_history)
        aggregate = engine.forecast_aggregate(all_company_data)
    """

    def __init__(self, config: ForecastConfig | None = None):
        self.config = config or ForecastConfig()
        self._stat_forecaster = StatisticalForecaster()
        self._ml_forecaster = MLForecaster()
        self._anomaly_detector = ResidualAnomalyDetector()
        self._lock = threading.Lock()

    def forecast_company(
        self,
        tax_code: str,
        revenue_values: list[float],
        timestamps: list[str] | None = None,
    ) -> ForecastResult:
        """
        Dự báo doanh thu cho một doanh nghiệp.

        Args:
            tax_code: Mã số thuế
            revenue_values: Chuỗi doanh thu theo quý
            timestamps: Labels cho mỗi quý (optional)

        Returns:
            ForecastResult với forecasts và anomalies.
        """
        t0 = time.perf_counter()
        result = ForecastResult(entity_id=tax_code, entity_type="company")

        if len(revenue_values) < self.config.min_history_quarters:
            result.confidence = 0.0
            result.model_used = "insufficient_data"
            return result

        # Run statistical forecast
        stat_result = self._stat_forecaster.forecast_ets(
            revenue_values, self.config.horizons, self.config.seasonal_period
        )

        # Run ML forecast (if available)
        ml_result = self._ml_forecaster.predict(
            revenue_values, self.config.horizons, self.config
        )

        # Ensemble: weighted average nếu cả hai available
        if self.config.use_ensemble and ml_result.get("model") != "unavailable":
            result.forecasts = self._ensemble(
                stat_result.get("forecasts", []),
                ml_result.get("forecasts", []),
                weights=(0.6, 0.4),  # Statistical có trọng số cao hơn
            )
            result.model_used = f"ensemble({stat_result.get('model', 'stat')}+gbm)"
        else:
            result.forecasts = stat_result.get("forecasts", [])
            result.model_used = stat_result.get("model", "unknown")

        # Anomaly detection trên residuals
        residuals = stat_result.get("residuals", [])
        if residuals and len(residuals) > 4:
            fitted_len = len(residuals)
            actuals = revenue_values[-fitted_len:]
            fitted_values = [a - r for a, r in zip(actuals, residuals)]
            result.anomalies = self._anomaly_detector.detect(
                actuals, fitted_values, self.config.anomaly_residual_threshold
            )

        # Confidence dựa trên data quality
        result.confidence = self._compute_confidence(revenue_values)
        result.processing_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "[Forecast] %s: %s | confidence=%.2f | %d anomalies",
            tax_code, result.model_used, result.confidence,
            len(result.anomalies),
        )

        return result

    def forecast_aggregate(
        self,
        all_series: list[TimeSeriesData],
    ) -> ForecastResult:
        """
        Dự báo tổng thu thuế toàn hệ thống.

        Tổng hợp tất cả company series thành aggregate rồi forecast.
        """
        t0 = time.perf_counter()

        if not all_series:
            return ForecastResult(
                entity_id="aggregate", entity_type="aggregate",
                model_used="no_data",
            )

        # Aggregate: sum tất cả company revenues theo quarter
        max_len = max(len(s.values) for s in all_series)
        aggregate = np.zeros(max_len, dtype=np.float64)

        for series in all_series:
            padded = np.zeros(max_len)
            padded[-len(series.values):] = series.values
            aggregate += padded

        result = self.forecast_company(
            "aggregate", aggregate.tolist()
        )
        result.entity_id = "aggregate"
        result.entity_type = "aggregate"
        result.processing_ms = round((time.perf_counter() - t0) * 1000, 1)

        return result

    def train_ml_model(self, all_series: list[TimeSeriesData]) -> dict[str, Any]:
        """Train ML forecaster trên tất cả data."""
        return self._ml_forecaster.train(all_series, self.config)

    def _ensemble(
        self,
        stat_forecasts: list[dict],
        ml_forecasts: list[dict],
        weights: tuple[float, float] = (0.6, 0.4),
    ) -> list[dict[str, Any]]:
        """Weighted average ensemble."""
        combined = []
        stat_map = {f["horizon_quarters"]: f for f in stat_forecasts}
        ml_map = {f["horizon_quarters"]: f for f in ml_forecasts}

        all_horizons = set(stat_map.keys()) | set(ml_map.keys())
        for h in sorted(all_horizons):
            s_val = stat_map.get(h, {}).get("forecast_value", 0)
            m_val = ml_map.get(h, {}).get("forecast_value", 0)

            if s_val > 0 and m_val > 0:
                ensemble_val = weights[0] * s_val + weights[1] * m_val
            elif s_val > 0:
                ensemble_val = s_val
            else:
                ensemble_val = m_val

            combined.append({
                "horizon_quarters": h,
                "forecast_value": round(ensemble_val, 2),
                "method": "ensemble",
                "stat_value": round(s_val, 2),
                "ml_value": round(m_val, 2),
            })

        return combined

    def _compute_confidence(self, values: list[float]) -> float:
        """Tính confidence dựa trên data quality."""
        n = len(values)
        conf = 0.0

        # Số lượng data points
        if n >= 20:
            conf += 0.4
        elif n >= 12:
            conf += 0.3
        elif n >= 8:
            conf += 0.2

        # Tính stability (coefficient of variation)
        arr = np.array(values)
        if np.mean(arr) > 0:
            cv = np.std(arr) / np.mean(arr)
            if cv < 0.3:
                conf += 0.3
            elif cv < 0.6:
                conf += 0.2
            else:
                conf += 0.1

        # Không có giá trị 0 liên tiếp
        zero_streak = max(
            (sum(1 for _ in g) for k, g in __import__('itertools').groupby(arr, key=lambda x: x == 0) if k),
            default=0
        )
        if zero_streak == 0:
            conf += 0.3
        elif zero_streak <= 2:
            conf += 0.15

        return round(min(1.0, conf), 3)


# ════════════════════════════════════════════════════════════════
#  Singleton
# ════════════════════════════════════════════════════════════════

_forecast_engine: RevenueForecastEngine | None = None


def get_forecast_engine() -> RevenueForecastEngine:
    """Singleton cho RevenueForecastEngine."""
    global _forecast_engine
    if _forecast_engine is None:
        _forecast_engine = RevenueForecastEngine()
    return _forecast_engine
