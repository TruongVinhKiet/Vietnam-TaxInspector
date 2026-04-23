"""
train_simulation.py – Training Script for Macro Simulation Regressor
====================================================================
Generates synthetic data from current industry baselines and trains a 
LightGBM regression model to predict the simulated delinquency rate.

Outputs:
    - data/models/simulation_lgbm.joblib
    - data/models/simulation_config.json
    - data/models/simulation_quality_report.json
    - data/models/simulation_drift_baseline.json
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "simulation-macro-v1"
FEATURE_NAMES = [
    "vat",
    "cit",
    "audit",
    "penalty",
    "interest",
    "growth",
    "base_rate",
    "avg_margin",
    "company_count",
]
SIM_ACCEPTANCE_R2_MIN = 0.90
SIM_ACCEPTANCE_RMSE_MAX = 0.08
SIM_MIN_TRAINING_SAMPLES = max(10_000, int(os.environ.get("SIMULATION_MIN_REQUIRED_SAMPLES", "10000")))

# Baseline Params
BASELINE_VAT = 10.0
BASELINE_CIT = 20.0
BASELINE_AUDIT = 5.0
BASELINE_PENALTY = 1.0
BASELINE_INTEREST = 6.0
BASELINE_GROWTH = 6.5


def _build_feature_stats(matrix: np.ndarray, feature_names: list[str]) -> dict[str, dict]:
    if matrix.size == 0:
        return {}

    stats: dict[str, dict] = {}
    for idx, name in enumerate(feature_names):
        values = np.asarray(matrix[:, idx], dtype=float)
        if values.size == 0:
            continue
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q0": float(np.min(values)),
            "q5": float(np.percentile(values, 5)),
            "q25": float(np.percentile(values, 25)),
            "q50": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
            "q95": float(np.percentile(values, 95)),
            "q100": float(np.max(values)),
        }
    return stats


def _build_drift_feature_matrix(industry_baselines: list[dict]) -> np.ndarray:
    rows = []
    for baseline in industry_baselines:
        rows.append([
            BASELINE_VAT,
            BASELINE_CIT,
            BASELINE_AUDIT,
            BASELINE_PENALTY,
            BASELINE_INTEREST,
            BASELINE_GROWTH,
            float(baseline.get("base_rate", 0.0)),
            float(baseline.get("avg_margin", 0.0)),
            float(baseline.get("company_count", 0.0)),
        ])
    return np.asarray(rows, dtype=float) if rows else np.asarray([])


def fetch_industry_baselines(db_url: str) -> list[dict]:
    import psycopg2
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Same logic as routers/simulation.py
    where_sql = "c.industry IS NOT NULL AND c.industry != '' AND c.industry != 'Offshore Entity'"
    
    cur.execute(f"""
        SELECT 
            c.industry,
            COUNT(DISTINCT c.tax_code) as company_count,
            COALESCE(AVG(tr.revenue), 0) as avg_revenue
        FROM companies c
        LEFT JOIN tax_returns tr ON tr.tax_code = c.tax_code
        WHERE {where_sql}
        GROUP BY c.industry
    """)
    rows = cur.fetchall()
    
    cur.execute(f"""
        SELECT
            c.industry,
            COUNT(DISTINCT dp.tax_code) as delinquent_count,
            COUNT(DISTINCT c.tax_code) as total_count
        FROM companies c
        LEFT JOIN delinquency_predictions dp ON dp.tax_code = c.tax_code AND dp.prob_90d >= 0.5
        WHERE {where_sql}
        GROUP BY c.industry
    """)
    delinq_rows = cur.fetchall()
    
    # Overdue fallback
    cur.execute("""
        SELECT c.industry, COUNT(DISTINCT tp.tax_code)
        FROM tax_payments tp
        JOIN companies c ON c.tax_code = tp.tax_code
        WHERE tp.status IN ('overdue', 'partial')
        GROUP BY c.industry
    """)
    overdue_rows = cur.fetchall()
    overdue_map = {r[0]: r[1] for r in overdue_rows}
    
    delinq_map = {r[0]: {"delinq_count": r[1], "total": r[2]} for r in delinq_rows}
    
    margins = {
        "Xây dựng": 0.06, "Bất động sản": 0.12, "Thương mại XNK": 0.04,
        "Sản xuất công nghiệp": 0.08, "Nông nghiệp": 0.05, "Vận tải & Logistics": 0.07,
        "Công nghệ thông tin": 0.15, "Dịch vụ tài chính": 0.18, "Y tế & Dược phẩm": 0.14,
        "Giáo dục & Đào tạo": 0.10, "Thực phẩm & Đồ uống": 0.09, "May mặc & Giầy da": 0.06,
        "Khoáng sản & Năng lượng": 0.11, "Du lịch & Khách sạn": 0.08, "Viễn thông": 0.13,
    }
    
    results = []
    for industry, count, avg_rev in rows:
        d = delinq_map.get(industry, {"delinq_count": 0, "total": count})
        rate = d["delinq_count"] / max(1, d["total"])
        if rate == 0:
            rate = overdue_map.get(industry, 0) / max(1, count)
            
        margin = margins.get(industry, 0.08)
        rate = max(0.02, min(0.95, rate)) if rate > 0 else max(0.05, margin * 1.5)
        
        results.append({
            "industry": industry,
            "company_count": count,
            "avg_margin": margin,
            "base_rate": rate
        })
        
    cur.close()
    conn.close()
    return results

def generate_synthetic_data(baselines: list[dict], n_samples: int = 10000):
    X = []
    y = []

    # Elasticity
    vat_elasticity = -0.08
    cit_elasticity = -0.05
    audit_elasticity = -0.015
    penalty_elasticity = -0.04
    interest_elasticity = 0.03
    growth_elasticity = -0.02
    
    for _ in range(n_samples):
        # sample macro params
        vat = np.random.uniform(5.0, 25.0)
        cit = np.random.uniform(10.0, 30.0)
        audit = np.random.uniform(1.0, 30.0)
        penalty = np.random.uniform(1.0, 5.0)
        interest = np.random.uniform(2.0, 15.0)
        growth = np.random.uniform(-5.0, 12.0)
        
        d_vat = vat - BASELINE_VAT
        d_cit = cit - BASELINE_CIT
        d_audit = audit - BASELINE_AUDIT
        d_penalty = penalty - BASELINE_PENALTY
        d_interest = interest - BASELINE_INTEREST
        d_growth = growth - BASELINE_GROWTH
        
        delinq_shift = (
            d_vat * vat_elasticity + d_cit * cit_elasticity
            + d_audit * audit_elasticity + d_penalty * penalty_elasticity
            + d_interest * interest_elasticity + d_growth * growth_elasticity
        )
        
        # sample an industry
        b = np.random.choice(baselines)
        margin_sensitivity = max(0.5, 1.0 - b["avg_margin"] * 3)
        
        noise = np.random.normal(0, 0.005) # Add 0.5% noise
        
        target = b["base_rate"] + delinq_shift * margin_sensitivity + noise
        target = max(0.01, min(0.99, target))
        
        X.append([
            vat, cit, audit, penalty, interest, growth,
            b["base_rate"], b["avg_margin"], b["company_count"]
        ])
        y.append(target)
        
    return np.array(X), np.array(y)

def train_model(
    db_url: str,
    sample_size: int = 10000,
    seed: int = 42,
    min_samples: int = SIM_MIN_TRAINING_SAMPLES,
):
    print("=" * 60)
    print("  SIMULATION MODEL TRAINING (Macro Elasticity)")
    print("=" * 60)

    np.random.seed(int(seed))
    required_samples = max(SIM_MIN_TRAINING_SAMPLES, int(min_samples))
    if int(sample_size) < required_samples:
        raise RuntimeError(
            f"Simulation training requires sample_size >= {required_samples:,}; got {int(sample_size):,}."
        )
    
    print("[1/4] Extracting industry baselines from DB...")
    env_url = os.environ.get("DATABASE_URL")
    if not env_url:
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "TaxInspector")
        env_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    if db_url == "default":
        db_url = env_url
        
    baselines = fetch_industry_baselines(db_url)
    if not baselines:
        raise RuntimeError("No industry baselines found.")
        
    print(f"      Matched {len(baselines)} industries.")
    
    print(f"\n[2/4] Generating {sample_size} synthetic data points...")
    X, y = generate_synthetic_data(baselines, sample_size)
    print(f"      X shape: {X.shape}, y shape: {y.shape}")

    drift_matrix = _build_drift_feature_matrix(baselines)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n[3/4] Training LightGBM Regressor...")
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        model_type = "lightgbm"
    except ImportError:
        print("      [WARN] LightGBM not available. Using sklearn GradientBoostingRegressor")
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        model_type = "sklearn_gb"
        
    # Evaluate
    preds = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    
    print(f"      RMSE: {rmse:.4f}")
    print(f"      MAE:  {mae:.4f}")
    print(f"      R2:   {r2:.4f}")
    
    # Feature importances
    features = FEATURE_NAMES
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        # normalize
        imps = imps / imps.sum()
        for f, i in sorted(zip(features, imps), key=lambda x: -x[1]):
            print(f"        {f:15s}: {i:.4f}")
            
    print("\n[4/4] Saving artifacts...")
    model_path = MODEL_DIR / "simulation_lgbm.joblib"
    joblib.dump(model, model_path)
    
    config = {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "features": features,
    }
    with open(MODEL_DIR / "simulation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    acceptance_criteria = {
        "training_samples_min": {
            "pass": bool(int(sample_size) >= int(required_samples)),
            "actual": int(sample_size),
            "threshold": int(required_samples),
        },
        "r2_min": {
            "pass": bool(r2 >= SIM_ACCEPTANCE_R2_MIN),
            "actual": round(r2, 4),
            "threshold": SIM_ACCEPTANCE_R2_MIN,
        },
        "rmse_max": {
            "pass": bool(rmse <= SIM_ACCEPTANCE_RMSE_MAX),
            "actual": round(rmse, 4),
            "threshold": SIM_ACCEPTANCE_RMSE_MAX,
        },
    }
        
    quality = {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4)
        },
        "acceptance_gates": {
            "overall_pass": bool(all(item.get("pass") for item in acceptance_criteria.values())),
            "criteria": acceptance_criteria,
        },
        "dataset": {
            "total_samples": sample_size,
            "required_min_samples": int(required_samples),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    }
    with open(MODEL_DIR / "simulation_quality_report.json", "w") as f:
        json.dump(quality, f, indent=2)

    drift_baseline = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "industry_baseline_snapshot",
        "industry_count": int(len(baselines)),
        "feature_names": features,
        "features": _build_feature_stats(drift_matrix, features),
    }
    with open(MODEL_DIR / "simulation_drift_baseline.json", "w", encoding="utf-8") as f:
        json.dump(drift_baseline, f, indent=2, ensure_ascii=False)

    print("      [OK] Saved artifacts to data/models/")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default="default")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=SIM_MIN_TRAINING_SAMPLES)
    args = parser.parse_args()

    min_samples = max(SIM_MIN_TRAINING_SAMPLES, int(args.min_samples))
    if int(args.sample_size) < min_samples:
        raise SystemExit(f"--sample-size must be >= --min-samples ({min_samples:,})")
    
    train_model(args.db_url, args.sample_size, seed=args.seed, min_samples=min_samples)
