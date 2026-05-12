# Experimental Evaluation Metrics

- Generated at: `2026-05-11T02:01:59Z`
- Seed: `42`
- Fraud records: `120000`
- Delinquency records: `120000`
- CV folds: `3`

## Fraud

| Model | Precision | Recall | F1 | AUC-ROC | Avg. Prec. |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.110±0.000 | 0.937±0.005 | 0.196±0.001 | 0.866±0.003 | 0.451±0.002 |
| Random Forest | 0.148±0.001 | 0.919±0.005 | 0.254±0.001 | 0.916±0.003 | 0.574±0.008 |
| XGBoost/GBM (standalone) | 0.835±0.022 | 0.411±0.006 | 0.551±0.009 | 0.921±0.003 | 0.597±0.008 |
| Isolation Forest (standalone) | 0.105±0.008 | 0.569±0.001 | 0.177±0.011 | 0.718±0.003 | 0.293±0.010 |
| XGBoost/GBM + Graph Features | 0.840±0.026 | 0.408±0.003 | 0.549±0.007 | 0.920±0.003 | 0.594±0.008 |
| XGBoost/GBM + IF + Calibrator | 0.153±0.004 | 0.907±0.014 | 0.262±0.006 | 0.919±0.001 | 0.594±0.007 |
| Hybrid (GBM+IF+Graph) | 0.151±0.005 | 0.905±0.016 | 0.258±0.007 | 0.918±0.002 | 0.591±0.008 |

## Delinquency

| Model | Precision | Recall | F1 | AUC-ROC | RMSE days |
|---|---:|---:|---:|---:|---:|
| Statistical Baseline (Z-score) | 0.474±0.015 | 0.402±0.034 | 0.434±0.013 | 0.696±0.003 | 12.5±0.1 |
| Logistic Regression | 0.317±0.001 | 0.913±0.004 | 0.471±0.001 | 0.735±0.003 | 31.4±0.1 |
| Random Forest | 0.310±0.001 | 0.926±0.005 | 0.465±0.001 | 0.728±0.004 | 29.0±0.0 |
| LightGBM-compatible GBDT (delinquency-temporal-v1) | 0.463±0.003 | 0.562±0.007 | 0.508±0.005 | 0.731±0.004 | 14.8±0.0 |
| Temporal sequence model (Transformer proxy) | 0.466±0.004 | 0.559±0.008 | 0.508±0.005 | 0.730±0.004 | 14.5±0.1 |

## Agent

```json
{
  "legal_route_accuracy": 0.9967,
  "faq_grounding_rate": 0.6967,
  "actionable_steps_rate": 1.0,
  "citation_or_reference_rate": 1.0,
  "mean_confidence": 0.571,
  "mean_latency_ms": 14.924
}
```