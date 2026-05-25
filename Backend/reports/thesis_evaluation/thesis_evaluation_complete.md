# TaxInspector Comprehensive Thesis Evaluation

- Generated at: `2026-05-21T01:48:38Z`
- Quick mode: `True`
- Rows: `5000`
- Folds: `2`
- Thesis score estimate: `9.35`

## Summary

### ablation

```json
{
  "status": "complete",
  "best_model": "C1_XGB_Graph",
  "key_finding": {
    "delta_auc_vs_B1": 0.050721,
    "relative_delta_pct": 5.6362
  }
}
```

### statistical_significance

```json
{
  "status": "complete",
  "baseline": "B1_XGBoost"
}
```

### fairness

```json
{
  "status": "complete",
  "disparate_impact_pass": false,
  "red_flag_count": 4
}
```

### rag_grounding

```json
{
  "status": "complete",
  "rate": 0.7704,
  "recommended_path": "ingest_curated_expansion_before_reranker_tuning"
}
```

### deep_learning

```json
{
  "status": "complete",
  "gat_f1": 0.692483,
  "vae_f1": 0.647059
}
```

### concept_drift

```json
{
  "status": "complete",
  "adwin_validated": true
}
```

### federated_learning

```json
{
  "status": "complete",
  "fed_vs_central_gap": 0.0
}
```

### user_study

```json
{
  "status": "complete",
  "sus_mean": 82.5,
  "simulated": true
}
```
