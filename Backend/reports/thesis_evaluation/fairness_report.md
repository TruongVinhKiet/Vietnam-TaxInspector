# Fairness Analysis

- Rows: `5001`
- Eval rows: `1251`
- Overall FPR: `0.009236`
- Overall TPR: `0.566667`
- Disparate impact pass: `False`

| Dimension | Groups | FPR DI | Selection DI | Equal Opp. Diff | Red flag |
|---|---:|---:|---:|---:|---:|
| industry | 15 | 0.0 | 0.0 | 1.0 | True |
| revenue_bucket | 3 | 0.0 | 0.59239 | 0.219048 | True |
| province | 32 | 0.0 | 0.0 | 1.0 | True |
| company_age | 2 | 0.0 | 0.607954 | 0.328841 | True |
