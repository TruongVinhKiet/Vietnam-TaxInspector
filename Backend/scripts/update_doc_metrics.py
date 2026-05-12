import json
import re

with open("e:/TaxInspector/Backend/reports/experimental_evaluation_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

def fmt(mean, std):
    return f"{mean:.3f}".replace(".", ",") + "±" + f"{std:.3f}".replace(".", ",")

def fmt_pct(val):
    return f"{val*100:.1f}%".replace(".", ",")

def fmt_ms(val):
    return f"{val:.1f} ms".replace(".", ",")

with open("e:/TaxInspector/doc.js", "r", encoding="utf-8") as f:
    content = f.read()

# Update Bảng 5: Fraud
fraud_models = metrics["fraud"]["models"]
content = re.sub(
    r"(tc\('Logistic Regression \(baseline\)',\{w:2800\}\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["Logistic Regression"]["precision"]["mean"], fraud_models["Logistic Regression"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["Logistic Regression"]["recall"]["mean"], fraud_models["Logistic Regression"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["Logistic Regression"]["f1"]["mean"], fraud_models["Logistic Regression"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["Logistic Regression"]["auc_roc"]["mean"], fraud_models["Logistic Regression"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["Logistic Regression"]["average_precision"]["mean"], fraud_models["Logistic Regression"]["average_precision"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tca\('Random Forest',2800\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["Random Forest"]["precision"]["mean"], fraud_models["Random Forest"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["Random Forest"]["recall"]["mean"], fraud_models["Random Forest"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["Random Forest"]["f1"]["mean"], fraud_models["Random Forest"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["Random Forest"]["auc_roc"]["mean"], fraud_models["Random Forest"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["Random Forest"]["average_precision"]["mean"], fraud_models["Random Forest"]["average_precision"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tc\('XGBoost/GBM \(standalone\)',\{bold:true,w:2800,shade:'D5E8F0'\}\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["XGBoost/GBM (standalone)"]["precision"]["mean"], fraud_models["XGBoost/GBM (standalone)"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["XGBoost/GBM (standalone)"]["recall"]["mean"], fraud_models["XGBoost/GBM (standalone)"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["XGBoost/GBM (standalone)"]["f1"]["mean"], fraud_models["XGBoost/GBM (standalone)"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["XGBoost/GBM (standalone)"]["auc_roc"]["mean"], fraud_models["XGBoost/GBM (standalone)"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["XGBoost/GBM (standalone)"]["average_precision"]["mean"], fraud_models["XGBoost/GBM (standalone)"]["average_precision"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tca\('Isolation Forest \(standalone\)',2800\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["Isolation Forest (standalone)"]["precision"]["mean"], fraud_models["Isolation Forest (standalone)"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["Isolation Forest (standalone)"]["recall"]["mean"], fraud_models["Isolation Forest (standalone)"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["Isolation Forest (standalone)"]["f1"]["mean"], fraud_models["Isolation Forest (standalone)"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["Isolation Forest (standalone)"]["auc_roc"]["mean"], fraud_models["Isolation Forest (standalone)"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["Isolation Forest (standalone)"]["average_precision"]["mean"], fraud_models["Isolation Forest (standalone)"]["average_precision"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tc\('XGBoost/GBM \+ Graph Feature Proxy',\{w:2800\}\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["XGBoost/GBM + Graph Feature Proxy"]["precision"]["mean"], fraud_models["XGBoost/GBM + Graph Feature Proxy"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["XGBoost/GBM + Graph Feature Proxy"]["recall"]["mean"], fraud_models["XGBoost/GBM + Graph Feature Proxy"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["XGBoost/GBM + Graph Feature Proxy"]["f1"]["mean"], fraud_models["XGBoost/GBM + Graph Feature Proxy"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["XGBoost/GBM + Graph Feature Proxy"]["auc_roc"]["mean"], fraud_models["XGBoost/GBM + Graph Feature Proxy"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["XGBoost/GBM + Graph Feature Proxy"]["average_precision"]["mean"], fraud_models["XGBoost/GBM + Graph Feature Proxy"]["average_precision"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tca\('GBM \+ IF \+ Calibrator',2800\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["XGBoost/GBM + IF + Calibrator"]["precision"]["mean"], fraud_models["XGBoost/GBM + IF + Calibrator"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["XGBoost/GBM + IF + Calibrator"]["recall"]["mean"], fraud_models["XGBoost/GBM + IF + Calibrator"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["XGBoost/GBM + IF + Calibrator"]["f1"]["mean"], fraud_models["XGBoost/GBM + IF + Calibrator"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["XGBoost/GBM + IF + Calibrator"]["auc_roc"]["mean"], fraud_models["XGBoost/GBM + IF + Calibrator"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["XGBoost/GBM + IF + Calibrator"]["average_precision"]["mean"], fraud_models["XGBoost/GBM + IF + Calibrator"]["average_precision"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tc\('Hybrid \(GBM\+IF\+Graph proxy\)',\{w:2800\}\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(fraud_models["Hybrid (GBM+IF+Graph)"]["precision"]["mean"], fraud_models["Hybrid (GBM+IF+Graph)"]["precision"]["std"]) +
              m.group(2) + fmt(fraud_models["Hybrid (GBM+IF+Graph)"]["recall"]["mean"], fraud_models["Hybrid (GBM+IF+Graph)"]["recall"]["std"]) +
              m.group(3) + fmt(fraud_models["Hybrid (GBM+IF+Graph)"]["f1"]["mean"], fraud_models["Hybrid (GBM+IF+Graph)"]["f1"]["std"]) +
              m.group(4) + fmt(fraud_models["Hybrid (GBM+IF+Graph)"]["auc_roc"]["mean"], fraud_models["Hybrid (GBM+IF+Graph)"]["auc_roc"]["std"]) +
              m.group(5) + fmt(fraud_models["Hybrid (GBM+IF+Graph)"]["average_precision"]["mean"], fraud_models["Hybrid (GBM+IF+Graph)"]["average_precision"]["std"]) + m.group(6),
    content
)


# Delinquency
del_models = metrics["delinquency"]["models"]

def fmt_rmse(mean, std):
    return f"{mean:.1f}".replace(".", ",") + "±" + f"{std:.1f}".replace(".", ",")

content = re.sub(
    r"(tc\('Statistical Baseline \(Z-score\)',\{w:2800\}\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(del_models["Statistical Baseline (Z-score)"]["precision"]["mean"], del_models["Statistical Baseline (Z-score)"]["precision"]["std"]) +
              m.group(2) + fmt(del_models["Statistical Baseline (Z-score)"]["recall"]["mean"], del_models["Statistical Baseline (Z-score)"]["recall"]["std"]) +
              m.group(3) + fmt(del_models["Statistical Baseline (Z-score)"]["f1"]["mean"], del_models["Statistical Baseline (Z-score)"]["f1"]["std"]) +
              m.group(4) + fmt(del_models["Statistical Baseline (Z-score)"]["auc_roc"]["mean"], del_models["Statistical Baseline (Z-score)"]["auc_roc"]["std"]) +
              m.group(5) + fmt_rmse(del_models["Statistical Baseline (Z-score)"]["rmse_days"]["mean"], del_models["Statistical Baseline (Z-score)"]["rmse_days"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tca\('Logistic Regression',2800\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1471.*?\} \])",
    lambda m: m.group(1) + fmt(del_models["Logistic Regression"]["precision"]["mean"], del_models["Logistic Regression"]["precision"]["std"]) +
              m.group(2) + fmt(del_models["Logistic Regression"]["recall"]["mean"], del_models["Logistic Regression"]["recall"]["std"]) +
              m.group(3) + fmt(del_models["Logistic Regression"]["f1"]["mean"], del_models["Logistic Regression"]["f1"]["std"]) +
              m.group(4) + fmt(del_models["Logistic Regression"]["auc_roc"]["mean"], del_models["Logistic Regression"]["auc_roc"]["std"]) +
              m.group(5) + fmt_rmse(del_models["Logistic Regression"]["rmse_days"]["mean"], del_models["Logistic Regression"]["rmse_days"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tc\('Random Forest',\{w:2800\}\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1200.*?\), tc\(')[^']+('\,\{w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(del_models["Random Forest"]["precision"]["mean"], del_models["Random Forest"]["precision"]["std"]) +
              m.group(2) + fmt(del_models["Random Forest"]["recall"]["mean"], del_models["Random Forest"]["recall"]["std"]) +
              m.group(3) + fmt(del_models["Random Forest"]["f1"]["mean"], del_models["Random Forest"]["f1"]["std"]) +
              m.group(4) + fmt(del_models["Random Forest"]["auc_roc"]["mean"], del_models["Random Forest"]["auc_roc"]["std"]) +
              m.group(5) + fmt_rmse(del_models["Random Forest"]["rmse_days"]["mean"], del_models["Random Forest"]["rmse_days"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tca\('LightGBM-compatible GBDT',2800\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1200.*?\), tca\(')[^']+('\.*?1471.*?\} \])",
    lambda m: m.group(1) + fmt(del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["precision"]["mean"], del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["precision"]["std"]) +
              m.group(2) + fmt(del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["recall"]["mean"], del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["recall"]["std"]) +
              m.group(3) + fmt(del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["f1"]["mean"], del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["f1"]["std"]) +
              m.group(4) + fmt(del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["auc_roc"]["mean"], del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["auc_roc"]["std"]) +
              m.group(5) + fmt_rmse(del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["rmse_days"]["mean"], del_models["LightGBM-compatible GBDT (delinquency-temporal-v1)"]["rmse_days"]["std"]) + m.group(6),
    content
)

content = re.sub(
    r"(tc\('Temporal sequence proxy',\{bold:true,w:2800,shade:'D5E8F0'\}\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1200.*?\), tc\(')[^']+('\,\{bold:true,w:1471.*?\} \])",
    lambda m: m.group(1) + fmt(del_models["Temporal sequence model (Transformer proxy)"]["precision"]["mean"], del_models["Temporal sequence model (Transformer proxy)"]["precision"]["std"]) +
              m.group(2) + fmt(del_models["Temporal sequence model (Transformer proxy)"]["recall"]["mean"], del_models["Temporal sequence model (Transformer proxy)"]["recall"]["std"]) +
              m.group(3) + fmt(del_models["Temporal sequence model (Transformer proxy)"]["f1"]["mean"], del_models["Temporal sequence model (Transformer proxy)"]["f1"]["std"]) +
              m.group(4) + fmt(del_models["Temporal sequence model (Transformer proxy)"]["auc_roc"]["mean"], del_models["Temporal sequence model (Transformer proxy)"]["auc_roc"]["std"]) +
              m.group(5) + fmt_rmse(del_models["Temporal sequence model (Transformer proxy)"]["rmse_days"]["mean"], del_models["Temporal sequence model (Transformer proxy)"]["rmse_days"]["std"]) + m.group(6),
    content
)


# Agent
agent = metrics["agent"]["metrics"]

content = re.sub(
    r"(tc\('Legal route accuracy',\{w:3300\}\), tc\(')[^']+('\,\{w:1700.*?\), tc\('> 95%',\{w:1700)",
    lambda m: m.group(1) + fmt_pct(agent["legal_route_accuracy"]) + m.group(2),
    content
)

content = re.sub(
    r"(tca\('FAQ grounding rate',3300\), tca\(')[^']+('\.*?1700.*?\), tca\('>= 75%',1700)",
    lambda m: m.group(1) + fmt_pct(agent["faq_grounding_rate"]) + m.group(2),
    content
)

content = re.sub(
    r"(tc\('Actionable steps rate',\{w:3300\}\), tc\(')[^']+('\,\{w:1700.*?\), tc\('> 95%',\{w:1700)",
    lambda m: m.group(1) + fmt_pct(agent["actionable_steps_rate"]) + m.group(2),
    content
)

content = re.sub(
    r"(tca\('Citation/reference rate',3300\), tca\(')[^']+('\.*?1700.*?\), tca\('> 95%',1700)",
    lambda m: m.group(1) + fmt_pct(agent["citation_or_reference_rate"]) + m.group(2),
    content
)

content = re.sub(
    r"(tc\('Mean confidence',\{w:3300\}\), tc\(')[^']+('\,\{w:1700.*?\), tc\('Monitor',\{w:1700)",
    lambda m: m.group(1) + f"{agent['mean_confidence']:.3f}".replace(".", ",") + m.group(2),
    content
)

content = re.sub(
    r"(tca\('Mean latency',3300\), tca\(')[^']+('\.*?1700.*?\), tca\('< 200 ms',1700)",
    lambda m: m.group(1) + fmt_ms(agent["mean_latency_ms"]) + m.group(2),
    content
)


# Sys perf
sys_perf = metrics["system_performance"]

content = re.sub(
    r"(tc\('Fraud scoring batch 5\.000 rows',\{w:3500\}\), tc\(')[^']+('\,\{w:2000.*?\), tc\(')[^']+('\,\{w:2000.*?\), tc\(')[^']+('\,\{w:1571.*?\} \])",
    lambda m: m.group(1) + fmt_ms(sys_perf["fraud_scoring_5000_rows_ms"]["p50"]) +
              m.group(2) + fmt_ms(sys_perf["fraud_scoring_5000_rows_ms"]["p95"]) +
              m.group(3) + fmt_ms(sys_perf["fraud_scoring_5000_rows_ms"]["p99"]) + m.group(4),
    content
)

content = re.sub(
    r"(tca\('Batch fraud scoring 120\.000 rows',3500\), tca\(')[^']+('\.*?2000.*?\), tca\(')[^']+('\.*?2000.*?\), tca\(')[^']+('\.*?1571.*?\} \])",
    lambda m: m.group(1) + fmt_ms(sys_perf["batch_scoring_all_rows_ms"]["p50"]) +
              m.group(2) + fmt_ms(sys_perf["batch_scoring_all_rows_ms"]["p95"]) +
              m.group(3) + fmt_ms(sys_perf["batch_scoring_all_rows_ms"]["p99"]) + m.group(4),
    content
)

content = re.sub(
    r"(tc\('Graph SCC 5\.000 nodes / 50K edges',\{w:3500\}\), tc\(')[^']+('\,\{w:2000.*?\), tc\(')[^']+('\,\{w:2000.*?\), tc\(')[^']+('\,\{w:1571.*?\} \])",
    lambda m: m.group(1) + fmt_ms(sys_perf["graph_scc_5000_nodes_50000_edges_ms"]["p50"]) +
              m.group(2) + fmt_ms(sys_perf["graph_scc_5000_nodes_50000_edges_ms"]["p95"]) +
              m.group(3) + fmt_ms(sys_perf["graph_scc_5000_nodes_50000_edges_ms"]["p99"]) + m.group(4),
    content
)

content = re.sub(
    r"(tca\('Agent legal template fallback',3500\), tca\(')[^']+('\.*?2000.*?\), tca\(')[^']+('\.*?2000.*?\), tca\(')[^']+('\.*?1571.*?\} \])",
    lambda m: m.group(1) + fmt_ms(sys_perf["agent_legal_template_ms"]["p50"]) +
              m.group(2) + fmt_ms(sys_perf["agent_legal_template_ms"]["p95"]) +
              m.group(3) + fmt_ms(sys_perf["agent_legal_template_ms"]["p99"]) + m.group(4),
    content
)

with open("e:/TaxInspector/doc.js", "w", encoding="utf-8") as f:
    f.write(content)
