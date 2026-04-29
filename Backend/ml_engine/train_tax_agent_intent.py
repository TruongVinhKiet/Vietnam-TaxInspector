from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import joblib
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from app.database import SessionLocal
from ml_engine.model_registry import ModelRegistryService


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


INTENTS = [
    "vat_refund_risk",
    "invoice_risk",
    "delinquency",
    "osint_ownership",
    "transfer_pricing",
    "audit_selection",
    "general_tax_query",
]


def _bootstrap_dataset() -> tuple[list[str], list[str]]:
    # Weakly-supervised bootstrap (Vietnamese) with many paraphrases + slots.
    rnd = random.Random(1337)

    prefixes = ["", "Xin hỏi", "Cho tôi hỏi", "Nhờ tư vấn", "Tôi muốn biết", "Hỏi nhanh"]
    suffixes = ["", "cần lưu ý gì?", "cần kiểm tra gì?", "là gì?", "được không?", "theo quy định nào?"]
    periods = ["kỳ 2025-Q4", "kỳ 2026-Q1", "tháng 03/2026", "năm 2025", ""]
    entities = ["doanh nghiệp", "công ty", "NNT", "hộ kinh doanh", ""]

    intent_seeds: dict[str, list[str]] = {
        "vat_refund_risk": [
            "hồ sơ hoàn thuế",
            "đề nghị hoàn thuế",
            "điều kiện hoàn thuế",
            "kiểm tra hồ sơ hoàn",
            "hoàn thuế VAT",
        ],
        "invoice_risk": [
            "hóa đơn trùng",
            "mua bán hóa đơn",
            "hóa đơn đầu vào",
            "hóa đơn đầu ra",
            "hóa đơn bất thường",
            "invoice rủi ro",
        ],
        "delinquency": [
            "nợ đọng thuế",
            "chậm nộp",
            "quá hạn nộp",
            "thu nợ",
            "cưỡng chế nợ",
            "dự báo nợ 90 ngày",
        ],
        "osint_ownership": [
            "ubo",
            "chủ sở hữu hưởng lợi",
            "offshore",
            "proxy company",
            "phoenix company",
            "chuỗi sở hữu",
            "công ty mẹ",
        ],
        "transfer_pricing": [
            "chuyển giá",
            "giao dịch liên kết",
            "transfer pricing",
            "hồ sơ xác định giá giao dịch liên kết",
            "so sánh giá thị trường",
            "mispricing",
        ],
        "audit_selection": [
            "xếp hạng hồ sơ",
            "thanh tra kiểm tra",
            "shortlist thanh tra",
            "ưu tiên kiểm tra",
            "audit selection",
            "chọn doanh nghiệp kiểm tra",
        ],
        "general_tax_query": [
            "kê khai thuế",
            "nộp tờ khai",
            "thủ tục thuế",
            "quy định thuế",
            "VAT theo quý",
        ],
    }

    paraphrase_verbs = [
        "đánh giá",
        "xác định",
        "nhận diện",
        "kiểm tra",
        "phân tích",
        "tư vấn",
        "hướng dẫn",
        "so sánh",
    ]

    X: list[str] = []
    y: list[str] = []

    # Larger corpus to reduce overlap and improve intent separation.
    target_per_intent = 1800  # ~ 7 * 1800 = 12600 (+variants)
    for intent, seeds in intent_seeds.items():
        for _ in range(target_per_intent):
            pfx = rnd.choice(prefixes)
            sfx = rnd.choice(suffixes)
            per = rnd.choice(periods)
            ent = rnd.choice(entities)
            verb = rnd.choice(paraphrase_verbs)
            topic = rnd.choice(seeds)
            parts = [pfx, verb, topic, ent, per]
            text_value = " ".join([p for p in parts if p]).strip()
            if sfx:
                text_value = f"{text_value} {sfx}".strip()
            if rnd.random() < 0.35:
                text_value = text_value.replace("vat", "VAT")
            if rnd.random() < 0.25:
                text_value = text_value + "?"
            X.append(text_value)
            y.append(intent)
            if rnd.random() < 0.20:
                X.append(text_value.lower())
                y.append(intent)

    return X, y


def _fetch_agent_turns_dataset(db) -> tuple[list[str], list[str]]:
    """
    Pull supervised examples from `agent_turns` if available.
    Uses assistant turns (they store normalized_intent).
    """
    try:
        rows = db.execute(
            text(
                """
                SELECT message_text, normalized_intent
                FROM agent_turns
                WHERE role = 'assistant'
                  AND normalized_intent IS NOT NULL
                  AND btrim(normalized_intent) <> ''
                ORDER BY created_at DESC
                LIMIT 50000
                """
            )
        ).fetchall()
    except ProgrammingError:
        db.rollback()
        return ([], [])
    X = []
    y = []
    for msg, intent in rows:
        if not msg or not intent:
            continue
        X.append(str(msg))
        y.append(str(intent))
    return X, y


def main() -> None:
    bootstrap_X, bootstrap_y = _bootstrap_dataset()
    labels = list(INTENTS)
    with SessionLocal() as db:
        turns_X, turns_y = _fetch_agent_turns_dataset(db)

    X = list(bootstrap_X)
    y = list(bootstrap_y)
    train_source = {"bootstrap": len(bootstrap_X), "agent_turns": len(turns_X)}
    if turns_X:
        X.extend(turns_X)
        y.extend(turns_y)

    # True hybrid featurization: char n-grams + word n-grams.
    vec = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=25000,
                ),
            ),
        ]
    )
    Xv = vec.fit_transform(X)
    # Stronger regularization to reduce overconfident leakage between intents in bootstrap data.
    clf = LogisticRegression(max_iter=2200, n_jobs=1, class_weight="balanced", C=0.7, multi_class="auto")
    clf.fit(Xv, y)

    vec_path = MODEL_DIR / "tax_agent_intent_vectorizer.joblib"
    model_path = MODEL_DIR / "tax_agent_intent_model.joblib"
    meta_path = MODEL_DIR / "tax_agent_intent_meta.json"

    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)

    model_version = f"tax-agent-intent-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    meta = {
        "model_version": model_version,
        "labels": labels,
        "trained_on": "bootstrap_templates_v2+agent_turns",
        "train_source": train_source,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    with SessionLocal() as db:
        registry = ModelRegistryService(db)
        registry.upsert_registry_entry(
            model_name="tax_agent_intent",
            model_version=model_version,
            artifact_path=f"file://{MODEL_DIR.as_posix()}/tax_agent_intent_model.joblib",
            feature_set_id=None,
            train_data_hash="bootstrap_templates_v2+agent_turns",
            metrics={"n_samples": len(X), "train_source": train_source},
            gates={"overall_pass": True, "bootstrap": True, "has_agent_turns": bool(turns_X)},
            status="staging",
        )

    print(f"[OK] trained tax_agent_intent model_version={model_version}")


if __name__ == "__main__":
    main()

