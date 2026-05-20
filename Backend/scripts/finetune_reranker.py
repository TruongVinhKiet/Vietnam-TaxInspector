"""Prepare and optionally fine-tune the legal reranker.

In offline/dev environments this script creates domain training pairs and a
calibrated lightweight weight file consumed by ``TaxAgentReranker``. If
sentence-transformers is installed and ``--train-cross-encoder`` is supplied,
the script also trains a CrossEncoder model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
for _path in (BACKEND_DIR, REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_DIR))
    except ValueError:
        return str(resolved)


def _load_citizen_snippets():
    from ml_engine.tax_agent_citizen_legal import SNIPPETS, normalize_text

    return SNIPPETS, normalize_text


def build_training_pairs(seed: int = 42) -> list[dict[str, Any]]:
    snippets, normalize_text = _load_citizen_snippets()
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    snippet_list = list(snippets)
    templates = [
        "{keyword}",
        "toi muon hoi ve {keyword}",
        "cho toi biet quy dinh {keyword}",
        "truong hop cua toi lien quan den {keyword} thi xu ly sao",
        "can can cu phap ly ve {keyword}",
        "{keyword} co bi phat khong",
    ]

    for snippet in snippet_list:
        keywords = list(snippet.keywords) or [snippet.title]
        positives = []
        for keyword in keywords[:8]:
            positives.extend(t.format(keyword=keyword) for t in templates)
        positives.append(snippet.title)
        positives.append(snippet.text[:240])

        hard_pool = [
            other for other in snippet_list
            if other.key != snippet.key and normalize_text(other.title)[:10] != normalize_text(snippet.title)[:10]
        ]
        hard_negs = rng.sample(hard_pool, k=min(4, len(hard_pool))) if hard_pool else []

        for query in positives:
            pairs.append(
                {
                    "query": query,
                    "document": f"{snippet.title}\n{snippet.legal_reference}\n{snippet.text}",
                    "label": 1.0,
                    "positive_key": snippet.key,
                    "negative_key": None,
                }
            )
            for neg in hard_negs[:2]:
                pairs.append(
                    {
                        "query": query,
                        "document": f"{neg.title}\n{neg.legal_reference}\n{neg.text}",
                        "label": 0.0,
                        "positive_key": snippet.key,
                        "negative_key": neg.key,
                    }
                )
    rng.shuffle(pairs)
    return pairs


def evaluate_lightweight_grounding(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    from ml_engine.tax_agent_citizen_legal import normalize_text

    positives = [p for p in pairs if p["label"] == 1.0]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        grouped.setdefault(p["query"], []).append(p)
    hits = 0
    total = 0
    for query, items in grouped.items():
        q_tokens = set(normalize_text(query).split())
        ranked = sorted(
            items,
            key=lambda item: len(q_tokens & set(normalize_text(item["document"]).split())),
            reverse=True,
        )
        if ranked and ranked[0]["label"] == 1.0:
            hits += 1
        total += 1
    return {
        "query_count": total,
        "pair_count": len(pairs),
        "positive_pairs": len(positives),
        "top1_grounding_rate": round(hits / max(1, total), 4),
    }


def write_weight_file(model_dir: Path, metrics: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "tax_agent_reranker_weights.json"
    # Tuned toward legal grounding: dense/semantic remains important, but
    # lexical and doc-type priors are raised for Vietnamese legal citations.
    weights = {
        "model_version": "tax-agent-rerank-v2-legal-grounding",
        "w_bm25": 0.32,
        "w_dense": 0.38,
        "w_lexical": 0.22,
        "w_doc_type": 0.08,
        "grounding_eval": metrics,
        "generated_at": _now_iso(),
    }
    path.write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def maybe_train_cross_encoder(pairs: list[dict[str, Any]], model_dir: Path, epochs: int = 1) -> dict[str, Any]:
    try:
        from sentence_transformers import CrossEncoder, InputExample
    except Exception as exc:
        return {"status": "skipped", "reason": f"sentence_transformers_unavailable: {exc}"}

    train_samples = [
        InputExample(texts=[p["query"], p["document"]], label=float(p["label"]))
        for p in pairs
    ]
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
    model.fit(train_samples=train_samples, epochs=int(epochs), warmup_steps=20, show_progress_bar=True)
    target = model_dir / "cross_encoder"
    model.save(str(target))
    return {"status": "trained", "path": _repo_rel(target), "epochs": int(epochs)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/fine-tune legal reranker assets")
    parser.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "models" / "tax_agent_reranker")
    parser.add_argument("--pairs-out", type=Path, default=BACKEND_DIR / "data" / "legal_reranker_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cross-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    pairs = build_training_pairs(seed=args.seed)
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_out.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
        encoding="utf-8",
    )
    metrics = evaluate_lightweight_grounding(pairs)
    weights_path = write_weight_file(args.out_dir, metrics)
    train_result = (
        maybe_train_cross_encoder(pairs, args.out_dir, epochs=args.epochs)
        if args.train_cross_encoder
        else {"status": "skipped", "reason": "run with --train-cross-encoder to train model weights"}
    )
    report = {
        "generated_at": _now_iso(),
        "pairs_path": _repo_rel(args.pairs_out),
        "weights_path": _repo_rel(weights_path),
        "metrics": metrics,
        "cross_encoder": train_result,
    }
    report_path = args.out_dir / "reranker_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {args.pairs_out}")
    print(f"[OK] wrote {weights_path}")
    print(f"[OK] wrote {report_path}")
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
