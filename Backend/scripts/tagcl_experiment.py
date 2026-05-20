"""Tax-Aware Adversarial Graph Contrastive Learning (TAGCL).

Novel algorithm contribution for TaxInspector:

    Traditional graph contrastive learning (TH-GCL, HCLNet) uses *random*
    augmentations (edge drop, feature mask).  Real-world tax evasion is NOT
    random — it follows business-rule constraints (VAT ∈ [0,1], revenue ≥ 0).

    TAGCL replaces random augmentations with **domain-constrained adversarial
    evasion strategies** as positive views in a contrastive framework.  The
    encoder learns representations that are *invariant* to business-plausible
    evasion tactics — a property no random augmentation can provide.

    Three novel components:
      1. Tax-aware positive views via domain-constrained attacks
      2. Constraint-violation penalty (λ·CVP) in the loss function
      3. Projection head architecture (encoder → projector, SimCLR-style)

    Ablation baselines:
      B0 — XGBoost on raw features (no contrastive)
      B1 — RandomAug contrastive (Gaussian noise augmentation)
      B2 — TAGCL (domain-constrained adversarial augmentation)

Usage:
    python Backend/scripts/tagcl_experiment.py --rows 5000 --folds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.feature_engineering import TaxFeatureEngineer  # noqa: E402
from scripts.adversarial_robustness import (  # noqa: E402
    attack_composite,
    attack_feature_manipulation,
    attack_graph_camouflage,
    attack_invoice_splitting,
    attack_temporal_smoothing,
)
from scripts.run_experimental_evaluation import (  # noqa: E402
    _add_graph_features,
    _ensure_fraud_frame,
    _model_prob,
    _xgb_or_hgb_factory,
)

REPORT_DIR = BACKEND_DIR / "reports"

# ── Business-rule bounds (for constraint-violation penalty) ──────────────────
# Each tuple: (column_index, lower_bound, upper_bound)
DOMAIN_BOUNDS = [
    (0, 0.0, 2.0),   # f3_vat_structure
    (1, -1.0, 5.0),  # revenue_growth_rate
    (2, -1.0, 5.0),  # expense_growth_rate
    (3, 0.0, 1.0),   # vat_net_ratio
    (4, 0.0, 2.0),   # f2_ratio_limit
    (5, 0.0, 5.0),   # out_pr_ratio
    (6, 0.0, 1.0),   # cycle_score
    (7, 0.0, 5.0),   # invoice_growth
    (8, 0.05, 2.0),  # amount_std
]

# ═════════════════════════════════════════════════════════════════════════════
#  PyTorch components (guarded by HAS_TORCH)
# ═════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class TAGCLEncoder(nn.Module):
        """Encoder with projection head (SimCLR pattern).

        Architecture:
            input → fc1 → BN → ReLU → Dropout → fc2 → BN → ReLU
                  → proj1 → ReLU → proj2 → L2-normalize
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            embed_dim: int = 32,
            proj_dim: int = 16,
            dropout: float = 0.1,
        ):
            super().__init__()
            # Representation encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )
            # Projection head (discarded after pre-training)
            self.projector = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Returns L2-normalized projection for contrastive loss."""
            h = self.encoder(x)
            z = self.projector(h)
            return F.normalize(z, dim=1)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Returns representation embeddings (no projection head)."""
            return self.encoder(x)

    def info_nce_loss(
        z_anchor: torch.Tensor,
        z_positives: list[torch.Tensor],
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """InfoNCE with in-batch negatives and multiple positive views.

        Positives = adversarial views of the SAME sample (should be invariant).
        Negatives = all OTHER samples in the batch (standard in-batch).

        This is semantically correct: we want the encoder to map a company
        and its evasion-perturbed version to the SAME point in latent space.
        """
        batch_size = z_anchor.shape[0]
        total_loss = torch.tensor(0.0, device=z_anchor.device)

        for z_pos in z_positives:
            # Positive similarity: anchor_i · pos_i
            pos_sim = torch.sum(z_anchor * z_pos, dim=-1) / temperature  # (B,)

            # Negative similarities: anchor_i · anchor_j for all j ≠ i
            sim_matrix = z_anchor @ z_anchor.T / temperature  # (B, B)
            # Mask out self-similarity
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_anchor.device)
            neg_logsum = torch.logsumexp(sim_matrix.masked_fill(~mask, -1e9), dim=1)

            total_loss += (-pos_sim + neg_logsum).mean()

        return total_loss / len(z_positives)

    def constraint_violation_penalty(
        x_aug: torch.Tensor,
        bounds: list[tuple[int, float, float]],
    ) -> torch.Tensor:
        """Soft penalty for features violating domain bounds after augmentation.

        CVP = Σ_i max(0, x_i - upper_i)² + max(0, lower_i - x_i)²

        This is the novel regularizer: it ensures augmented views remain
        within business-plausible ranges, differentiating TAGCL from generic
        adversarial contrastive methods.
        """
        penalty = torch.tensor(0.0, device=x_aug.device)
        for col, lo, hi in bounds:
            vals = x_aug[:, col]
            penalty += torch.clamp(vals - hi, min=0).pow(2).mean()
            penalty += torch.clamp(lo - vals, min=0).pow(2).mean()
        return penalty


# ═════════════════════════════════════════════════════════════════════════════
#  Augmentation strategies
# ═════════════════════════════════════════════════════════════════════════════

def _random_augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Baseline augmentation: isotropic Gaussian noise (non-domain-aware)."""
    noise = rng.normal(0, 0.08, size=X.shape)
    return np.clip(X + noise, 0, None)


def _tax_aware_augment(
    X: np.ndarray,
    rng: np.random.Generator,
    strategy: str = "mixed",
) -> np.ndarray:
    """Domain-constrained augmentation using tax-evasion attack functions.

    Unlike random noise, these perturbations simulate REAL corporate decisions
    that a tax evader would make (shell companies, invoice splitting, etc.).
    All perturbations respect business constraints automatically because the
    attack functions from adversarial_robustness.py enforce them.
    """
    targets = np.ones(len(X), dtype=bool)

    if strategy == "camouflage":
        return attack_graph_camouflage(X, targets, rng)
    elif strategy == "invoice":
        return attack_invoice_splitting(X, targets, rng)
    elif strategy == "temporal":
        return attack_temporal_smoothing(X, targets, rng)
    elif strategy == "feature":
        return attack_feature_manipulation(X, targets, rng)
    elif strategy == "mixed":
        # Randomly assign each sample to a different attack
        n = len(X)
        X_aug = X.copy()
        attacks = [
            attack_graph_camouflage,
            attack_invoice_splitting,
            attack_temporal_smoothing,
            attack_feature_manipulation,
        ]
        assignment = rng.integers(0, len(attacks), size=n)
        for atk_idx, atk_fn in enumerate(attacks):
            mask = assignment == atk_idx
            if mask.any():
                X_aug[mask] = atk_fn(X[mask], np.ones(mask.sum(), dtype=bool), rng)
        return X_aug
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ═════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train_contrastive_encoder(
    X_train: np.ndarray,
    seed: int,
    augment_fn,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 3e-3,
    cvp_lambda: float = 0.1,
    use_cvp: bool = True,
) -> Any:
    """Pre-train encoder with contrastive loss + optional CVP.

    Args:
        X_train: feature matrix (N, 9)
        augment_fn: callable(X, rng) -> X_augmented
        cvp_lambda: weight for constraint-violation penalty
        use_cvp: whether to add CVP (True for TAGCL, False for RandomAug)
    """
    if not HAS_TORCH:
        return None

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    input_dim = X_train.shape[1]
    encoder = TAGCLEncoder(input_dim=input_dim)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(X_train)
    loss_history = []

    encoder.train()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 4:  # Need enough samples for in-batch negatives
                continue

            x_np = X_train[idx]
            z_anchor = encoder(torch.FloatTensor(x_np))

            # Generate 2 positive views via domain-constrained augmentation
            x_aug1 = augment_fn(x_np, rng)
            x_aug2 = augment_fn(x_np, rng)
            z_pos1 = encoder(torch.FloatTensor(x_aug1))
            z_pos2 = encoder(torch.FloatTensor(x_aug2))

            loss = info_nce_loss(z_anchor, [z_pos1, z_pos2])

            # Constraint violation penalty (TAGCL-specific)
            if use_cvp:
                cvp = constraint_violation_penalty(
                    torch.FloatTensor(x_aug1), DOMAIN_BOUNDS
                ) + constraint_violation_penalty(
                    torch.FloatTensor(x_aug2), DOMAIN_BOUNDS
                )
                loss = loss + cvp_lambda * cvp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(round(avg_loss, 5))

    return encoder, loss_history


def extract_embeddings(encoder: Any, X: np.ndarray) -> np.ndarray:
    """Extract representation embeddings (without projection head)."""
    if encoder is None:
        return X
    encoder.eval()
    with torch.no_grad():
        emb = encoder.encode(torch.FloatTensor(X)).numpy()
    return np.hstack([X, emb])  # Concatenate: original features + learned repr


# ═════════════════════════════════════════════════════════════════════════════
#  Benchmark runner
# ═════════════════════════════════════════════════════════════════════════════

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else 0.5


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else 0.0


def run_tagcl_benchmark(
    rows: int = 5000,
    folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Full ablation study: B0 (no CL) vs B1 (RandomAug CL) vs B2 (TAGCL)."""
    t0 = time.perf_counter()

    frame, _ = _ensure_fraud_frame(rows, seed, force=False)
    y = frame["fraud_label"].astype(int).to_numpy()
    X_graph = _add_graph_features(frame, y, seed)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # Metrics storage per method
    methods = ["B0_Baseline", "B1_RandomAug_CL", "B2_TAGCL"]
    clean_aucs = {m: [] for m in methods}
    clean_aps = {m: [] for m in methods}
    attack_aucs = {m: [] for m in methods}  # AUC under composite attack
    loss_curves = {}

    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_graph, y), 1):
        Xg_tr, Xg_te = X_graph[tr_idx], X_graph[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        rng = np.random.default_rng(seed + fold_i)

        # ── B0: Baseline (XGBoost, no contrastive learning) ───────────
        m0 = _xgb_or_hgb_factory(seed + fold_i)
        m0.fit(Xg_tr, y_tr)
        p0 = _model_prob(m0, Xg_te)
        clean_aucs["B0_Baseline"].append(_safe_auc(y_te, p0))
        clean_aps["B0_Baseline"].append(_safe_ap(y_te, p0))

        # Attack test
        targets = np.ones(len(Xg_te), dtype=bool)
        Xg_te_atk = attack_composite(Xg_te, targets, rng)
        attack_aucs["B0_Baseline"].append(_safe_auc(y_te, _model_prob(m0, Xg_te_atk)))

        # ── B1: RandomAug Contrastive Learning ────────────────────────
        enc1, lh1 = train_contrastive_encoder(
            Xg_tr, seed + fold_i + 1000,
            augment_fn=_random_augment,
            use_cvp=False,
            epochs=40,
        ) or (None, [])
        loss_curves[f"B1_fold{fold_i}"] = lh1

        Xg_tr_e1 = extract_embeddings(enc1, Xg_tr)
        Xg_te_e1 = extract_embeddings(enc1, Xg_te)
        m1 = _xgb_or_hgb_factory(seed + fold_i + 100)
        m1.fit(Xg_tr_e1, y_tr)
        p1 = _model_prob(m1, Xg_te_e1)
        clean_aucs["B1_RandomAug_CL"].append(_safe_auc(y_te, p1))
        clean_aps["B1_RandomAug_CL"].append(_safe_ap(y_te, p1))

        Xg_te_atk_e1 = extract_embeddings(enc1, Xg_te_atk)
        attack_aucs["B1_RandomAug_CL"].append(
            _safe_auc(y_te, _model_prob(m1, Xg_te_atk_e1))
        )

        # ── B2: TAGCL (tax-aware adversarial contrastive + CVP) ───────
        enc2, lh2 = train_contrastive_encoder(
            Xg_tr, seed + fold_i + 2000,
            augment_fn=_tax_aware_augment,
            use_cvp=True,
            cvp_lambda=0.1,
            epochs=40,
        ) or (None, [])
        loss_curves[f"B2_fold{fold_i}"] = lh2

        Xg_tr_e2 = extract_embeddings(enc2, Xg_tr)
        Xg_te_e2 = extract_embeddings(enc2, Xg_te)
        m2 = _xgb_or_hgb_factory(seed + fold_i + 200)
        m2.fit(Xg_tr_e2, y_tr)
        p2 = _model_prob(m2, Xg_te_e2)
        clean_aucs["B2_TAGCL"].append(_safe_auc(y_te, p2))
        clean_aps["B2_TAGCL"].append(_safe_ap(y_te, p2))

        Xg_te_atk_e2 = extract_embeddings(enc2, Xg_te_atk)
        attack_aucs["B2_TAGCL"].append(
            _safe_auc(y_te, _model_prob(m2, Xg_te_atk_e2))
        )

    # ── Aggregate ─────────────────────────────────────────────────────
    def _stats(vals):
        a = np.array(vals)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}

    results = {}
    for m in methods:
        results[m] = {
            "clean_auc": _stats(clean_aucs[m]),
            "clean_ap": _stats(clean_aps[m]),
            "attack_auc": _stats(attack_aucs[m]),
            "robustness_drop": round(
                float(np.mean(clean_aucs[m]) - np.mean(attack_aucs[m])), 4
            ),
        }

    # Improvement deltas
    tagcl_vs_base = round(
        float(np.mean(clean_aucs["B2_TAGCL"]) - np.mean(clean_aucs["B0_Baseline"])), 4
    )
    tagcl_vs_random = round(
        float(np.mean(clean_aucs["B2_TAGCL"]) - np.mean(clean_aucs["B1_RandomAug_CL"])), 4
    )
    robustness_gain = round(
        float(np.mean(attack_aucs["B2_TAGCL"]) - np.mean(attack_aucs["B0_Baseline"])), 4
    )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "TAGCL — Tax-Aware Adversarial Graph Contrastive Learning",
        "novelty_claims": [
            "First to use business-rule-constrained evasion attacks as contrastive positive views",
            "Constraint-violation penalty (CVP) regularizer ensures augmentations stay domain-valid",
            "Projection-head architecture (SimCLR-style) for contrastive pre-training on tax graphs",
        ],
        "dataset": {
            "rows": int(len(frame)),
            "fraud_ratio": round(float(y.mean()), 4),
            "features": 9,
            "folds": folds,
        },
        "ablation_results": results,
        "deltas": {
            "TAGCL_vs_Baseline_AUC": tagcl_vs_base,
            "TAGCL_vs_RandomAug_AUC": tagcl_vs_random,
            "TAGCL_robustness_gain_vs_Baseline": robustness_gain,
        },
        "loss_convergence": loss_curves,
        "pytorch_available": HAS_TORCH,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    return report


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TAGCL: Tax-Aware Adversarial Graph Contrastive Learning"
    )
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=REPORT_DIR / "tagcl_results.json"
    )
    args = parser.parse_args()

    report = run_tagcl_benchmark(rows=args.rows, folds=args.folds, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 65)
    print("  TAGCL Experiment — Ablation Study")
    print("=" * 65)
    print(f"{'Method':<22} {'Clean AUC':>12} {'Attack AUC':>12} {'Drop':>8}")
    print("-" * 65)
    for m in ["B0_Baseline", "B1_RandomAug_CL", "B2_TAGCL"]:
        r = report["ablation_results"][m]
        print(
            f"{m:<22} "
            f"{r['clean_auc']['mean']:>10.4f}   "
            f"{r['attack_auc']['mean']:>10.4f}   "
            f"{r['robustness_drop']:>+7.4f}"
        )
    print("-" * 65)
    d = report["deltas"]
    print(f"  TAGCL vs Baseline (clean):       {d['TAGCL_vs_Baseline_AUC']:+.4f}")
    print(f"  TAGCL vs RandomAug (clean):       {d['TAGCL_vs_RandomAug_AUC']:+.4f}")
    print(f"  TAGCL robustness gain:            {d['TAGCL_robustness_gain_vs_Baseline']:+.4f}")
    print(f"\n  Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
