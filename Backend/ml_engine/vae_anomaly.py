"""
vae_anomaly.py – Variational Autoencoder for Transaction Anomaly Detection
============================================================================
Architecture:
    Encoder:  Input(D) → 128 → 64 → [μ(8), σ(8)]
    Decoder:  Latent(8) → 64 → 128 → Output(D)
    Loss:     Reconstruction (MSE) + KL Divergence (β-VAE)

Purpose:
    Unsupervised deep anomaly detection for invoice/transaction patterns.
    High reconstruction error → anomalous transaction.

Input Features (per invoice, D=16):
    - amount_log:           log1p(amount) normalized
    - vat_rate:             VAT rate / 100
    - seller_risk:          seller company risk score
    - buyer_risk:           buyer company risk score
    - days_since_reg:       seller age (normalized)
    - degree_seller:        seller's transaction degree
    - degree_buyer:         buyer's transaction degree
    - in_out_ratio:         seller's in/out amount ratio
    - is_reciprocal:        bidirectional flag
    - delta_recip_days:     time to nearest reciprocal
    - day_of_week_sin:      sin(2π * dow / 7)
    - day_of_week_cos:      cos(2π * dow / 7)
    - month_sin:            sin(2π * month / 12)
    - month_cos:            cos(2π * month / 12)
    - amount_zscore:        z-score vs. industry median
    - near_dup_count:       near-duplicate cluster size

Design:
    - β-VAE formulation for disentangled latent space
    - Reconstruction error serves as anomaly score
    - Latent space can be visualized via t-SNE for analyst review
    - Threshold calibrated on validation set (P95 of normal)

Reference:
    Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
    An & Cho, "Variational Autoencoder based Anomaly Detection", 2015
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

INPUT_DIM = 16
LATENT_DIM = 8
BETA = 1.0  # KL weight (β-VAE); >1 for more disentanglement


# ════════════════════════════════════════════════════════════════
#  1. VAE Model Definition
# ════════════════════════════════════════════════════════════════

class TransactionVAE(nn.Module):
    """
    β-Variational Autoencoder for anomaly detection on transaction features.
    
    The encoder maps input features to a latent distribution q(z|x).
    The decoder reconstructs the input from sampled latent codes.
    Anomaly score = reconstruction error (MSE per sample).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
        latent_dim: int = LATENT_DIM,
        dropout: float = 0.15,
        beta: float = BETA,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # ── Encoder ──
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space: mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # ── Decoder ──
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Deterministic at inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            {"reconstruction": ..., "mu": ..., "logvar": ..., "z": ...}
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }


# ════════════════════════════════════════════════════════════════
#  2. Loss Function
# ════════════════════════════════════════════════════════════════

def vae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE loss = Reconstruction + β * KL Divergence.
    
    Returns: (total_loss, recon_loss, kl_loss)
    """
    recon_loss = F.mse_loss(reconstruction, x, reduction="mean")

    # KL divergence: -0.5 * Σ(1 + logvar - μ² - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ════════════════════════════════════════════════════════════════
#  3. Feature Engineering
# ════════════════════════════════════════════════════════════════

class TransactionFeatureBuilder:
    """Build standardized feature vectors for VAE from raw invoice data."""

    def __init__(self, input_dim: int = INPUT_DIM):
        self.input_dim = input_dim
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def build_features(
        self,
        invoices: list[dict],
        company_map: dict[str, dict],
    ) -> np.ndarray:
        """
        Build feature matrix from raw invoice + company data.
        
        Returns: (N, input_dim) numpy array.
        """
        from datetime import date as date_cls

        rows = []
        for inv in invoices:
            seller = inv.get("seller_tax_code", "")
            buyer = inv.get("buyer_tax_code", "")
            s_info = company_map.get(seller, {})
            b_info = company_map.get(buyer, {})

            amount = max(0.0, float(inv.get("amount", 0) or 0))
            vat_rate = float(inv.get("vat_rate", 10) or 10)

            inv_date = inv.get("date")
            if isinstance(inv_date, str):
                try:
                    inv_date = date_cls.fromisoformat(inv_date)
                except ValueError:
                    inv_date = date_cls.today()

            # Seller age
            reg_date = s_info.get("registration_date")
            if reg_date:
                if isinstance(reg_date, str):
                    try:
                        reg_date = date_cls.fromisoformat(reg_date)
                    except ValueError:
                        reg_date = None
            days_since_reg = (
                (date_cls.today() - reg_date).days if reg_date else 1000
            )

            dow = inv_date.weekday() if inv_date else 0
            month = inv_date.month if inv_date else 1

            rows.append([
                math.log1p(amount) / 25.0,
                vat_rate / 100.0,
                float(s_info.get("risk_score", 0) or 0) / 100.0,
                float(b_info.get("risk_score", 0) or 0) / 100.0,
                min(1.0, days_since_reg / 3650.0),
                min(1.0, float(s_info.get("degree", 0) or 0) / 100.0),
                min(1.0, float(b_info.get("degree", 0) or 0) / 100.0),
                0.5,  # in_out_ratio placeholder
                float(inv.get("is_reciprocal", 0) or 0),
                0.5,  # delta_recip placeholder
                math.sin(2 * math.pi * dow / 7.0),
                math.cos(2 * math.pi * dow / 7.0),
                math.sin(2 * math.pi * (month - 1) / 12.0),
                math.cos(2 * math.pi * (month - 1) / 12.0),
                0.0,  # amount_zscore placeholder
                float(inv.get("near_dup_count", 0) or 0) / 10.0,
            ])

        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, self.input_dim), dtype=np.float32)

    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit mean/std normalization on training data."""
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds < 1e-8] = 1.0  # prevent div by zero

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalization."""
        if self.means is not None and self.stds is not None:
            return (X - self.means) / self.stds
        return X


# ════════════════════════════════════════════════════════════════
#  4. Training Pipeline
# ════════════════════════════════════════════════════════════════

class VAEAnomalyTrainer:
    """Train and persist the TransactionVAE for anomaly detection."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
        beta: float = BETA,
        lr: float = 0.001,
    ):
        self.model = TransactionVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.config = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "beta": beta,
            "hidden_dims": [128, 64],
        }
        self.anomaly_threshold: float = 0.0

    def train(
        self,
        X: np.ndarray,
        epochs: int = 150,
        batch_size: int = 128,
        val_ratio: float = 0.15,
    ) -> dict[str, float]:
        """
        Train VAE on normal transaction data.
        
        The anomaly threshold is calibrated at P95 of validation
        reconstruction error (assuming validation is mostly normal).
        """
        n = X.shape[0]
        n_val = int(n * val_ratio)
        indices = np.random.permutation(n)
        train_X = torch.tensor(X[indices[n_val:]], dtype=torch.float32)
        val_X = torch.tensor(X[indices[:n_val]], dtype=torch.float32)

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(train_X.shape[0])
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, train_X.shape[0], batch_size):
                batch = train_X[perm[start : start + batch_size]]
                if batch.shape[0] < 2:
                    continue

                self.optimizer.zero_grad()
                out = self.model(batch)
                total, recon, kl = vae_loss(
                    batch, out["reconstruction"], out["mu"], out["logvar"],
                    beta=self.config["beta"],
                )
                total.backward()
                self.optimizer.step()

                epoch_loss += total.item()
                n_batches += 1

            if (epoch + 1) % 50 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(val_X)
                    val_total, val_recon, val_kl = vae_loss(
                        val_X, val_out["reconstruction"],
                        val_out["mu"], val_out["logvar"],
                        beta=self.config["beta"],
                    )
                    # Per-sample reconstruction error
                    per_sample_error = torch.mean(
                        (val_X - val_out["reconstruction"]) ** 2, dim=1
                    )
                    p95 = float(torch.quantile(per_sample_error, 0.95).item())

                avg_loss = epoch_loss / max(1, n_batches)
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"TrainLoss={avg_loss:.4f} | "
                    f"ValRecon={val_recon.item():.4f} | "
                    f"ValKL={val_kl.item():.4f} | "
                    f"P95_threshold={p95:.4f}"
                )

        # Calibrate threshold
        self.model.eval()
        with torch.no_grad():
            val_out = self.model(val_X)
            per_sample_error = torch.mean(
                (val_X - val_out["reconstruction"]) ** 2, dim=1
            )
            self.anomaly_threshold = float(
                torch.quantile(per_sample_error, 0.95).item()
            )

        self.config["anomaly_threshold"] = self.anomaly_threshold

        return {
            "anomaly_threshold_p95": round(self.anomaly_threshold, 6),
            "val_samples": int(val_X.shape[0]),
            "train_samples": int(train_X.shape[0]),
        }

    def save(self, path: str | Path | None = None) -> None:
        save_dir = Path(path) if path else MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "vae_anomaly.pt")
        with open(save_dir / "vae_anomaly_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"[OK] VAE Anomaly model saved to {save_dir}")


# ════════════════════════════════════════════════════════════════
#  5. Inference Pipeline
# ════════════════════════════════════════════════════════════════

class VAEAnomalyInference:
    """Load and run anomaly scoring with the trained VAE."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model: TransactionVAE | None = None
        self.anomaly_threshold: float = 0.0
        self._loaded = False

    def load(self) -> bool:
        config_path = self.model_dir / "vae_anomaly_config.json"
        model_path = self.model_dir / "vae_anomaly.pt"

        if not config_path.exists() or not model_path.exists():
            logger.warning("VAE Anomaly model not found.")
            return False

        with open(config_path) as f:
            config = json.load(f)

        self.model = TransactionVAE(
            input_dim=config.get("input_dim", INPUT_DIM),
            latent_dim=config.get("latent_dim", LATENT_DIM),
            beta=config.get("beta", BETA),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.anomaly_threshold = config.get("anomaly_threshold", 0.1)
        self._loaded = True
        logger.info(f"VAE Anomaly model loaded from {self.model_dir}")
        return True

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for a batch of transactions.
        
        Returns: (N,) array of reconstruction errors.
        Higher = more anomalous.
        """
        if not self._loaded or self.model is None:
            return np.zeros(X.shape[0])

        x_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x_tensor)
            per_sample_error = torch.mean(
                (x_tensor - out["reconstruction"]) ** 2, dim=1
            )
        return per_sample_error.numpy()

    def detect(self, X: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect anomalies with threshold-based classification.
        
        Returns list of dicts with anomaly_score, is_anomaly, percentile_rank.
        """
        scores = self.score(X)
        results = []
        for i, score in enumerate(scores):
            results.append({
                "index": i,
                "anomaly_score": round(float(score), 6),
                "is_anomaly": bool(score > self.anomaly_threshold),
                "severity": (
                    "critical" if score > self.anomaly_threshold * 2
                    else "high" if score > self.anomaly_threshold
                    else "normal"
                ),
            })
        return results

    def get_latent_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extract latent embeddings for visualization (t-SNE, UMAP).
        
        Returns: (N, latent_dim) array.
        """
        if not self._loaded or self.model is None:
            return np.zeros((X.shape[0], LATENT_DIM))

        x_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            mu, _ = self.model.encode(x_tensor)
        return mu.numpy()
