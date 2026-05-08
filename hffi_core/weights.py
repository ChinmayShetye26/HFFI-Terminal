"""
Weight learning for HFFI.

The original draft used hand-set weights (w1=0.25, w2=0.25, ...). For an
academic submission this is the single biggest weakness — reviewers will ask
"why these numbers?". This module replaces them with two principled approaches:

1. PCA-based weights (unsupervised)
   ----------------------------------
   Run PCA on the five normalized component matrix (L, D, E, P, M). The first
   principal component captures the direction of maximum variance in fragility
   space. We use the absolute loadings on PC1 as weights, normalized to sum
   to 1 (excluding the interaction term, which keeps its policy-driven λ).

   This answers: "given the joint distribution of these five fragility
   dimensions in the population, which directions matter most?"

2. Logistic-regression weights (supervised)
   ----------------------------------
   If you have a labeled "distress" outcome y_i ∈ {0,1} for each household
   (e.g. defaulted, missed payment, hit by stress event), fit a logistic
   regression and use the coefficients (rescaled to be non-negative and sum
   to 1) as weights.

   This answers: "which fragility dimensions actually predict distress?"

Either set of weights is saved to YAML and loaded by the scoring engine.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


COMPONENT_COLS = ["L", "D", "E", "P", "M"]
WEIGHT_KEYS = ["w1_liquidity", "w2_debt", "w3_expense", "w4_portfolio", "w5_macro"]


def _normalize_to_unit_sum(weights: np.ndarray) -> np.ndarray:
    """Clip negatives to 0 and rescale so the vector sums to 1."""
    w = np.maximum(weights, 0.0)
    s = w.sum()
    if s <= 0:
        # Degenerate case — fall back to equal weights.
        return np.full_like(w, 1.0 / len(w))
    return w / s


# --------------------------------------------------------------------------- #
# PCA-based weights
# --------------------------------------------------------------------------- #
def learn_weights_pca(
    components_df: pd.DataFrame,
    lambda_LD: float = 0.10,
    component_scale: float = 0.90,
) -> Tuple[dict, dict]:
    """Learn weights from the first principal component of the components matrix.

    Parameters
    ----------
    components_df : DataFrame with columns ['L', 'D', 'E', 'P', 'M']
        One row per household; values in [0, 1].
    lambda_LD : float
        Interaction-term weight (kept as a policy choice, not learned).
    component_scale : float
        Total mass assigned to the five components (the rest goes to the
        interaction). Must sum with lambda_LD-mass to ≤ 1.

    Returns
    -------
    weights : dict ready for hffi_score()
    diagnostics : dict with explained variance, loadings, and notes
    """
    X = components_df[COMPONENT_COLS].to_numpy(dtype=float)

    # Standardize before PCA (variances differ wildly otherwise)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, Xs.shape[0]))
    pca.fit(Xs)

    # PC1 is the direction of maximum variance. Take absolute loadings —
    # we want each component's importance regardless of sign.
    loadings = np.abs(pca.components_[0])
    raw_weights = _normalize_to_unit_sum(loadings) * component_scale

    weights = {k: float(v) for k, v in zip(WEIGHT_KEYS, raw_weights)}
    weights["lambda_LD"] = float(lambda_LD)

    diagnostics = {
        "method": "pca_pc1",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_loadings_raw": pca.components_[0].tolist(),
        "n_samples": int(Xs.shape[0]),
        "note": (
            "Weights derived from absolute PC1 loadings, rescaled to sum to "
            f"{component_scale} (interaction λ = {lambda_LD})."
        ),
    }
    return weights, diagnostics


# --------------------------------------------------------------------------- #
# Logistic-regression-based weights
# --------------------------------------------------------------------------- #
def learn_weights_logreg(
    components_df: pd.DataFrame,
    distress_label: pd.Series,
    lambda_LD: float = 0.10,
    component_scale: float = 0.90,
    include_interaction: bool = True,
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[dict, dict]:
    """Learn weights from a logistic regression predicting a distress label.

    Parameters
    ----------
    components_df : DataFrame with columns ['L', 'D', 'E', 'P', 'M']
    distress_label : Series of 0/1 labels (same index as components_df)
    include_interaction : if True, fits an L*D interaction term and uses its
        coefficient to set lambda_LD.

    Returns
    -------
    weights : dict ready for hffi_score()
    diagnostics : dict with coefficients, accuracy, AUC, and notes
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import cross_val_score

    X = components_df[COMPONENT_COLS].copy()
    if include_interaction:
        X["LD"] = components_df["L"] * components_df["D"]

    y = distress_label.astype(int).to_numpy()

    # Fit logistic regression. We want non-negative coefficients in spirit
    # (more fragility → more risk), so we rely on data sign-correctness and
    # then clip negatives in the post-step.
    model = LogisticRegression(
        C=C, solver="lbfgs", max_iter=2000, random_state=random_state
    )
    model.fit(X.to_numpy(), y)

    coefs = model.coef_[0]
    component_coefs = coefs[: len(COMPONENT_COLS)]
    interaction_coef = coefs[-1] if include_interaction else 0.0

    # Cross-validated AUC — the headline diagnostic
    try:
        auc_cv = float(np.mean(cross_val_score(
            LogisticRegression(C=C, solver="lbfgs", max_iter=2000),
            X.to_numpy(), y, cv=5, scoring="roc_auc",
        )))
    except Exception:
        auc_cv = float("nan")

    # Convert coefficients → weights. Take absolute values, then rescale.
    raw = _normalize_to_unit_sum(np.abs(component_coefs)) * component_scale
    weights = {k: float(v) for k, v in zip(WEIGHT_KEYS, raw)}

    if include_interaction:
        # Use interaction coefficient (rescaled) as λ if positive; else fall
        # back to the policy default.
        if interaction_coef > 0:
            total_mass = float(np.sum(np.abs(component_coefs)) + abs(interaction_coef))
            weights["lambda_LD"] = float(abs(interaction_coef) / total_mass) * (1 - component_scale)
        else:
            weights["lambda_LD"] = float(lambda_LD)
    else:
        weights["lambda_LD"] = float(lambda_LD)

    diagnostics = {
        "method": "logreg",
        "raw_coefficients": coefs.tolist(),
        "feature_names": list(X.columns),
        "intercept": float(model.intercept_[0]),
        "training_auc": float(roc_auc_score(y, model.predict_proba(X)[:, 1])),
        "cv5_auc": auc_cv,
        "training_accuracy": float(accuracy_score(y, model.predict(X))),
        "n_samples": int(X.shape[0]),
        "n_positive": int(y.sum()),
    }
    return weights, diagnostics


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_weights(weights: dict, path: str | Path, diagnostics: Optional[dict] = None) -> None:
    """Save weights (and optional diagnostics) to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": weights}
    if diagnostics:
        payload["diagnostics"] = diagnostics
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_weights(path: str | Path) -> dict:
    """Load weights from YAML; returns the weights dict only."""
    with Path(path).open() as f:
        payload = yaml.safe_load(f)
    return payload["weights"]
