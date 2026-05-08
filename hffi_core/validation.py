"""
Validation harness for HFFI.

Three pillars, each addressing a specific reviewer concern:

1. Out-of-sample evaluation
   ------------------------
   K-fold cross-validation with strict train/test separation. Reports AUC,
   Brier score, and calibration. Defends against the "p-hacked backtest"
   concern flagged by the medical Fragility Index philosophy.

2. Baseline comparison
   -------------------
   The HFFI must beat simple baselines or it has no claim to a contribution.
   Baselines:
     - DSR-only (debt service ratio alone)
     - Liquidity-only (months of buffer)
     - FICO-style equal-weight composite (simple sum of normalized components)
   For each, we report AUC and ΔAUC vs HFFI.

3. Sensitivity analysis
   --------------------
   The reviewer's question "what if you set the weights differently?" must
   have a defensible answer. We perturb each weight by ±20% and report how
   much the household ranking changes (Spearman correlation) and how much
   the band assignments flip.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from .scoring import hffi_score, DEFAULT_WEIGHTS, risk_band, distress_probability


# --------------------------------------------------------------------------- #
# 1. Out-of-sample evaluation
# --------------------------------------------------------------------------- #
def out_of_sample_eval(
    components_df: pd.DataFrame,
    distress_label: pd.Series,
    weights: Optional[dict] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """K-fold out-of-sample evaluation of HFFI as a distress predictor.

    For each fold, scores are computed on the held-out data using the SAME
    weights (so we are evaluating the formula, not refitting). Returns mean
    AUC, Brier score, and calibration diagnostic.

    Note: if the weights themselves were learned from data, you must split
    BEFORE learning the weights for an honest evaluation. This function
    assumes weights are already fixed.
    """
    w = weights or DEFAULT_WEIGHTS
    X = components_df[["L", "D", "E", "P", "M"]].to_numpy()
    y = distress_label.astype(int).to_numpy()

    aucs, briers = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for _, test_idx in kf.split(X):
        X_test, y_test = X[test_idx], y[test_idx]
        scores = np.array([
            hffi_score(*row, weights=w) for row in X_test
        ])
        probs = np.array([distress_probability(s) for s in scores])
        try:
            aucs.append(roc_auc_score(y_test, scores))
            briers.append(brier_score_loss(y_test, probs))
        except ValueError:
            # Single-class fold — skip
            continue

    # Calibration: bin households into deciles by HFFI, compare avg predicted
    # vs avg actual distress rate.
    full_scores = np.array([hffi_score(*row, weights=w) for row in X])
    full_probs = np.array([distress_probability(s) for s in full_scores])
    deciles = pd.qcut(full_scores, q=10, duplicates="drop")
    cal = pd.DataFrame({"score": full_scores, "prob": full_probs, "y": y, "decile": deciles})
    calibration_table = cal.groupby("decile", observed=True).agg(
        avg_score=("score", "mean"),
        avg_predicted=("prob", "mean"),
        actual_rate=("y", "mean"),
        n=("y", "size"),
    ).round(3)

    return {
        "n_splits": n_splits,
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc": float(np.std(aucs)) if aucs else float("nan"),
        "mean_brier": float(np.mean(briers)) if briers else float("nan"),
        "calibration_table": calibration_table,
    }


# --------------------------------------------------------------------------- #
# 2. Baseline comparison
# --------------------------------------------------------------------------- #
def baseline_comparison(
    components_df: pd.DataFrame,
    distress_label: pd.Series,
    weights: Optional[dict] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare HFFI against simpler baselines on the same prediction task."""
    w = weights or DEFAULT_WEIGHTS
    X = components_df[["L", "D", "E", "P", "M"]].to_numpy()
    y = distress_label.astype(int).to_numpy()

    # Predictors to compare
    predictors = {
        "HFFI (full)":           np.array([hffi_score(*row, weights=w) for row in X]),
        "Liquidity only":        X[:, 0] * 100,                  # L
        "Debt only (DSR proxy)": X[:, 1] * 100,                  # D
        "Equal-weight composite": np.mean(X, axis=1) * 100,       # naive average
        "Logistic on components": _logreg_baseline(X, y, n_splits, random_state),
    }

    rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for name, pred in predictors.items():
        fold_aucs = []
        for _, test_idx in kf.split(X):
            try:
                fold_aucs.append(roc_auc_score(y[test_idx], pred[test_idx]))
            except ValueError:
                continue
        rows.append({
            "model": name,
            "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
            "auc_std":  float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        })
    df = pd.DataFrame(rows)
    df["delta_vs_hffi"] = df["auc_mean"] - df.loc[df["model"] == "HFFI (full)", "auc_mean"].values[0]
    return df.round(4)


def _logreg_baseline(X, y, n_splits, random_state):
    """Out-of-fold predictions from a logistic regression — strong baseline."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        model = LogisticRegression(max_iter=2000)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


# --------------------------------------------------------------------------- #
# 3. Sensitivity analysis
# --------------------------------------------------------------------------- #
def sensitivity_analysis(
    components_df: pd.DataFrame,
    weights: Optional[dict] = None,
    perturbation: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """How robust are HFFI rankings to ±perturbation in each weight?

    For each weight key, we perturb it by ±perturbation (e.g. ±20%),
    recompute scores for every household, and report:
        - Spearman correlation with the baseline ranking
        - % of households whose risk band changed
    """
    base_w = weights or DEFAULT_WEIGHTS.copy()
    X = components_df[["L", "D", "E", "P", "M"]].to_numpy()
    base_scores = np.array([hffi_score(*row, weights=base_w) for row in X])
    base_bands = np.array([risk_band(s) for s in base_scores])

    rows = []
    for key in base_w:
        for direction in (-1, +1):
            new_w = base_w.copy()
            new_w[key] = base_w[key] * (1 + direction * perturbation)
            new_scores = np.array([hffi_score(*row, weights=new_w) for row in X])
            new_bands = np.array([risk_band(s) for s in new_scores])

            rho, _ = spearmanr(base_scores, new_scores)
            band_flip_rate = float((base_bands != new_bands).mean())

            rows.append({
                "weight_perturbed": key,
                "direction": "+" if direction > 0 else "-",
                "perturbation_pct": int(perturbation * 100),
                "spearman_rho": round(float(rho), 4),
                "band_flip_pct": round(band_flip_rate * 100, 2),
            })
    return pd.DataFrame(rows)
