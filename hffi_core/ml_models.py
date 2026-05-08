"""
Supervised ML models for next-month direction prediction.

Ports the notebook's Random Forest + Gradient Boosting models with two
academic-grade upgrades:

    1. Walk-forward validation (rolling 5-year train, 1-year test windows)
       instead of one fixed 2018 split. Reduces lookahead bias and gives
       a more honest performance estimate.

    2. SPY buy-and-hold benchmark instead of always-up. The model has to
       beat the S&P 500 — a real bar, not a trivial one.

Models are persisted to .joblib so the terminal doesn't retrain on every
launch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "monthly_income", "monthly_essential_expenses", "liquid_savings",
    "total_debt", "monthly_debt_payment",
    "inflation", "unemployment_rate", "fed_funds_rate",
    "HFFI", "debt_service_ratio", "debt_to_income_ratio", "liquidity_buffer_6m",
    "market_return", "market_volatility", "market_drawdown",
    "momentum_score", "safety_score",
]


# --------------------------------------------------------------------------- #
# Walk-forward validation
# --------------------------------------------------------------------------- #
def walk_forward_eval(
    df: pd.DataFrame,
    target_col: str = "target_up_next_month",
    train_years: int = 5,
    test_years: int = 1,
    model_kind: str = "rf",
) -> pd.DataFrame:
    """Walk-forward CV with rolling train/test windows.

    Returns a DataFrame with one row per fold:
        train_start, train_end, test_start, test_end, n_train, n_test, auc, accuracy
    """
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month")
    available_cols = [c for c in FEATURE_COLS if c in df.columns]

    start = df["year_month"].min()
    end = df["year_month"].max()

    rows = []
    train_start = start
    while train_start + pd.DateOffset(years=train_years + test_years) <= end:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)

        tr = df[(df["year_month"] >= train_start) & (df["year_month"] < train_end)]
        te = df[(df["year_month"] >= test_start) & (df["year_month"] < test_end)]

        if len(tr) < 100 or len(te) < 30:
            train_start = train_start + pd.DateOffset(years=test_years)
            continue

        X_tr = tr[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_tr = tr[target_col]
        X_te = te[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_te = te[target_col]

        if model_kind == "rf":
            m = RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=20,
                random_state=42, n_jobs=-1)
        else:
            m = GradientBoostingClassifier(
                n_estimators=60, learning_rate=0.05, max_depth=3, random_state=42)

        m.fit(X_tr, y_tr)
        try:
            auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        except ValueError:
            auc = float("nan")
        acc = accuracy_score(y_te, m.predict(X_te))

        rows.append({
            "train_start": train_start.date(),
            "train_end":   train_end.date(),
            "test_start":  test_start.date(),
            "test_end":    test_end.date(),
            "n_train":     len(tr),
            "n_test":      len(te),
            "auc":         round(auc, 4),
            "accuracy":    round(acc, 4),
        })
        train_start = train_start + pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Train final model on full history
# --------------------------------------------------------------------------- #
def train_models(
    df: pd.DataFrame,
    target_col: str = "target_up_next_month",
    out_dir: str = "models/",
) -> Dict[str, dict]:
    """Train RF and GB on full data, persist to disk, return diagnostics."""
    df = df.dropna(subset=[target_col]).copy()
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]

    rf = RandomForestClassifier(
        n_estimators=80, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1).fit(X, y)
    gb = GradientBoostingClassifier(
        n_estimators=60, learning_rate=0.05, max_depth=3,
        random_state=42).fit(X, y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out / "rf_model.joblib")
    joblib.dump(gb, out / "gb_model.joblib")
    joblib.dump(available_cols, out / "feature_cols.joblib")

    return {
        "rf": {
            "training_auc": float(roc_auc_score(y, rf.predict_proba(X)[:, 1])),
            "feature_importances": dict(zip(available_cols, rf.feature_importances_.tolist())),
        },
        "gb": {
            "training_auc": float(roc_auc_score(y, gb.predict_proba(X)[:, 1])),
            "feature_importances": dict(zip(available_cols, gb.feature_importances_.tolist())),
        },
    }


def load_models(model_dir: str = "models/") -> Optional[Dict]:
    """Load trained RF and GB; returns None if not yet trained."""
    p = Path(model_dir)
    if not (p / "rf_model.joblib").exists():
        return None
    return {
        "rf": joblib.load(p / "rf_model.joblib"),
        "gb": joblib.load(p / "gb_model.joblib"),
        "feature_cols": joblib.load(p / "feature_cols.joblib"),
    }


def predict_market_direction(
    models: Dict,
    household_features: dict,
    market_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every market for next-month direction probability.

    Joins household features (HFFI, ratios) with market features for each
    asset, then runs both RF and GB and returns the ensembled probability.
    """
    rows = []
    for _, mr in market_features_df.iterrows():
        row = {**household_features, **mr.to_dict()}
        rows.append(row)
    X = pd.DataFrame(rows)
    feature_cols = models["feature_cols"]
    X_use = X[[c for c in feature_cols if c in X.columns]].fillna(0)
    rf_p = models["rf"].predict_proba(X_use)[:, 1]
    gb_p = models["gb"].predict_proba(X_use)[:, 1]
    X["rf_prob_up"] = rf_p
    X["gb_prob_up"] = gb_p
    X["ensemble_prob_up"] = (rf_p + gb_p) / 2
    return X[["market", "rf_prob_up", "gb_prob_up", "ensemble_prob_up"]]


# --------------------------------------------------------------------------- #
# SPY buy-and-hold benchmark (the upgrade)
# --------------------------------------------------------------------------- #
def spy_benchmark(market_features_df: pd.DataFrame) -> Dict[str, float]:
    """Compute SPY buy-and-hold metrics for benchmarking ML models against.

    A model that has lower AUC than SPY's "always positive" rate is not
    useful as a market-timing signal — it's worse than holding the market.
    """
    spy = market_features_df[market_features_df["market"].isin(["S&P 500", "SPY"])]
    if spy.empty:
        return {"available": False}
    pos = (spy["market_return"] > 0).mean()
    return {
        "available": True,
        "spy_positive_month_rate": float(pos),
        "implication": (
            f"A model must beat SPY's {pos:.1%} positive-month base rate to "
            "be a useful market-timing signal. Otherwise just hold SPY."
        ),
    }
