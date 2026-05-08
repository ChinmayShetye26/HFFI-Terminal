"""Data-science recommendation layer for HFFI Terminal.

This module adds a supervised suitability model, household segmentation,
explainable scoring, and position sizing on top of the existing HFFI rules.
It is intentionally additive: the rule-based HFFI framework remains the guardrail
and the ML layer provides probability, ranking, and evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from hffi_core.ml_models import FEATURE_COLS
from hffi_core.validation_runner import make_synthetic_market_panel


@dataclass
class ModelBundle:
    rf: RandomForestClassifier
    gb: GradientBoostingClassifier
    feature_cols: list[str]
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    segment_centers: pd.DataFrame


def engineer_household_features(inputs, result, macro: dict, holdings_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Create household features used by the ML suitability model."""
    monthly_income = float(inputs.monthly_income or 0)
    annual_income = monthly_income * 12
    essential = float(inputs.monthly_essential_expenses or 0)
    total_expenses = float(inputs.monthly_total_expenses or 0)
    liquid_savings = float(inputs.liquid_savings or 0)
    total_debt = float(inputs.total_debt or 0)
    debt_payment = float(inputs.monthly_debt_payment or 0)

    debt_service_ratio = debt_payment / monthly_income if monthly_income else 1.0
    debt_to_income_ratio = total_debt / annual_income if annual_income else 1.0
    liquidity_buffer_6m = liquid_savings / (6 * essential) if essential else 0.0
    monthly_buying_capacity = max(monthly_income - total_expenses - debt_payment, 0.0)
    employment_type = str(getattr(inputs, "employment_type", "full_time") or "full_time").lower()
    employment_risk = {
        "full_time": 0.0,
        "salaried": 0.0,
        "part_time": 0.12,
        "contract": 0.18,
        "self_employed": 0.16,
        "unemployed": 0.35,
        "retired": 0.06,
    }.get(employment_type, 0.08)
    dependents = max(int(getattr(inputs, "dependents", 0) or 0), 0)

    equity_pct = float(inputs.portfolio_weights.get("equity", 0.0))
    bond_pct = float(inputs.portfolio_weights.get("bond", 0.0))
    cash_pct = float(inputs.portfolio_weights.get("cash", 0.0))
    concentration = 0.0
    if holdings_df is not None and not holdings_df.empty and "allocation_weight" in holdings_df.columns:
        concentration = float(holdings_df["allocation_weight"].max())

    return {
        "monthly_income": monthly_income,
        "monthly_essential_expenses": essential,
        "liquid_savings": liquid_savings,
        "total_debt": total_debt,
        "monthly_debt_payment": debt_payment,
        "inflation": float(macro.get("inflation_rate", 0.0)),
        "unemployment_rate": float(macro.get("unemployment_rate", 0.0)),
        "fed_funds_rate": float(macro.get("fed_funds_rate", 0.0)),
        "HFFI": float(result.score),
        "debt_service_ratio": debt_service_ratio,
        "debt_to_income_ratio": debt_to_income_ratio,
        "liquidity_buffer_6m": liquidity_buffer_6m,
        "equity_pct": equity_pct,
        "bond_pct": bond_pct,
        "liquid_savings_pct": cash_pct,
        "monthly_buying_capacity": monthly_buying_capacity,
        "employment_risk": employment_risk,
        "dependents": float(dependents),
        "dependent_pressure": min(dependents * 0.05, 0.35),
        "portfolio_volatility": float(inputs.portfolio_volatility),
        "expected_drawdown": float(inputs.expected_drawdown),
        "portfolio_concentration": concentration,
        "macro_stress_index": (
            0.4 * float(macro.get("inflation_rate", 0.0))
            + 0.4 * float(macro.get("unemployment_rate", 0.0))
            + 0.2 * float(macro.get("fed_funds_rate", 0.0))
        ),
    }


def train_suitability_model(seed: int = 42) -> ModelBundle:
    """Train RF/GB suitability models on the synthetic household-market panel."""
    panel, _ = make_synthetic_market_panel(seed=seed)
    df = panel.dropna(subset=["target_up_next_month"]).copy()
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["target_up_next_month"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=120, max_depth=8, min_samples_leaf=15,
        random_state=seed, n_jobs=1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=90, learning_rate=0.05, max_depth=3, random_state=seed,
    )
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    rf_prob = rf.predict_proba(X_test)[:, 1]
    gb_prob = gb.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + gb_prob) / 2
    metrics = {
        "rf_auc": float(roc_auc_score(y_test, rf_prob)),
        "gb_auc": float(roc_auc_score(y_test, gb_prob)),
        "ensemble_auc": float(roc_auc_score(y_test, ens_prob)),
        "ensemble_accuracy": float(accuracy_score(y_test, ens_prob >= 0.5)),
        "training_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
    }

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    seg_cols = ["debt_service_ratio", "debt_to_income_ratio", "liquidity_buffer_6m", "HFFI"]
    seg_df = df[seg_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(seg_df)
    centers = pd.DataFrame(km.cluster_centers_, columns=seg_cols)
    centers["segment"] = centers.apply(_segment_name_from_row, axis=1)

    return ModelBundle(
        rf=rf,
        gb=gb,
        feature_cols=feature_cols,
        metrics=metrics,
        feature_importance=importances,
        segment_centers=centers,
    )


def _segment_name_from_row(row: pd.Series) -> str:
    if row["HFFI"] >= 70:
        return "Fragile debt-stressed"
    if row["liquidity_buffer_6m"] < 0.35:
        return "Liquidity-stressed"
    if row["debt_service_ratio"] < 0.12 and row["liquidity_buffer_6m"] > 1.0:
        return "Stable growth-ready"
    return "Balanced saver"


def assign_household_segment(features: Dict[str, float]) -> str:
    """Rule-readable household segment for dashboard explanations."""
    if features["HFFI"] >= 70 or features["debt_service_ratio"] >= 0.35:
        return "Fragile debt-stressed"
    if features["liquidity_buffer_6m"] < 0.35:
        return "Liquidity-stressed"
    if features["HFFI"] < 30 and features["monthly_buying_capacity"] > 0:
        return "Stable growth-ready"
    return "Balanced saver"


def score_market_recommendations(
    model: ModelBundle,
    household_features: Dict[str, float],
    market_scores: pd.DataFrame,
    target_weights: Dict[str, float],
    actual_weights: Dict[str, float],
) -> pd.DataFrame:
    """Score assets with ML probability + HFFI suitability + allocation gap."""
    if market_scores.empty:
        return pd.DataFrame()

    rows = []
    for _, market in market_scores.iterrows():
        row = {**household_features, **market.to_dict()}
        rows.append(row)
    X = pd.DataFrame(rows)
    for col in model.feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X_model = X[model.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    rf_prob = model.rf.predict_proba(X_model)[:, 1]
    gb_prob = model.gb.predict_proba(X_model)[:, 1]
    X["ml_probability"] = (rf_prob + gb_prob) / 2

    category = X.get("category", pd.Series(["equity"] * len(X))).fillna("equity").astype(str)
    X["allocation_gap"] = [
        float(target_weights.get(cat, 0.0) - actual_weights.get(cat, 0.0))
        for cat in category
    ]
    hffi_capacity = np.clip((100.0 - float(household_features["HFFI"])) / 100.0, 0.0, 1.0)
    downside_score = 1.0 - np.clip(
        X["market_volatility"].fillna(0.0).abs() + X["market_drawdown"].fillna(0.0).abs(),
        0.0, 1.0,
    )
    market_fit = X["suitability_score"].fillna(0.0)
    market_fit_norm = (market_fit - market_fit.min()) / (market_fit.max() - market_fit.min()) if market_fit.max() != market_fit.min() else 0.5
    allocation_score = np.clip(0.5 + X["allocation_gap"], 0.0, 1.0)

    X["ds_score"] = (
        0.35 * X["ml_probability"]
        + 0.25 * hffi_capacity
        + 0.20 * allocation_score
        + 0.20 * downside_score
    )
    X["recommendation"] = np.select(
        [X["ds_score"] >= 0.68, X["ds_score"] >= 0.42],
        ["BUY CANDIDATE", "HOLD / WATCH"],
        default="AVOID / REDUCE",
    )
    monthly_capacity = float(household_features.get("monthly_buying_capacity", 0.0))
    buy_mask = X["recommendation"].eq("BUY CANDIDATE")
    positive_scores = X.loc[buy_mask, "ds_score"].clip(lower=0)
    score_sum = float(positive_scores.sum()) or 1.0
    X["suggested_monthly_amount"] = 0.0
    X.loc[buy_mask, "suggested_monthly_amount"] = monthly_capacity * positive_scores / score_sum
    X["segment"] = assign_household_segment(household_features)
    X["comment"] = X.apply(lambda r: _recommendation_comment(r, household_features), axis=1)
    cols = [
        "ticker", "name", "category", "recommendation", "suggested_monthly_amount",
        "ml_probability", "suitability_score", "allocation_gap", "ds_score",
        "segment", "comment",
    ]
    return X[[c for c in cols if c in X.columns]].sort_values("ds_score", ascending=False).reset_index(drop=True)


def _recommendation_comment(row: pd.Series, household_features: Dict[str, float]) -> str:
    hffi = household_features["HFFI"]
    capacity = household_features.get("monthly_buying_capacity", 0.0)
    if hffi >= 60 and str(row.get("category")) in {"equity", "sector"}:
        guardrail = "HFFI guardrail is defensive, so equity risk needs a high score before adding."
    elif hffi < 30:
        guardrail = "Low fragility allows growth exposure when market and allocation scores agree."
    else:
        guardrail = "Moderate fragility favors staged allocation instead of aggressive buying."
    capacity_text = f"Monthly buying capacity is ${capacity:,.0f}."
    return (
        f"{guardrail} ML probability={row.get('ml_probability', 0):.1%}, "
        f"HFFI suitability={row.get('suitability_score', 0):+.3f}, "
        f"allocation gap={row.get('allocation_gap', 0):+.1%}. {capacity_text}"
    )


def build_model_performance_summary(model: ModelBundle) -> pd.DataFrame:
    return pd.DataFrame([
        {"metric": "RF AUC", "value": model.metrics["rf_auc"]},
        {"metric": "GB AUC", "value": model.metrics["gb_auc"]},
        {"metric": "Ensemble AUC", "value": model.metrics["ensemble_auc"]},
        {"metric": "Ensemble Accuracy", "value": model.metrics["ensemble_accuracy"]},
        {"metric": "Training rows", "value": model.metrics["training_rows"]},
        {"metric": "Test rows", "value": model.metrics["test_rows"]},
    ])
