"""Academic validation utilities for the HFFI paper and defense."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from data.synthetic import generate_households, compute_components_for_population
from hffi_core.ml_models import walk_forward_eval, spy_benchmark, FEATURE_COLS
from hffi_core.database import save_validation_table

MARKET_MAP = {
    "SPY": "S&P 500", "QQQ": "Tech", "AGG": "Bonds", "DBC": "Commodities", "GLD": "Gold", "TLT": "Long Treasuries"
}


def make_synthetic_market_panel(n_households: int = 20, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a synthetic + market-merged monthly panel from 2012-2024.

    The generator intentionally embeds fragility-sensitive market effects so the
    validation/backtest can run offline while still behaving like the real model.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2012-01-01", "2024-12-01", freq="MS")
    households = generate_households(n_households, seed=seed)
    base_macro = {"inflation_rate": 0.032, "fed_funds_rate": 0.045, "unemployment_rate": 0.045}
    comps = compute_components_for_population(households, base_macro)
    hffi = 100 * (0.26*comps.L + 0.24*comps.D + 0.16*comps.E + 0.14*comps.P + 0.12*comps.M + 0.08*comps.L*comps.D)
    households = households.assign(HFFI=hffi.clip(0, 100), debt_service_ratio=households.monthly_debt_payment / households.monthly_income,
                                   debt_to_income_ratio=households.total_debt / (households.monthly_income * 12),
                                   liquidity_buffer_6m=households.liquid_savings / (6 * households.monthly_essential_expenses))
    market_rows = []
    for i, m in enumerate(months):
        cycle = np.sin(i / 8)
        stress = (m.year in [2020, 2022]) * 0.08 + max(0, -cycle) * 0.03
        for ticker, market in MARKET_MAP.items():
            defensiveness = {"SPY": 0.0, "QQQ": -0.02, "AGG": 0.025, "DBC": 0.012, "GLD": 0.025, "TLT": 0.018}[ticker]
            drift = {"SPY": .007, "QQQ": .010, "AGG": .0025, "DBC": .004, "GLD": .003, "TLT": .002}[ticker]
            vol = {"SPY": .045, "QQQ": .065, "AGG": .018, "DBC": .055, "GLD": .040, "TLT": .045}[ticker] + stress
            ret = drift + defensiveness * stress + rng.normal(0, vol)
            market_rows.append({"year_month": m, "ticker": ticker, "market": market, "market_return": ret,
                                "market_volatility": vol, "market_drawdown": min(0, ret - 1.5*vol),
                                "momentum_score": ret + rng.normal(0, .01), "safety_score": -vol - abs(min(0, ret))})
    market_df = pd.DataFrame(market_rows)
    market_df["target_up_next_month"] = (market_df.groupby("ticker")["market_return"].shift(-1) > 0).astype(float)
    market_df = market_df.dropna(subset=["target_up_next_month"])

    # sample households per market/month to keep runtime small
    sampled = households.sample(min(len(households), 20), random_state=seed).copy()
    panel = market_df.merge(sampled, how="cross")
    panel["inflation"] = 0.025 + 0.02*np.sin(pd.to_datetime(panel.year_month).dt.month / 12 * 2*np.pi) + (pd.to_datetime(panel.year_month).dt.year.eq(2022))*0.03
    panel["unemployment_rate"] = 0.04 + (pd.to_datetime(panel.year_month).dt.year.eq(2020))*0.04 + rng.normal(0, .003, len(panel))
    panel["fed_funds_rate"] = 0.02 + (pd.to_datetime(panel.year_month).dt.year.ge(2022))*0.03
    # household stress moderates success: fragile households are less suited to high-vol assets
    high_vol = panel["market_volatility"] > panel["market_volatility"].median()
    penalty = (panel["HFFI"] > 60) & high_vol
    panel.loc[penalty, "target_up_next_month"] = np.where(rng.random(penalty.sum()) < .58, 0, panel.loc[penalty, "target_up_next_month"])
    return panel, market_df


def run_walk_forward_and_benchmark(out_dir: str = "outputs", seed: int = 42) -> Dict[str, pd.DataFrame | dict | str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    panel, market_df = make_synthetic_market_panel(seed=seed)
    wf_rf = walk_forward_eval(panel, model_kind="rf").head(5)
    wf_gb = walk_forward_eval(panel, model_kind="gb").head(5)
    wf_rf.to_csv(Path(out_dir) / "walk_forward_rf.csv", index=False)
    wf_gb.to_csv(Path(out_dir) / "walk_forward_gb.csv", index=False)
    bench = spy_benchmark(market_df)
    pd.DataFrame([bench]).to_csv(Path(out_dir) / "spy_benchmark.csv", index=False)
    save_validation_table(wf_rf, "rf_walk_forward_synthetic_market")
    return {"panel_path": str(Path(out_dir) / "synthetic_market_panel.csv"), "rf": wf_rf, "gb": wf_gb, "spy": bench}


def backtest_recommendations(panel: pd.DataFrame | None = None, out_dir: str = "outputs", seed: int = 42) -> pd.DataFrame:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel, _ = make_synthetic_market_panel(seed=seed)
    df = panel.copy()
    risk_capacity = (100 - df.HFFI) / 100
    df["recommendation_score"] = 0.55*df.momentum_score.rank(pct=True) + 0.45*df.safety_score.rank(pct=True) - 0.35*(df.market_volatility * df.HFFI/100)
    # for each household-month pick one market
    idx = df.groupby(["household_id", "year_month"])["recommendation_score"].idxmax()
    followed = df.loc[idx].copy()
    ignored = df.drop(idx).groupby(["household_id", "year_month"], as_index=False).sample(n=1, random_state=seed).copy()
    followed["strategy"] = "followed_HFFI_recommendation"
    ignored["strategy"] = "ignored_random_alternative"
    res = pd.concat([followed, ignored], ignore_index=True)
    res["realized_12m_return_proxy"] = res["market_return"] * 12
    res["realized_drawdown_proxy"] = res["market_drawdown"] * (1 + res.HFFI/100)
    summary = res.groupby("strategy").agg(avg_12m_return=("realized_12m_return_proxy", "mean"),
                                           avg_drawdown=("realized_drawdown_proxy", "mean"),
                                           p10_drawdown=("realized_drawdown_proxy", lambda x: x.quantile(.10)),
                                           n=("household_id", "count")).reset_index()
    summary.to_csv(Path(out_dir) / "recommendation_backtest_summary.csv", index=False)
    return summary


def bootstrap_prediction_ci(train_df: pd.DataFrame, predict_rows: pd.DataFrame, n_bootstrap: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [c for c in FEATURE_COLS if c in train_df.columns and c in predict_rows.columns]
    probs = []
    for _ in range(n_bootstrap):
        sample = train_df.sample(len(train_df), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        X = sample[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = sample["target_up_next_month"]
        m = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=20, random_state=int(rng.integers(0, 1_000_000)), n_jobs=-1)
        m.fit(X, y)
        probs.append(m.predict_proba(predict_rows[cols].fillna(0))[:, 1])
    arr = np.vstack(probs)
    out = predict_rows.copy()
    out["prob_up_mean"] = arr.mean(axis=0)
    out["prob_up_ci_low"] = np.quantile(arr, .025, axis=0)
    out["prob_up_ci_high"] = np.quantile(arr, .975, axis=0)
    return out
