"""
Market suitability recommender.

Ports the notebook's portfolio + market recommender into the production engine:
    - Portfolio recommender (Conservative / Balanced / Growth) with HFFI-allowed sets
    - Market suitability scorer across asset categories with regime detection
    - Concrete buy/hold/sell signal per asset given the household's HFFI

The output is structured so the dashboard and report generator can consume it
directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Portfolio templates (from notebook cell 16)
# --------------------------------------------------------------------------- #
PORTFOLIO_CANDIDATES = pd.DataFrame([
    {"portfolio": "Conservative", "exp_return": 0.05, "volatility": 0.08,
     "max_drawdown": 0.10, "liquidity_score": 0.95},
    {"portfolio": "Balanced",     "exp_return": 0.07, "volatility": 0.13,
     "max_drawdown": 0.18, "liquidity_score": 0.80},
    {"portfolio": "Growth",       "exp_return": 0.10, "volatility": 0.20,
     "max_drawdown": 0.30, "liquidity_score": 0.60},
    {"portfolio": "Aggressive Growth", "exp_return": 0.12, "volatility": 0.25,
     "max_drawdown": 0.40, "liquidity_score": 0.50},
])


def allowed_portfolios(hffi: float) -> List[str]:
    """HFFI-conditional set of allowed portfolios (notebook cell 16)."""
    if hffi >= 80:
        return ["Conservative"]
    if hffi >= 60:
        return ["Conservative", "Balanced"]
    if hffi >= 30:
        return ["Conservative", "Balanced", "Growth"]
    return ["Balanced", "Growth", "Aggressive Growth"]


def fragility_target_profile(hffi: float) -> Dict[str, float]:
    """Target volatility/drawdown/liquidity profile by HFFI band."""
    if hffi <= 30:
        return {"preferred_volatility": 0.20, "preferred_drawdown": 0.30,
                "preferred_liquidity": 0.60}
    if hffi <= 60:
        return {"preferred_volatility": 0.13, "preferred_drawdown": 0.18,
                "preferred_liquidity": 0.80}
    if hffi <= 80:
        return {"preferred_volatility": 0.09, "preferred_drawdown": 0.12,
                "preferred_liquidity": 0.90}
    return {"preferred_volatility": 0.07, "preferred_drawdown": 0.08,
            "preferred_liquidity": 0.95}


def score_portfolios(hffi: float) -> pd.DataFrame:
    """Score the four portfolio templates for a given HFFI score."""
    target = fragility_target_profile(hffi)
    allowed = allowed_portfolios(hffi)
    scored = PORTFOLIO_CANDIDATES[PORTFOLIO_CANDIDATES["portfolio"].isin(allowed)].copy()

    if len(scored) == 1:
        scored["suitability_score"] = 1.0
        return scored.reset_index(drop=True)

    rng = scored["exp_return"].max() - scored["exp_return"].min()
    if rng == 0:
        scored["return_score"] = 1.0
    else:
        scored["return_score"] = (
            (scored["exp_return"] - scored["exp_return"].min()) / rng
        )

    scored["vol_penalty"]   = abs(scored["volatility"]   - target["preferred_volatility"])
    scored["dd_penalty"]    = abs(scored["max_drawdown"] - target["preferred_drawdown"])
    scored["liq_alignment"] = 1 - abs(scored["liquidity_score"] - target["preferred_liquidity"])

    scored["suitability_score"] = (
        0.35 * scored["return_score"]
        - 0.25 * scored["vol_penalty"]
        - 0.20 * scored["dd_penalty"]
        + 0.20 * scored["liq_alignment"]
    )
    return scored.sort_values("suitability_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Market features (from notebook cell 4) — vectorized
# --------------------------------------------------------------------------- #
def compute_market_features(
    monthly_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute momentum, safety, drawdown, sharpe-proxy for each asset.

    monthly_prices and monthly_returns must share columns (one per asset)
    indexed by month-end dates.
    """
    rows = []
    for col in monthly_returns.columns:
        df = pd.DataFrame({
            "date":   monthly_returns.index,
            "market": col,
            "market_return": monthly_returns[col],
        }).dropna()

        df["market_volatility"] = df["market_return"].rolling(12).std()
        dd_series = (monthly_prices[col] / monthly_prices[col].cummax()) - 1
        df["market_drawdown"] = dd_series.reindex(df["date"]).rolling(12).min().values
        df["market_sharpe_proxy"] = df["market_return"] / df["market_volatility"]
        df["return_3m"] = df["market_return"].rolling(3).mean()
        df["volatility_3m"] = df["market_return"].rolling(3).std()
        df["momentum_score"] = 0.6 * df["market_return"].fillna(0) + 0.4 * df["return_3m"].fillna(0)
        df["safety_score"] = (
            -0.6 * df["volatility_3m"].fillna(0)
            -0.4 * df["market_drawdown"].fillna(0)
        )
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Per-household market suitability (from notebook cell 20)
# --------------------------------------------------------------------------- #
def score_markets_for_household(
    hffi: float,
    risk_band: str,
    debt_service_ratio: float,
    macro_stress_index: float,
    liquidity_buffer_6m: float,
    market_snapshot: pd.DataFrame,
    risk_off_regime: bool,
) -> pd.DataFrame:
    """Score each asset for suitability given the household's profile.

    market_snapshot must contain one row per asset with columns:
        market, market_return, market_volatility, market_drawdown,
        market_sharpe_proxy, momentum_score, safety_score.
    """
    s = market_snapshot.copy()
    target = fragility_target_profile(hffi)

    s["preferred_volatility"] = target["preferred_volatility"]
    s["preferred_drawdown"]   = target["preferred_drawdown"]

    s["vol_penalty"] = (s["market_volatility"] - target["preferred_volatility"]).abs()
    s["dd_penalty"]  = (s["market_drawdown"]   - target["preferred_drawdown"]).abs()

    s["household_risk_penalty"] = (
        0.30 * debt_service_ratio
        + 0.20 * macro_stress_index
        - 0.15 * liquidity_buffer_6m
    )

    def normalize(col: pd.Series) -> pd.Series:
        rng = col.max() - col.min()
        return (col - col.min()) / rng if rng != 0 else pd.Series(0.0, index=col.index)

    s["return_score"]  = normalize(s["market_return"].fillna(0))
    s["sharpe_score"]  = normalize(s["market_sharpe_proxy"].fillna(0))
    s["momentum_norm"] = normalize(s["momentum_score"].fillna(0))
    s["safety_norm"]   = normalize(s["safety_score"].fillna(0))

    if risk_off_regime:
        # Risk-off: heavier on safety/sharpe than return
        w_return = 0.15
        w_sharpe = 0.25
        w_momentum = 0.10
        w_safety = 0.25
        w_vol_pen = 0.15
        w_dd_pen = 0.10
        w_household_pen = 0.10
    else:
        # Risk-on
        w_return = 0.25
        w_sharpe = 0.20
        w_momentum = 0.20
        w_safety = 0.10
        w_vol_pen = 0.10
        w_dd_pen = 0.05
        w_household_pen = 0.10

    # When the household is fragile, override toward safety regardless of regime.
    # This fixes a notebook artifact where return-side weights dominated the
    # score for a 2-asset universe even when the household couldn't afford risk.
    if hffi >= 80:
        w_return = 0.05
        w_sharpe = 0.10
        w_momentum = 0.05
        w_safety = 0.40
        w_vol_pen = 0.20
        w_dd_pen = 0.15
        w_household_pen = 0.05
    elif hffi >= 60:
        w_return = 0.10
        w_sharpe = 0.20
        w_momentum = 0.10
        w_safety = 0.30
        w_vol_pen = 0.15
        w_dd_pen = 0.10
        w_household_pen = 0.05

    s["suitability_score"] = (
        w_return * s["return_score"]
        + w_sharpe * s["sharpe_score"]
        + w_momentum * s["momentum_norm"]
        + w_safety * s["safety_norm"]
        - w_vol_pen * s["vol_penalty"]
        - w_dd_pen * s["dd_penalty"]
        - w_household_pen * s["household_risk_penalty"]
    )

    s["recommendation_confidence"] = s["suitability_score"] - s["suitability_score"].mean()
    s["hffi"] = hffi
    s["risk_band"] = risk_band
    s["regime"] = "risk_off" if risk_off_regime else "risk_on"
    return s.sort_values("suitability_score", ascending=False).reset_index(drop=True)


def explain_market_recommendation(row: pd.Series) -> str:
    """Plain-English reason for each market recommendation (notebook cell 21)."""
    band = row.get("risk_band", "")
    market = row.get("market", "the asset")
    score = row.get("suitability_score", 0.0)

    if band == "Stable":
        prefix = "Low fragility supports growth-oriented exposure."
    elif band == "Moderate Fragility":
        prefix = "Moderate fragility favors balanced risk-return assets."
    elif band == "High Fragility":
        prefix = "High fragility favors lower-volatility assets with downside protection."
    else:
        prefix = "Severe fragility mandates the safest assets only."

    if score > 0.5:
        verdict = f"{market} ranks as a strong fit"
    elif score > 0.0:
        verdict = f"{market} is a moderate fit"
    elif score > -0.3:
        verdict = f"{market} is a weak fit"
    else:
        verdict = f"{market} is a poor fit"

    return f"{prefix} {verdict} (suitability={score:+.2f})."


# --------------------------------------------------------------------------- #
# Buy / hold / sell signals
# --------------------------------------------------------------------------- #
@dataclass
class TradeSignal:
    ticker: str
    name: str
    action: str             # 'BUY' | 'HOLD' | 'AVOID'
    target_allocation_pct: float  # of total portfolio
    suggested_horizon: str  # '1-3 months' | '6-12 months' | '12+ months'
    suitability: float
    confidence: float
    rationale: str


def generate_trade_signals(
    portfolio_choice: str,
    market_scores: pd.DataFrame,
    asset_metadata: pd.DataFrame,
    top_n: int = 10,
) -> List[TradeSignal]:
    """Convert market suitability scores into concrete buy/hold/avoid signals.

    portfolio_choice: 'Conservative' | 'Balanced' | 'Growth' | 'Aggressive Growth'
    market_scores: output of score_markets_for_household (per asset)
    asset_metadata: ticker → name + category (from asset_universe.to_dict_records)
    """
    cat_target = {
        "Conservative":     {"bond": 0.55, "equity": 0.15, "sector": 0.10,
                             "commodity": 0.10, "index": 0.10},
        "Balanced":         {"bond": 0.40, "equity": 0.25, "sector": 0.15,
                             "commodity": 0.10, "index": 0.10},
        "Growth":           {"bond": 0.20, "equity": 0.40, "sector": 0.25,
                             "commodity": 0.05, "index": 0.10},
        "Aggressive Growth":{"bond": 0.10, "equity": 0.55, "sector": 0.25,
                             "commodity": 0.05, "index": 0.05},
    }
    targets = cat_target.get(portfolio_choice, cat_target["Balanced"])

    # Join metadata. Prefer ticker when available; fall back to market label.
    join_key = "ticker" if "ticker" in market_scores.columns and "ticker" in asset_metadata.columns else "market"
    merged = market_scores.merge(
        asset_metadata, left_on=join_key, right_on="ticker", how="left", suffixes=("", "_meta")
    )

    signals: List[TradeSignal] = []
    for category, bucket_pct in targets.items():
        bucket = merged[merged["category"] == category].copy()
        if bucket.empty:
            continue
        bucket = bucket.sort_values("suitability_score", ascending=False).head(top_n)
        # Distribute the bucket allocation across top-N assets, weighted by suitability
        total = bucket["suitability_score"].clip(lower=0).sum()
        if total <= 0:
            weights = np.ones(len(bucket)) / len(bucket)
        else:
            weights = bucket["suitability_score"].clip(lower=0).to_numpy() / total

        for (_, row), w in zip(bucket.iterrows(), weights):
            sc = row["suitability_score"]
            if sc > 0.4 and row["recommendation_confidence"] > 0:
                action, horizon = "BUY", "6-12 months"
            elif sc > 0.0:
                action, horizon = "HOLD", "1-3 months"
            else:
                action, horizon = "AVOID", "—"

            signals.append(TradeSignal(
                ticker=row["ticker"] if pd.notna(row.get("ticker")) else row["market"],
                name=row.get("name", row["market"]) if pd.notna(row.get("name")) else row["market"],
                action=action,
                target_allocation_pct=float(w * bucket_pct * 100),
                suggested_horizon=horizon,
                suitability=float(sc),
                confidence=float(row.get("recommendation_confidence", 0.0)),
                rationale=explain_market_recommendation(row),
            ))

    signals.sort(key=lambda x: (x.action != "BUY", -x.target_allocation_pct))
    return signals



def select_one_market_recommendation(market_scores: pd.DataFrame) -> Dict[str, object]:
    """Pick exactly one market for the household and return advisor-ready text.

    This implements the paper claim: every household receives one actionable
    market, selected from the joint household-fragility × market-condition score.
    """
    if market_scores.empty:
        return {"market": None, "recommendation_score": 0.0, "plain_language_reason": "No market data available."}
    row = market_scores.sort_values("suitability_score", ascending=False).iloc[0]
    hffi = float(row.get("hffi", 0))
    market = row.get("market", "selected market")
    score = float(row.get("suitability_score", 0))
    band = row.get("risk_band", "Unknown")
    if hffi < 30:
        stance = "The household appears stable, so the system can accept more growth exposure when market signals are strong."
    elif hffi < 60:
        stance = "The household sits in the Moderate risk band, so the system balances upside with resilience rather than chasing maximum return."
    elif hffi < 80:
        stance = "The household is financially stretched, so safer and lower-volatility opportunities receive priority."
    else:
        stance = "The household is severely fragile, so the recommendation prioritizes capital preservation and liquidity."
    reason = f"{stance} {market} has the best combined score after considering momentum, safety, drawdown, volatility, debt burden, liquidity buffer, and macro regime. Recommendation score: {score:.3f}."
    return {
        "market": market,
        "ticker": row.get("ticker"),
        "risk_band": band,
        "hffi": hffi,
        "recommendation_score": score,
        "plain_language_reason": reason,
    }


def demo_hffi_42_commodities_case() -> Dict[str, object]:
    """Deterministic test case requested for the paper/defense.

    A household with HFFI≈42 is Moderate risk; commodities win because their
    recent performance is stable enough and their risk profile is manageable.
    """
    market_snapshot = pd.DataFrame([
        {"market":"Tech", "ticker":"QQQ", "market_return":0.035, "market_volatility":0.075, "market_drawdown":-0.18, "market_sharpe_proxy":0.47, "momentum_score":0.030, "safety_score":-0.12},
        {"market":"Commodities", "ticker":"DBC", "market_return":0.018, "market_volatility":0.038, "market_drawdown":-0.07, "market_sharpe_proxy":0.47, "momentum_score":0.021, "safety_score":-0.035},
        {"market":"Long Treasuries", "ticker":"TLT", "market_return":-0.004, "market_volatility":0.050, "market_drawdown":-0.11, "market_sharpe_proxy":-0.08, "momentum_score":-0.003, "safety_score":-0.055},
        {"market":"S&P 500", "ticker":"SPY", "market_return":0.014, "market_volatility":0.045, "market_drawdown":-0.09, "market_sharpe_proxy":0.31, "momentum_score":0.013, "safety_score":-0.052},
    ])
    scored = score_markets_for_household(
        hffi=42, risk_band="Moderate Fragility", debt_service_ratio=0.18,
        macro_stress_index=0.045, liquidity_buffer_6m=0.55,
        market_snapshot=market_snapshot, risk_off_regime=True)
    return select_one_market_recommendation(scored)
