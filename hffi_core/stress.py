"""
Stress simulation for households.

Two regimes:

1. Deterministic shock scenarios — apply a named shock (income drop, inflation
   spike, market crash) to the household inputs and recompute HFFI. Produces
   a comparison table.

2. Monte Carlo — sample many shock combinations from probability distributions
   and trace out a distribution of post-shock HFFI scores. Outputs include
   mean, standard deviation, 5th/95th percentile (Value-at-Risk style).
"""

from __future__ import annotations
from dataclasses import replace, dataclass
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

from .scoring import HouseholdInputs, compute_household_hffi


# --------------------------------------------------------------------------- #
# Named scenarios
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "baseline":       dict(income_drop=0.0, inflation_add=0.0, rate_add=0.0,
                           equity_drawdown=0.0, unemp_add=0.0),
    "mild_recession": dict(income_drop=0.10, inflation_add=0.01, rate_add=0.01,
                           equity_drawdown=0.15, unemp_add=0.02),
    "severe_recession": dict(income_drop=0.25, inflation_add=0.03, rate_add=0.02,
                             equity_drawdown=0.35, unemp_add=0.05),
    "stagflation":    dict(income_drop=0.05, inflation_add=0.06, rate_add=0.04,
                           equity_drawdown=0.20, unemp_add=0.03),
    "job_loss":       dict(income_drop=1.0,  inflation_add=0.0, rate_add=0.0,
                           equity_drawdown=0.0,  unemp_add=0.0),
    "rate_shock":     dict(income_drop=0.0,  inflation_add=0.0, rate_add=0.03,
                           equity_drawdown=0.10, unemp_add=0.01),
    "market_crash":   dict(income_drop=0.0,  inflation_add=0.0, rate_add=0.0,
                           equity_drawdown=0.40, unemp_add=0.0),
}


def _apply_shock(
    household: HouseholdInputs,
    macro: dict,
    shock: dict,
) -> tuple[HouseholdInputs, dict]:
    """Apply a single shock dict to a household + macro snapshot. Returns shocked copies."""
    new_income = household.monthly_income * (1.0 - shock.get("income_drop", 0.0))
    new_dd = min(1.0, household.expected_drawdown + shock.get("equity_drawdown", 0.0))

    new_household = replace(
        household,
        monthly_income=max(new_income, 0.0),
        expected_drawdown=new_dd,
    )
    new_macro = {
        "inflation_rate":     macro.get("inflation_rate", 0.02) + shock.get("inflation_add", 0.0),
        "fed_funds_rate":     macro.get("fed_funds_rate", 0.03) + shock.get("rate_add", 0.0),
        "unemployment_rate":  macro.get("unemployment_rate", 0.04) + shock.get("unemp_add", 0.0),
    }
    return new_household, new_macro


# --------------------------------------------------------------------------- #
# Deterministic scenarios
# --------------------------------------------------------------------------- #
def apply_shock_scenarios(
    household: HouseholdInputs,
    macro: dict,
    weights: Optional[dict] = None,
    scenarios: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Run all named scenarios, return a DataFrame with HFFI under each."""
    scns = scenarios or SCENARIOS
    rows = []
    for name, shock in scns.items():
        h_shocked, m_shocked = _apply_shock(household, macro, shock)
        result = compute_household_hffi(h_shocked, m_shocked, weights=weights)
        rows.append({
            "scenario": name,
            "HFFI": round(result.score, 1),
            "band": result.band,
            "distress_prob": round(result.distress_probability, 3),
            "L": round(result.L, 3),
            "D": round(result.D, 3),
            "E": round(result.E, 3),
            "P": round(result.P, 3),
            "M": round(result.M, 3),
        })
    return pd.DataFrame(rows).set_index("scenario")


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass
class MonteCarloResult:
    scores: np.ndarray
    distress_probs: np.ndarray
    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    prob_severe: float  # P(score > 80)


def monte_carlo_stress(
    household: HouseholdInputs,
    macro: dict,
    weights: Optional[dict] = None,
    n_sims: int = 5000,
    seed: int = 42,
    # Distribution parameters for shock variables
    income_drop_mean: float = 0.05, income_drop_std: float = 0.15,
    inflation_add_mean: float = 0.01, inflation_add_std: float = 0.02,
    rate_add_mean: float = 0.005, rate_add_std: float = 0.015,
    equity_dd_mean: float = 0.10, equity_dd_std: float = 0.18,
    unemp_add_mean: float = 0.005, unemp_add_std: float = 0.02,
) -> MonteCarloResult:
    """Run a Monte Carlo over plausible shock combinations.

    All shocks are drawn from independent truncated normals (clipped to
    sensible ranges to avoid negative inflation, etc.). For a more refined
    model, replace with a multivariate distribution capturing correlations
    between, e.g., recession-induced income drops and equity drawdowns.
    """
    rng = np.random.default_rng(seed)

    # Independent draws (truncated)
    income_drops = np.clip(rng.normal(income_drop_mean, income_drop_std, n_sims), 0, 1)
    inflation_adds = np.clip(rng.normal(inflation_add_mean, inflation_add_std, n_sims), -0.02, 0.15)
    rate_adds = np.clip(rng.normal(rate_add_mean, rate_add_std, n_sims), -0.02, 0.10)
    equity_dds = np.clip(rng.normal(equity_dd_mean, equity_dd_std, n_sims), 0, 0.80)
    unemp_adds = np.clip(rng.normal(unemp_add_mean, unemp_add_std, n_sims), 0, 0.15)

    scores = np.empty(n_sims)
    distress = np.empty(n_sims)
    for i in range(n_sims):
        shock = {
            "income_drop":     income_drops[i],
            "inflation_add":   inflation_adds[i],
            "rate_add":        rate_adds[i],
            "equity_drawdown": equity_dds[i],
            "unemp_add":       unemp_adds[i],
        }
        h_shocked, m_shocked = _apply_shock(household, macro, shock)
        result = compute_household_hffi(h_shocked, m_shocked, weights=weights)
        scores[i] = result.score
        distress[i] = result.distress_probability

    return MonteCarloResult(
        scores=scores,
        distress_probs=distress,
        mean=float(scores.mean()),
        std=float(scores.std()),
        p05=float(np.percentile(scores, 5)),
        p50=float(np.percentile(scores, 50)),
        p95=float(np.percentile(scores, 95)),
        prob_severe=float((scores > 80).mean()),
    )
