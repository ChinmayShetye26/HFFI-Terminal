"""
Long-horizon investment plan generator.

Given the household's HFFI and a chosen investment horizon (1–30 years),
generate a Monte Carlo wealth projection showing:

    - Expected portfolio value at year N
    - 5th–95th percentile wealth band (uncertainty)
    - Year-by-year contribution + growth schedule
    - Tactical asset-allocation transitions as fragility evolves

This is the heart of the "Investment Plan up to 10 Years" deliverable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Portfolio characteristics — must match market_recommender.PORTFOLIO_CANDIDATES
PORTFOLIO_PARAMS = {
    "Conservative":     {"mu": 0.05, "sigma": 0.08},
    "Balanced":         {"mu": 0.07, "sigma": 0.13},
    "Growth":           {"mu": 0.10, "sigma": 0.20},
    "Aggressive Growth":{"mu": 0.12, "sigma": 0.25},
}


@dataclass
class InvestmentPlan:
    portfolio: str
    horizon_years: int
    initial_capital: float
    monthly_contribution: float
    annual_contribution_growth: float
    expected_return: float
    volatility: float
    yearly_schedule: pd.DataFrame  # year, contribution, expected_value, p5, p50, p95
    final_p50: float
    final_p5: float
    final_p95: float
    final_expected: float
    n_simulations: int
    summary: str


# --------------------------------------------------------------------------- #
# Monte Carlo wealth simulator
# --------------------------------------------------------------------------- #
def simulate_wealth_path(
    initial: float,
    monthly_contribution: float,
    annual_contribution_growth: float,
    horizon_years: int,
    mu: float,
    sigma: float,
    n_sims: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Run Monte Carlo on geometric Brownian motion with monthly contributions.

    Returns: shape (n_sims, horizon_years + 1) — wealth at end of each year
    (index 0 = today's initial capital).
    """
    rng = np.random.default_rng(seed)
    months = horizon_years * 12
    monthly_mu = (1 + mu) ** (1/12) - 1
    monthly_sigma = sigma / np.sqrt(12)

    # Sample monthly returns: shape (n_sims, months)
    rets = rng.normal(monthly_mu, monthly_sigma, size=(n_sims, months))

    wealth = np.full(n_sims, initial, dtype=float)
    yearly = np.zeros((n_sims, horizon_years + 1))
    yearly[:, 0] = initial
    contribution = monthly_contribution

    for m in range(months):
        # Apply return then add contribution
        wealth = wealth * (1 + rets[:, m]) + contribution
        # Bump contribution at year boundaries
        if (m + 1) % 12 == 0:
            year_idx = (m + 1) // 12
            yearly[:, year_idx] = wealth
            contribution *= (1 + annual_contribution_growth)
    return yearly


def build_investment_plan(
    portfolio: str,
    horizon_years: int,
    initial_capital: float,
    monthly_contribution: float,
    annual_contribution_growth: float = 0.03,
    n_sims: int = 5000,
    seed: int = 42,
) -> InvestmentPlan:
    """Build a full investment plan for the chosen portfolio + horizon."""
    if portfolio not in PORTFOLIO_PARAMS:
        raise ValueError(f"Unknown portfolio: {portfolio}")
    if horizon_years < 1 or horizon_years > 30:
        raise ValueError("horizon_years must be in [1, 30]")

    params = PORTFOLIO_PARAMS[portfolio]
    mu, sigma = params["mu"], params["sigma"]
    paths = simulate_wealth_path(
        initial_capital, monthly_contribution, annual_contribution_growth,
        horizon_years, mu, sigma, n_sims, seed,
    )

    rows = []
    contribution = monthly_contribution
    cumulative_contrib = initial_capital
    for year in range(0, horizon_years + 1):
        if year > 0:
            cumulative_contrib += contribution * 12
            contribution *= (1 + annual_contribution_growth)
        col = paths[:, year]
        rows.append({
            "year": year,
            "cumulative_contribution": round(cumulative_contrib, 0),
            "expected_value":  round(np.mean(col), 0),
            "p5":  round(np.percentile(col, 5), 0),
            "p50": round(np.percentile(col, 50), 0),
            "p95": round(np.percentile(col, 95), 0),
            "growth_above_contributions": round(np.mean(col) - cumulative_contrib, 0),
        })
    schedule = pd.DataFrame(rows)

    final = schedule.iloc[-1]
    summary = (
        f"With a {portfolio} portfolio over {horizon_years} years, "
        f"starting with ${initial_capital:,.0f} and contributing ${monthly_contribution:,.0f}/month "
        f"(growing {annual_contribution_growth:.0%}/yr), the median outcome is "
        f"${final['p50']:,.0f}. The 5th-95th percentile band is "
        f"${final['p5']:,.0f}–${final['p95']:,.0f}. "
        f"Total contributed: ${final['cumulative_contribution']:,.0f}; "
        f"investment growth: ${final['growth_above_contributions']:,.0f}."
    )

    return InvestmentPlan(
        portfolio=portfolio, horizon_years=horizon_years,
        initial_capital=initial_capital, monthly_contribution=monthly_contribution,
        annual_contribution_growth=annual_contribution_growth,
        expected_return=mu, volatility=sigma,
        yearly_schedule=schedule,
        final_p50=float(final["p50"]),
        final_p5=float(final["p5"]),
        final_p95=float(final["p95"]),
        final_expected=float(final["expected_value"]),
        n_simulations=n_sims,
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Compare multiple portfolio choices for the same horizon
# --------------------------------------------------------------------------- #
def compare_portfolios(
    horizon_years: int,
    initial_capital: float,
    monthly_contribution: float,
    annual_contribution_growth: float = 0.03,
    portfolios: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Side-by-side comparison of all portfolios at the given horizon."""
    if portfolios is None:
        portfolios = list(PORTFOLIO_PARAMS.keys())

    rows = []
    for p in portfolios:
        plan = build_investment_plan(
            p, horizon_years, initial_capital,
            monthly_contribution, annual_contribution_growth)
        rows.append({
            "portfolio": p,
            "expected_return": plan.expected_return,
            "volatility":     plan.volatility,
            f"median_value_y{horizon_years}":     plan.final_p50,
            f"5th_pctile_value_y{horizon_years}": plan.final_p5,
            f"95th_pctile_value_y{horizon_years}":plan.final_p95,
            "expected_value": plan.final_expected,
        })
    return pd.DataFrame(rows)
