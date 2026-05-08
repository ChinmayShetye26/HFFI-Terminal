"""
Synthetic household data generator.

Produces a population of N households with realistic income/debt/savings
distributions plus a synthetic "distress" label generated from a known
ground-truth fragility process. Useful for:

    1. Calibrating weight-learning code before real SCF data is available
    2. Demoing the validation harness in the IEEE paper
    3. Unit tests

The generator is deliberately simple — the GOAL is not to capture the true
SCF distribution, but to give downstream code something to chew on. For
real validation you must use SCF microdata.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def generate_households(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic household-level table.

    Returns columns: household_id, monthly_income, monthly_essential_expenses,
    monthly_total_expenses, liquid_savings, total_debt, monthly_debt_payment,
    portfolio_volatility, expected_drawdown, equity_pct, bond_pct, cash_pct,
    rate_sensitivity.

    Plus a distress label generated from a noisy ground-truth fragility
    function — useful as a test target for logreg weight learning.
    """
    rng = np.random.default_rng(seed)

    # Income from a log-normal distribution centered at $5k/month
    monthly_income = rng.lognormal(mean=np.log(5000), sigma=0.55, size=n).clip(800, 60000)

    # Expenses scale with income but with substantial spread
    expense_ratio = rng.beta(5, 3, n) * 0.95  # mostly 0.4-0.85
    monthly_total_expenses = monthly_income * expense_ratio

    # Essential share is between 50% and 100% of total
    essential_share = np.clip(rng.beta(7, 3, n), 0.5, 1.0)
    monthly_essential_expenses = monthly_total_expenses * essential_share

    # Savings: heavy right-skew, many households have very little
    liquid_savings = rng.lognormal(mean=np.log(3000), sigma=2.0, size=n).clip(0, 500_000)

    # Debt: zero-inflated; ~75% have some debt
    has_debt = rng.random(n) < 0.75
    total_debt = np.where(
        has_debt,
        rng.lognormal(mean=np.log(15000), sigma=1.5, size=n),
        0.0,
    ).clip(0, 1_500_000)
    # Monthly debt payment ~ 1-2% of total debt for those with debt
    monthly_debt_payment = total_debt * rng.uniform(0.012, 0.025, n)

    # Portfolio characteristics
    equity_pct = rng.beta(2, 2, n)
    bond_pct = (1 - equity_pct) * rng.beta(3, 2, n)
    cash_pct = 1 - equity_pct - bond_pct
    portfolio_volatility = 0.04 + equity_pct * 0.20 + rng.normal(0, 0.03, n).clip(-0.05, 0.05)
    expected_drawdown = portfolio_volatility * 2 + rng.normal(0, 0.05, n).clip(-0.10, 0.10)
    expected_drawdown = expected_drawdown.clip(0, 0.6)

    rate_sensitivity = rng.beta(2, 3, n)

    df = pd.DataFrame({
        "household_id": [f"H{i:06d}" for i in range(n)],
        "monthly_income": monthly_income.round(2),
        "monthly_essential_expenses": monthly_essential_expenses.round(2),
        "monthly_total_expenses": monthly_total_expenses.round(2),
        "liquid_savings": liquid_savings.round(2),
        "total_debt": total_debt.round(2),
        "monthly_debt_payment": monthly_debt_payment.round(2),
        "portfolio_volatility": portfolio_volatility.round(4),
        "expected_drawdown": expected_drawdown.round(4),
        "equity_pct": equity_pct.round(3),
        "bond_pct": bond_pct.round(3),
        "cash_pct": cash_pct.round(3),
        "rate_sensitivity": rate_sensitivity.round(3),
    })
    return df


def add_distress_label(
    df: pd.DataFrame,
    components_df: pd.DataFrame,
    base_rate: float = 0.12,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic 0/1 distress label from the true fragility components.

    Distress is sampled from a logistic with known coefficients on (L, D, E,
    P, M, L*D). This is the ground truth that weight-learning should recover.
    """
    rng = np.random.default_rng(seed)
    # True coefficients (the values weight-learning should recover)
    beta = np.array([1.4, 1.6, 0.8, 0.9, 0.7, 1.2])  # L, D, E, P, M, L*D
    L = components_df["L"].to_numpy()
    D = components_df["D"].to_numpy()
    E = components_df["E"].to_numpy()
    P = components_df["P"].to_numpy()
    M = components_df["M"].to_numpy()
    LD = L * D

    Z = beta[0]*L + beta[1]*D + beta[2]*E + beta[3]*P + beta[4]*M + beta[5]*LD
    # Calibrate intercept so overall distress rate ~= base_rate
    intercept = np.log(base_rate / (1 - base_rate)) - Z.mean()
    p = 1.0 / (1.0 + np.exp(-(Z + intercept)))
    y = (rng.random(len(p)) < p).astype(int)
    return pd.Series(y, index=df.index, name="distress")


def compute_components_for_population(df: pd.DataFrame, macro: dict | None = None) -> pd.DataFrame:
    """Compute the L, D, E, P, M components for every row of df."""
    from hffi_core.components import (
        liquidity_fragility, debt_fragility, expense_fragility,
        portfolio_fragility, macro_fragility,
    )
    macro = macro or {"inflation_rate": 0.032, "fed_funds_rate": 0.05, "unemployment_rate": 0.038}

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "L": liquidity_fragility(r["liquid_savings"], r["monthly_essential_expenses"]),
            "D": debt_fragility(r["monthly_debt_payment"], r["monthly_income"],
                                r["total_debt"], r["monthly_income"] * 12),
            "E": expense_fragility(r["monthly_essential_expenses"], r["monthly_total_expenses"]),
            "P": portfolio_fragility(
                annualized_volatility=r["portfolio_volatility"],
                weights=[r["equity_pct"], r["bond_pct"], r["cash_pct"]],
                expected_drawdown=r["expected_drawdown"],
            ),
            "M": macro_fragility(
                inflation_rate=macro["inflation_rate"],
                fed_funds_rate=macro["fed_funds_rate"],
                unemployment_rate=macro["unemployment_rate"],
                rate_sensitivity=r["rate_sensitivity"],
            ),
        })
    return pd.DataFrame(rows, index=df.index)
