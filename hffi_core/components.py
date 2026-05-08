"""
Component-level fragility formulas.

Each component returns a value in [0, 1]:
    0 = no fragility on this dimension
    1 = maximum fragility on this dimension

References to the original PDF: these formulas refine the ones in section 9 of
the existing draft, with explicit input validation and bounded output.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Liquidity Fragility
# --------------------------------------------------------------------------- #
def liquidity_fragility(
    liquid_savings: float,
    monthly_essential_expenses: float,
    target_months: float = 6.0,
) -> float:
    """Liquidity fragility based on emergency-fund coverage.

    Formula:
        L = 1 - min(liquid_savings / (target_months * essential_expenses), 1)

    A household with 6 months of essential expenses saved → L = 0.
    A household with zero savings → L = 1.
    """
    if monthly_essential_expenses <= 0:
        # No essential expenses recorded — treat as undefined; assume safe.
        return 0.0
    if liquid_savings < 0:
        liquid_savings = 0.0
    buffer_ratio = liquid_savings / (target_months * monthly_essential_expenses)
    return float(1.0 - min(buffer_ratio, 1.0))


# --------------------------------------------------------------------------- #
# 2. Debt Fragility
# --------------------------------------------------------------------------- #
def debt_fragility(
    monthly_debt_payment: float,
    monthly_income: float,
    total_debt: float,
    annual_income: float,
    short_term_weight: float = 0.6,
) -> float:
    """Debt fragility combining short-term cash-flow burden and long-term leverage.

    Formula:
        D = α * (monthly_debt_payment / monthly_income)
          + (1 - α) * (total_debt / annual_income)
        with α = 0.6 by default.
    """
    if monthly_income <= 0 or annual_income <= 0:
        return 1.0
    if total_debt < 0:
        total_debt = 0.0
    if monthly_debt_payment < 0:
        monthly_debt_payment = 0.0

    dsr = monthly_debt_payment / monthly_income
    leverage = total_debt / annual_income
    raw = short_term_weight * dsr + (1 - short_term_weight) * leverage
    # Cap at 1.0 (a household with DSR > 1 is already maximally fragile on this axis)
    return float(min(raw, 1.0))


# --------------------------------------------------------------------------- #
# 3. Expense Fragility
# --------------------------------------------------------------------------- #
def expense_fragility(
    monthly_essential_expenses: float,
    monthly_total_expenses: float,
) -> float:
    """Expense fragility = share of essential (non-discretionary) spending.

    A household where 100% of spending is essential cannot cut back when shocked.
    """
    if monthly_total_expenses <= 0:
        return 0.0
    if monthly_essential_expenses < 0:
        monthly_essential_expenses = 0.0
    return float(min(monthly_essential_expenses / monthly_total_expenses, 1.0))


# --------------------------------------------------------------------------- #
# 4. Portfolio Fragility
# --------------------------------------------------------------------------- #
def _herfindahl(weights: Iterable[float]) -> float:
    """Herfindahl-Hirschman concentration index, normalized to [0, 1].

    HHI ranges from 1/N (equal weights) to 1 (single asset).
    Returns 0 when fully diversified, 1 when fully concentrated.
    """
    w = np.asarray(list(weights), dtype=float)
    if w.size == 0 or w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    hhi = float(np.sum(w**2))
    n = w.size
    if n <= 1:
        return 1.0
    # Normalize so equal-weight portfolio → 0, single-asset → 1
    return (hhi - 1.0 / n) / (1.0 - 1.0 / n)


def portfolio_fragility(
    annualized_volatility: float,
    weights: Iterable[float],
    expected_drawdown: float,
    vol_scale: float = 0.30,
    dd_scale: float = 0.50,
    w_vol: float = 0.4,
    w_conc: float = 0.3,
    w_dd: float = 0.3,
) -> float:
    """Portfolio fragility from volatility, concentration, and drawdown exposure.

    All three inputs are normalized to [0, 1] before combining:
        vol_norm = min(annualized_vol / vol_scale, 1)        # 30% vol → 1
        dd_norm  = min(expected_drawdown / dd_scale, 1)      # 50% DD → 1
        conc_norm = Herfindahl-based concentration

    P = w_vol * vol_norm + w_conc * conc_norm + w_dd * dd_norm
    """
    if annualized_volatility < 0:
        annualized_volatility = 0.0
    if expected_drawdown < 0:
        expected_drawdown = 0.0

    vol_norm = min(annualized_volatility / vol_scale, 1.0)
    dd_norm = min(expected_drawdown / dd_scale, 1.0)
    conc_norm = _herfindahl(weights)

    return float(w_vol * vol_norm + w_conc * conc_norm + w_dd * dd_norm)


# --------------------------------------------------------------------------- #
# 5. Macroeconomic Fragility
# --------------------------------------------------------------------------- #
def macro_fragility(
    inflation_rate: float,
    fed_funds_rate: float,
    unemployment_rate: float,
    *,
    inflation_baseline: float = 0.02,
    inflation_stress: float = 0.06,
    rate_baseline: float = 0.025,
    rate_stress: float = 0.06,
    unemp_baseline: float = 0.04,
    unemp_stress: float = 0.08,
    w_infl: float = 0.4,
    w_rate: float = 0.3,
    w_unemp: float = 0.3,
    rate_sensitivity: float = 1.0,
) -> float:
    """Macro fragility from current inflation, rates, unemployment.

    Each macro input is mapped to [0, 1] using a piecewise-linear ramp from
    a "baseline" (calm) level to a "stress" (high-pressure) level.

    `rate_sensitivity` ∈ [0, 1] scales how exposed THIS household is to rate
    moves (e.g. 1.0 for ARM mortgage holders, 0.0 for renters with no debt).
    """
    def ramp(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    infl_norm = ramp(inflation_rate, inflation_baseline, inflation_stress)
    rate_norm = ramp(fed_funds_rate, rate_baseline, rate_stress) * rate_sensitivity
    unemp_norm = ramp(unemployment_rate, unemp_baseline, unemp_stress)

    return float(w_infl * infl_norm + w_rate * rate_norm + w_unemp * unemp_norm)
