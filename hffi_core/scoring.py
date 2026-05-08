"""
HFFI composite scoring.

The Household Financial Fragility Index combines five normalized components
plus an interaction term that captures the well-known nonlinearity:
    low liquidity AND high debt is much worse than either alone.

    HFFI = 100 * (w1*L + w2*D + w3*E + w4*P + w5*M + λ*(L*D))

The score is bounded to [0, 100].
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import numpy as np

from .components import (
    liquidity_fragility,
    debt_fragility,
    expense_fragility,
    portfolio_fragility,
    macro_fragility,
)


# --------------------------------------------------------------------------- #
# Default weights (initial baseline; replaced after weight learning)
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {
    "w1_liquidity": 0.25,
    "w2_debt":      0.25,
    "w3_expense":   0.15,
    "w4_portfolio": 0.20,
    "w5_macro":     0.15,
    "lambda_LD":    0.10,
}


@dataclass
class HouseholdInputs:
    """Raw household inputs the user provides through the terminal."""
    monthly_income: float
    monthly_essential_expenses: float
    monthly_total_expenses: float
    liquid_savings: float
    total_debt: float
    monthly_debt_payment: float
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    portfolio_volatility: float = 0.0
    expected_drawdown: float = 0.0
    rate_sensitivity: float = 0.5  # 0 = renter no debt, 1 = ARM mortgage
    # Optional metadata
    age: Optional[int] = None
    dependents: Optional[int] = None
    employment_type: Optional[str] = None


@dataclass
class FragilityResult:
    """Output of HFFI computation, suitable for display + explanation."""
    L: float
    D: float
    E: float
    P: float
    M: float
    interaction_LD: float
    score: float
    band: str
    distress_probability: float
    contributions: Dict[str, float]  # each component's contribution to final score

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def hffi_score(
    L: float, D: float, E: float, P: float, M: float,
    weights: Optional[dict] = None,
) -> float:
    """Compute the composite HFFI score on [0, 100]."""
    w = weights or DEFAULT_WEIGHTS
    raw = (
        w["w1_liquidity"] * L
        + w["w2_debt"]    * D
        + w["w3_expense"] * E
        + w["w4_portfolio"] * P
        + w["w5_macro"]   * M
        + w["lambda_LD"]  * (L * D)
    )
    return float(np.clip(100.0 * raw, 0.0, 100.0))


def risk_band(score: float) -> str:
    """Map HFFI score to qualitative band."""
    if score <= 30:
        return "Stable"
    elif score <= 60:
        return "Moderate Fragility"
    elif score <= 80:
        return "High Fragility"
    return "Severe Fragility"


def distress_probability(score: float) -> float:
    """Map HFFI score (0-100) to a 12-month distress probability via logistic.

    Calibrated so that:
        score = 30 → ~5% probability
        score = 60 → ~30% probability
        score = 80 → ~70% probability
        score = 95 → ~95% probability
    These calibration points are placeholders and should be re-fit on real data.
    """
    # Logistic with slope k and midpoint x0 fitted to the calibration points above.
    k, x0 = 0.10, 65.0
    return float(1.0 / (1.0 + np.exp(-k * (score - x0))))


# --------------------------------------------------------------------------- #
# End-to-end pipeline for one household
# --------------------------------------------------------------------------- #
def compute_household_hffi(
    inputs: HouseholdInputs,
    macro: dict,
    weights: Optional[dict] = None,
) -> FragilityResult:
    """Take raw household inputs + current macro snapshot → FragilityResult.

    `macro` should contain at least:
        inflation_rate, fed_funds_rate, unemployment_rate
    All as decimal fractions (0.06 not 6).
    """
    L = liquidity_fragility(
        inputs.liquid_savings, inputs.monthly_essential_expenses
    )
    D = debt_fragility(
        inputs.monthly_debt_payment, inputs.monthly_income,
        inputs.total_debt, inputs.monthly_income * 12,
    )
    E = expense_fragility(
        inputs.monthly_essential_expenses, inputs.monthly_total_expenses
    )
    dependents = max(int(inputs.dependents or 0), 0)
    if dependents:
        # Dependents increase spending rigidity because fewer expenses can be
        # safely cut during stress; cap the adjustment so the paper's core
        # expense formula remains the primary driver.
        E = float(np.clip(E + min(dependents * 0.025, 0.125), 0.0, 1.0))
    P = portfolio_fragility(
        annualized_volatility=inputs.portfolio_volatility,
        weights=list(inputs.portfolio_weights.values()),
        expected_drawdown=inputs.expected_drawdown,
    )
    M = macro_fragility(
        inflation_rate=macro.get("inflation_rate", 0.02),
        fed_funds_rate=macro.get("fed_funds_rate", 0.03),
        unemployment_rate=macro.get("unemployment_rate", 0.04),
        rate_sensitivity=inputs.rate_sensitivity,
    )
    employment_type = str(inputs.employment_type or "full_time").lower().replace("-", "_").replace(" ", "_")
    employment_risk = {
        "full_time": 0.00,
        "salaried": 0.00,
        "part_time": 0.06,
        "contract": 0.10,
        "self_employed": 0.10,
        "unemployed": 0.22,
        "retired": 0.03,
    }.get(employment_type, 0.03)
    if employment_risk:
        unemp_pressure = float(np.clip(macro.get("unemployment_rate", 0.04) / 0.08, 0.0, 1.0))
        M = float(np.clip(M + employment_risk * unemp_pressure, 0.0, 1.0))

    score = hffi_score(L, D, E, P, M, weights)
    w = weights or DEFAULT_WEIGHTS
    contributions = {
        "Liquidity":  100 * w["w1_liquidity"] * L,
        "Debt":       100 * w["w2_debt"] * D,
        "Expenses":   100 * w["w3_expense"] * E,
        "Portfolio":  100 * w["w4_portfolio"] * P,
        "Macro":      100 * w["w5_macro"] * M,
        "Interaction (L×D)": 100 * w["lambda_LD"] * (L * D),
    }

    return FragilityResult(
        L=L, D=D, E=E, P=P, M=M,
        interaction_LD=L * D,
        score=score,
        band=risk_band(score),
        distress_probability=distress_probability(score),
        contributions=contributions,
    )
