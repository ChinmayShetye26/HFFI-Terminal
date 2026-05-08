"""
Recommendation engine.

Two layers:

1. Rule-based recommendations
   ---------------------------
   Direct, deterministic, fully explainable. Each fragility component above
   a threshold triggers a specific action. Suitable for the IEEE paper as
   a robust baseline, and for end-user trust ("here's exactly why").

2. Content-based filtering (asset matching)
   -----------------------------------------
   Maps the household's fragility profile to an asset-allocation template.
   Each band has a target portfolio (cash / bonds / equities / alts) chosen
   to address the dominant fragility dimension.

Reinforcement learning is documented as a future extension (see docs/) but
is not in the prototype — it requires sequential interaction data the
project doesn't have yet.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from .scoring import FragilityResult


# --------------------------------------------------------------------------- #
# Allocation templates by risk band
# --------------------------------------------------------------------------- #
ALLOCATION_TEMPLATES = {
    "Stable": {
        "cash_emergency": 0.10,
        "short_bonds":    0.20,
        "us_equities":    0.45,
        "intl_equities":  0.15,
        "alternatives":   0.10,
        "rationale": (
            "Low fragility. Standard growth-oriented allocation with a normal "
            "emergency reserve."
        ),
    },
    "Moderate Fragility": {
        "cash_emergency": 0.20,
        "short_bonds":    0.30,
        "us_equities":    0.30,
        "intl_equities":  0.10,
        "alternatives":   0.10,
        "rationale": (
            "Moderate fragility. Build emergency reserves before adding equity "
            "risk; keep duration short so rate moves don't compound stress."
        ),
    },
    "High Fragility": {
        "cash_emergency": 0.35,
        "short_bonds":    0.40,
        "us_equities":    0.15,
        "intl_equities":  0.05,
        "alternatives":   0.05,
        "rationale": (
            "High fragility. Capital preservation comes before growth. Heavy "
            "cash + short-bond tilt; minimal equity until liquidity rebuilds."
        ),
    },
    "Severe Fragility": {
        "cash_emergency": 0.50,
        "short_bonds":    0.40,
        "us_equities":    0.05,
        "intl_equities":  0.00,
        "alternatives":   0.05,
        "rationale": (
            "Severe fragility. The household is one shock away from distress. "
            "Pause new investing, focus on debt reduction and cash buffer."
        ),
    },
}


# --------------------------------------------------------------------------- #
# Rule-based recommendations
# --------------------------------------------------------------------------- #
@dataclass
class Recommendation:
    priority: int            # 1 = highest, 5 = lowest
    component: str           # which fragility dimension triggered this
    action: str              # short imperative
    detail: str              # explanation
    expected_impact: str     # rough estimate of fragility reduction


def _rule_based(result: FragilityResult, inputs) -> List[Recommendation]:
    """Generate rule-based recommendations from a FragilityResult."""
    recs: List[Recommendation] = []

    # --- Liquidity rules -------------------------------------------------- #
    if result.L > 0.5:
        recs.append(Recommendation(
            priority=1,
            component="Liquidity",
            action="Build a 3–6 month emergency fund as priority #1",
            detail=(
                f"Liquid savings cover less than {int((1-result.L)*6)} months of "
                f"essential expenses. Until this gap closes, every other shock "
                f"compounds. Target: at least 3 months in a high-yield savings "
                f"account before adding equity risk."
            ),
            expected_impact="Drops L by ~0.3–0.5; reduces overall HFFI by ~10–15 points."
        ))

    # --- Debt rules ------------------------------------------------------- #
    if result.D > 0.5:
        recs.append(Recommendation(
            priority=1 if result.D > 0.7 else 2,
            component="Debt",
            action="Aggressive debt paydown — start with highest interest rate first",
            detail=(
                "Debt service ratio is high enough that a modest income shock "
                "becomes a default scenario. Prioritize: credit cards → unsecured "
                "loans → car loans → mortgage. If feasible, consolidate or "
                "refinance to lower the monthly payment."
            ),
            expected_impact="Each 5pt drop in DSR reduces D by ~0.10."
        ))

    # --- Interaction term: low liquidity AND high debt is the danger zone - #
    if result.L > 0.5 and result.D > 0.5:
        recs.append(Recommendation(
            priority=1,
            component="Interaction (L×D)",
            action="DANGER ZONE — pause discretionary spending entirely until L or D drops",
            detail=(
                "Low liquidity + high debt is the configuration most strongly "
                "associated with household financial distress. The interaction "
                "term in HFFI flags this. Until at least one of these improves, "
                "treat all non-essential spending as suspended."
            ),
            expected_impact="Eliminating L×D risk drops HFFI by ~5–10 points alone."
        ))

    # --- Expense rigidity ------------------------------------------------- #
    if result.E > 0.85:
        recs.append(Recommendation(
            priority=3,
            component="Expenses",
            action="Find cuttable expenses — almost all spending is currently essential",
            detail=(
                "Over 85% of spending is essential, leaving no buffer to absorb "
                "shocks. Audit: subscription stacking, insurance comparison, "
                "renegotiable contracts (phone, internet, utilities). Even a "
                "5pt drop in essential share buys flexibility."
            ),
            expected_impact="Each 5pt drop reduces E by ~0.05."
        ))

    # --- Portfolio rules -------------------------------------------------- #
    if result.P > 0.5:
        recs.append(Recommendation(
            priority=2,
            component="Portfolio",
            action="Reduce portfolio risk — diversify or de-risk",
            detail=(
                "Portfolio fragility is elevated — likely due to volatility, "
                "concentration, or both. Rebalance toward the allocation "
                f"template for {result.band}: more short bonds, less single-name "
                f"equity exposure."
            ),
            expected_impact="Diversification alone can drop P by 0.15–0.25."
        ))

    # --- Macro rules ------------------------------------------------------ #
    if result.M > 0.5:
        recs.append(Recommendation(
            priority=3,
            component="Macro",
            action="Hedge against the dominant macro stress",
            detail=(
                "Current macro environment (inflation, rates, unemployment) is "
                "elevated relative to baseline. If you have variable-rate debt, "
                "lock fixed rates now. If inflation is the driver, tilt toward "
                "TIPS or short-duration credit."
            ),
            expected_impact="Macro hedges typically trim M by 0.10–0.20."
        ))

    # If the household is fundamentally healthy, give a positive nudge.
    if not recs and result.score < 30:
        recs.append(Recommendation(
            priority=5,
            component="General",
            action="Maintain — the fundamentals look solid",
            detail=(
                "All five fragility dimensions are within healthy bands. Continue "
                "regular contributions, rebalance annually, and re-check this "
                "score after major life events."
            ),
            expected_impact="No urgent action needed."
        ))

    recs.sort(key=lambda r: r.priority)
    return recs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_recommendations(
    result: FragilityResult,
    inputs,
) -> Dict:
    """Top-level function: returns rule-based actions + allocation template.

    Returns
    -------
    dict with keys:
        - 'actions': List[Recommendation]
        - 'allocation': dict of asset class → target weight
        - 'rationale': string explaining the allocation choice
        - 'band': risk band name
    """
    actions = _rule_based(result, inputs)
    allocation = ALLOCATION_TEMPLATES.get(result.band, ALLOCATION_TEMPLATES["Moderate Fragility"]).copy()
    rationale = allocation.pop("rationale")

    return {
        "actions": actions,
        "allocation": allocation,
        "rationale": rationale,
        "band": result.band,
    }
