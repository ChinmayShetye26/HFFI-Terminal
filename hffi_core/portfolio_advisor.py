"""Holdings-aware portfolio allocation and trade guidance.

This module turns user-entered Equity/Bond holdings plus household liquid
savings into:
    - invested amount and optional current value
    - actual Equity/Bond/Liquid Savings percentages
    - target allocation comparison from the HFFI recommendation template
    - rule-based buy-more / hold / trim / sell guidance

The outputs are decision-support signals only. The application never places
orders and does not provide tax, legal, or individualized investment advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import re

import numpy as np
import pandas as pd


VALID_CATEGORIES = {"equity", "bond", "cash"}
CASH_ALIASES = {"CASH", "USD", "US DOLLAR", "DOLLAR", "SAVINGS", "CHECKING"}
TICKER_RE = re.compile(r"^[A-Z0-9.^=_-]{1,24}$")


@dataclass
class PortfolioHolding:
    category: str
    ticker: str
    quantity: float
    buy_price: float
    name: str = ""


@dataclass
class HoldingAction:
    category: str
    ticker: str
    name: str
    action: str
    strategy: str
    suggested_timing: str
    rationale: str
    invested_amount: float
    current_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float
    allocation_weight_pct: float
    target_category_pct: float
    suitability: Optional[float] = None


def sanitize_ticker(value: object) -> str:
    """Normalize a ticker/account label and reject unsafe symbols."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if text in CASH_ALIASES:
        return "CASH"
    compact = text.replace(" ", "")
    return compact if TICKER_RE.fullmatch(compact) else ""


def _positive_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return max(out, 0.0)


def target_core_allocation(allocation_template: Dict[str, float]) -> Dict[str, float]:
    """Map the paper's allocation template into Equity/Bond/Cash buckets.

    The recommendation template contains a small alternatives bucket. For the
    Equity/Bond/Cash calculator we normalize only the three core buckets so the
    percentages add to 100%.
    """
    raw = {
        "equity": (
            allocation_template.get("equity", 0.0)
            + allocation_template.get("us_equities", 0.0)
            + allocation_template.get("intl_equities", 0.0)
        ),
        "bond": allocation_template.get("bond", allocation_template.get("short_bonds", 0.0)),
        "cash": allocation_template.get("cash", allocation_template.get("cash_emergency", 0.0)),
    }
    total = sum(raw.values())
    if total <= 0:
        return {"equity": 0.60, "bond": 0.30, "cash": 0.10}
    return {k: float(v / total) for k, v in raw.items()}


def build_holdings_dataframe(
    holdings: Iterable[PortfolioHolding],
    current_prices: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Return one clean row per holding with invested amount and optional P/L.

    Allocation percentages intentionally use invested amount:

        invested_amount = number_of_units * buy_price

    This matches the HFFI portfolio input definition. Current market prices are
    retained only for display and optional P/L context.
    """
    current_prices = current_prices or {}
    rows = []
    for holding in holdings:
        category = str(holding.category or "").strip().lower()
        if category not in VALID_CATEGORIES:
            continue

        ticker = sanitize_ticker(holding.ticker)
        if not ticker:
            continue

        quantity = _positive_float(holding.quantity)
        buy_price = _positive_float(holding.buy_price)
        if ticker == "CASH" and buy_price <= 0:
            buy_price = 1.0

        live_price = _positive_float(current_prices.get(ticker), default=0.0)
        current_price = live_price or buy_price or (1.0 if category == "cash" else 0.0)
        if quantity <= 0 or current_price <= 0:
            continue

        cost_basis = quantity * (buy_price or current_price)
        current_value = quantity * current_price
        unrealized_pl = current_value - cost_basis
        unrealized_pl_pct = unrealized_pl / cost_basis if cost_basis else 0.0
        rows.append({
            "category": category,
            "ticker": ticker,
            "name": holding.name or ticker,
            "quantity": quantity,
            "buy_price": buy_price or current_price,
            "current_price": current_price,
            "invested_amount": cost_basis,
            "current_value": current_value,
            "cost_basis": cost_basis,
            "unrealized_pl": unrealized_pl,
            "unrealized_pl_pct": unrealized_pl_pct,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "category", "ticker", "name", "quantity", "buy_price", "current_price",
            "invested_amount", "current_value", "cost_basis", "unrealized_pl",
            "unrealized_pl_pct", "allocation_weight",
        ])
    total_invested = float(df["invested_amount"].sum())
    df["allocation_weight"] = df["invested_amount"] / total_invested if total_invested > 0 else 0.0
    return df.sort_values(["category", "invested_amount"], ascending=[True, False]).reset_index(drop=True)


def summarize_allocation(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual allocation by Equity/Bond/Liquid Savings.

    Formula:

        Equity % = Equity invested amount / total portfolio value * 100
        Bond % = Bond invested amount / total portfolio value * 100
        Liquid Savings % = Liquid Savings / total portfolio value * 100

    where total portfolio value is Equity invested + Bond invested + Liquid
    Savings.
    """
    if holdings_df.empty:
        return pd.DataFrame({
            "category": ["equity", "bond", "cash"],
            "invested_amount": [0.0, 0.0, 0.0],
            "actual_weight": [0.0, 0.0, 0.0],
        })

    summary = (
        holdings_df.groupby("category", as_index=False)["invested_amount"].sum()
        .set_index("category")
        .reindex(["equity", "bond", "cash"], fill_value=0.0)
        .reset_index()
    )
    total = float(summary["invested_amount"].sum())
    summary["actual_weight"] = summary["invested_amount"] / total if total > 0 else 0.0
    return summary


def allocation_weights_from_holdings(holdings_df: pd.DataFrame) -> Dict[str, float]:
    summary = summarize_allocation(holdings_df)
    return {
        str(row["category"]): float(row["actual_weight"])
        for _, row in summary.iterrows()
    }


def market_score_lookup(market_scores: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Build ticker -> suitability map from market recommender output."""
    if market_scores is None or market_scores.empty:
        return {}
    lookup: Dict[str, float] = {}
    for _, row in market_scores.iterrows():
        for col in ("ticker", "market"):
            ticker = sanitize_ticker(row.get(col))
            if ticker and pd.notna(row.get("suitability_score")):
                lookup[ticker] = float(row["suitability_score"])
    return lookup


def _strategy_for_hffi(hffi: float) -> str:
    if hffi < 30:
        return "Growth Rebalance"
    if hffi < 60:
        return "Balanced Resilience"
    if hffi < 80:
        return "Capital Preservation"
    return "Liquidity Defense"


def _market_phrase(suitability: Optional[float]) -> str:
    if suitability is None:
        return "live market score is unavailable, so the rule leans on HFFI and allocation gap"
    if suitability >= 0.35:
        return f"live market score is strong ({suitability:+.3f})"
    if suitability >= 0.05:
        return f"live market score is acceptable ({suitability:+.3f})"
    if suitability >= -0.10:
        return f"live market score is neutral ({suitability:+.3f})"
    return f"live market score is weak ({suitability:+.3f})"


def _gap_phrase(category_gap: float) -> str:
    direction = "under target" if category_gap > 0 else "over target"
    return f"{abs(category_gap):.1%} {direction}"


def recommend_holding_actions(
    holdings_df: pd.DataFrame,
    target_weights: Dict[str, float],
    market_scores: Optional[pd.DataFrame] = None,
    hffi: float = 0.0,
    rebalance_band: float = 0.03,
    buying_capacity: float = 0.0,
) -> List[HoldingAction]:
    """Recommend simple BUY / SELL / HOLD actions for current holdings."""
    if holdings_df.empty:
        return []

    summary = summarize_allocation(holdings_df)
    actual_weights = {
        str(row["category"]): float(row["actual_weight"])
        for _, row in summary.iterrows()
    }
    total_value = float(holdings_df["invested_amount"].sum())
    score_by_ticker = market_score_lookup(market_scores)
    strategy = _strategy_for_hffi(hffi)
    actions: List[HoldingAction] = []

    for _, row in holdings_df.iterrows():
        category = str(row["category"])
        ticker = str(row["ticker"])
        current_weight = float(row["invested_amount"] / total_value) if total_value else 0.0
        actual_category_weight = actual_weights.get(category, 0.0)
        target_category_weight = float(target_weights.get(category, 0.0))
        category_gap = target_category_weight - actual_category_weight
        suitability = score_by_ticker.get(ticker)
        category_band = (
            0.10 if category == "bond" and 30 <= hffi < 60 else
            0.15 if category == "bond" and hffi >= 60 else
            rebalance_band
        )
        market_text = _market_phrase(suitability)
        gap_text = _gap_phrase(category_gap)
        rationale_parts = [
            f"Strategy: {strategy}.",
            f"{category.title()} allocation is {actual_category_weight:.1%} versus target {target_category_weight:.1%} ({gap_text}).",
            f"The {market_text}.",
        ]

        if category == "cash":
            if category_gap > rebalance_band:
                action = "BUY"
                timing = "Now: route new surplus to Liquid Savings before increasing market exposure."
                rationale_parts.append("Liquidity is below the HFFI target buffer, so resilience comes before new risk.")
            elif category_gap < -0.05:
                action = "HOLD"
                timing = "Hold cash while reviewing Equity/Bond gaps; deploy only through staged monthly contributions."
                rationale_parts.append("Liquid Savings is above target, but it still supports household shock absorption.")
            else:
                action = "HOLD"
                timing = "No action now; review after the next income, expense, or HFFI change."
                rationale_parts.append("Liquid Savings is close to the HFFI target buffer.")
        else:
            fragile_equity = category == "equity" and hffi >= 60
            if category_gap > category_band:
                if fragile_equity:
                    action = "HOLD"
                    timing = "Do not add equity while HFFI is High/Severe; review after HFFI falls below 60."
                    rationale_parts.append("Equity is underweight, but household fragility is too high for added equity risk.")
                elif suitability is not None and suitability < -0.05:
                    action = "HOLD"
                    timing = "Wait 30 days or until the live score improves before adding."
                    rationale_parts.append("The category is underweight, but the market signal is not strong enough for a fresh buy.")
                elif buying_capacity <= 0:
                    action = "HOLD"
                    timing = "Hold because monthly buying capacity is not positive; revisit after cash flow improves."
                    rationale_parts.append("The allocation gap supports buying, but the household has no available monthly surplus.")
                else:
                    action = "BUY"
                    monthly_amount = max(buying_capacity, 0.0)
                    timing = f"Buy gradually using up to ${monthly_amount:,.0f}/month, split into 4 weekly tranches, then re-check HFFI."
                    rationale_parts.append("The category is below target and the household has positive buying capacity.")
            elif category_gap < -category_band:
                if category == "bond" and hffi >= 30 and actual_category_weight <= target_category_weight + 0.15:
                    action = "HOLD"
                    timing = "Hold; broad bonds are the defensive sleeve for Moderate-or-higher HFFI unless overweight becomes extreme."
                    rationale_parts.append("Bond exposure is above target but still within the resilience tolerance band.")
                elif suitability is not None and suitability > 0.25:
                    action = "HOLD"
                    timing = "Hold for now; trim only if the category remains materially overweight after 30 days."
                    rationale_parts.append("The category is overweight, but this holding still has a strong live market score.")
                else:
                    action = "SELL"
                    timing = "Sell 25% of the excess allocation weekly until the target weight is reached."
                    rationale_parts.append("The category is above the HFFI target allocation.")
            else:
                if suitability is not None and suitability < -0.20:
                    action = "SELL"
                    timing = "Reduce within 1-2 weeks if the weak recommendation score persists."
                    rationale_parts.append("Allocation is near target, but this holding has weak suitability.")
                elif suitability is not None and suitability > 0.35:
                    action = "HOLD"
                    timing = "Hold now; add only on planned contribution dates if the category becomes underweight."
                    rationale_parts.append("Allocation is near target and the holding scores well.")
                else:
                    action = "HOLD"
                    timing = "No trade now; re-check in 30 days or after an HFFI band change."
                    rationale_parts.append("Allocation is close to target.")

            concentration_limit = 0.10 if hffi >= 60 else 0.15
            if category == "bond":
                concentration_limit = 0.60 if hffi >= 30 else 0.45
            if current_weight > concentration_limit and action in {"BUY", "HOLD"}:
                action = "SELL"
                timing = "Reduce concentration gradually over 1-4 weeks until position weight is inside the limit."
                rationale_parts.append(
                    f"Position weight is {current_weight:.1%}, above the {concentration_limit:.0%} concentration limit for this risk band."
                )

        actions.append(HoldingAction(
            category=category,
            ticker=ticker,
            name=str(row.get("name", ticker)),
            action=action,
            strategy=strategy,
            suggested_timing=timing,
            rationale=" ".join(rationale_parts),
            invested_amount=float(row["invested_amount"]),
            current_value=float(row["current_value"]),
            cost_basis=float(row["cost_basis"]),
            unrealized_pl=float(row["unrealized_pl"]),
            unrealized_pl_pct=float(row["unrealized_pl_pct"]),
            allocation_weight_pct=current_weight * 100.0,
            target_category_pct=target_category_weight * 100.0,
            suitability=suitability,
        ))

    priority = {
        "SELL": 0,
        "BUY": 1,
        "HOLD": 2,
    }
    return sorted(actions, key=lambda a: (priority.get(a.action, 9), -a.invested_amount))
