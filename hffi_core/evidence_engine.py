"""Evidence and counterfactual layer for HFFI recommendations.

The Evidence Lab is designed for project defense and data-science review. It
keeps the recommendation output explainable by showing:
    - counterfactual HFFI impact for realistic household actions
    - confidence and evidence behind BUY/HOLD/SELL decisions
    - top model features used by the ML suitability layer
    - model-card and security guardrails
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from hffi_core.portfolio_advisor import HoldingAction
from hffi_core.scoring import HouseholdInputs, FragilityResult, compute_household_hffi


def monthly_buying_capacity(inputs: HouseholdInputs) -> float:
    """Cash flow left after expenses and debt service."""
    return max(
        float(inputs.monthly_income or 0)
        - float(inputs.monthly_total_expenses or 0)
        - float(inputs.monthly_debt_payment or 0),
        0.0,
    )


def build_counterfactual_table(
    inputs: HouseholdInputs,
    macro: dict,
    base_result: Optional[FragilityResult] = None,
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """Rank what-if actions by expected HFFI improvement.

    Positive HFFI impact means the action lowers fragility.
    """
    base_result = base_result or compute_household_hffi(inputs, macro, weights=weights)
    base_score = float(base_result.score)
    base_prob = float(base_result.distress_probability)
    scenarios: list[dict] = []
    capacity = monthly_buying_capacity(inputs)
    essential = float(inputs.monthly_essential_expenses or 0)
    debt = float(inputs.total_debt or 0)

    liquidity_amount = max(500.0, min(max(essential, 500.0), max(capacity * 3, essential * 0.25)))
    debt_paydown = min(max(debt * 0.05, 500.0), max(debt, 0.0)) if debt > 0 else 0.0

    def add(label: str, changed: HouseholdInputs, method: str, why: str) -> None:
        new_result = compute_household_hffi(changed, macro, weights=weights)
        scenarios.append({
            "Action": label,
            "Method": method,
            "New HFFI": new_result.score,
            "HFFI improvement": base_score - float(new_result.score),
            "New band": new_result.band,
            "Distress probability change": float(new_result.distress_probability) - base_prob,
            "Why it matters": why,
        })

    add(
        "Add one-month liquidity buffer",
        replace(inputs, liquid_savings=float(inputs.liquid_savings or 0) + liquidity_amount),
        f"Increase Liquid Savings by ${liquidity_amount:,.0f}.",
        "Liquidity directly lowers L and weakens the L x D danger interaction.",
    )

    if debt_paydown > 0:
        debt_ratio = max((debt - debt_paydown) / debt, 0.0)
        add(
            "Pay down high-interest debt",
            replace(
                inputs,
                total_debt=max(debt - debt_paydown, 0.0),
                monthly_debt_payment=float(inputs.monthly_debt_payment or 0) * debt_ratio,
            ),
            f"Reduce total debt by ${debt_paydown:,.0f} and scale the monthly payment.",
            "Debt paydown reduces D and also reduces the interaction penalty when liquidity is low.",
        )

    if float(inputs.monthly_essential_expenses or 0) > 0:
        add(
            "Convert 5% essential spending to flexible spending",
            replace(inputs, monthly_essential_expenses=essential * 0.95),
            "Lower essential expenses while keeping total spending unchanged.",
            "Expense flexibility lowers E because more spending can be cut during a shock.",
        )

    weights_now = _normalize_core_weights(inputs.portfolio_weights)
    if weights_now["equity"] >= 0.10:
        shifted = dict(weights_now)
        shift = min(0.10, shifted["equity"])
        shifted["equity"] -= shift
        shifted["bond"] += shift * 0.70
        shifted["cash"] += shift * 0.30
        add(
            "Shift 10% from equity to bonds/cash",
            replace(
                inputs,
                portfolio_weights=_normalize_core_weights(shifted),
                portfolio_volatility=max(float(inputs.portfolio_volatility) * 0.88, 0.0),
                expected_drawdown=max(float(inputs.expected_drawdown) * 0.85, 0.0),
            ),
            "Move part of the portfolio from equity risk into defensive assets.",
            "This tests whether capital preservation improves P for the current HFFI band.",
        )

    add(
        "Diversify to lower portfolio risk",
        replace(
            inputs,
            portfolio_volatility=max(float(inputs.portfolio_volatility) * 0.80, 0.0),
            expected_drawdown=max(float(inputs.expected_drawdown) * 0.80, 0.0),
        ),
        "Reduce modeled volatility and max drawdown by 20%.",
        "Diversification lowers portfolio fragility without changing household debt or income.",
    )

    if capacity > 0 and weights_now["equity"] < 0.90:
        risk_up = dict(weights_now)
        shift = min(0.05, 0.90 - risk_up["equity"])
        risk_up["equity"] += shift
        risk_up["cash"] = max(risk_up["cash"] - shift, 0.0)
        add(
            "Stress-test adding equity now",
            replace(
                inputs,
                portfolio_weights=_normalize_core_weights(risk_up),
                portfolio_volatility=min(float(inputs.portfolio_volatility) * 1.08, 0.50),
                expected_drawdown=min(float(inputs.expected_drawdown) * 1.08, 0.80),
            ),
            "Increase equity weight by 5% and raise modeled risk.",
            "This is a guardrail test: it shows whether a new buy would raise household fragility.",
        )

    out = pd.DataFrame(scenarios)
    if out.empty:
        return out
    out["Rank"] = out["HFFI improvement"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values(["HFFI improvement", "Action"], ascending=[False, True]).reset_index(drop=True)


def build_decision_evidence_table(
    holding_actions: Iterable[HoldingAction] | None = None,
    ds_signals: Optional[pd.DataFrame] = None,
    hffi: float = 0.0,
) -> pd.DataFrame:
    """Create an auditable evidence table for holding and market decisions."""
    rows: list[dict] = []
    for action in holding_actions or []:
        confidence = _holding_confidence(action, hffi)
        rows.append({
            "Decision": f"Holding {action.ticker}",
            "Recommendation": action.action,
            "Confidence": confidence,
            "Confidence band": _confidence_band(confidence),
            "Primary evidence": _holding_primary_evidence(action, hffi),
            "When": action.suggested_timing,
            "Comment": action.rationale,
        })

    if ds_signals is not None and not ds_signals.empty:
        for _, row in ds_signals.head(8).iterrows():
            confidence = _ds_confidence(row)
            rows.append({
                "Decision": f"Market {row.get('ticker', row.get('market', ''))}",
                "Recommendation": str(row.get("recommendation", "HOLD / WATCH")),
                "Confidence": confidence,
                "Confidence band": _confidence_band(confidence),
                "Primary evidence": (
                    f"DS score {float(row.get('ds_score', 0.0)):.3f}, "
                    f"ML probability {float(row.get('ml_probability', 0.0)):.1%}, "
                    f"allocation gap {float(row.get('allocation_gap', 0.0)):+.1%}."
                ),
                "When": _ds_timing(row, hffi),
                "Comment": str(row.get("comment", "")),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "Decision", "Recommendation", "Confidence", "Confidence band",
            "Primary evidence", "When", "Comment",
        ])
    return out.sort_values("Confidence", ascending=False).reset_index(drop=True)


def build_feature_evidence_table(model, household_features: dict, top_n: int = 10) -> pd.DataFrame:
    """Show the strongest model features with current household values."""
    rows = []
    importance = getattr(model, "feature_importance", pd.DataFrame()).head(top_n)
    for _, row in importance.iterrows():
        feature = str(row["feature"])
        value = household_features.get(feature, np.nan)
        rows.append({
            "Feature": feature,
            "Current value": value,
            "Importance": float(row["importance"]),
            "Interpretation": _feature_interpretation(feature, value),
        })
    return pd.DataFrame(rows)


def build_model_card_table(model) -> pd.DataFrame:
    """Compact model card for academic and security review."""
    metrics = getattr(model, "metrics", {})
    return pd.DataFrame([
        {
            "Section": "Purpose",
            "Evidence": "Rank household-aware investment candidates and explain BUY/HOLD/SELL guidance.",
        },
        {
            "Section": "Model",
            "Evidence": "Random Forest + Gradient Boosting ensemble trained on synthetic household-market panel.",
        },
        {
            "Section": "Validation",
            "Evidence": (
                f"Ensemble AUC {metrics.get('ensemble_auc', 0):.3f}; "
                f"accuracy {metrics.get('ensemble_accuracy', 0):.3f}; "
                f"train rows {metrics.get('training_rows', 0):,.0f}."
            ),
        },
        {
            "Section": "Guardrail",
            "Evidence": "HFFI band, liquidity, debt burden, and allocation gap can override market momentum.",
        },
        {
            "Section": "Limitations",
            "Evidence": "Prototype uses fallback/live market proxies and synthetic labels; not a brokerage or licensed advice engine.",
        },
        {
            "Section": "Security",
            "Evidence": "No trade execution, ticker sanitization, API keys loaded from environment, and fallback mode during provider rate limits.",
        },
    ])


def summarize_strategy_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    """Add readable deltas to the recommendation backtest summary."""
    if backtest is None or backtest.empty or "strategy" not in backtest.columns:
        return pd.DataFrame()
    out = backtest.copy()
    followed = out[out["strategy"].eq("followed_HFFI_recommendation")]
    ignored = out[out["strategy"].eq("ignored_random_alternative")]
    if not followed.empty and not ignored.empty:
        return_delta = float(followed["avg_12m_return"].iloc[0] - ignored["avg_12m_return"].iloc[0])
        drawdown_delta = float(followed["avg_drawdown"].iloc[0] - ignored["avg_drawdown"].iloc[0])
        out["Return edge vs random"] = np.where(
            out["strategy"].eq("followed_HFFI_recommendation"), return_delta, 0.0
        )
        out["Drawdown edge vs random"] = np.where(
            out["strategy"].eq("followed_HFFI_recommendation"), drawdown_delta, 0.0
        )
    return out


def _normalize_core_weights(weights: dict) -> dict:
    raw = {
        "equity": max(float((weights or {}).get("equity", 0.0)), 0.0),
        "bond": max(float((weights or {}).get("bond", 0.0)), 0.0),
        "cash": max(float((weights or {}).get("cash", 0.0)), 0.0),
    }
    total = sum(raw.values())
    if total <= 0:
        return {"equity": 0.0, "bond": 0.0, "cash": 1.0}
    return {k: v / total for k, v in raw.items()}


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.78:
        return "High"
    if confidence >= 0.58:
        return "Medium"
    return "Low"


def _holding_confidence(action: HoldingAction, hffi: float) -> float:
    suitability = 0.0 if action.suitability is None else abs(float(action.suitability))
    market_strength = min(suitability / 0.45, 1.0)
    concentration = min(float(action.allocation_weight_pct) / 25.0, 1.0)
    fragility_certainty = min(abs(float(hffi) - 50.0) / 50.0, 1.0)
    action_boost = 0.12 if action.action in {"BUY", "SELL"} else 0.04
    return float(np.clip(0.38 + 0.24 * market_strength + 0.18 * concentration + 0.08 * fragility_certainty + action_boost, 0.0, 0.95))


def _ds_confidence(row: pd.Series) -> float:
    ds_score = float(row.get("ds_score", 0.0))
    ml_prob = float(row.get("ml_probability", 0.5))
    allocation_gap = abs(float(row.get("allocation_gap", 0.0)))
    threshold_margin = min(abs(ds_score - 0.42), abs(ds_score - 0.68))
    return float(np.clip(
        0.35
        + 0.25 * min(threshold_margin / 0.20, 1.0)
        + 0.22 * min(abs(ml_prob - 0.50) / 0.30, 1.0)
        + 0.18 * min(allocation_gap / 0.20, 1.0),
        0.0,
        0.95,
    ))


def _holding_primary_evidence(action: HoldingAction, hffi: float) -> str:
    suitability = "unavailable" if action.suitability is None else f"{action.suitability:+.3f}"
    return (
        f"HFFI {hffi:.1f}; strategy {action.strategy}; "
        f"position weight {action.allocation_weight_pct:.1f}%; "
        f"target category {action.target_category_pct:.1f}%; "
        f"market suitability {suitability}."
    )


def _ds_timing(row: pd.Series, hffi: float) -> str:
    recommendation = str(row.get("recommendation", ""))
    if recommendation.startswith("BUY") and hffi < 60:
        return "Use staged monthly contribution; re-check after each HFFI update."
    if recommendation.startswith("BUY"):
        return "Only consider after liquidity and debt guardrails improve."
    if recommendation.startswith("AVOID"):
        return "Avoid or reduce during the next rebalance window."
    return "Hold/watch; review in 30 days or after a market-regime change."


def _feature_interpretation(feature: str, value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = np.nan
    if feature in {"HFFI", "debt_service_ratio", "debt_to_income_ratio", "macro_stress_index"}:
        return "Higher value increases caution in the recommendation engine."
    if feature in {"liquidity_buffer_6m", "monthly_buying_capacity"}:
        return "Higher value increases capacity for staged investing."
    if feature in {"market_volatility", "market_drawdown", "expected_drawdown", "portfolio_volatility"}:
        return "Higher risk value lowers suitability for fragile households."
    if feature in {"momentum_score", "market_return", "market_sharpe_proxy"}:
        return "Higher market quality can support BUY/HOLD if HFFI guardrails allow it."
    if np.isfinite(numeric):
        return "Used as a supporting model signal for suitability ranking."
    return "Feature is available in training and filled safely when missing."
