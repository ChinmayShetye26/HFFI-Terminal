"""
HFFI Terminal — Expanded Streamlit dashboard.

Run with:
    streamlit run app/streamlit_app.py

Tabs:
    1. Score          — HFFI gauge + breakdown + waterfall
    2. Stress test    — scenarios + Monte Carlo
    3. Recommendations — rule-based actions + portfolio + trade signals
    4. Markets        — categorized live tickers (red/green) + live charts
    5. Investment Plan — Monte Carlo wealth projection
    6. Macro          — live FRED indicators
    7. News           — live headlines
    8. Validation     — academic validation outputs
    9. Report         — Excel export
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

from dotenv import load_dotenv
load_dotenv()

from hffi_core.scoring import HouseholdInputs, compute_household_hffi, DEFAULT_WEIGHTS
from hffi_core.stress import apply_shock_scenarios, monte_carlo_stress
from hffi_core.recommendations import generate_recommendations
from hffi_core.market_recommender import (
    score_portfolios, score_markets_for_household, generate_trade_signals,
    allowed_portfolios, select_one_market_recommendation, demo_hffi_42_commodities_case,
)
from hffi_core.investment_plan import build_investment_plan, compare_portfolios, PORTFOLIO_PARAMS
from hffi_core.report_generator import generate_report
from hffi_core.database import init_db, save_household_run, save_recommendations
from hffi_core.ds_recommender import (
    assign_household_segment,
    build_model_performance_summary,
    engineer_household_features,
    score_market_recommendations,
    train_suitability_model,
)
from hffi_core.evidence_engine import (
    build_counterfactual_table,
    build_decision_evidence_table,
    build_feature_evidence_table,
    build_model_card_table,
    summarize_strategy_backtest,
)
from hffi_core.portfolio_advisor import (
    PortfolioHolding,
    allocation_weights_from_holdings,
    build_holdings_dataframe,
    recommend_holding_actions,
    sanitize_ticker,
    summarize_allocation,
    target_core_allocation,
)
from hffi_core.validation_runner import run_walk_forward_and_benchmark, backtest_recommendations, make_synthetic_market_panel

st.set_page_config(page_title="HFFI Terminal", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------- #
# Bloomberg-style terminal skin
# --------------------------------------------------------------------------- #
def apply_terminal_theme(mode: str = "Dark"):
    dark = mode == "Dark"
    bg = "#05070A" if dark else "#F7F9FC"
    panel = "#101820" if dark else "#FFFFFF"
    text = "#E8F0F2" if dark else "#111827"
    accent = "#FFB000" if dark else "#0057D9"
    border = "#23313D" if dark else "#D0D7E2"
    st.markdown(f"""
    <style>
    .stApp {{ background: {bg}; color: {text}; }}
    [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
    h1, h2, h3, .stMetric label {{ color: {accent} !important; letter-spacing: .03em; }}
    div[data-testid="stMetric"] {{ background: {panel}; border: 1px solid {border}; border-radius: 10px; padding: 12px; }}
    .terminal-card {{ background:{panel}; border:1px solid {border}; border-left:4px solid {accent}; border-radius:10px; padding:14px; margin:8px 0; }}
    .ticker-tape {{ font-family: 'Courier New', monospace; color:{accent}; background:{panel}; border:1px solid {border}; border-radius:8px; padding:8px; white-space:nowrap; overflow:hidden; }}
    .stButton>button {{ border:1px solid {accent}; color:{accent}; background:transparent; }}
    </style>
    """, unsafe_allow_html=True)



# --------------------------------------------------------------------------- #
# Cached data fetchers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600)
def get_macro():
    try:
        from data.macro_fetcher import fetch_macro_snapshot
        return fetch_macro_snapshot()
    except Exception as e:
        st.sidebar.warning(f"Live macro unavailable ({type(e).__name__}). Using fallback.")
        return {
            "inflation_rate": 0.032, "fed_funds_rate": 0.05, "unemployment_rate": 0.038,
            "mortgage_rate": 0.072, "treasury_10y": 0.044, "treasury_2y": 0.048,
            "yield_curve_spread": -0.004, "vix": 14.5, "real_gdp_growth": 0.024,
            "timestamp": "fallback",
        }


@st.cache_data(ttl=300)
def get_categorized_market(category: str):
    try:
        from data.asset_universe import get_assets_by_category
        from data.market_fetcher import fetch_market_snapshot
        assets = get_assets_by_category(category)
        if not assets:
            return pd.DataFrame()
        symbols = [a.fetch_symbol() for a in assets]
        df = fetch_market_snapshot(symbols)
        df = df.drop(columns=["name", "subcategory", "ticker_display"], errors="ignore")
        meta = pd.DataFrame([
            {"fetch_symbol": a.fetch_symbol(), "name": a.name,
             "subcategory": a.subcategory, "ticker_display": a.ticker}
            for a in assets
        ]).set_index("fetch_symbol")
        df = df.join(meta, how="left")
        return df
    except Exception as e:
        st.warning(f"Live data for {category} unavailable: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_chart(ticker: str, period: str, interval: str):
    try:
        from data.chart_data import fetch_history
        return fetch_history(ticker, period=period, interval=interval)
    except Exception as e:
        st.warning(f"Chart fetch failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_news():
    try:
        from data.news_fetcher import fetch_market_news
        return fetch_market_news()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_market_universe_snapshot():
    """Build the market_features DataFrame the recommender needs."""
    tickers = ["SPY", "AGG", "QQQ", "VYM", "DBC", "GLD", "TLT", "IEF", "VEA", "VWO"]
    names = {"SPY": "S&P 500", "AGG": "Bonds", "QQQ": "Tech", "VYM": "Dividend",
             "DBC": "Commodities", "GLD": "Gold", "TLT": "Long Treasuries",
             "IEF": "Mid Treasuries", "VEA": "Intl Developed", "VWO": "Emerging Markets"}
    try:
        from data.market_fetcher import (
            _extract_yfinance_errors,
            _is_yfinance_in_cooldown,
            _looks_rate_limited,
            _mark_yfinance_rate_limited,
            _quiet_yfinance,
        )
        if _is_yfinance_in_cooldown():
            return _fallback_market_universe_snapshot(tickers, names)
        import yfinance as yf
        with _quiet_yfinance() as yf_output:
            downloaded = yf.download(
                tickers, period="2y", interval="1mo",
                auto_adjust=True, progress=False, threads=False,
            )
        diagnostic_text = f"{yf_output.getvalue()}\n{_extract_yfinance_errors(yf)}"
        if _looks_rate_limited(diagnostic_text):
            _mark_yfinance_rate_limited()
            return _fallback_market_universe_snapshot(tickers, names)
        if isinstance(downloaded.columns, pd.MultiIndex):
            data = downloaded["Close"] if "Close" in downloaded.columns.get_level_values(0) else pd.DataFrame()
        else:
            close = downloaded.get("Close", pd.Series(dtype=float))
            data = close.to_frame(tickers[0]) if isinstance(close, pd.Series) else close
        if data.empty:
            return _fallback_market_universe_snapshot(tickers, names)
        rets = data.pct_change()
        rows = []
        for col in data.columns:
            if data[col].dropna().empty:
                continue
            rolling_vol = rets[col].rolling(12).std().iloc[-1]
            running_max = data[col].cummax()
            dd = (data[col] / running_max - 1).rolling(12).min().iloc[-1]
            sharpe = rets[col].iloc[-1] / rolling_vol if rolling_vol else 0
            mom = 0.6 * rets[col].iloc[-1] + 0.4 * rets[col].iloc[-3:].mean()
            saf = -0.6 * rets[col].iloc[-3:].std() - 0.4 * dd
            rows.append({
                "market": names.get(col, col), "ticker": col,
                "category": _category_for(col), "name": names.get(col, col),
                "market_return": rets[col].iloc[-1],
                "market_volatility": rolling_vol,
                "market_drawdown": dd,
                "market_sharpe_proxy": sharpe,
                "momentum_score": mom, "safety_score": saf,
            })
        if not rows:
            return _fallback_market_universe_snapshot(tickers, names)
        return pd.DataFrame(rows)
    except Exception as e:
        try:
            from data.market_fetcher import _looks_rate_limited, _mark_yfinance_rate_limited
            if _looks_rate_limited(e):
                _mark_yfinance_rate_limited()
        except Exception:
            pass
        return _fallback_market_universe_snapshot(tickers, names)


def _fallback_market_universe_snapshot(tickers: list[str], names: dict[str, str]) -> pd.DataFrame:
    """Small fallback feature table for recommendation continuity."""
    try:
        from data.market_fetcher import fetch_market_snapshot
        snap = fetch_market_snapshot(tickers)
    except Exception:
        snap = pd.DataFrame()
    rows = []
    for ticker in tickers:
        row = snap.loc[ticker] if not snap.empty and ticker in snap.index else {}
        day_ret = float(row.get("change_pct", 0.0)) if hasattr(row, "get") and pd.notna(row.get("change_pct", 0.0)) else 0.0
        category = _category_for(ticker)
        base_vol = {"bond": 0.035, "commodity": 0.060, "index": 0.055, "equity": 0.075}.get(category, 0.06)
        drawdown = -min(0.35, abs(day_ret) * 8 + base_vol)
        rows.append({
            "market": names.get(ticker, ticker),
            "ticker": ticker,
            "category": category,
            "name": names.get(ticker, ticker),
            "market_return": day_ret,
            "market_volatility": base_vol,
            "market_drawdown": drawdown,
            "market_sharpe_proxy": day_ret / base_vol if base_vol else 0.0,
            "momentum_score": day_ret,
            "safety_score": -0.6 * base_vol - 0.4 * drawdown,
            "data_source": "fallback",
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def get_holding_market_snapshot(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Build live/fallback market features for the user's own holdings."""
    safe_symbols = tuple(
        dict.fromkeys(s for s in (sanitize_ticker(sym) for sym in symbols) if s and s != "CASH")
    )
    rows = []
    for ticker in safe_symbols:
        try:
            from data.chart_data import fetch_history
            hist = fetch_history(ticker, period="1y", interval="1mo")
            closes = hist["Close"].dropna() if not hist.empty and "Close" in hist.columns else pd.Series(dtype=float)
            rets = closes.pct_change().dropna()
            if closes.empty or rets.empty:
                raise ValueError("no history")
            rolling_vol = float(rets.tail(12).std()) if len(rets) > 1 else 0.05
            running_max = closes.cummax()
            dd = float((closes / running_max - 1).tail(12).min())
            latest_ret = float(rets.iloc[-1])
            sharpe = latest_ret / rolling_vol if rolling_vol else 0.0
            mom = float(0.6 * latest_ret + 0.4 * rets.tail(3).mean())
            safety = float(-0.6 * rets.tail(3).std() - 0.4 * dd) if len(rets) > 2 else -0.4 * dd
        except Exception:
            fallback = _fallback_market_universe_snapshot([ticker], {ticker: ticker})
            if fallback.empty:
                continue
            row = fallback.iloc[0]
            rolling_vol = float(row["market_volatility"])
            dd = float(row["market_drawdown"])
            latest_ret = float(row["market_return"])
            sharpe = float(row["market_sharpe_proxy"])
            mom = float(row["momentum_score"])
            safety = float(row["safety_score"])

        rows.append({
            "market": ticker,
            "ticker": ticker,
            "category": _category_for(ticker),
            "name": ticker,
            "market_return": latest_ret,
            "market_volatility": rolling_vol,
            "market_drawdown": dd,
            "market_sharpe_proxy": sharpe,
            "momentum_score": mom,
            "safety_score": safety,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def get_ds_model():
    return train_suitability_model()


@st.cache_data(ttl=3600)
def get_strategy_backtest_summary():
    panel, _ = make_synthetic_market_panel()
    return summarize_strategy_backtest(backtest_recommendations(panel))


@st.cache_data(ttl=300)
def get_holding_price_map(symbols: tuple[str, ...]) -> dict:
    """Fetch current prices for user-entered holdings with safe fallbacks."""
    safe_symbols = tuple(
        dict.fromkeys(
            s for s in (sanitize_ticker(sym) for sym in symbols)
            if s and s != "CASH"
        )
    )
    if not safe_symbols:
        return {"CASH": 1.0}
    try:
        from data.market_fetcher import fetch_market_snapshot
        df = fetch_market_snapshot(list(safe_symbols))
        prices = {
            str(idx).upper(): float(row["price"])
            for idx, row in df.iterrows()
            if pd.notna(row.get("price"))
        }
        prices["CASH"] = 1.0
        return prices
    except Exception:
        return {"CASH": 1.0}


def _category_for(ticker: str) -> str:
    if ticker in {"AGG", "TLT", "IEF", "BND", "SHY", "LQD", "HYG", "TIP", "MUB", "EMB"}:
        return "bond"
    if ticker in {"GLD", "SLV", "USO", "UNG", "DBA", "CORN", "WEAT", "DBC", "CPER"}:
        return "commodity"
    if ticker in {"SPY", "QQQ", "VYM", "DIA", "IWM", "VEA", "VWO"}:
        return "index"
    return "equity"


@st.cache_data(ttl=3600)
def get_holding_asset_options(category: str) -> tuple[list[str], dict[str, str]]:
    from data.asset_universe import get_assets_by_category
    assets = get_assets_by_category(category)
    options = [""] + [a.ticker for a in assets]
    names = {a.ticker: a.name for a in assets}
    return options, names


def _render_holding_rows(category: str, title: str, max_rows: int = 30) -> list[PortfolioHolding]:
    """Render numbered holdings rows with per-row duplicate prevention."""
    options, name_map = get_holding_asset_options(category)
    ticker_options = [t for t in options if t]
    holdings: list[PortfolioHolding] = []

    st.markdown(f"**{title}**")
    h_no, h_ticker, h_name, h_units, h_price = st.columns([0.35, 1.0, 1.55, 0.95, 0.95])
    h_no.caption("#")
    h_ticker.caption("Ticker")
    h_name.caption("Name")
    h_units.caption("Units")
    h_price.caption("Buy price")

    selected_so_far: list[str] = []
    for row_no in range(1, max_rows + 1):
        ticker_key = f"{category}_ticker_{row_no}"
        units_key = f"{category}_units_{row_no}"
        price_key = f"{category}_price_{row_no}"

        current = sanitize_ticker(st.session_state.get(ticker_key, ""))
        if current and current in selected_so_far:
            st.session_state[ticker_key] = ""
            current = ""

        row_options = [""] + [t for t in ticker_options if t not in selected_so_far or t == current]
        index = row_options.index(current) if current in row_options else 0

        c_no, c_ticker, c_name, c_units, c_price = st.columns([0.35, 1.0, 1.55, 0.95, 0.95])
        c_no.write(row_no)
        ticker = c_ticker.selectbox(
            f"{title} row {row_no} ticker",
            row_options,
            index=index,
            key=ticker_key,
            label_visibility="collapsed",
        )
        ticker = sanitize_ticker(ticker)
        company_name = name_map.get(ticker, "")
        c_name.text_input(
            f"{title} row {row_no} name",
            value=company_name,
            key=f"{category}_name_{row_no}_{ticker or 'blank'}",
            disabled=True,
            label_visibility="collapsed",
        )
        quantity = c_units.number_input(
            f"{title} row {row_no} units",
            min_value=0.0,
            step=1.0,
            key=units_key,
            label_visibility="collapsed",
        )
        buy_price = c_price.number_input(
            f"{title} row {row_no} buy price",
            min_value=0.0,
            step=1.0,
            key=price_key,
            label_visibility="collapsed",
        )

        if not ticker:
            continue
        selected_so_far.append(ticker)
        if quantity <= 0 or buy_price <= 0:
            continue

        holdings.append(PortfolioHolding(
            category=category,
            ticker=ticker,
            name=company_name or ticker,
            quantity=float(quantity),
            buy_price=float(buy_price),
        ))
    return holdings


def _monthly_buying_capacity(inputs) -> float:
    return max(
        float(inputs.monthly_income) - float(inputs.monthly_total_expenses) - float(inputs.monthly_debt_payment),
        0.0,
    )


def _market_scoring_inputs(inputs, macro):
    debt_service_ratio = (inputs.monthly_debt_payment / inputs.monthly_income) if inputs.monthly_income else 0
    liquidity_buffer_6m = (
        inputs.liquid_savings / (6 * inputs.monthly_essential_expenses)
        if inputs.monthly_essential_expenses else 0
    )
    macro_stress = (
        0.4 * macro.get("inflation_rate", 0)
        + 0.4 * macro.get("unemployment_rate", 0)
        + 0.2 * macro.get("fed_funds_rate", 0)
    )
    risk_off = (
        macro.get("unemployment_rate", 0) > 0.045
        or macro.get("inflation_rate", 0) > 0.04
    )
    return debt_service_ratio, liquidity_buffer_6m, macro_stress, risk_off


def _build_evidence_context(result, inputs, macro, portfolio_context=None):
    portfolio_context = portfolio_context or {}
    recs = generate_recommendations(result, inputs)
    target_weights = target_core_allocation(recs["allocation"])
    holdings_df = portfolio_context.get("holdings_df", pd.DataFrame())
    if holdings_df is not None and not holdings_df.empty:
        actual_weights = allocation_weights_from_holdings(holdings_df)
    else:
        actual_weights = {
            "equity": float(inputs.portfolio_weights.get("equity", 0.0)),
            "bond": float(inputs.portfolio_weights.get("bond", 0.0)),
            "cash": float(inputs.portfolio_weights.get("cash", 0.0)),
        }

    household_features = engineer_household_features(inputs, result, macro, holdings_df)
    debt_service_ratio, liquidity_buffer_6m, macro_stress, risk_off = _market_scoring_inputs(inputs, macro)
    market_df = get_market_universe_snapshot()
    scored = pd.DataFrame()
    if not market_df.empty:
        scored = score_markets_for_household(
            hffi=result.score,
            risk_band=result.band,
            debt_service_ratio=debt_service_ratio,
            macro_stress_index=macro_stress,
            liquidity_buffer_6m=liquidity_buffer_6m,
            market_snapshot=market_df,
            risk_off_regime=risk_off,
        )

    ds_model = get_ds_model()
    ds_signals = pd.DataFrame()
    if not scored.empty:
        ds_signals = score_market_recommendations(
            model=ds_model,
            household_features=household_features,
            market_scores=scored,
            target_weights=target_weights,
            actual_weights=actual_weights,
        )

    holding_actions = []
    if portfolio_context.get("use_detailed_holdings") and holdings_df is not None and not holdings_df.empty:
        holding_symbols = tuple(
            holdings_df.loc[
                holdings_df["category"].isin(["equity", "bond"]),
                "ticker",
            ].dropna().astype(str).unique()
        )
        holding_market_df = get_holding_market_snapshot(holding_symbols)
        holding_scored = pd.DataFrame()
        if not holding_market_df.empty:
            holding_scored = score_markets_for_household(
                hffi=result.score,
                risk_band=result.band,
                debt_service_ratio=debt_service_ratio,
                macro_stress_index=macro_stress,
                liquidity_buffer_6m=liquidity_buffer_6m,
                market_snapshot=holding_market_df,
                risk_off_regime=risk_off,
            )
        holding_actions = [
            action for action in recommend_holding_actions(
                holdings_df=holdings_df,
                target_weights=target_weights,
                market_scores=holding_scored if not holding_scored.empty else scored,
                hffi=result.score,
                buying_capacity=_monthly_buying_capacity(inputs),
            )
            if action.category in {"equity", "bond"}
        ]

    return {
        "recs": recs,
        "target_weights": target_weights,
        "actual_weights": actual_weights,
        "holdings_df": holdings_df,
        "household_features": household_features,
        "household_segment": assign_household_segment(household_features),
        "market_scores": scored,
        "ds_model": ds_model,
        "ds_signals": ds_signals,
        "holding_actions": holding_actions,
        "risk_off": risk_off,
    }


def _trade_signal_comment(signal, result, capacity: float) -> str:
    if result.score >= 80:
        posture = "Liquidity Defense: preserve cash first and only consider the safest signals."
    elif result.score >= 60:
        posture = "Capital Preservation: avoid adding equity risk unless the household balance sheet improves."
    elif result.score >= 30:
        posture = "Balanced Resilience: use staged buys and keep enough liquid savings intact."
    else:
        posture = "Growth Rebalance: the household can accept more growth exposure when market quality is strong."

    if capacity <= 0:
        capacity_note = "Monthly buying capacity is $0, so treat this as a watchlist item rather than a new purchase."
    else:
        capacity_note = f"Estimated monthly buying capacity is ${capacity:,.0f}; size any new position within that limit."

    if signal.action == "BUY":
        signal_note = f"{signal.ticker} is a buy candidate because suitability is {signal.suitability:+.3f} and confidence is {signal.confidence:+.3f}."
    elif signal.action == "HOLD":
        signal_note = f"{signal.ticker} is a hold/watch candidate because the score is positive but not strong enough for aggressive buying."
    else:
        signal_note = f"{signal.ticker} is not preferred right now because suitability is below the current HFFI-adjusted threshold."
    return f"{posture} {signal_note} {capacity_note}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_inputs():
    st.sidebar.header("Household profile")
    with st.sidebar.expander("Income & expenses", expanded=True):
        monthly_income = st.number_input("Monthly income ($)", 0.0, 50000.0, 6000.0, step=100.0)
        monthly_total_expenses = st.number_input("Total monthly expenses ($)", 0.0, 50000.0, 4500.0, step=100.0)
        monthly_essential_expenses = st.number_input("Of which essential ($)", 0.0, 50000.0, 3500.0, step=100.0)
    with st.sidebar.expander("Savings & debt", expanded=True):
        liquid_savings = st.number_input("Liquid savings ($)", 0.0, 1_000_000.0, 8000.0, step=500.0)
        total_debt = st.number_input("Total debt ($)", 0.0, 5_000_000.0, 22000.0, step=500.0)
        monthly_debt_payment = st.number_input("Monthly debt payment ($)", 0.0, 50000.0, 800.0, step=50.0)
    portfolio_context = {
        "use_detailed_holdings": False,
        "holdings_df": pd.DataFrame(),
        "allocation_summary": pd.DataFrame(),
        "price_source": "manual percentages",
    }
    with st.sidebar.expander("Portfolio", expanded=False):
        use_detailed_holdings = st.checkbox(
            "Use detailed Equity/Bond/Cash holdings",
            value=False,
            help="Enter ticker, buying price, and shares/units to calculate allocation percentages automatically.",
        )
        if use_detailed_holdings:
            st.caption("Enter up to 30 Equity rows and 30 Bond rows. Liquid Savings comes from the Savings & debt section.")
            holdings = (
                _render_holding_rows("equity", "Equity Holdings", max_rows=30)
                + _render_holding_rows("bond", "Bond Holdings", max_rows=30)
            )
            if liquid_savings > 0:
                holdings.append(PortfolioHolding("cash", "CASH", liquid_savings, 1.0, "Liquid Savings"))
            price_map = get_holding_price_map(tuple(h.ticker for h in holdings))
            holdings_df = build_holdings_dataframe(holdings, price_map)
            allocation_summary = summarize_allocation(holdings_df)
            weights_from_holdings = allocation_weights_from_holdings(holdings_df)
            equity_weight = weights_from_holdings.get("equity", 0.0)
            bond_weight = weights_from_holdings.get("bond", 0.0)
            cash_weight = weights_from_holdings.get("cash", 0.0)
            equity_pct = int(round(equity_weight * 100))
            bond_pct = int(round(bond_weight * 100))
            cash_pct = int(round(cash_weight * 100))
            if holdings_df.empty:
                st.info("Add at least one Equity/Bond holding or enter Liquid Savings to calculate allocation.")
            else:
                st.caption(
                    f"Calculated mix: Equity {equity_pct}% | Bond {bond_pct}% | Liquid Savings {cash_pct}%"
                )
            portfolio_context = {
                "use_detailed_holdings": True,
                "holdings_df": holdings_df,
                "allocation_summary": allocation_summary,
                "price_source": "allocation uses invested amount; live prices are P/L display only",
            }
        else:
            equity_pct = st.slider("Equity %", 0, 100, 60)
            bond_pct = st.slider("Bond %", 0, 100, 30)
            cash_pct = st.slider("Cash %", 0, 100, 10)
            total = equity_pct + bond_pct + cash_pct or 100
            equity_weight = equity_pct / total
            bond_weight = bond_pct / total
            cash_weight = cash_pct / total
        portfolio_volatility = st.slider("Portfolio volatility", 0.0, 0.50, 0.16, step=0.01)
        expected_drawdown = st.slider("Expected max drawdown", 0.0, 0.80, 0.20, step=0.05)
        rate_sensitivity = st.slider("Rate sensitivity (0=renter, 1=ARM)", 0.0, 1.0, 0.5, step=0.1)
    inputs = HouseholdInputs(
        monthly_income=monthly_income,
        monthly_essential_expenses=monthly_essential_expenses,
        monthly_total_expenses=monthly_total_expenses,
        liquid_savings=liquid_savings, total_debt=total_debt,
        monthly_debt_payment=monthly_debt_payment,
        portfolio_weights={"equity": equity_weight, "bond": bond_weight, "cash": cash_weight},
        portfolio_volatility=portfolio_volatility,
        expected_drawdown=expected_drawdown,
        rate_sensitivity=rate_sensitivity,
    )
    return inputs, portfolio_context


# --------------------------------------------------------------------------- #
# Score panel
# --------------------------------------------------------------------------- #
def render_score_panel(result):
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=result.score, number={"font": {"size": 60}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkblue"},
                   "steps": [{"range": [0, 30], "color": "#cdebc5"},
                             {"range": [30, 60], "color": "#fde7a4"},
                             {"range": [60, 80], "color": "#f8c082"},
                             {"range": [80, 100], "color": "#f08080"}]},
            title={"text": f"HFFI — {result.band}"}))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("12-month distress probability", f"{result.distress_probability:.1%}")
    with c2:
        comp = pd.DataFrame({
            "Component": ["Liquidity (L)", "Debt (D)", "Expenses (E)", "Portfolio (P)", "Macro (M)"],
            "Value (0-1)": [result.L, result.D, result.E, result.P, result.M]})
        fig = px.bar(comp, x="Component", y="Value (0-1)",
                     color="Value (0-1)", color_continuous_scale="RdYlGn_r", range_y=[0, 1])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        contrib_df = pd.DataFrame({
            "Component": list(result.contributions.keys()),
            "Points": list(result.contributions.values())
        }).sort_values("Points", ascending=True)
        fig2 = px.bar(contrib_df, x="Points", y="Component", orientation="h",
                      color="Points", color_continuous_scale="RdYlGn_r")
        fig2.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


def render_stress_panel(household, macro, weights):
    st.subheader("Stress simulator")
    scn_df = apply_shock_scenarios(household, macro, weights=weights)
    c1, c2 = st.columns([2, 1])
    with c1:
        chart_df = scn_df.reset_index()
        fig = px.bar(chart_df, x="scenario", y="HFFI", color="band",
                     color_discrete_map={"Stable": "#5DCAA5", "Moderate Fragility": "#EF9F27",
                                          "High Fragility": "#E24B4A", "Severe Fragility": "#791F1F"})
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(scn_df[["HFFI", "band", "distress_prob"]], height=360)
    with st.expander("Monte Carlo (2000 random shocks)"):
        mc = monte_carlo_stress(household, macro, weights=weights, n_sims=2000)
        fig = px.histogram(mc.scores, nbins=40, title="Distribution of post-shock HFFI")
        fig.add_vline(x=mc.p05, line_dash="dash", annotation_text=f"5th: {mc.p05:.1f}")
        fig.add_vline(x=mc.p95, line_dash="dash", annotation_text=f"95th: {mc.p95:.1f}")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mc.mean:.1f}")
        c2.metric("Std", f"{mc.std:.1f}")
        c3.metric("5th-95th", f"{mc.p05:.0f}-{mc.p95:.0f}")
        c4.metric("P(severe)", f"{mc.prob_severe:.1%}")


def render_recommendations_panel(result, inputs, macro, portfolio_context=None):
    st.subheader("Recommendations")
    st.caption("Educational decision-support only. The app does not place trades or replace a licensed advisor.")
    recs = generate_recommendations(result, inputs)
    buying_capacity = _monthly_buying_capacity(inputs)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Priority actions**")
        for r in recs["actions"]:
            with st.container(border=True):
                st.markdown(f"**Priority {r.priority}: {r.action}**")
                st.caption(f"Triggered by: {r.component}")
                st.write(r.detail)
                st.caption(f"Expected impact — {r.expected_impact}")
    with c2:
        st.markdown("**Recommended allocation**")
        alloc_df = pd.DataFrame({"Asset class": list(recs["allocation"].keys()),
                                  "Weight": list(recs["allocation"].values())})
        fig = px.pie(alloc_df, values="Weight", names="Asset class", hole=0.45)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(recs["rationale"])

    target_weights = target_core_allocation(recs["allocation"])
    portfolio_context = portfolio_context or {}
    holdings_df = portfolio_context.get("holdings_df", pd.DataFrame())
    if holdings_df is not None and not holdings_df.empty:
        actual_weights = allocation_weights_from_holdings(holdings_df)
    else:
        actual_weights = {
            "equity": float(inputs.portfolio_weights.get("equity", 0.0)),
            "bond": float(inputs.portfolio_weights.get("bond", 0.0)),
            "cash": float(inputs.portfolio_weights.get("cash", 0.0)),
        }
    household_features = engineer_household_features(inputs, result, macro, holdings_df)
    household_segment = assign_household_segment(household_features)
    st.caption(f"Household data-science segment: **{household_segment}**")
    if portfolio_context.get("use_detailed_holdings") and not holdings_df.empty:
        st.markdown("---")
        st.subheader("Your Equity/Bond/Cash allocation")
        allocation_summary = summarize_allocation(holdings_df)
        allocation_summary["target_weight"] = allocation_summary["category"].map(target_weights).fillna(0.0)
        allocation_summary["gap"] = allocation_summary["target_weight"] - allocation_summary["actual_weight"]
        c_alloc, c_basis = st.columns([1, 2])
        with c_alloc:
            fig_alloc = go.Figure(data=[
                go.Bar(name="Actual", x=allocation_summary["category"], y=allocation_summary["actual_weight"]),
                go.Bar(name="HFFI target", x=allocation_summary["category"], y=allocation_summary["target_weight"]),
            ])
            fig_alloc.update_layout(
                height=280,
                barmode="group",
                yaxis_tickformat=".0%",
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
        with c_basis:
            display_summary = allocation_summary.rename(columns={
                "category": "Category",
                "invested_amount": "Formula value",
                "actual_weight": "Actual %",
                "target_weight": "Target %",
                "gap": "Gap",
            })
            st.dataframe(display_summary.style.format({
                "Formula value": "${:,.2f}",
                "Actual %": "{:.1%}",
                "Target %": "{:.1%}",
                "Gap": "{:+.1%}",
            }), use_container_width=True)

            cost_df = holdings_df.rename(columns={
                "category": "Category", "ticker": "Ticker", "name": "Name",
                "quantity": "Shares/Units", "buy_price": "Buy price",
                "invested_amount": "Invested amount",
                "current_price": "Current price", "current_value": "Current value",
                "cost_basis": "Cost basis", "unrealized_pl": "Unrealized P/L ($)",
                "unrealized_pl_pct": "Unrealized return",
                "allocation_weight": "Portfolio %",
            })
            st.dataframe(cost_df[[
                "Category", "Ticker", "Name", "Shares/Units", "Buy price",
                "Invested amount", "Current price", "Current value", "Unrealized P/L ($)",
                "Unrealized return", "Portfolio %",
            ]].style.format({
                "Shares/Units": "{:,.4f}",
                "Buy price": "${:,.2f}",
                "Invested amount": "${:,.2f}",
                "Current price": "${:,.2f}",
                "Current value": "${:,.2f}",
                "Unrealized P/L ($)": "${:+,.2f}",
                "Unrealized return": "{:+.1%}",
                "Portfolio %": "{:.1%}",
            }), height=260, use_container_width=True)

    st.markdown("---")
    st.subheader("Portfolio choice (HFFI-conditional)")
    port_scored = score_portfolios(result.score)
    cols_to_show = ["portfolio", "exp_return", "volatility", "max_drawdown",
                    "liquidity_score", "suitability_score"]
    available = [c for c in cols_to_show if c in port_scored.columns]
    st.dataframe(port_scored[available].style.format({
        "exp_return": "{:.1%}", "volatility": "{:.1%}", "max_drawdown": "{:.1%}",
        "liquidity_score": "{:.2f}", "suitability_score": "{:.3f}"}))
    top_portfolio = port_scored.iloc[0]["portfolio"]
    st.success(f"Recommended portfolio for HFFI = {result.score:.0f}: **{top_portfolio}**")

    st.markdown("---")
    st.subheader("Specific trade signals")
    st.caption(f"Strategy uses HFFI={result.score:.1f} ({result.band}), live/fallback market suitability, and estimated monthly buying capacity of ${buying_capacity:,.0f}.")
    market_df = get_market_universe_snapshot()
    scored = pd.DataFrame()
    if not market_df.empty:
        debt_service_ratio = (inputs.monthly_debt_payment / inputs.monthly_income) if inputs.monthly_income else 0
        liquidity_buffer_6m = (inputs.liquid_savings / (6 * inputs.monthly_essential_expenses)) if inputs.monthly_essential_expenses else 0
        macro_stress = (0.4 * macro.get("inflation_rate", 0) + 0.4 * macro.get("unemployment_rate", 0)
                        + 0.2 * macro.get("fed_funds_rate", 0))
        risk_off = (macro.get("unemployment_rate", 0) > 0.045 or macro.get("inflation_rate", 0) > 0.04)
        scored = score_markets_for_household(
            hffi=result.score, risk_band=result.band,
            debt_service_ratio=debt_service_ratio,
            macro_stress_index=macro_stress,
            liquidity_buffer_6m=liquidity_buffer_6m,
            market_snapshot=market_df, risk_off_regime=risk_off)

        selected_market = select_one_market_recommendation(scored)
        st.markdown("**One-market household recommendation**")
        st.markdown(f"<div class='terminal-card'><b>{selected_market['market']}</b><br>{selected_market['plain_language_reason']}</div>", unsafe_allow_html=True)
        if st.checkbox("Show deterministic HFFI≈42 test case"):
            st.json(demo_hffi_42_commodities_case())

        ds_model = get_ds_model()
        ds_signals = score_market_recommendations(
            model=ds_model,
            household_features=household_features,
            market_scores=scored,
            target_weights=target_weights,
            actual_weights=actual_weights,
        ).head(12)
        if not ds_signals.empty:
            sig_df = ds_signals.rename(columns={
                "ticker": "Ticker",
                "name": "Name",
                "category": "Category",
                "recommendation": "Recommendation",
                "suggested_monthly_amount": "Suggested monthly $",
                "ml_probability": "ML probability",
                "suitability_score": "HFFI suitability",
                "allocation_gap": "Allocation gap",
                "ds_score": "DS score",
                "segment": "Segment",
                "comment": "Comment",
            })
            st.dataframe(sig_df.style.format({
                "Suggested monthly $": "${:,.0f}",
                "ML probability": "{:.1%}",
                "HFFI suitability": "{:+.3f}",
                "Allocation gap": "{:+.1%}",
                "DS score": "{:.3f}",
            }), height=430, use_container_width=True)
    else:
        st.info("Live market data unavailable — trade signals require yfinance access.")
    if portfolio_context.get("use_detailed_holdings") and not holdings_df.empty:
        st.markdown("---")
        st.subheader("Buy more / sell guidance for your holdings")
        holding_symbols = tuple(
            holdings_df.loc[holdings_df["category"].isin(["equity", "bond"]), "ticker"].dropna().astype(str).unique()
        )
        holding_market_df = get_holding_market_snapshot(holding_symbols)
        holding_scored = pd.DataFrame()
        if not holding_market_df.empty:
            debt_service_ratio = (inputs.monthly_debt_payment / inputs.monthly_income) if inputs.monthly_income else 0
            liquidity_buffer_6m = (inputs.liquid_savings / (6 * inputs.monthly_essential_expenses)) if inputs.monthly_essential_expenses else 0
            macro_stress = (0.4 * macro.get("inflation_rate", 0) + 0.4 * macro.get("unemployment_rate", 0)
                            + 0.2 * macro.get("fed_funds_rate", 0))
            risk_off = (macro.get("unemployment_rate", 0) > 0.045 or macro.get("inflation_rate", 0) > 0.04)
            holding_scored = score_markets_for_household(
                hffi=result.score, risk_band=result.band,
                debt_service_ratio=debt_service_ratio,
                macro_stress_index=macro_stress,
                liquidity_buffer_6m=liquidity_buffer_6m,
                market_snapshot=holding_market_df, risk_off_regime=risk_off)
        st.caption(f"Running holding strategy: HFFI-aware allocation gap + live market momentum/safety + buying capacity (${buying_capacity:,.0f}/month).")
        actions = recommend_holding_actions(
            holdings_df=holdings_df,
            target_weights=target_weights,
            market_scores=holding_scored if not holding_scored.empty else scored,
            hffi=result.score,
            buying_capacity=buying_capacity,
        )
        holding_actions = [a for a in actions if a.category in {"equity", "bond"}]
        if holding_actions:
            action_df = pd.DataFrame([{
                "Status": a.action,
                "Strategy": a.strategy,
                "Ticker": a.ticker,
                "Name": a.name,
                "Category": a.category,
                "Invested amount": a.invested_amount,
                "Portfolio %": a.allocation_weight_pct,
                "Target category %": a.target_category_pct,
                "Suitability": a.suitability,
                "When": a.suggested_timing,
                "Comment": a.rationale,
            } for a in holding_actions])

            def _color_holding_action(val):
                if val == "BUY":
                    return "background-color: #c6efce; color: #0a5c0a"
                if val == "SELL":
                    return "background-color: #ffc7ce; color: #9c0006"
                if val == "HOLD":
                    return "background-color: #ffeb9c; color: #7a5d00"
                return ""

            st.dataframe(action_df.style.map(_color_holding_action, subset=["Status"]).format({
                "Invested amount": "${:,.2f}",
                "Portfolio %": "{:.1f}%",
                "Target category %": "{:.1f}%",
                "Suitability": lambda x: "" if pd.isna(x) else f"{x:+.3f}",
            }), height=420, use_container_width=True)
        else:
            st.info("Add valid Equity or Bond holdings to receive Buy/Sell/Hold status.")
    return top_portfolio, recs


def render_markets_panel():
    st.subheader("Markets — categorized live snapshot")
    category = st.selectbox(
        "Category",
        ["sector", "commodity", "forex", "bond", "index", "equity"],
        format_func=lambda c: {
            "sector": "🏭 Sector ETFs", "commodity": "🛢️ Commodities",
            "forex": "💱 Forex", "bond": "💵 Bonds & Treasuries",
            "index": "📈 Major Indices", "equity": "🏢 Mega-cap Equities",
        }.get(c, c))

    df = get_categorized_market(category)
    if df.empty:
        st.info(f"Live snapshot for {category} not available — check API keys / network.")
    else:
        if "data_source" in df.columns and df["data_source"].astype(str).str.startswith("fallback").any():
            st.caption("Live provider is unavailable or rate-limited. Showing fallback data so the terminal stays usable.")
        df = df.dropna(subset=["change_pct"]).copy()
        df["change_pct_disp"] = (df["change_pct"] * 100).round(2)
        df = df.sort_values("change_pct", ascending=False)
        tile_cols = st.columns(4)
        for i, (sym, row) in enumerate(df.head(20).iterrows()):
            with tile_cols[i % 4]:
                color = "#0a5c0a" if row["change_pct"] >= 0 else "#9c0006"
                bg = "#c6efce" if row["change_pct"] >= 0 else "#ffc7ce"
                arrow = "▲" if row["change_pct"] >= 0 else "▼"
                # Defensive formatting: yfinance can return NaN/None/float values for
                # fields like name, price, and change. Convert everything safely before
                # slicing or formatting so the Streamlit app does not crash.
                ticker_raw = row.get("ticker_display", sym)
                name_raw = row.get("name", "")
                price_raw = row.get("price", 0.0)
                change_raw = row.get("change", 0.0)
                change_pct_raw = row.get("change_pct_disp", 0.0)

                ticker = str(ticker_raw if pd.notna(ticker_raw) else sym)
                name = str(name_raw if pd.notna(name_raw) else "")[:24]
                price = float(price_raw) if pd.notna(price_raw) else 0.0
                change = float(change_raw) if pd.notna(change_raw) else 0.0
                change_pct_disp = float(change_pct_raw) if pd.notna(change_pct_raw) else 0.0

                st.markdown(
                    f"<div style='background:{bg}; padding:10px; border-radius:8px; "
                    f"margin-bottom:8px; border-left:4px solid {color}'>"
                    f"<div style='font-weight:bold; font-size:14px'>{ticker}</div>"
                    f"<div style='font-size:11px; color:#666'>{name}</div>"
                    f"<div style='font-size:18px; font-weight:bold'>${price:.2f}</div>"
                    f"<div style='color:{color}; font-weight:bold'>{arrow} "
                    f"{change_pct_disp:+.2f}% (${change:+.2f})</div>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Live chart")
    components.html(
        """
        <div style="font-family:Courier New,monospace;background:#101820;color:#ffb000;
                    border:1px solid #23313d;border-radius:8px;padding:8px 10px;
                    display:flex;justify-content:space-between;align-items:center;">
          <span>MARKET CHART CONSOLE</span>
          <span id="hffi-clock"></span>
        </div>
        <script>
          const el = document.getElementById("hffi-clock");
          function tick() { el.textContent = new Date().toLocaleTimeString(); }
          tick(); setInterval(tick, 1000);
        </script>
        """,
        height=42,
    )
    from data.asset_universe import get_assets_by_category
    assets = get_assets_by_category(category)
    asset_options = {
        f"{a.ticker} - {a.name}": a.fetch_symbol()
        for a in assets
    }
    default_label = next(iter(asset_options), "SPY - S&P 500 ETF")
    c1, c2, c3, c4 = st.columns([2.4, 1.2, 1.2, 1])
    with c1:
        selected_asset = st.selectbox(
            "Stock / asset name",
            list(asset_options.keys()) or [default_label],
            help="Start typing to search within the selected category.",
        )
    with c2:
        custom_symbol = st.text_input(
            "Search / custom ticker",
            value="",
            help="Optional yfinance symbol: AAPL, GLD, EURUSD=X, ^GSPC, BTC-USD",
        )
    with c3:
        period = st.selectbox("Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
    with c4:
        interval = st.selectbox("Interval",
            ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"], index=4)

    generate_chart = st.button("Generate", type="primary", use_container_width=True)
    selected_symbol = custom_symbol.strip() or asset_options.get(selected_asset, "SPY")
    if generate_chart or "chart_symbol" not in st.session_state:
        st.session_state.chart_symbol = selected_symbol
        st.session_state.chart_period = period
        st.session_state.chart_interval = interval

    ticker_input = st.session_state.get("chart_symbol", selected_symbol)
    period = st.session_state.get("chart_period", period)
    interval = st.session_state.get("chart_interval", interval)
    chart_df = get_chart(ticker_input, period, interval)
    if chart_df.empty:
        st.info("No chart data — check the symbol or try a different period/interval.")
    else:
        if str(chart_df.attrs.get("data_source", "")).startswith("fallback"):
            st.caption("Live chart provider is rate-limited/unavailable. Showing fallback OHLC data.")
        fig = go.Figure(data=[go.Candlestick(
            x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"], name=ticker_input)])
        fig.update_layout(height=500, title=f"{ticker_input} — {period} / {interval}",
            xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        vol_fig = px.bar(chart_df, x=chart_df.index, y="Volume", title="Volume")
        vol_fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(vol_fig, use_container_width=True)


def render_investment_plan_panel(result):
    st.subheader("Investment Plan — Monte Carlo wealth projection")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon = st.slider("Horizon (years)", 1, 30, 10)
    with c2:
        initial = st.number_input("Initial capital ($)", 0.0, 5_000_000.0, 10000.0, step=1000.0)
    with c3:
        monthly = st.number_input("Monthly contribution ($)", 0.0, 50000.0, 500.0, step=50.0)
    with c4:
        contrib_growth = st.slider("Annual contribution growth", 0.0, 0.10, 0.03, step=0.01)

    allowed = allowed_portfolios(result.score)
    portfolio_choice = st.radio(
        f"Portfolio choice (allowed for HFFI={result.score:.0f}: {', '.join(allowed)})",
        allowed, horizontal=True, index=0)

    plan = build_investment_plan(portfolio_choice, horizon, initial, monthly, contrib_growth)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Median (year {horizon})", f"${plan.final_p50:,.0f}")
    c2.metric("5th percentile", f"${plan.final_p5:,.0f}")
    c3.metric("95th percentile", f"${plan.final_p95:,.0f}")
    c4.metric("Mean expected", f"${plan.final_expected:,.0f}")

    sched = plan.yearly_schedule
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sched["year"], y=sched["p95"], fill=None, mode="lines",
                             line=dict(color="lightblue"), name="95th percentile"))
    fig.add_trace(go.Scatter(x=sched["year"], y=sched["p5"], fill="tonexty", mode="lines",
                             line=dict(color="lightblue"), name="5th–95th band",
                             fillcolor="rgba(135,206,250,0.3)"))
    fig.add_trace(go.Scatter(x=sched["year"], y=sched["p50"], mode="lines+markers",
                             line=dict(color="darkblue", width=3), name="Median"))
    fig.add_trace(go.Scatter(x=sched["year"], y=sched["cumulative_contribution"],
                             mode="lines", line=dict(color="gray", dash="dash"),
                             name="Total contributed"))
    fig.update_layout(title=f"{portfolio_choice} — wealth projection over {horizon} years",
                      xaxis_title="Year", yaxis_title="Portfolio value ($)",
                      height=460, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Year-by-year schedule**")
    st.dataframe(sched.style.format({
        "cumulative_contribution": "${:,.0f}", "expected_value": "${:,.0f}",
        "p5": "${:,.0f}", "p50": "${:,.0f}", "p95": "${:,.0f}",
        "growth_above_contributions": "${:,.0f}"}))
    st.info(plan.summary)

    with st.expander("Compare all portfolios at this horizon"):
        comp = compare_portfolios(horizon, initial, monthly, contrib_growth)
        st.dataframe(comp.style.format({
            "expected_return": "{:.1%}", "volatility": "{:.1%}",
            f"median_value_y{horizon}": "${:,.0f}",
            f"5th_pctile_value_y{horizon}": "${:,.0f}",
            f"95th_pctile_value_y{horizon}": "${:,.0f}",
            "expected_value": "${:,.0f}"}))

    return plan, portfolio_choice, initial, monthly, horizon


def render_macro_panel(macro):
    st.subheader("Macro tape")
    cols = st.columns(6)
    cols[0].metric("Inflation (CPI YoY)", f"{macro.get('inflation_rate', np.nan):.2%}")
    cols[1].metric("Fed Funds Rate", f"{macro.get('fed_funds_rate', np.nan):.2%}")
    cols[2].metric("Unemployment", f"{macro.get('unemployment_rate', np.nan):.2%}")
    cols[3].metric("30y Mortgage", f"{macro.get('mortgage_rate', np.nan):.2%}")
    cols[4].metric("10y Treasury", f"{macro.get('treasury_10y', np.nan):.2%}")
    cols[5].metric("VIX", f"{macro.get('vix', np.nan):.1f}")
    spread = macro.get("yield_curve_spread", float("nan"))
    if not np.isnan(spread):
        sign = "INVERTED" if spread < 0 else "normal"
        st.caption(f"10y–2y spread: {spread:.2%} ({sign})")
    st.caption(f"Last fetch: {macro.get('timestamp', 'unknown')}")


def render_news_panel():
    st.subheader("Market news")
    news_df = get_news()
    if news_df.empty:
        st.info("Set NEWSAPI_KEY in `.env` to see live news.")
        return
    for _, row in news_df.head(20).iterrows():
        with st.container(border=True):
            st.markdown(f"**[{row['title']}]({row['url']})**")
            st.caption(f"{row['source']} • {row['published']} • topic: {row['query']}")
            if row.get("description"):
                st.write(row["description"])


def render_evidence_lab_panel(result, inputs, macro, portfolio_context=None):
    st.subheader("Evidence Lab")
    st.caption("Counterfactuals, confidence, model evidence, and guardrails for defending the recommendation system.")
    ctx = _build_evidence_context(result, inputs, macro, portfolio_context)
    counterfactuals = build_counterfactual_table(inputs, macro, base_result=result)
    evidence_df = build_decision_evidence_table(
        holding_actions=ctx["holding_actions"],
        ds_signals=ctx["ds_signals"],
        hffi=result.score,
    )
    feature_df = build_feature_evidence_table(ctx["ds_model"], ctx["household_features"])

    best_action = counterfactuals.iloc[0] if not counterfactuals.empty else None
    avg_conf = float(evidence_df["Confidence"].mean()) if not evidence_df.empty else 0.0
    perf = build_model_performance_summary(ctx["ds_model"])
    metric_lookup = dict(zip(perf["metric"], perf["value"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", ctx["household_segment"])
    c2.metric("Best HFFI improvement", f"{best_action['HFFI improvement']:+.1f}" if best_action is not None else "n/a")
    c3.metric("Avg decision confidence", f"{avg_conf:.0%}" if avg_conf else "n/a")
    c4.metric("Ensemble AUC", f"{metric_lookup.get('Ensemble AUC', 0):.3f}")

    with st.expander("Counterfactual simulator", expanded=True):
        st.caption("This ranks actions by estimated HFFI reduction. Positive improvement means lower household fragility.")
        if counterfactuals.empty:
            st.info("Not enough household inputs to simulate counterfactuals.")
        else:
            chart_df = counterfactuals.sort_values("HFFI improvement", ascending=True)
            fig = px.bar(
                chart_df,
                x="HFFI improvement",
                y="Action",
                orientation="h",
                color="HFFI improvement",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(counterfactuals.style.format({
                "New HFFI": "{:.1f}",
                "HFFI improvement": "{:+.1f}",
                "Distress probability change": "{:+.1%}",
            }), use_container_width=True, height=310)

    with st.expander("Decision audit: BUY / HOLD / SELL evidence", expanded=True):
        if evidence_df.empty:
            st.info("Add detailed holdings or wait for market signals to populate decision evidence.")
        else:
            st.dataframe(evidence_df.style.format({"Confidence": "{:.0%}"}), use_container_width=True, height=420)

    with st.expander("Explainable ML features", expanded=True):
        st.caption("Top model features are shown with the current household value and business interpretation.")
        if feature_df.empty:
            st.info("Model feature evidence is unavailable.")
        else:
            st.dataframe(feature_df.style.format({
                "Current value": lambda x: "" if pd.isna(x) else f"{x:,.3f}",
                "Importance": "{:.4f}",
            }), use_container_width=True, height=360)

    with st.expander("Strategy backtest summary", expanded=False):
        st.caption("Offline validation compares following the HFFI recommendation against a random ignored alternative.")
        backtest = get_strategy_backtest_summary()
        if backtest.empty:
            st.info("Backtest summary is unavailable.")
        else:
            st.dataframe(backtest.style.format({
                "avg_12m_return": "{:.2%}",
                "avg_drawdown": "{:.2%}",
                "p10_drawdown": "{:.2%}",
                "Return edge vs random": "{:+.2%}",
                "Drawdown edge vs random": "{:+.2%}",
            }), use_container_width=True)

    with st.expander("Model card and security guardrails", expanded=False):
        st.dataframe(build_model_card_table(ctx["ds_model"]), use_container_width=True, hide_index=True)


def render_validation_panel():
    st.subheader("Academic validation and defense outputs")
    st.caption("Runs the built-in walk-forward validation, SPY buy-and-hold benchmark, and recommendation backtest on the synthetic+market merged panel.")
    with st.expander("Data Science recommendation model", expanded=True):
        with st.spinner("Loading ML suitability model..."):
            ds_model = get_ds_model()
        perf = build_model_performance_summary(ds_model)
        c1, c2, c3, c4 = st.columns(4)
        metric_lookup = dict(zip(perf["metric"], perf["value"]))
        c1.metric("Ensemble AUC", f"{metric_lookup.get('Ensemble AUC', 0):.3f}")
        c2.metric("Accuracy", f"{metric_lookup.get('Ensemble Accuracy', 0):.3f}")
        c3.metric("Train rows", f"{metric_lookup.get('Training rows', 0):,.0f}")
        c4.metric("Test rows", f"{metric_lookup.get('Test rows', 0):,.0f}")
        st.markdown("**Top model features**")
        st.dataframe(ds_model.feature_importance.head(12).style.format({"importance": "{:.4f}"}), use_container_width=True)
        st.markdown("**Household segments learned from synthetic panel**")
        st.dataframe(ds_model.segment_centers.style.format({
            "debt_service_ratio": "{:.2f}",
            "debt_to_income_ratio": "{:.2f}",
            "liquidity_buffer_6m": "{:.2f}",
            "HFFI": "{:.1f}",
        }), use_container_width=True)
    if st.button("Run validation suite", type="primary"):
        with st.spinner("Running offline validation suite..."):
            panel, _ = make_synthetic_market_panel()
            results = run_walk_forward_and_benchmark()
            backtest = backtest_recommendations(panel)
        st.markdown("**5-fold rolling walk-forward validation — Random Forest**")
        st.dataframe(results["rf"].style.format({"auc":"{:.4f}", "accuracy":"{:.4f}"}))
        st.markdown("**Gradient Boosting robustness check**")
        st.dataframe(results["gb"].style.format({"auc":"{:.4f}", "accuracy":"{:.4f}"}))
        st.markdown("**SPY buy-and-hold benchmark**")
        st.json(results["spy"])
        st.markdown("**Recommendation backtest: following vs ignoring HFFI-conditional recommendations**")
        st.dataframe(backtest.style.format({"avg_12m_return":"{:.2%}", "avg_drawdown":"{:.2%}", "p10_drawdown":"{:.2%}"}))
        st.success("CSV outputs saved under outputs/ and validation folds persisted to SQLite.")
    st.info("Paper wording: 5-fold rolling validation replaces the one-shot 2018+ split; SPY positive-month rate replaces the always-up baseline; the recommendation backtest compares realized 12-month return/drawdown proxies for followed vs ignored recommendations.")


def render_chatbot_panel(user_state):
    st.subheader("Restricted HFFI chatbot")
    st.caption("Ask about your fragility profile, portfolio choices, or finance concepts. "
               "Powered by OpenAI if OPENAI_API_KEY is set; otherwise uses restricted local fallback.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if has_key:
        st.success("✅ OpenAI API connected.")
    else:
        st.warning("⚠️ No OPENAI_API_KEY set — using restricted rule-based fallback.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    if user_msg := st.chat_input("Ask about your fragility, portfolio, or finance concepts…"):
        st.session_state.chat_history.append(ChatMessage(role="user", content=user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = chat(st.session_state.chat_history, user_state, prefer_api=has_key)
            st.markdown(reply)
        save_chat(user_state.get("run_id"), user_msg, reply, not reply.startswith("I can only answer"))
        st.session_state.chat_history.append(ChatMessage(role="assistant", content=reply))

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("Try these starter questions"):
        suggestions = [
            "Why is my HFFI score what it is?",
            "Explain the L×D interaction term.",
            "What does the recommender suggest and why?",
            "How does the Monte Carlo work?",
            "What is the yield curve telling us right now?",
            "Why is debt service ratio important?",
        ]
        for s in suggestions:
            st.code(s, language=None)


def render_report_panel(user_state):
    st.subheader("Generate Excel investment report")
    st.caption("Multi-sheet workbook: executive summary, fragility breakdown, "
               "year-by-year wealth schedule, portfolio comparison, trade signals, "
               "stress tests, recommendations, macro snapshot.")

    if st.button("📥 Generate report", type="primary"):
        with st.spinner("Building Excel report…"):
            output_path = Path("reports") / f"hffi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            from data.asset_universe import to_dict_records
            asset_meta = pd.DataFrame(to_dict_records())
            market_df = get_market_universe_snapshot()
            signals = []
            if not market_df.empty:
                debt_service_ratio = ((user_state["inputs"].monthly_debt_payment / user_state["inputs"].monthly_income)
                                       if user_state["inputs"].monthly_income else 0)
                liquidity_buffer_6m = ((user_state["inputs"].liquid_savings / (6 * user_state["inputs"].monthly_essential_expenses))
                                        if user_state["inputs"].monthly_essential_expenses else 0)
                macro = user_state["macro"]
                macro_stress = (0.4 * macro.get("inflation_rate", 0)
                                + 0.4 * macro.get("unemployment_rate", 0)
                                + 0.2 * macro.get("fed_funds_rate", 0))
                risk_off = (macro.get("unemployment_rate", 0) > 0.045
                            or macro.get("inflation_rate", 0) > 0.04)
                scored = score_markets_for_household(
                    hffi=user_state["fragility_result"].score,
                    risk_band=user_state["fragility_result"].band,
                    debt_service_ratio=debt_service_ratio,
                    macro_stress_index=macro_stress,
                    liquidity_buffer_6m=liquidity_buffer_6m,
                    market_snapshot=market_df, risk_off_regime=risk_off)
                signals = generate_trade_signals(user_state["portfolio_choice"], scored, asset_meta)

            comparison = compare_portfolios(
                user_state["horizon_years"], user_state["initial_capital"],
                user_state["monthly_contribution"], 0.03)
            stress = apply_shock_scenarios(user_state["inputs"], user_state["macro"])
            recs_dict = generate_recommendations(user_state["fragility_result"], user_state["inputs"])

            generate_report(
                output_path=output_path,
                fragility_result=user_state["fragility_result"],
                investment_plan=user_state["investment_plan"],
                macro=user_state["macro"],
                portfolio_choice=user_state["portfolio_choice"],
                initial_capital=user_state["initial_capital"],
                monthly_contribution=user_state["monthly_contribution"],
                horizon_years=user_state["horizon_years"],
                portfolio_comparison=comparison,
                trade_signals=signals,
                stress_scenarios=stress,
                recommendations=recs_dict,
            )

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Excel report",
                data=f, file_name=output_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success(f"Report generated: {output_path}")


def main():
    init_db()
    theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    apply_terminal_theme(theme_mode)
    st.markdown("<div class='ticker-tape'>HFFI TERMINAL │ FRAGILITY │ MACRO │ RECOMMENDATIONS │ VALIDATION │ RISK CONTROL</div>", unsafe_allow_html=True)
    st.title("HFFI Terminal")
    st.caption("Household Financial Fragility Index — Quant Strategy Terminal")

    inputs, portfolio_context = sidebar_inputs()
    macro = get_macro()
    weights = DEFAULT_WEIGHTS
    result = compute_household_hffi(inputs, macro, weights=weights)

    tabs = st.tabs([
        "Score", "Stress", "Recommendations", "Markets",
        "Investment Plan", "Macro", "News", "Evidence Lab", "Validation", "Report",
    ])

    with tabs[0]:
        render_score_panel(result)
    with tabs[1]:
        render_stress_panel(inputs, macro, weights)
    with tabs[2]:
        portfolio_choice, recs_dict = render_recommendations_panel(result, inputs, macro, portfolio_context)
    with tabs[3]:
        render_markets_panel()
    with tabs[4]:
        plan, portfolio_choice, initial, monthly, horizon = render_investment_plan_panel(result)
    with tabs[5]:
        render_macro_panel(macro)
    with tabs[6]:
        render_news_panel()
    with tabs[7]:
        render_evidence_lab_panel(result, inputs, macro, portfolio_context)
    with tabs[8]:
        render_validation_panel()

    run_id = save_household_run(result, inputs, macro)
    user_state = {
        "fragility_result": result, "inputs": inputs, "macro": macro,
        "portfolio_choice": portfolio_choice, "investment_plan": plan,
        "initial_capital": initial, "monthly_contribution": monthly,
        "horizon_years": horizon, "run_id": run_id,
        "portfolio_context": portfolio_context,
    }

    with tabs[9]:
        render_report_panel(user_state)


if __name__ == "__main__":
    main()
