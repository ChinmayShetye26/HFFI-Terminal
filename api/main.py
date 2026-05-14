"""FastAPI service for the React HFFI Investment Terminal.

The API keeps the data-science and recommendation system in Python while the
React app owns the interactive frontend. Endpoints are intentionally narrow and
return JSON-ready objects so they can be consumed by a browser, test client, or
future mobile/desktop shell.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional
import sys
import time

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from data.asset_universe import get_assets_by_category, get_categories, to_dict_records
from data.chart_data import fetch_history
from data.macro_fetcher import fetch_macro_snapshot
from data.market_fetcher import fetch_market_snapshot
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
)
from hffi_core.investment_plan import (
    build_investment_plan as build_long_horizon_investment_plan,
    compare_portfolios,
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
from hffi_core.report_generator import generate_report
from hffi_core.recommendations import generate_recommendations
from hffi_core.scoring import DEFAULT_WEIGHTS, HouseholdInputs, compute_household_hffi
from hffi_core.stress import apply_shock_scenarios, monte_carlo_stress
from hffi_core.market_recommender import (
    generate_trade_signals,
    score_markets_for_household,
    score_portfolios,
    select_one_market_recommendation,
)
from api.security import (
    AuthenticatedUser,
    LoginRequest,
    allowed_origins,
    audit_event,
    authenticate_user,
    create_access_token,
    get_current_user,
    payload_fingerprint,
    public_rate_limit,
    rate_limit,
    require_roles,
    security_config,
    security_headers,
)


class HoldingIn(BaseModel):
    category: Literal["equity", "bond"]
    ticker: str = Field(max_length=24)
    units: float = Field(default=0, ge=0)
    buyPrice: float = Field(default=0, ge=0)
    name: str = ""

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = sanitize_ticker(value)
        if value and not ticker:
            raise ValueError("Ticker contains unsupported characters.")
        return ticker


class PortfolioWeightsIn(BaseModel):
    equity: float = Field(default=0.6, ge=0, le=1)
    bond: float = Field(default=0.3, ge=0, le=1)
    cash: float = Field(default=0.1, ge=0, le=1)


class HouseholdIn(BaseModel):
    monthlyIncome: float = Field(default=6000, ge=0, le=1_000_000)
    monthlyEssentialExpenses: float = Field(default=3500, ge=0, le=1_000_000)
    monthlyTotalExpenses: float = Field(default=4500, ge=0, le=1_000_000)
    liquidSavings: float = Field(default=8000, ge=0, le=1_000_000_000)
    totalDebt: float = Field(default=22000, ge=0, le=1_000_000_000)
    monthlyDebtPayment: float = Field(default=800, ge=0, le=1_000_000)
    portfolioVolatility: float = Field(default=0.16, ge=0, le=0.5)
    expectedDrawdown: float = Field(default=0.2, ge=0, le=0.8)
    rateSensitivity: float = Field(default=0.5, ge=0, le=1)
    employmentType: Literal["full_time", "part_time", "contract", "self_employed", "unemployed", "retired"] = "full_time"
    dependents: int = Field(default=0, ge=0, le=12)
    portfolioWeights: PortfolioWeightsIn = Field(default_factory=PortfolioWeightsIn)

    @model_validator(mode="after")
    def validate_expense_consistency(self) -> "HouseholdIn":
        if self.monthlyEssentialExpenses > self.monthlyTotalExpenses:
            raise ValueError("Essential expenses cannot exceed total expenses.")
        if self.monthlyDebtPayment > self.monthlyIncome * 2 and self.monthlyIncome > 0:
            raise ValueError("Monthly debt payment is outside the supported range for this prototype.")
        return self


class AnalyzeRequest(BaseModel):
    household: HouseholdIn
    holdings: list[HoldingIn] = Field(default_factory=list, max_length=60)

    @model_validator(mode="after")
    def validate_holdings_count(self) -> "AnalyzeRequest":
        counts = {"equity": 0, "bond": 0}
        for holding in self.holdings:
            counts[holding.category] += 1
        if counts["equity"] > 30 or counts["bond"] > 30:
            raise ValueError("Maximum 30 equity holdings and 30 bond holdings are supported.")
        return self


class BacktestRequest(BaseModel):
    household: HouseholdIn
    holdings: list[HoldingIn] = Field(default_factory=list, max_length=60)
    startDate: date = Field(default=date(2021, 1, 1))
    endDate: date = Field(default_factory=date.today)
    initialCapital: float = Field(default=100000, gt=0, le=100_000_000)
    frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    transactionCostPct: float = Field(default=0.001, ge=0, le=0.05)
    benchmarkTicker: str = Field(default="SPY", max_length=24)

    @field_validator("benchmarkTicker")
    @classmethod
    def validate_benchmark(cls, value: str) -> str:
        ticker = sanitize_ticker(value)
        if value and not ticker:
            raise ValueError("Benchmark ticker contains unsupported characters.")
        return ticker or "SPY"

    @model_validator(mode="after")
    def validate_backtest_scope(self) -> "BacktestRequest":
        if self.endDate <= self.startDate:
            raise ValueError("End date must be after start date.")
        if (self.endDate - self.startDate).days > 3650:
            raise ValueError("Backtest date range is limited to 10 years.")
        counts = {"equity": 0, "bond": 0}
        for holding in self.holdings:
            counts[holding.category] += 1
        if counts["equity"] > 30 or counts["bond"] > 30:
            raise ValueError("Maximum 30 equity holdings and 30 bond holdings are supported.")
        return self


class ReportRequest(AnalyzeRequest):
    initialCapital: float = Field(default=100000, gt=0, le=100_000_000)
    monthlyContribution: float = Field(default=500, ge=0, le=1_000_000)
    horizonYears: int = Field(default=10, ge=1, le=30)
    annualContributionGrowth: float = Field(default=0.03, ge=0, le=0.25)


app = FastAPI(
    title="HFFI Terminal API",
    version="1.0.0",
    description="Household Financial Fragility Index recommendation API for the React terminal.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def add_security_headers_and_audit(request: Request, call_next):
    started = time.perf_counter()
    response: Response
    try:
        response = await call_next(request)
    except Exception:
        audit_event(
            "request_error",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        raise
    for key, value in security_headers().items():
        response.headers.setdefault(key, value)
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    audit_event(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=elapsed_ms,
        client=request.client.host if request.client else "unknown",
    )
    return response


@lru_cache(maxsize=1)
def _ds_model():
    return train_suitability_model()


@lru_cache(maxsize=1)
def _asset_records() -> list[dict[str, Any]]:
    return to_dict_records()


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _macro_snapshot() -> dict[str, Any]:
    try:
        return fetch_macro_snapshot()
    except Exception:
        return {
            "inflation_rate": 0.032,
            "fed_funds_rate": 0.05,
            "unemployment_rate": 0.038,
            "mortgage_rate": 0.072,
            "treasury_10y": 0.044,
            "treasury_2y": 0.048,
            "yield_curve_spread": -0.004,
            "vix": 14.5,
            "real_gdp_growth": 0.024,
            "timestamp": "fallback",
        }


def _category_for(ticker: str) -> str:
    lookup = _asset_lookup()
    normalized = sanitize_ticker(ticker)
    if normalized in lookup:
        return str(lookup[normalized].get("category", "equity"))
    if ticker in {"AGG", "TLT", "IEF", "BND", "SHY", "LQD", "HYG", "TIP", "MUB", "EMB"}:
        return "bond"
    if ticker in {"GLD", "SLV", "USO", "UNG", "DBA", "CORN", "WEAT", "DBC", "CPER"}:
        return "commodity"
    if ticker in {"SPY", "QQQ", "VYM", "DIA", "IWM", "VEA", "VWO"}:
        return "index"
    return "equity"


def _asset_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in _asset_records():
        ticker = sanitize_ticker(row.get("ticker"))
        fetch_symbol = sanitize_ticker(row.get("fetch_symbol"))
        if ticker:
            lookup[ticker] = row
        if fetch_symbol:
            lookup[fetch_symbol] = row
    return lookup


@lru_cache(maxsize=1)
def _market_universe_snapshot() -> pd.DataFrame:
    recommendation_tickers = {
        "AAPL", "MSFT", "NVDA",
        "AGG", "TLT", "TIP",
        "XLK", "XLV", "XLU",
        "GLD", "DBC", "SLV",
        "EURUSD", "USDJPY", "DXY",
        "SPX", "IXIC", "VIX",
    }
    assets = [asset for asset in _asset_records() if str(asset.get("ticker", "")).upper() in recommendation_tickers]
    rows = []
    for asset in assets:
        ticker = str(asset.get("ticker", "")).upper()
        fetch_symbol = str(asset.get("fetch_symbol") or ticker).upper()
        hist = fetch_history(fetch_symbol, period="1y", interval="1d")
        if hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"].dropna()
        if close.empty:
            continue
        rets = close.pct_change().dropna()
        latest_ret = _safe_number(rets.iloc[-1] if not rets.empty else 0)
        rolling_vol = _safe_number(rets.tail(60).std() * np.sqrt(252), 0.12)
        running_max = close.cummax()
        dd = _safe_number((close / running_max - 1).tail(126).min(), -0.08)
        sharpe = latest_ret / rolling_vol if rolling_vol else 0.0
        mom = _safe_number(0.6 * latest_ret + 0.4 * rets.tail(20).mean(), 0.0)
        safety = _safe_number(-0.6 * rets.tail(60).std() - 0.4 * abs(dd), -0.03)
        rows.append({
            "market": asset.get("name", ticker),
            "ticker": ticker,
            "fetch_symbol": fetch_symbol,
            "category": asset.get("category", _category_for(ticker)),
            "subcategory": asset.get("subcategory", ""),
            "name": asset.get("name", ticker),
            "current_price": _safe_number(close.iloc[-1]),
            "market_return": latest_ret,
            "market_volatility": rolling_vol,
            "market_drawdown": dd,
            "market_sharpe_proxy": sharpe,
            "momentum_score": mom,
            "safety_score": safety,
            "data_source": hist.attrs.get("data_source", "live:yfinance"),
        })
    return pd.DataFrame(rows)


def _to_household_inputs(payload: HouseholdIn, weights: dict[str, float]) -> HouseholdInputs:
    return HouseholdInputs(
        monthly_income=payload.monthlyIncome,
        monthly_essential_expenses=payload.monthlyEssentialExpenses,
        monthly_total_expenses=payload.monthlyTotalExpenses,
        liquid_savings=payload.liquidSavings,
        total_debt=payload.totalDebt,
        monthly_debt_payment=payload.monthlyDebtPayment,
        portfolio_weights=weights,
        portfolio_volatility=payload.portfolioVolatility,
        expected_drawdown=payload.expectedDrawdown,
        rate_sensitivity=payload.rateSensitivity,
        dependents=payload.dependents,
        employment_type=payload.employmentType,
    )


def _holdings_frame(holdings: list[HoldingIn], liquid_savings: float) -> pd.DataFrame:
    portfolio_holdings = [
        PortfolioHolding(
            category=h.category,
            ticker=h.ticker,
            quantity=float(h.units),
            buy_price=float(h.buyPrice),
            name=h.name or h.ticker,
        )
        for h in holdings
        if h.ticker and h.units > 0 and h.buyPrice > 0
    ]
    if liquid_savings > 0:
        portfolio_holdings.append(
            PortfolioHolding("cash", "CASH", float(liquid_savings), 1.0, "Liquid Savings")
        )
    symbols = [h.ticker for h in portfolio_holdings if h.ticker != "CASH"]
    price_map = {"CASH": 1.0}
    if symbols:
        try:
            snapshot = fetch_market_snapshot(symbols)
            for ticker, row in snapshot.iterrows():
                if pd.notna(row.get("price")):
                    price_map[str(ticker).upper()] = float(row["price"])
        except Exception:
            pass
    return build_holdings_dataframe(portfolio_holdings, price_map)


def _market_scoring_inputs(inputs: HouseholdInputs, macro: dict[str, Any]) -> tuple[float, float, float, bool]:
    debt_service_ratio = inputs.monthly_debt_payment / inputs.monthly_income if inputs.monthly_income else 0
    liquidity_buffer_6m = (
        inputs.liquid_savings / (6 * inputs.monthly_essential_expenses)
        if inputs.monthly_essential_expenses else 0
    )
    macro_stress = (
        0.4 * macro.get("inflation_rate", 0)
        + 0.4 * macro.get("unemployment_rate", 0)
        + 0.2 * macro.get("fed_funds_rate", 0)
    )
    risk_off = macro.get("unemployment_rate", 0) > 0.045 or macro.get("inflation_rate", 0) > 0.04
    return debt_service_ratio, liquidity_buffer_6m, macro_stress, risk_off


def _history_period_for(start: pd.Timestamp, end: pd.Timestamp) -> str:
    today = pd.Timestamp.today().tz_localize(None)
    days = max(int((max(end, today) - start).days), 1)
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    if days <= 1825:
        return "5y"
    if days <= 3650:
        return "10y"
    return "max"


def _price_matrix(symbols: list[str], start: date, end: date) -> tuple[pd.DataFrame, str]:
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    period = _history_period_for(start_ts, end_ts)
    series_by_symbol: dict[str, pd.Series] = {}
    sources: set[str] = set()
    for symbol in dict.fromkeys(sanitize_ticker(symbol) for symbol in symbols):
        if not symbol:
            continue
        hist = fetch_history(symbol, period=period, interval="1d")
        if hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index()
        close = close[(close.index >= start_ts) & (close.index <= end_ts)].dropna()
        if close.empty:
            continue
        series_by_symbol[symbol] = close.astype(float)
        sources.add(str(hist.attrs.get("data_source", "live:yfinance")))
    if not series_by_symbol:
        raise HTTPException(status_code=422, detail="No historical price data available for the selected backtest window.")
    prices = pd.concat(series_by_symbol, axis=1).sort_index().ffill().bfill()
    prices = prices.dropna(how="all")
    if prices.empty or len(prices) < 3:
        raise HTTPException(status_code=422, detail="Not enough historical price rows for backtesting.")
    return prices, ", ".join(sorted(sources)) or "unknown"


def _fallback_macro_for_date(ts: pd.Timestamp) -> dict[str, Any]:
    macro = {
        "inflation_rate": 0.025,
        "fed_funds_rate": 0.025,
        "unemployment_rate": 0.042,
        "mortgage_rate": 0.045,
        "treasury_10y": 0.032,
        "treasury_2y": 0.028,
        "yield_curve_spread": 0.004,
        "vix": 18.0,
        "real_gdp_growth": 0.02,
        "timestamp": ts.date().isoformat(),
    }
    if pd.Timestamp("2020-03-01") <= ts <= pd.Timestamp("2021-06-30"):
        macro.update({
            "inflation_rate": 0.018,
            "fed_funds_rate": 0.0025,
            "unemployment_rate": 0.075,
            "mortgage_rate": 0.032,
            "treasury_10y": 0.009,
            "treasury_2y": 0.0015,
            "yield_curve_spread": 0.0075,
            "vix": 32.0,
            "real_gdp_growth": -0.015,
        })
    elif pd.Timestamp("2022-01-01") <= ts <= pd.Timestamp("2023-12-31"):
        macro.update({
            "inflation_rate": 0.062,
            "fed_funds_rate": 0.043,
            "unemployment_rate": 0.037,
            "mortgage_rate": 0.063,
            "treasury_10y": 0.037,
            "treasury_2y": 0.043,
            "yield_curve_spread": -0.006,
            "vix": 24.0,
            "real_gdp_growth": 0.012,
        })
    elif ts >= pd.Timestamp("2024-01-01"):
        macro.update({
            "inflation_rate": 0.033,
            "fed_funds_rate": 0.050,
            "unemployment_rate": 0.040,
            "mortgage_rate": 0.069,
            "treasury_10y": 0.043,
            "treasury_2y": 0.046,
            "yield_curve_spread": -0.003,
            "vix": 16.5,
            "real_gdp_growth": 0.022,
        })
    return macro


def _backtest_universe(payload: BacktestRequest) -> tuple[list[str], list[str], dict[str, float], float]:
    rows = []
    for holding in payload.holdings:
        ticker = sanitize_ticker(holding.ticker)
        if holding.category in {"equity", "bond"} and ticker and holding.units > 0 and holding.buyPrice > 0:
            rows.append({
                "category": holding.category,
                "ticker": ticker,
                "basis": float(holding.units * holding.buyPrice),
            })
    if not any(row["category"] == "equity" for row in rows):
        rows.append({"category": "equity", "ticker": "SPY", "basis": float(payload.household.portfolioWeights.equity)})
    if not any(row["category"] == "bond" for row in rows):
        rows.append({"category": "bond", "ticker": "AGG", "basis": float(payload.household.portfolioWeights.bond)})

    market_basis = sum(row["basis"] for row in rows)
    cash_basis = float(payload.household.liquidSavings)
    total_basis = market_basis + cash_basis
    if total_basis <= 0:
        total_basis = 1.0
        cash_basis = float(payload.household.portfolioWeights.cash)

    symbol_weights: dict[str, float] = {}
    for row in rows:
        symbol_weights[row["ticker"]] = symbol_weights.get(row["ticker"], 0.0) + row["basis"] / total_basis
    cash_weight = max(0.0, 1.0 - sum(symbol_weights.values()))
    total = sum(symbol_weights.values()) + cash_weight or 1.0
    symbol_weights = {symbol: weight / total for symbol, weight in symbol_weights.items()}
    cash_weight /= total
    equity_symbols = sorted({row["ticker"] for row in rows if row["category"] == "equity"})
    bond_symbols = sorted({row["ticker"] for row in rows if row["category"] == "bond"})
    return equity_symbols, bond_symbols, symbol_weights, cash_weight


def _category_symbol_weights(
    symbols: list[str],
    base_total_weights: dict[str, float],
    price_window: pd.DataFrame,
    hffi: float,
    category: str,
) -> dict[str, float]:
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0]: 1.0}
    base = pd.Series({symbol: max(base_total_weights.get(symbol, 0.0), 0.0001) for symbol in symbols})
    prices = price_window[symbols].dropna(how="all").ffill().bfill()
    if len(prices) < 5:
        return (base / base.sum()).to_dict()
    returns = prices.pct_change().dropna()
    momentum = prices.iloc[-1] / prices.iloc[max(0, len(prices) - 63)] - 1.0
    vol = returns.tail(63).std() * np.sqrt(252)
    if category == "equity":
        risk_weight = 0.8 if hffi < 60 else 1.4
        raw = base * np.clip(1.0 + 2.5 * momentum - risk_weight * vol, 0.2, 2.5)
    else:
        raw = base * np.clip(1.0 + 1.0 * momentum - 0.9 * vol, 0.3, 2.0)
    total = float(raw.sum())
    return (raw / total if total > 0 else base / base.sum()).to_dict()


def _performance_metrics(values: pd.Series, label: str, trades: int = 0, turnover: float = 0.0) -> dict[str, Any]:
    values = values.dropna()
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if len(values) > 1 else 0.0
    days = max((values.index[-1] - values.index[0]).days, 1)
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (365.25 / days) - 1.0) if values.iloc[0] else 0.0
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
    max_drawdown = float((values / values.cummax() - 1.0).min()) if len(values) else 0.0
    sharpe = float((returns.mean() * 252) / volatility) if volatility > 0 and len(returns) else 0.0
    return {
        "strategy": label,
        "finalValue": float(values.iloc[-1]) if len(values) else 0.0,
        "totalReturn": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "maxDrawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": int(trades),
        "turnover": float(turnover),
    }


def _holding_period(category: str, hffi: float, recommendation: str) -> str:
    if recommendation == "AVOID / REDUCE":
        return "Reduce or avoid until the score improves for 30-60 days."
    if hffi >= 60:
        return "1-3 months, then re-check HFFI and market score."
    if category in {"forex", "commodity", "sector"}:
        return "1-3 months with tighter review because the category is tactical."
    if category == "bond":
        return "3-6 months, reviewed after rate or macro changes."
    return "6-12 months if HFFI stays Stable/Moderate and price stays above the downside trigger."


def _build_investment_plan(
    ds_signals: pd.DataFrame,
    market_scores: pd.DataFrame,
    hffi: float,
    buying_capacity: float,
) -> list[dict[str, Any]]:
    if ds_signals.empty:
        return []
    price_lookup = {}
    if not market_scores.empty:
        for _, row in market_scores.iterrows():
            ticker = sanitize_ticker(row.get("ticker"))
            if ticker:
                price_lookup[ticker] = {
                    "price": _safe_number(row.get("current_price"), 0.0),
                    "momentum": _safe_number(row.get("momentum_score"), 0.0),
                    "vol": _safe_number(row.get("market_volatility"), 0.0),
                    "drawdown": _safe_number(row.get("market_drawdown"), 0.0),
                }
    rows = []
    for _, row in ds_signals.head(18).iterrows():
        ticker = sanitize_ticker(row.get("ticker"))
        category = str(row.get("category", "equity"))
        price_info = price_lookup.get(ticker, {})
        current_price = price_info.get("price") or None
        ds_score = _safe_number(row.get("ds_score"), 0.0)
        ml_probability = _safe_number(row.get("ml_probability"), 0.0)
        suitability = _safe_number(row.get("suitability_score"), 0.0)
        momentum = price_info.get("momentum", 0.0)
        vol = abs(price_info.get("vol", 0.0))
        recommendation = str(row.get("recommendation", "HOLD / WATCH"))
        if recommendation == "BUY CANDIDATE":
            suggested = min(_safe_number(row.get("suggested_monthly_amount"), 0.0), max(buying_capacity, 0.0))
        else:
            suggested = 0.0
        risk_guardrail = 0.65 if hffi >= 60 and category in {"equity", "sector", "commodity", "forex"} else 1.0
        expected_move = float(np.clip(
            (0.05 * ml_probability + 0.08 * max(suitability, 0.0) + 1.5 * momentum) * risk_guardrail,
            -0.12,
            0.22,
        ))
        entry_discount = 0.015 if category in {"bond", "forex"} else 0.03
        if current_price:
            expected_entry = current_price * (1 - entry_discount) if recommendation == "BUY CANDIDATE" else current_price
            target_price = current_price * (1 + max(expected_move, 0.01))
            downside_trigger = current_price * (1 - min(max(vol * 0.6, 0.03), 0.12))
        else:
            expected_entry = None
            target_price = None
            downside_trigger = None
        rows.append({
            "ticker": ticker,
            "name": row.get("name", ticker),
            "category": category,
            "recommendation": recommendation,
            "modelStrategy": "RF/GB suitability + HFFI guardrail + momentum/safety + allocation gap",
            "suggestedAmount": suggested,
            "currentPrice": current_price,
            "expectedEntryPrice": expected_entry,
            "targetPrice": target_price,
            "downsideTriggerPrice": downside_trigger,
            "expectedMovePct": expected_move,
            "holdingPeriod": _holding_period(category, hffi, recommendation),
            "confidence": float(np.clip(0.45 * ds_score + 0.35 * ml_probability + 0.20 * max(suitability, 0.0), 0.0, 1.0)),
            "comment": (
                f"{recommendation}: score={ds_score:.3f}, ML={ml_probability:.1%}, "
                f"suitability={suitability:+.3f}. Suggested amount is capped by monthly buying capacity "
                f"(${buying_capacity:,.0f}) and HFFI {hffi:.1f}."
            ),
        })
    return rows


def _live_holding_signal_audit(
    household: HouseholdIn,
    holdings: list[HoldingIn],
    latest_trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    macro = _macro_snapshot()
    holdings_df = _holdings_frame(holdings, household.liquidSavings)
    if holdings_df.empty:
        return []
    weights = allocation_weights_from_holdings(holdings_df)
    inputs = _to_household_inputs(household, weights)
    result = compute_household_hffi(inputs, macro, weights=DEFAULT_WEIGHTS)
    recs = generate_recommendations(result, inputs)
    target_weights = target_core_allocation(recs["allocation"])
    debt_service_ratio, liquidity_buffer_6m, macro_stress, risk_off = _market_scoring_inputs(inputs, macro)
    holding_symbols = tuple(
        holdings_df.loc[
            holdings_df["category"].isin(["equity", "bond"]),
            "ticker",
        ].dropna().astype(str).unique()
    )
    holding_rows = []
    for ticker in holding_symbols:
        hist = fetch_history(ticker, period="6mo", interval="1d")
        if hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"].dropna()
        if close.empty:
            continue
        rets = close.pct_change().dropna()
        running_max = close.cummax()
        drawdown = _safe_number((close / running_max - 1).tail(126).min(), -0.08)
        vol = _safe_number(rets.tail(60).std() * np.sqrt(252), 0.12)
        latest = _safe_number(rets.iloc[-1] if not rets.empty else 0)
        holding_rows.append({
            "market": ticker,
            "ticker": ticker,
            "name": ticker,
            "category": _category_for(ticker),
            "current_price": _safe_number(close.iloc[-1]),
            "market_return": latest,
            "market_volatility": vol,
            "market_drawdown": drawdown,
            "market_sharpe_proxy": latest / vol if vol else 0,
            "momentum_score": _safe_number(0.6 * latest + 0.4 * rets.tail(20).mean(), latest),
            "safety_score": -vol - abs(drawdown),
        })
    market_scores = pd.DataFrame()
    if holding_rows:
        market_scores = score_markets_for_household(
            hffi=result.score,
            risk_band=result.band,
            debt_service_ratio=debt_service_ratio,
            macro_stress_index=macro_stress,
            liquidity_buffer_6m=liquidity_buffer_6m,
            market_snapshot=pd.DataFrame(holding_rows),
            risk_off_regime=risk_off,
        )
    actions = recommend_holding_actions(
        holdings_df=holdings_df,
        target_weights=target_weights,
        market_scores=market_scores,
        hffi=result.score,
        buying_capacity=max(inputs.monthly_income - inputs.monthly_total_expenses - inputs.monthly_debt_payment, 0),
    )
    latest_by_category: dict[str, str] = {}
    for trade in latest_trades:
        category = str(trade.get("category", "")).lower()
        raw_action = str(trade.get("action", "HOLD")).upper()
        if raw_action in {"SELL", "RAISE CASH"}:
            latest_by_category[category] = "SELL"
        elif raw_action in {"BUY", "DEPLOY CASH"}:
            latest_by_category[category] = "BUY"
    rows = []
    for action in actions:
        if action.category not in {"equity", "bond"}:
            continue
        backtest_action = latest_by_category.get(action.category, "HOLD")
        matched = action.action == backtest_action or (action.action == "HOLD" and backtest_action == "HOLD")
        rows.append({
            "ticker": action.ticker,
            "category": action.category,
            "liveAction": action.action,
            "latestBacktestAction": backtest_action,
            "matched": matched,
            "hffi": result.score,
            "suitability": action.suitability,
            "comment": (
                f"Live holding engine says {action.action}; latest backtest replay says {backtest_action}. "
                f"{'Matched' if matched else 'Mismatch to review'} because live data, current HFFI, and backtest period may differ. "
                f"{action.rationale}"
            ),
        })
    return rows


def _rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> set[pd.Timestamp]:
    if frequency == "weekly":
        periods = dates.to_period("W-FRI")
    elif frequency == "quarterly":
        periods = dates.to_period("Q")
    else:
        periods = dates.to_period("M")
    grouped = pd.Series(dates, index=dates).groupby(periods).last()
    return set(pd.Timestamp(value) for value in grouped.iloc[:-1])


def _run_backtest(payload: BacktestRequest) -> dict[str, Any]:
    if payload.endDate <= payload.startDate:
        raise HTTPException(status_code=422, detail="End date must be after start date.")

    equity_symbols, bond_symbols, initial_symbol_weights, initial_cash_weight = _backtest_universe(payload)
    benchmark = sanitize_ticker(payload.benchmarkTicker) or "SPY"
    symbols = equity_symbols + bond_symbols
    prices, data_source = _price_matrix(symbols + [benchmark], payload.startDate, payload.endDate)
    symbols = [symbol for symbol in symbols if symbol in prices.columns]
    equity_symbols = [symbol for symbol in equity_symbols if symbol in prices.columns]
    bond_symbols = [symbol for symbol in bond_symbols if symbol in prices.columns]
    if not symbols:
        raise HTTPException(status_code=422, detail="No selected holdings have usable price history.")
    if benchmark not in prices.columns:
        benchmark = symbols[0]

    dates = prices.index
    start_prices = prices.iloc[0]
    rec_units = {
        symbol: payload.initialCapital * initial_symbol_weights.get(symbol, 0.0) / float(start_prices[symbol])
        for symbol in symbols
    }
    buy_hold_units = dict(rec_units)
    rec_cash = payload.initialCapital * initial_cash_weight
    buy_hold_cash = rec_cash
    benchmark_units = payload.initialCapital / float(start_prices[benchmark])
    signal_dates = _rebalance_dates(dates, payload.frequency)

    equity_curve = []
    allocation_rows = []
    trade_rows = []
    pending: dict[str, Any] | None = None
    turnover_total = 0.0
    recommendation_values: list[float] = []

    for idx, dt in enumerate(dates):
        today_prices = prices.loc[dt]
        if pending is not None:
            total_before = sum(rec_units.get(symbol, 0.0) * float(today_prices[symbol]) for symbol in symbols) + rec_cash
            desired_before_cost = {
                symbol: pending["symbol_targets"].get(symbol, 0.0) * total_before
                for symbol in symbols
            }
            current_values = {
                symbol: rec_units.get(symbol, 0.0) * float(today_prices[symbol])
                for symbol in symbols
            }
            turnover = sum(abs(desired_before_cost[symbol] - current_values[symbol]) for symbol in symbols)
            cost = turnover * payload.transactionCostPct
            total_after_cost = max(total_before - cost, 0.0)
            for symbol in symbols:
                rec_units[symbol] = pending["symbol_targets"].get(symbol, 0.0) * total_after_cost / float(today_prices[symbol])
            rec_cash = pending["cash_target"] * total_after_cost
            turnover_total += turnover
            trade_rows.extend(pending["trades"])
            pending = None

        rec_value = sum(rec_units.get(symbol, 0.0) * float(today_prices[symbol]) for symbol in symbols) + rec_cash
        buy_hold_value = sum(buy_hold_units.get(symbol, 0.0) * float(today_prices[symbol]) for symbol in symbols) + buy_hold_cash
        benchmark_value = benchmark_units * float(today_prices[benchmark])
        recommendation_values.append(rec_value)
        equity_curve.append({
            "date": dt.date().isoformat(),
            "recommendation": rec_value,
            "buyHold": buy_hold_value,
            "benchmark": benchmark_value,
        })

        current_values = {
            symbol: rec_units.get(symbol, 0.0) * float(today_prices[symbol])
            for symbol in symbols
        }
        equity_value = sum(current_values.get(symbol, 0.0) for symbol in equity_symbols)
        bond_value = sum(current_values.get(symbol, 0.0) for symbol in bond_symbols)
        total_value = equity_value + bond_value + rec_cash or 1.0
        actual_weights = {
            "equity": equity_value / total_value,
            "bond": bond_value / total_value,
            "cash": rec_cash / total_value,
        }

        value_series = pd.Series(recommendation_values, index=dates[: idx + 1])
        daily_returns = value_series.pct_change().dropna()
        rolling_vol = float(daily_returns.tail(63).std() * np.sqrt(252)) if len(daily_returns) > 5 else payload.household.portfolioVolatility
        drawdown_series = value_series / value_series.cummax() - 1.0
        rolling_drawdown = float(abs(drawdown_series.tail(126).min())) if len(drawdown_series) > 5 else payload.household.expectedDrawdown
        macro = _fallback_macro_for_date(dt)
        inputs = _to_household_inputs(payload.household, actual_weights)
        inputs = HouseholdInputs(
            monthly_income=inputs.monthly_income,
            monthly_essential_expenses=inputs.monthly_essential_expenses,
            monthly_total_expenses=inputs.monthly_total_expenses,
            liquid_savings=rec_cash,
            total_debt=inputs.total_debt,
            monthly_debt_payment=inputs.monthly_debt_payment,
            portfolio_weights=actual_weights,
            portfolio_volatility=rolling_vol,
            expected_drawdown=rolling_drawdown,
            rate_sensitivity=inputs.rate_sensitivity,
            dependents=inputs.dependents,
            employment_type=inputs.employment_type,
        )
        result = compute_household_hffi(inputs, macro, weights=DEFAULT_WEIGHTS)
        allocation_rows.append({
            "date": dt.date().isoformat(),
            "equity": actual_weights["equity"],
            "bond": actual_weights["bond"],
            "cash": actual_weights["cash"],
            "hffi": result.score,
            "band": result.band,
        })

        if dt in signal_dates and idx + 1 < len(dates):
            recs = generate_recommendations(result, inputs)
            target_weights = target_core_allocation(recs["allocation"])
            price_window = prices.loc[:dt]
            equity_inner = _category_symbol_weights(equity_symbols, initial_symbol_weights, price_window, result.score, "equity")
            bond_inner = _category_symbol_weights(bond_symbols, initial_symbol_weights, price_window, result.score, "bond")
            symbol_targets = {
                symbol: target_weights.get("equity", 0.0) * equity_inner.get(symbol, 0.0)
                for symbol in equity_symbols
            }
            symbol_targets.update({
                symbol: target_weights.get("bond", 0.0) * bond_inner.get(symbol, 0.0)
                for symbol in bond_symbols
            })
            cash_target = float(target_weights.get("cash", 0.0))
            target_sum = sum(symbol_targets.values()) + cash_target or 1.0
            symbol_targets = {symbol: weight / target_sum for symbol, weight in symbol_targets.items()}
            cash_target /= target_sum

            trades = []
            for category, current_weight, target_weight in (
                ("equity", actual_weights["equity"], target_weights.get("equity", 0.0)),
                ("bond", actual_weights["bond"], target_weights.get("bond", 0.0)),
                ("cash", actual_weights["cash"], target_weights.get("cash", 0.0)),
            ):
                diff = (target_weight - current_weight) * total_value
                if abs(diff) < max(50.0, total_value * 0.005):
                    continue
                action = "BUY" if diff > 0 else "SELL"
                if category == "cash":
                    action = "RAISE CASH" if diff > 0 else "DEPLOY CASH"
                trades.append({
                    "signalDate": dt.date().isoformat(),
                    "executionDate": dates[idx + 1].date().isoformat(),
                    "category": category,
                    "action": action,
                    "amount": abs(diff),
                    "fromWeight": current_weight,
                    "targetWeight": float(target_weight),
                    "hffi": result.score,
                    "band": result.band,
                    "comment": (
                        f"HFFI {result.score:.1f} ({result.band}) set target "
                        f"{category} weight to {target_weight:.1%} from {current_weight:.1%}; "
                        f"trade executes on the next trading day to avoid look-ahead bias."
                    ),
                })
            pending = {
                "symbol_targets": symbol_targets,
                "cash_target": cash_target,
                "trades": trades,
            }

    curve_df = pd.DataFrame(equity_curve)
    curve_df.index = pd.to_datetime(curve_df["date"])
    drawdown_df = pd.DataFrame({
        "date": curve_df["date"],
        "recommendation": curve_df["recommendation"] / curve_df["recommendation"].cummax() - 1.0,
        "buyHold": curve_df["buyHold"] / curve_df["buyHold"].cummax() - 1.0,
        "benchmark": curve_df["benchmark"] / curve_df["benchmark"].cummax() - 1.0,
    })
    metrics = [
        _performance_metrics(curve_df.set_index(pd.to_datetime(curve_df["date"]))["recommendation"], "HFFI Recommendation", len(trade_rows), turnover_total),
        _performance_metrics(curve_df.set_index(pd.to_datetime(curve_df["date"]))["buyHold"], "Buy and Hold"),
        _performance_metrics(curve_df.set_index(pd.to_datetime(curve_df["date"]))["benchmark"], f"Benchmark {benchmark}"),
    ]
    latest_signal_date = max((str(trade.get("signalDate")) for trade in trade_rows), default="")
    latest_trades = [trade for trade in trade_rows if str(trade.get("signalDate")) == latest_signal_date]
    signal_audit = _live_holding_signal_audit(payload.household, payload.holdings, latest_trades)
    return _jsonable({
        "settings": {
            "startDate": payload.startDate.isoformat(),
            "endDate": payload.endDate.isoformat(),
            "initialCapital": payload.initialCapital,
            "frequency": payload.frequency,
            "transactionCostPct": payload.transactionCostPct,
            "benchmarkTicker": benchmark,
        },
        "dataSource": data_source,
        "macroSource": "historical regime fallback for offline replay; live FRED can be added when an API key is configured",
        "metrics": metrics,
        "equityCurve": equity_curve,
        "drawdown": drawdown_df,
        "allocations": allocation_rows,
        "trades": trade_rows,
        "signalAudit": signal_audit,
    })


def _generate_excel_report(payload: ReportRequest) -> Path:
    macro = _macro_snapshot()
    holdings_df = _holdings_frame(payload.holdings, payload.household.liquidSavings)
    if not holdings_df.empty:
        weights = allocation_weights_from_holdings(holdings_df)
    else:
        weights = payload.household.portfolioWeights.model_dump()
        total = sum(weights.values()) or 1.0
        weights = {key: float(value) / total for key, value in weights.items()}

    inputs = _to_household_inputs(payload.household, weights)
    result = compute_household_hffi(inputs, macro, weights=DEFAULT_WEIGHTS)
    portfolio_scores = score_portfolios(result.score)
    portfolio_choice = str(portfolio_scores.iloc[0]["portfolio"]) if not portfolio_scores.empty else "Balanced"
    plan = build_long_horizon_investment_plan(
        portfolio=portfolio_choice,
        horizon_years=payload.horizonYears,
        initial_capital=payload.initialCapital,
        monthly_contribution=payload.monthlyContribution,
        annual_contribution_growth=payload.annualContributionGrowth,
        n_sims=3000,
    )
    comparison = compare_portfolios(
        horizon_years=payload.horizonYears,
        initial_capital=payload.initialCapital,
        monthly_contribution=payload.monthlyContribution,
        annual_contribution_growth=payload.annualContributionGrowth,
    )
    recs = generate_recommendations(result, inputs)
    stress = apply_shock_scenarios(inputs, macro)

    trade_signals = []
    market_df = _market_universe_snapshot()
    if not market_df.empty:
        debt_service_ratio, liquidity_buffer_6m, macro_stress, risk_off = _market_scoring_inputs(inputs, macro)
        market_scores = score_markets_for_household(
            hffi=result.score,
            risk_band=result.band,
            debt_service_ratio=debt_service_ratio,
            macro_stress_index=macro_stress,
            liquidity_buffer_6m=liquidity_buffer_6m,
            market_snapshot=market_df,
            risk_off_regime=risk_off,
        )
        trade_signals = generate_trade_signals(
            portfolio_choice=portfolio_choice,
            market_scores=market_scores,
            asset_metadata=pd.DataFrame(_asset_records()),
        )

    output_path = ROOT / "reports" / f"hffi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return generate_report(
        output_path=output_path,
        fragility_result=result,
        investment_plan=plan,
        macro=macro,
        portfolio_choice=portfolio_choice,
        initial_capital=payload.initialCapital,
        monthly_contribution=payload.monthlyContribution,
        horizon_years=payload.horizonYears,
        portfolio_comparison=comparison,
        trade_signals=trade_signals,
        stress_scenarios=stress,
        recommendations=recs,
    )


@app.get("/api/security/config")
def api_security_config() -> dict[str, Any]:
    return security_config()


@app.post("/api/auth/login", dependencies=[Depends(public_rate_limit("auth_login", 8, 60))])
def login(payload: LoginRequest, request: Request) -> dict[str, Any]:
    user = authenticate_user(payload.username, payload.password)
    client = request.client.host if request.client else "unknown"
    if user is None:
        audit_event("login_failed", username=payload.username, client=client)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_access_token(user)
    audit_event("login_success", username=user.username, role=user.role, client=client)
    return {
        "accessToken": token,
        "tokenType": "bearer",
        "expiresIn": security_config()["tokenTtlMinutes"] * 60,
        "user": {"username": user.username, "role": user.role},
    }


@app.get("/api/auth/me", dependencies=[Depends(rate_limit("auth_me", 60, 60))])
def auth_me(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    return {"user": {"username": user.username, "role": user.role}}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "hffi-terminal-api"}


@app.get("/api/assets", dependencies=[Depends(rate_limit("assets", 120, 60))])
def assets(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    grouped = {}
    for category in get_categories():
        grouped[category] = [
            {
                "ticker": asset.ticker,
                "name": asset.name,
                "category": asset.category,
                "subcategory": asset.subcategory,
                "fetchSymbol": asset.fetch_symbol(),
            }
            for asset in get_assets_by_category(category)
        ]
    return {"categories": get_categories(), "assets": grouped}


@app.get("/api/market/{category}", dependencies=[Depends(rate_limit("market", 40, 60))])
def market_snapshot(category: str, user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    assets = get_assets_by_category(category)
    if not assets:
        raise HTTPException(status_code=404, detail="Unknown category.")
    symbols = [asset.fetch_symbol() for asset in assets]
    snapshot = fetch_market_snapshot(symbols)
    rows = []
    by_fetch = {asset.fetch_symbol(): asset for asset in assets}
    for symbol, row in snapshot.iterrows():
        asset = by_fetch.get(str(symbol))
        rows.append({
            "ticker": asset.ticker if asset else str(symbol),
            "fetchSymbol": str(symbol),
            "name": asset.name if asset else row.get("name", str(symbol)),
            "category": category,
            "subcategory": asset.subcategory if asset else "",
            "price": row.get("price"),
            "change": row.get("change"),
            "changePct": row.get("change_pct"),
            "dataSource": row.get("data_source", "unknown"),
        })
    return {"category": category, "rows": _jsonable(rows)}


@app.get("/api/chart/{ticker}", dependencies=[Depends(rate_limit("chart", 20, 60))])
def chart(ticker: str, period: str = "6mo", interval: str = "1d", user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    safe = sanitize_ticker(ticker)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid ticker.")
    allowed_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
    allowed_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
    if period not in allowed_periods:
        raise HTTPException(status_code=422, detail="Unsupported chart period.")
    if interval not in allowed_intervals:
        raise HTTPException(status_code=422, detail="Unsupported chart interval.")
    df = fetch_history(safe, period=period, interval=interval)
    if df.empty:
        return {"ticker": safe, "dataSource": "empty", "rows": []}
    rows = []
    for idx, row in df.tail(1000).iterrows():
        rows.append({
            "date": pd.Timestamp(idx).isoformat(),
            "open": row.get("Open"),
            "high": row.get("High"),
            "low": row.get("Low"),
            "close": row.get("Close"),
            "volume": row.get("Volume"),
        })
    return {"ticker": safe, "dataSource": df.attrs.get("data_source", "live:yfinance"), "rows": _jsonable(rows)}


@app.post(
    "/api/backtest",
    dependencies=[
        Depends(rate_limit("backtest", 5, 60)),
        Depends(require_roles("admin", "analyst")),
    ],
)
def backtest(payload: BacktestRequest, user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    result = _run_backtest(payload)
    audit_event(
        "backtest_completed",
        user=user.username,
        role=user.role,
        fingerprint=payload_fingerprint(payload.model_dump()),
        trades=len(result.get("trades", [])),
        metrics=len(result.get("metrics", [])),
    )
    return result


@app.get("/api/backtest", dependencies=[Depends(rate_limit("backtest_info", 60, 60))])
def backtest_info(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "status": "available",
        "method": "POST",
        "message": "Use the React Backtesting tab or POST a BacktestRequest JSON payload to run a backtest.",
    }


@app.post(
    "/api/report/excel",
    dependencies=[
        Depends(rate_limit("report_excel", 5, 60)),
        Depends(require_roles("admin", "analyst")),
    ],
)
def excel_report(payload: ReportRequest, user: AuthenticatedUser = Depends(get_current_user)) -> FileResponse:
    output_path = _generate_excel_report(payload)
    audit_event(
        "excel_report_generated",
        user=user.username,
        role=user.role,
        fingerprint=payload_fingerprint(payload.model_dump()),
        file=output_path.name,
    )
    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.post(
    "/api/analyze",
    dependencies=[
        Depends(rate_limit("analyze", 30, 60)),
        Depends(require_roles("admin", "analyst")),
    ],
)
def analyze(payload: AnalyzeRequest, user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
    macro = _macro_snapshot()
    holdings_df = _holdings_frame(payload.holdings, payload.household.liquidSavings)
    if not holdings_df.empty:
        weights = allocation_weights_from_holdings(holdings_df)
    else:
        weights = payload.household.portfolioWeights.model_dump()
        total = sum(weights.values()) or 1.0
        weights = {key: float(value) / total for key, value in weights.items()}

    inputs = _to_household_inputs(payload.household, weights)
    result = compute_household_hffi(inputs, macro, weights=DEFAULT_WEIGHTS)
    recs = generate_recommendations(result, inputs)
    target_weights = target_core_allocation(recs["allocation"])
    actual_weights = weights
    household_features = engineer_household_features(inputs, result, macro, holdings_df)
    ds_model = _ds_model()
    segment = assign_household_segment(household_features)
    debt_service_ratio, liquidity_buffer_6m, macro_stress, risk_off = _market_scoring_inputs(inputs, macro)
    buying_capacity = max(
        inputs.monthly_income - inputs.monthly_total_expenses - inputs.monthly_debt_payment,
        0,
    )

    market_df = _market_universe_snapshot()
    market_scores = pd.DataFrame()
    ds_signals = pd.DataFrame()
    selected_market: dict[str, Any] = {}
    if not market_df.empty:
        market_scores = score_markets_for_household(
            hffi=result.score,
            risk_band=result.band,
            debt_service_ratio=debt_service_ratio,
            macro_stress_index=macro_stress,
            liquidity_buffer_6m=liquidity_buffer_6m,
            market_snapshot=market_df,
            risk_off_regime=risk_off,
        )
        selected_market = select_one_market_recommendation(market_scores)
        ds_signals = score_market_recommendations(
            model=ds_model,
            household_features=household_features,
            market_scores=market_scores,
            target_weights=target_weights,
            actual_weights=actual_weights,
        )

    holding_actions = []
    if not holdings_df.empty:
        holding_symbols = tuple(
            holdings_df.loc[
                holdings_df["category"].isin(["equity", "bond"]),
                "ticker",
            ].dropna().astype(str).unique()
        )
        holding_market_df = pd.DataFrame()
        holding_rows = []
        for ticker in holding_symbols:
            hist = fetch_history(ticker, period="6mo", interval="1d")
            if hist.empty or "Close" not in hist.columns:
                continue
            close = hist["Close"].dropna()
            rets = close.pct_change().dropna()
            if close.empty:
                continue
            running_max = close.cummax()
            drawdown = _safe_number((close / running_max - 1).min(), -0.08)
            vol = _safe_number(rets.std() * np.sqrt(21), 0.05)
            latest = _safe_number(rets.iloc[-1] if not rets.empty else 0)
            holding_rows.append({
                "market": ticker,
                "ticker": ticker,
                "name": ticker,
                "category": _category_for(ticker),
                "market_return": latest,
                "market_volatility": vol,
                "market_drawdown": drawdown,
                "market_sharpe_proxy": latest / vol if vol else 0,
                "momentum_score": _safe_number(rets.tail(20).mean(), latest),
                "safety_score": -vol - abs(drawdown),
            })
        if holding_rows:
            holding_market_df = pd.DataFrame(holding_rows)
            holding_market_df = score_markets_for_household(
                hffi=result.score,
                risk_band=result.band,
                debt_service_ratio=debt_service_ratio,
                macro_stress_index=macro_stress,
                liquidity_buffer_6m=liquidity_buffer_6m,
                market_snapshot=holding_market_df,
                risk_off_regime=risk_off,
            )
        holding_actions = recommend_holding_actions(
            holdings_df=holdings_df,
            target_weights=target_weights,
            market_scores=holding_market_df if not holding_market_df.empty else market_scores,
            hffi=result.score,
            buying_capacity=buying_capacity,
        )

    counterfactuals = build_counterfactual_table(inputs, macro, base_result=result)
    decision_evidence = build_decision_evidence_table(
        holding_actions=[action for action in holding_actions if action.category in {"equity", "bond"}],
        ds_signals=ds_signals,
        hffi=result.score,
    )
    feature_evidence = build_feature_evidence_table(ds_model, household_features)
    stress = apply_shock_scenarios(inputs, macro)
    monte_carlo = monte_carlo_stress(inputs, macro, n_sims=2000)
    hist_counts, hist_edges = np.histogram(monte_carlo.scores, bins=32, range=(0, 100))
    monte_carlo_payload = {
        "mean": monte_carlo.mean,
        "std": monte_carlo.std,
        "p05": monte_carlo.p05,
        "p50": monte_carlo.p50,
        "p95": monte_carlo.p95,
        "probSevere": monte_carlo.prob_severe,
        "histogram": [
            {
                "binStart": float(hist_edges[i]),
                "binEnd": float(hist_edges[i + 1]),
                "count": int(count),
            }
            for i, count in enumerate(hist_counts)
        ],
    }
    portfolio_scores = score_portfolios(result.score)
    investment_plan = _build_investment_plan(ds_signals, market_scores, result.score, buying_capacity)
    allocation_summary = summarize_allocation(holdings_df) if not holdings_df.empty else pd.DataFrame([
        {"category": "equity", "invested_amount": 0, "actual_weight": actual_weights.get("equity", 0)},
        {"category": "bond", "invested_amount": 0, "actual_weight": actual_weights.get("bond", 0)},
        {"category": "cash", "invested_amount": 0, "actual_weight": actual_weights.get("cash", 0)},
    ])

    response_payload = _jsonable({
        "macro": macro,
        "hffi": {
            "score": result.score,
            "band": result.band,
            "distressProbability": result.distress_probability,
            "components": {"L": result.L, "D": result.D, "E": result.E, "P": result.P, "M": result.M},
            "contributions": result.contributions,
        },
        "segment": segment,
        "recommendations": [
            {
                "priority": rec.priority,
                "component": rec.component,
                "action": rec.action,
                "detail": rec.detail,
                "expectedImpact": rec.expected_impact,
            }
            for rec in recs["actions"]
        ],
        "targetAllocation": target_weights,
        "actualAllocation": actual_weights,
        "allocationSummary": allocation_summary,
        "holdings": holdings_df,
        "holdingActions": [
            {
                "category": action.category,
                "ticker": action.ticker,
                "name": action.name,
                "status": action.action,
                "strategy": action.strategy,
                "when": action.suggested_timing,
                "comment": action.rationale,
                "investedAmount": action.invested_amount,
                "currentValue": action.current_value,
                "unrealizedPl": action.unrealized_pl,
                "unrealizedPlPct": action.unrealized_pl_pct,
                "allocationWeightPct": action.allocation_weight_pct,
                "targetCategoryPct": action.target_category_pct,
                "suitability": action.suitability,
            }
            for action in holding_actions if action.category in {"equity", "bond"}
        ],
        "selectedMarket": selected_market,
        "marketSignals": ds_signals.head(12),
        "investmentPlan": investment_plan,
        "decisionEvidence": decision_evidence,
        "counterfactuals": counterfactuals,
        "featureEvidence": feature_evidence,
        "modelCard": build_model_card_table(ds_model),
        "modelPerformance": build_model_performance_summary(ds_model),
        "stress": stress.reset_index(),
        "monteCarlo": monte_carlo_payload,
        "portfolioScores": portfolio_scores,
        "riskOff": risk_off,
    })
    audit_event(
        "analysis_completed",
        user=user.username,
        role=user.role,
        fingerprint=payload_fingerprint(payload.model_dump()),
        hffi=round(float(result.score), 2),
        band=str(result.band).replace(" ", "_"),
        holding_actions=len(holding_actions),
        investment_candidates=len(investment_plan),
    )
    return response_payload
