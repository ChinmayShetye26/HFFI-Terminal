# React + FastAPI HFFI Terminal

This interface is an additive professional frontend for the existing HFFI
project. Streamlit remains available, while React provides the interactive
terminal experience and FastAPI exposes the Python scoring/recommendation
engine.

## Architecture

```text
frontend/ React + TypeScript + Vite
        |
        | HTTP JSON
        v
api/ FastAPI service
        |
        v
hffi_core/ + data/ Python HFFI, ML, Evidence Lab, market fallback
```

## Run Locally

Install Python API dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm.cmd install
```

Fastest launcher:

```powershell
.\run_react_terminal.bat
```

Manual API start:

```powershell
.\scripts\run_terminal_api.ps1
```

Manual React frontend start in another terminal:

```powershell
.\scripts\run_terminal_frontend.ps1
```

Open:

```text
http://127.0.0.1:5173
```

## Interactive Sections

- Fragility: HFFI gauge, component heatmap, distress probability, priority actions.
- Portfolio: searchable Equity/Bond holdings, duplicate ticker prevention, holding guidance.
- Markets: category snapshot, ticker search, custom period/interval, fullscreen candlestick chart, fallback/live source label.
- Backtesting: historical HFFI recommendation replay versus buy-and-hold and benchmark portfolios.
- Evidence: counterfactual simulator, decision confidence audit, and recommendation analytics.
- Scenarios: recession/job-loss/rate-shock stress cards and Monte Carlo stress distribution.
- Model: feature evidence, model card, validation and security guardrails.

## Backtesting Checks

1. Open the Backtesting tab.
2. Set Start date, End date, Initial capital, Frequency, Transaction cost %, and Benchmark ticker.
3. Click Run Backtest.
4. Review Performance Summary, Equity Curve, Drawdown, Allocation And HFFI Replay, and Trade Log.

The backtest calculates recommendation signals on each rebalance date and
executes them on the next trading day to avoid look-ahead bias. Historical
price data uses the configured market provider, with offline fallback when live
data is unavailable.

## Security Notes

- The frontend never receives API keys.
- The API sanitizes ticker symbols before using provider calls.
- No endpoint places trades or mutates brokerage accounts.
- Yahoo Finance rate limits are handled by the existing fallback/cooldown logic.
- CORS is restricted to local React development origins.
