# Architecture

## Layered design

The system is organized in five horizontal layers; data flows top to bottom.

### 1. Data layer
External sources we don't control:
- **FRED** — macro indicators (inflation, fed funds, unemployment, mortgage,
  treasury yields, VIX, GDP). Free API key required.
- **yfinance / Polygon / Alpaca** — market data (prices, volumes, change %).
  Provider switch via `MARKET_PROVIDER` env var.
- **NewsAPI** — recent market and macro news headlines.
- **SCF / CES (offline)** — household survey data for weight calibration.
  Bulk-loaded, not streamed.

### 2. Ingestion + cache layer
`data/macro_fetcher.py`, `data/market_fetcher.py`, `data/news_fetcher.py`

Each fetcher:
- Pulls from the upstream API
- Normalizes units (rates as decimals, not percentages)
- Caches to `.cache/*.parquet` with a TTL
- Falls back gracefully when keys/network are unavailable

### 3. Core engine layer
`hffi_core/`

Pure-Python, no UI or network. Inputs come from the ingestion layer,
outputs go to the strategy layer. Every public function is unit-testable.

- `components.py` — five fragility formulas, each [0, 1] bounded
- `scoring.py` — composite score, risk band, distress probability
- `weights.py` — PCA + LogReg learning + YAML persistence
- `stress.py` — scenarios + Monte Carlo

### 4. Strategy + recommendation layer
`hffi_core/recommendations.py`

Translates fragility scores into actionable output:
- Priority-ranked rule-based recommendations (with explanation)
- Asset allocation template by risk band
- Explainability: each recommendation cites which component triggered it

### 5. Terminal UI layer
`app/streamlit_app.py`

Five tabs, all reactive:
- **Score** — gauge + component bar + contribution waterfall
- **Stress test** — bar chart per scenario + Monte Carlo histogram
- **Recommendations** — priority list + allocation pie
- **Macro** — six live indicators + yield-curve callout
- **Markets & News** — top movers + headline feed

---

## Data contract

The boundary between layers is enforced by two dataclasses:

### `HouseholdInputs` (user → engine)
```python
@dataclass
class HouseholdInputs:
    monthly_income: float
    monthly_essential_expenses: float
    monthly_total_expenses: float
    liquid_savings: float
    total_debt: float
    monthly_debt_payment: float
    portfolio_weights: Dict[str, float]
    portfolio_volatility: float
    expected_drawdown: float
    rate_sensitivity: float
    age, dependents, employment_type: optional metadata
```

### `FragilityResult` (engine → UI)
```python
@dataclass
class FragilityResult:
    L, D, E, P, M: float                 # individual components in [0, 1]
    interaction_LD: float                # L × D
    score: float                         # composite, in [0, 100]
    band: str                            # "Stable" | "Moderate" | "High" | "Severe"
    distress_probability: float          # 12-month, calibrated
    contributions: Dict[str, float]      # per-component points contributed
```

This contract means the UI never touches raw fragility math, and the
engine never knows how the score is displayed. That separation is what
makes the codebase testable — `tests/test_pipeline.py` exercises the full
core engine with zero UI dependencies.

---

## Caching policy

| Data       | Cache TTL   | Justification                    |
|------------|-------------|----------------------------------|
| Macro      | 60 minutes  | Macro releases are daily-ish     |
| Market     | 5 minutes   | Need freshness for movers panel  |
| News       | 30 minutes  | Headlines change quickly         |
| Tickers    | 24 hours    | Universe membership rarely changes |
| Weights    | Persistent  | YAML, only updated on retrain    |

Caches live in `.cache/` and are gitignored. First-run is slower because
the cache is cold.

---

## Failure modes

| Failure                          | What happens                              |
|----------------------------------|-------------------------------------------|
| `FRED_API_KEY` missing           | Macro panel shows defaults; UI flags it   |
| `NEWSAPI_KEY` missing            | News falls back to RSS; partial coverage  |
| Market provider rate-limited     | UI shows last cached snapshot             |
| User inputs zero income          | Debt component returns 1.0 (max fragile)  |
| All-cash portfolio               | Portfolio volatility/drawdown = 0 → P low |
| Network down                     | UI uses cached data; banner shows "offline" |

The system is designed to degrade — every external dependency has a
fallback so you can demo the prototype even if APIs are unreachable.
