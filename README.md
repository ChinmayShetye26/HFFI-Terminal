# HFFI Terminal

**Household Financial Fragility Index — Quant Strategy Terminal for Households**

A research prototype that:

1. Computes a composite **Household Financial Fragility Index (HFFI)** from
   five fragility dimensions (liquidity, debt, expenses, portfolio, macro)
   plus a debt × liquidity interaction term.
2. Maps the score to a **portfolio allocation strategy** and a set of
   **personalized recommendations**.
3. Streams **live macro indicators** (FRED), **market data** (yfinance /
   Polygon / Alpaca), and **news** (NewsAPI) into a Bloomberg-style
   Streamlit terminal.
4. Provides a **validation harness** with out-of-sample AUC, baseline
   comparisons, and weight-sensitivity analysis — designed to defend the
   academic claims in the IEEE paper.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (free for FRED + NewsAPI)
cp .env.example .env
# then edit .env and fill in:
#   FRED_API_KEY    (free at https://fred.stlouisfed.org/docs/api/api_key.html)
#   NEWSAPI_KEY     (free at https://newsapi.org/)

# 3. Run the smoke test (no network required)
python tests/test_pipeline.py

# 4. Launch the terminal
streamlit run app/streamlit_app.py
```

---

## Project structure

```
hffi_terminal/
├── hffi_core/              # The HFFI library (pure Python, no UI)
│   ├── components.py       # L, D, E, P, M formulas
│   ├── scoring.py          # Composite score + risk bands + distress prob
│   ├── weights.py          # PCA + LogReg weight learning
│   ├── stress.py           # Deterministic + Monte Carlo stress tests
│   ├── recommendations.py  # Rule-based actions + allocation templates
│   └── validation.py       # OOS evaluation + baselines + sensitivity
├── data/                   # Live data fetchers
│   ├── macro_fetcher.py    # FRED — inflation, fed funds, mortgage, etc.
│   ├── market_fetcher.py   # yfinance / Polygon / Alpaca with universe selection
│   ├── news_fetcher.py     # NewsAPI + RSS fallback
│   └── synthetic.py        # Generator for testing without real data
├── app/
│   └── streamlit_app.py    # The terminal UI
├── tests/
│   └── test_pipeline.py    # End-to-end smoke test
├── docs/                   # Design docs, IEEE paper notes
│   ├── SCOPE.md            # Tightened semester scope
│   ├── ARCHITECTURE.md     # System architecture
│   ├── PAPER_REVIEW.md     # IEEE draft review and fixes
│   └── EXTENSIONS.md       # Future enhancements
├── requirements.txt
└── .env.example
```

---

## Configuring market data

The terminal supports three data providers and four ticker universes,
selected at runtime from the UI or via `.env`:

| Provider  | Cost | Universe support           | Notes                          |
|-----------|------|----------------------------|--------------------------------|
| yfinance  | Free | sp500, russell1000         | Slow above ~500 tickers        |
| polygon   | Paid | sp500, russell1000, full_us | Fast, real-time, requires key |
| alpaca    | Free | sp500, russell1000, full_us | Free tier good for paper trading |

For "every US company", you need a Polygon or Alpaca key
(`MARKET_PROVIDER=polygon` and `TICKER_UNIVERSE=full_us` in `.env`).
Without that, the terminal defaults to the S&P 500.

---

## Citation / reproducibility

The HFFI formula:

    HFFI = 100 × (w1·L + w2·D + w3·E + w4·P + w5·M + λ·(L·D))

The default weights are an initial baseline; the production weights should
be learned from data using `hffi_core.weights.learn_weights_logreg()` on
a labeled household dataset (e.g., the SCF microdata with a constructed
distress label). See `docs/PAPER_REVIEW.md` for details.
"# HFFI-Terminal" 
"# HFFI-Terminal" 
