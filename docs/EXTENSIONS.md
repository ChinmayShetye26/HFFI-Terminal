# Extensions — what else can you do with this project

A concrete menu of things this project can grow into. Each is sized so
you can pick and choose what fits your time, interest, and ambitions.

---

## A. Academic extensions (paper material)

### A1. Multi-entity fragility
Apply the same five-component framework to companies, sectors, and
countries, with entity-specific component definitions. The composite
formula stays the same; only the inputs change. This is a clean follow-up
paper.

### A2. Network contagion
If you can get household-network data (employer clusters, neighborhoods),
extend HFFI with a graph-based contagion model. Stressed households with
strong network ties to other stressed households become more fragile.
This is what the original transcript called the "GNN extension".

### A3. Dynamic weights
The current weights are static. In reality, the importance of each
fragility dimension varies by macro regime — debt fragility matters more
when rates are rising, portfolio fragility matters more in market
crashes. Use a Hidden Markov Model on macro indicators to detect regimes,
then maintain regime-specific weight vectors.

### A4. Real SCF calibration
Download the SCF microdata from the Federal Reserve, handle the five
implicates and survey weights properly, and recalibrate. This alone is
publishable as an empirical paper.

### A5. Counterfactual recommendations
The current recommender is rule-based. Upgrade to a counterfactual
explanation: "if you increased your liquid savings to $X, your HFFI
would drop from 65 to 42". This is a well-studied area in interpretable
ML (DiCE, MACE algorithms).

---

## B. Engineering extensions (FinTech / startup material)

### B1. Authentication + multi-tenant
Add user accounts so different households can log in and track HFFI
over time. Use Supabase / Firebase / Auth0 for the auth, postgres for
the historical scores.

### B2. Mobile app
Wrap the API behind a thin React Native / Flutter UI. The terminal is
desktop-friendly; a mobile experience needs different information
density.

### B3. Bank/FinTech B2B integration
Provide REST APIs that return HFFI scores given a household profile.
Banks could use this for credit decisions; robo-advisors for portfolio
recommendations. This is the actual commercial wedge.

### B4. Real broker integration
Connect to Plaid (account aggregation) and Alpaca/IBKR (trading) so
recommendations can be one-click executed. This crosses into KYC /
broker-dealer regulatory territory — get a lawyer.

### B5. Continuous monitoring + alerts
Schedule daily HFFI recomputation per user; send push notifications
when scores cross band thresholds. AWS EventBridge + Lambda + SNS
makes this straightforward.

### B6. SCF-trained pre-built model
Distribute a pre-trained weights file calibrated on SCF as the default,
so users don't need to do any setup. Update annually.

---

## C. Data Science extensions (CV + portfolio material)

### C1. NLP-based macro sentiment
Currently the news panel is just headlines. Add FinBERT or a similar
financial sentiment model to score each headline; use aggregate
sentiment as an additional macro fragility input.

### C2. Time-series forecasting of macro inputs
Forecast inflation, fed funds, unemployment 6 months out using ARIMA
or Prophet. Show the household what their HFFI will look like under
the forecasted macro path.

### C3. Causal evaluation of recommendations
If you have longitudinal data, evaluate whether households that
followed each recommendation type actually saw HFFI improvement. Use
propensity score matching or difference-in-differences.

### C4. Fairness audit
Check whether HFFI assigns systematically higher fragility scores to
demographic subgroups (race, age, income bracket). If so, document the
disparity and discuss whether it's a feature or a bug.

### C5. SHAP / LIME explanations
The current explainability is a contribution waterfall. Upgrade to
proper SHAP values per household — this is what the IEEE draft
already promises but doesn't deliver.

---

## D. Quant strategy extensions (hedge fund material)

### D1. HFFI-as-alpha
Aggregate HFFI by ZIP code or state. Test whether regions with rapidly
rising HFFI underperform in consumer-discretionary stocks 6–12 months
later. If yes, this is a publishable alpha signal — and the bridge from
"household fragility tool" to "macro hedge fund signal" the original
transcript imagined.

### D2. Backtested portfolio strategies
For each risk band, run a 20-year backtest of the suggested allocation
template against benchmarks (60/40, S&P 500). Report Sharpe, max
drawdown, time-to-recovery. This makes the recommendation engine
defensible to a quant audience.

### D3. Regime-conditional alpha
Show that different recommendation templates outperform in different
macro regimes — the recession template beats 60/40 in 2008/2020, the
growth template beats it in 2017/2019. This is a clean story for any
quant interview.

### D4. Risk parity meets fragility
Replace the static allocation templates with risk-parity allocations
that target a household-fragility-adjusted risk budget. The household's
HFFI score sets the maximum risk contribution per asset class.

---

## E. Productization (if this becomes a real product)

### E1. Branding + positioning
"Bloomberg Terminal for Households" is the elevator pitch. The actual
positioning depends on the buyer:
  - Direct-to-consumer: "Free financial vulnerability score. Like Credit
    Karma, but for actual financial resilience."
  - B2B (banks): "Pre-default early-warning signal for your loan book."
  - B2B (regulators): "Macro fragility heatmap for policy targeting."

### E2. Compliance
Anything that suggests specific securities or actions to a retail user
is regulated as financial advice. Either get a Registered Investment
Advisor license, or position the recommendations as "general
educational guidance" and have an attorney review the disclaimers.

### E3. Pricing
- B2C: Freemium. Free basic score, paid tier for what-if simulator and
  alerts. ~$5–15/month.
- B2B: Per-API-call (e.g., $0.01 per household scored) or annual seat
  license ($10k–$100k/year).

---

## F. What I'd actually do next (recommendations)

If I were sitting where you're sitting, I'd prioritize like this:

1. **Run the prototype and screenshot it** — for the paper figures.
2. **Calibrate on real SCF data** — strengthens every empirical claim
   in the paper without changing any code structure.
3. **Add HFFI-as-alpha (D1)** — backtests are easy; if the signal works,
   you have a Quant Finance project, not just a Data Science project.
4. **Submit the paper.**
5. **Build the multi-tenant version (B1) only if a startup is the goal.**

The first four steps fit in your remaining semester. The rest is for
after.
