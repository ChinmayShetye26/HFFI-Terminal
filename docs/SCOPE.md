# HFFI Terminal — Tightened Scope for Final Semester

## Honest assessment of the original scope

The conversation in `Quant_Research_in_Finance.pdf` accumulated, over 446
pages, a vision that included:

- Households + companies + sectors + countries (multi-entity)
- Bloomberg-style terminal with multi-user access layers
- Recommendation engine with reinforcement learning
- Graph Neural Network contagion modeling
- Real-time data ingestion across all global markets
- Mobile app, web app, desktop terminal, REST API, B2B integrations

**This is a 2–3 year build by a small team.** It is not a final-semester
data science project. Trying to ship all of it would produce a thin,
unfinished version of everything and fail the academic submission.

The scope below is what **one student can actually finish**, defend, and
turn in — while still demonstrating every Data Science discipline the
project is meant to showcase.

---

## In scope (semester deliverables)

### A. Core HFFI library — DONE in this codebase
- ✅ Five-component fragility model (L, D, E, P, M)
- ✅ Interaction term (L × D)
- ✅ Risk-band mapping + distress probability calibration
- ✅ Weight learning via PCA (unsupervised) and LogReg (supervised)
- ✅ Persistent weights (YAML)

### B. Validation harness — DONE
- ✅ Out-of-sample 5-fold AUC
- ✅ Baseline comparison (Liquidity-only, Debt-only, Equal-weight, LogReg)
- ✅ Sensitivity analysis (±20% weight perturbations, Spearman ρ, band-flip rate)
- ✅ Calibration table (deciles)

### C. Stress testing — DONE
- ✅ Seven named scenarios (recessions, stagflation, job loss, rate shock, market crash)
- ✅ Monte Carlo with 5,000 sampled shock combinations
- ✅ VaR-style outputs (5th/95th percentiles, P(severe))

### D. Recommendation engine — DONE (rule-based + content-based)
- ✅ Rule-based: priority-ranked actions per fragility component
- ✅ Content-based: allocation templates per risk band
- ✅ Explainability: each rec links to which component triggered it
- ❌ Reinforcement learning — **deferred** (requires sequential interaction data we don't have)

### E. Live data integration — DONE (with provider flexibility)
- ✅ FRED macro (inflation, fed funds, unemployment, mortgage, treasury, VIX)
- ✅ Market data (yfinance free / Polygon paid / Alpaca paid — runtime switch)
- ✅ News (NewsAPI + RSS fallback)
- ✅ Caching + rate-limit awareness

### F. Terminal UI — DONE
- ✅ Five-panel Streamlit app (Score / Stress / Recommendations / Macro / Markets+News)
- ✅ Interactive sliders for shock scenarios
- ✅ Plotly visualizations (gauge, breakdown, waterfall, scenario bar, MC histogram)

### G. Empirical validation on synthetic data — DONE
- ✅ Population generator (5,000 households, configurable)
- ✅ Synthetic distress label with known ground-truth coefficients
- ✅ Demonstrates HFFI beats all simple baselines (0.751 vs 0.665–0.679)

### H. IEEE paper updates — DOCUMENTED in `PAPER_REVIEW.md`
- ✅ Concrete fixes to existing draft (mapped section by section)
- ✅ Empirical results section (validation tables ready to paste)
- ✅ Limitations section
- ✅ Proper literature review

---

## Out of scope (defer to future work, mention in paper)

These are listed as future enhancements in `EXTENSIONS.md` and in the
paper's "Future Work" section:

- **Multi-entity fragility (companies, sectors, countries).** Different
  data, different formulas, different stakeholders. A separate paper.
- **Real SCF microdata calibration.** SCF is publicly available but
  requires careful imputation (5 implicates), survey weighting, and
  cohort tracking. A 1–2 month project on its own.
- **Reinforcement learning for sequential recommendations.** Requires
  longitudinal household interaction data we don't have access to.
- **Graph neural networks for contagion.** Requires household-network
  data (zip-code clusters, employer networks). Not available at scale.
- **Mobile app / multi-tenant SaaS.** Engineering, not research.
- **Real broker integration for portfolio rebalancing.** Compliance + KYC
  requirements outside the academic project.

---

## What this means for the paper

The paper claims the project delivers:

1. **A novel composite fragility index** combining five dimensions and
   an interaction term — *demonstrated*.
2. **A principled weight-learning approach** (PCA + LogReg) replacing
   arbitrary weights — *demonstrated*.
3. **A validation methodology** showing the index beats simpler
   baselines and is robust to weight perturbation — *demonstrated*.
4. **An end-to-end implementation** with live macro + market integration
   — *demonstrated*.

Every claim in the paper must map to a working piece of code in this
repository. No hand-waving.
