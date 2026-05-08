# Data Science Recommendation Layer

The HFFI Terminal now has an additive data-science recommendation layer on top
of the original rule-based HFFI system.

Pipeline:

1. Household feature engineering
2. Market feature engineering
3. Household segmentation
4. Supervised ML suitability model
5. HFFI-aware recommendation scoring
6. Position sizing from monthly buying capacity
7. Explainable comments and model-performance reporting

Core engineered features:

- debt service ratio
- debt-to-income ratio
- six-month liquidity buffer
- Equity/Bond/Liquid Savings allocation
- monthly buying capacity
- macro stress index
- portfolio volatility
- expected drawdown
- concentration
- market return, volatility, drawdown, momentum, and safety

Model:

- Random Forest
- Gradient Boosting
- Ensemble probability

Final data-science score:

```text
DS score =
35% ML probability
+ 25% HFFI risk capacity
+ 20% allocation gap
+ 20% downside protection
```

Recommendations remain guarded by HFFI. A fragile household will not be pushed
into equity risk just because a market signal is positive.
