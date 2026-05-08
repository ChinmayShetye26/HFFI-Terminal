"""HFFI Core: Household Financial Fragility Index computation, weight learning,
stress simulation, recommendations, and validation."""

from .components import (
    liquidity_fragility,
    debt_fragility,
    expense_fragility,
    portfolio_fragility,
    macro_fragility,
)
from .scoring import (
    hffi_score,
    risk_band,
    distress_probability,
    compute_household_hffi,
)
from .weights import learn_weights_pca, learn_weights_logreg, save_weights, load_weights
from .stress import monte_carlo_stress, apply_shock_scenarios
from .recommendations import generate_recommendations
from .validation import (
    out_of_sample_eval,
    baseline_comparison,
    sensitivity_analysis,
)

__version__ = "0.1.0"
