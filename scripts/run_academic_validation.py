from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hffi_core.validation_runner import make_synthetic_market_panel, run_walk_forward_and_benchmark, backtest_recommendations

if __name__ == "__main__":
    panel, _ = make_synthetic_market_panel()
    Path("outputs").mkdir(exist_ok=True)
    panel.to_csv("outputs/synthetic_market_panel.csv", index=False)
    results = run_walk_forward_and_benchmark()
    print("\nRF walk-forward validation")
    print(results["rf"].to_string(index=False))
    print("\nGB walk-forward validation")
    print(results["gb"].to_string(index=False))
    print("\nSPY benchmark")
    print(results["spy"])
    print("\nRecommendation backtest")
    print(backtest_recommendations(panel).to_string(index=False))
