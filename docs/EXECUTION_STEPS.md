# Execution steps for this version

```bash
cd hffi_terminal
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add optional keys:

```bash
NEWSAPI_KEY=optional
MARKET_PROVIDER=yfinance
```

Initialize/run the validation suite:

```bash
python scripts/run_academic_validation.py
```

Outputs are written to:

- `outputs/walk_forward_rf.csv`
- `outputs/walk_forward_gb.csv`
- `outputs/spy_benchmark.csv`
- `outputs/recommendation_backtest_summary.csv`
- `data/hffi_terminal.sqlite3`

Run the terminal:

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar theme switch for Light/Dark mode. Open the **Validation** tab to rerun the walk-forward table, SPY benchmark, and recommendation backtest from the UI.

For SCF calibration:

1. Download public SCF microdata.
2. Clean it into `data/scf_households.csv` with `L,D,E,P,M,financial_distress` columns.
3. Run:

```python
from hffi_core.scf_calibration import load_scf_microdata, calibrate_hffi_weights_from_scf
scf = load_scf_microdata()
print(calibrate_hffi_weights_from_scf(scf))
```
