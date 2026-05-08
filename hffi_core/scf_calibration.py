"""SCF calibration hook for reviewer-facing external validity.

The public Survey of Consumer Finances files are not bundled in this project.
Place cleaned SCF household microdata at `data/scf_households.csv` with columns
that map to the HFFI household fields, then run `calibrate_hffi_weights_from_scf`.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SCF_COMPONENT_COLS = ["L", "D", "E", "P", "M", "LD"]


def load_scf_microdata(path: str | Path = "data/scf_households.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("SCF microdata not found. Download public SCF data, clean it, and save as data/scf_households.csv")
    return pd.read_csv(p)


def calibrate_hffi_weights_from_scf(df: pd.DataFrame, target_col: str = "financial_distress") -> Dict[str, float]:
    """Fit logistic weights on real SCF distress indicators.

    Required columns: L, D, E, P, M and target_col. LD is created if absent.
    Returns normalized absolute coefficient weights suitable for DEFAULT_WEIGHTS review.
    """
    d = df.copy()
    if "LD" not in d.columns:
        d["LD"] = d["L"] * d["D"]
    missing = [c for c in SCF_COMPONENT_COLS + [target_col] if c not in d.columns]
    if missing:
        raise ValueError(f"Missing SCF calibration columns: {missing}")
    X = d[SCF_COMPONENT_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = d[target_col].astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X, y)
    coefs = model.named_steps["logisticregression"].coef_[0]
    weights = np.abs(coefs) / np.abs(coefs).sum()
    out = dict(zip(["liquidity", "debt", "expenses", "portfolio", "macro", "interaction"], weights.round(4)))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/scf_calibrated_weights.json").write_text(json.dumps(out, indent=2))
    return out
