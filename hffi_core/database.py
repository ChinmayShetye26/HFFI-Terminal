"""SQLite persistence for HFFI Terminal.

Stores household runs, recommendations, and validation results locally.
This gives the project a reproducible audit trail without requiring
external infrastructure.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DB_PATH = Path("data/hffi_terminal.sqlite3")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS household_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    household_id TEXT,
    hffi_score REAL NOT NULL,
    risk_band TEXT NOT NULL,
    distress_probability REAL,
    inputs_json TEXT NOT NULL,
    macro_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    created_at TEXT NOT NULL,
    household_id TEXT,
    market TEXT NOT NULL,
    ticker TEXT,
    recommendation_score REAL NOT NULL,
    action TEXT,
    rationale TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES household_runs(id)
);
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    fold INTEGER,
    train_start TEXT,
    train_end TEXT,
    test_start TEXT,
    test_end TEXT,
    auc REAL,
    accuracy REAL,
    n_train INTEGER,
    n_test INTEGER,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id INTEGER,
    user_message TEXT NOT NULL,
    assistant_reply TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    reason TEXT,
    FOREIGN KEY(run_id) REFERENCES household_runs(id)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str = DB_PATH) -> Path:
    with connect(db_path):
        pass
    return Path(db_path)


def save_household_run(result: Any, inputs: Any, macro: Dict[str, Any], household_id: Optional[str] = None, db_path: Path | str = DB_PATH) -> int:
    inputs_json = json.dumps(getattr(inputs, "__dict__", str(inputs)), default=str)
    macro_json = json.dumps(macro, default=str)
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO household_runs(created_at, household_id, hffi_score, risk_band, distress_probability, inputs_json, macro_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), household_id, float(result.score), str(result.band), float(result.distress_probability), inputs_json, macro_json),
        )
        return int(cur.lastrowid)


def save_recommendations(records: Iterable[Dict[str, Any]], run_id: Optional[int] = None, household_id: Optional[str] = None, db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as conn:
        for r in records:
            conn.execute(
                """INSERT INTO recommendations(created_at, run_id, household_id, market, ticker, recommendation_score, action, rationale, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now_iso(), run_id, household_id, r.get("market"), r.get("ticker"), float(r.get("recommendation_score", r.get("suitability_score", 0))), r.get("action"), r.get("rationale", r.get("plain_language_reason", "")), json.dumps(r, default=str)),
            )


def save_validation_table(df, experiment_name: str, db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as conn:
        for i, row in df.reset_index(drop=True).iterrows():
            d = row.to_dict()
            conn.execute(
                """INSERT INTO validation_results(created_at, experiment_name, fold, train_start, train_end, test_start, test_end, auc, accuracy, n_train, n_test, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now_iso(), experiment_name, int(i + 1), str(d.get("train_start")), str(d.get("train_end")), str(d.get("test_start")), str(d.get("test_end")), float(d.get("auc", 0)), float(d.get("accuracy", 0)), int(d.get("n_train", 0)), int(d.get("n_test", 0)), json.dumps(d, default=str)),
            )


def save_chat(run_id: Optional[int], user_message: str, assistant_reply: str, allowed: bool, reason: str = "", db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO chat_audit(created_at, run_id, user_message, assistant_reply, allowed, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now_iso(), run_id, user_message[:2000], assistant_reply[:4000], int(allowed), reason[:500]),
        )
