"""Filesystem locations shared by every stage."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
OUTPUT_DIR = Path(os.environ.get("TRADE_PIPELINE_OUTPUT", REPO_ROOT / "spark" / "output"))

BRONZE_DIR = OUTPUT_DIR / "bronze"
SILVER_DIR = OUTPUT_DIR / "silver"
GOLD_DIR = OUTPUT_DIR / "gold"
RESULTS_DIR = OUTPUT_DIR / "results"

# Analysis window shared by the firm-level financial data and GDP.
FIRST_YEAR, LAST_YEAR = 2013, 2023
