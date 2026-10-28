"""Central path definitions for the Tc-screening project (single source of truth)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_KAGGLE = DATA_RAW / "supercon_kaggle"
DATA_UCI = DATA_RAW / "supercon_uci"
DATA_CURATED = DATA_DIR / "curation"

DOCS_DIR = PROJECT_ROOT / "docs"
GATE_REPORTS = DOCS_DIR / "gate_reports"

SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_LOGS = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FUTURE_SCOPE_DIR = PROJECT_ROOT / "future_scope"

MANIFEST_PATH = DATA_CURATED / "data_manifest.json"

for _d in (DATA_KAGGLE, DATA_UCI, DATA_CURATED, GATE_REPORTS, RESULTS_FIGURES,
           RESULTS_TABLES, RESULTS_LOGS, CHECKPOINTS_DIR, NOTEBOOKS_DIR,
           FUTURE_SCOPE_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
