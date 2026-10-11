"""Paths and shared settings. Override locations with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    duckdb: Path
    reports: Path

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def app_data(self) -> Path:
        return self.reports / "app_data"

    @property
    def results_json(self) -> Path:
        return self.reports / "results.json"


def get_paths() -> Paths:
    return Paths(
        raw_dir=Path(os.environ.get("CRC_RAW_DIR", ROOT / "data" / "raw")),
        duckdb=Path(os.environ.get("CRC_DUCKDB_PATH", ROOT / "data" / "warehouse.duckdb")),
        reports=Path(os.environ.get("CRC_REPORTS_DIR", ROOT / "reports")),
    )
