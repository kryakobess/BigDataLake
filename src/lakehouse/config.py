from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BronzeConfig:
    source_path: Path
    target_uri: str
    years: list[int] | None = None
    dates: list[date] | None = None
    force: bool = False
    infer_schema_length: int = 10_000
    log_path: Path = Path("logs/bronze.log")


@dataclass(slots=True)
class SilverConfig:
    source_uri: str
    target_uri: str
    delay_threshold_minutes: int = 15
    log_path: Path = Path("logs/silver.log")


@dataclass(slots=True)
class GoldConfig:
    source_uri: str
    analytics_target_uri: str
    features_target_uri: str
    log_path: Path = Path("logs/gold.log")


@dataclass(slots=True)
class MLConfig:
    source_uri: str
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "flight-delay-lakehouse"
    log_path: Path = Path("logs/ml.log")


@dataclass(slots=True)
class ReportPlotsConfig:
    analytics_source_uri: str
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "flight-delay-lakehouse"
    run_id: str | None = None
    output_dir: Path = Path("reports/figures")
    log_path: Path = Path("logs/report_plots.log")
