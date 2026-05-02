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


@dataclass(slots=True)
class DeltaOptimizeConfig:
    source_uri: str
    z_order_columns: list[str] | None = None
    target_size_bytes: int | None = None
    log_path: Path = Path("logs/delta_optimize.log")


@dataclass(slots=True)
class DeltaVacuumConfig:
    source_uri: str
    retention_hours: int = 168
    dry_run: bool = True
    enforce_retention_duration: bool = True
    full: bool = False
    log_path: Path = Path("logs/delta_vacuum.log")


@dataclass(slots=True)
class DeltaTimeTravelConfig:
    source_uri: str
    version: int = 0
    limit: int = 5
    log_path: Path = Path("logs/delta_time_travel.log")
