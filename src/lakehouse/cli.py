from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from lakehouse.bronze import run_bronze
from lakehouse.config import (
    BronzeConfig,
    DeltaOptimizeConfig,
    DeltaTimeTravelConfig,
    DeltaVacuumConfig,
    GoldConfig,
    MLConfig,
    ReportPlotsConfig,
    SilverConfig,
)
from lakehouse.delta_ops import run_delta_optimize, run_delta_time_travel, run_delta_vacuum
from lakehouse.gold import run_gold
from lakehouse.silver import run_silver


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bronze":
        config = BronzeConfig(
            source_path=Path(args.source),
            target_uri=args.target,
            years=parse_years(args.years),
            dates=parse_dates(args.dates),
            force=args.force,
        )
        run_bronze(config)
        return 0

    if args.command == "silver":
        config = SilverConfig(
            source_uri=args.source,
            target_uri=args.target,
            delay_threshold_minutes=args.delay_threshold,
        )
        run_silver(config)
        return 0

    if args.command == "gold":
        config = GoldConfig(
            source_uri=args.source,
            analytics_target_uri=args.analytics_target,
            features_target_uri=args.features_target,
        )
        run_gold(config)
        return 0

    if args.command == "ml":
        from lakehouse.ml import run_ml

        config = MLConfig(
            source_uri=args.source,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
        run_ml(config)
        return 0

    if args.command == "report-plots":
        from lakehouse.report_plots import run_report_plots

        config = ReportPlotsConfig(
            analytics_source_uri=args.analytics_source,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
        )
        run_report_plots(config)
        return 0

    if args.command == "delta-optimize":
        config = DeltaOptimizeConfig(
            source_uri=args.source,
            z_order_columns=parse_csv_items(args.z_order_cols),
            target_size_bytes=megabytes_to_bytes(args.target_size_mb),
        )
        run_delta_optimize(config)
        return 0

    if args.command == "delta-vacuum":
        config = DeltaVacuumConfig(
            source_uri=args.source,
            retention_hours=args.retention_hours,
            dry_run=not args.execute,
            enforce_retention_duration=not args.disable_retention_check,
            full=args.full,
        )
        run_delta_vacuum(config)
        return 0

    if args.command == "delta-time-travel":
        config = DeltaTimeTravelConfig(
            source_uri=args.source,
            version=args.version,
            limit=args.limit,
        )
        run_delta_time_travel(config)
        return 0

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lakehouse CLI")
    subparsers = parser.add_subparsers(dest="command")

    bronze_parser = subparsers.add_parser(
        "bronze",
        help="Load raw flight CSV batches into a bronze Delta table.",
    )
    bronze_parser.add_argument(
        "--source",
        default="flight_data_2018_2024.csv",
        help="Path to the source CSV file.",
    )
    bronze_parser.add_argument(
        "--target",
        default="data/lakehouse/bronze/flights",
        help="Destination Delta table URI.",
    )
    bronze_parser.add_argument(
        "--years",
        default=None,
        help="Optional comma-separated list of years to read from the source.",
    )
    bronze_parser.add_argument(
        "--dates",
        default=None,
        help="Optional comma-separated list of flight dates to load, for example 2024-01-14,2024-01-15.",
    )
    bronze_parser.add_argument(
        "--force",
        action="store_true",
        help="Append the selected daily batches even if they already exist in bronze.",
    )

    silver_parser = subparsers.add_parser(
        "silver",
        help="Transform bronze Delta data into a curated silver Delta table.",
    )
    silver_parser.add_argument(
        "--source",
        default="data/lakehouse/bronze/flights",
        help="Source bronze Delta table URI.",
    )
    silver_parser.add_argument(
        "--target",
        default="data/lakehouse/silver/flights",
        help="Destination silver Delta table URI.",
    )
    silver_parser.add_argument(
        "--delay-threshold",
        type=int,
        default=15,
        help="Arrival delay threshold in minutes for the is_delayed flag.",
    )

    gold_parser = subparsers.add_parser(
        "gold",
        help="Build gold analytics and ML feature marts from the silver Delta table.",
    )
    gold_parser.add_argument(
        "--source",
        default="data/lakehouse/silver/flights",
        help="Source silver Delta table URI.",
    )
    gold_parser.add_argument(
        "--analytics-target",
        default="data/lakehouse/gold/analytics",
        help="Destination Delta table URI for the analytics mart.",
    )
    gold_parser.add_argument(
        "--features-target",
        default="data/lakehouse/gold/features",
        help="Destination Delta table URI for the ML feature mart.",
    )

    ml_parser = subparsers.add_parser(
        "ml",
        help="Train and compare ML models on the gold feature Delta table and log runs to MLflow.",
    )
    ml_parser.add_argument(
        "--source",
        default="data/lakehouse/gold/features",
        help="Source gold feature Delta table URI.",
    )
    ml_parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    ml_parser.add_argument(
        "--experiment-name",
        default="flight-delay-lakehouse",
        help="MLflow experiment name.",
    )

    report_plots_parser = subparsers.add_parser(
        "report-plots",
        help="Generate analytics and ML plots for the final README report.",
    )
    report_plots_parser.add_argument(
        "--analytics-source",
        default="data/lakehouse/gold/analytics",
        help="Source gold analytics Delta table URI.",
    )
    report_plots_parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    report_plots_parser.add_argument(
        "--experiment-name",
        default="flight-delay-lakehouse",
        help="MLflow experiment name used to locate the latest parent run.",
    )
    report_plots_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional MLflow parent run id. If omitted, the latest finished ml_pipeline run is used.",
    )
    report_plots_parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory where report figures will be saved.",
    )

    delta_optimize_parser = subparsers.add_parser(
        "delta-optimize",
        help="Run Delta OPTIMIZE compaction and optional Z-ORDER on a table.",
    )
    delta_optimize_parser.add_argument(
        "--source",
        default="data/lakehouse/bronze/flights",
        help="Target Delta table URI.",
    )
    delta_optimize_parser.add_argument(
        "--z-order-cols",
        default=None,
        help="Optional comma-separated list of columns for Z-ORDER, for example source_date,origin,dest.",
    )
    delta_optimize_parser.add_argument(
        "--target-size-mb",
        type=int,
        default=None,
        help="Optional target file size in megabytes for OPTIMIZE.",
    )

    delta_vacuum_parser = subparsers.add_parser(
        "delta-vacuum",
        help="Run Delta VACUUM on a table.",
    )
    delta_vacuum_parser.add_argument(
        "--source",
        default="data/lakehouse/bronze/flights",
        help="Target Delta table URI.",
    )
    delta_vacuum_parser.add_argument(
        "--retention-hours",
        type=int,
        default=168,
        help="Retention period in hours.",
    )
    delta_vacuum_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete stale files. Without this flag VACUUM runs in dry-run mode.",
    )
    delta_vacuum_parser.add_argument(
        "--disable-retention-check",
        action="store_true",
        help="Disable the minimum retention duration safety check.",
    )
    delta_vacuum_parser.add_argument(
        "--full",
        action="store_true",
        help="Scan the full table directory during VACUUM.",
    )

    delta_time_travel_parser = subparsers.add_parser(
        "delta-time-travel",
        help="Read a historical Delta table version and print a sample.",
    )
    delta_time_travel_parser.add_argument(
        "--source",
        default="data/lakehouse/bronze/flights",
        help="Target Delta table URI.",
    )
    delta_time_travel_parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Historical Delta version to load.",
    )
    delta_time_travel_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows to show from the historical snapshot.",
    )

    return parser


def parse_years(value: str | None) -> list[int] | None:
    if value is None:
        return None
    years = [item.strip() for item in value.split(",") if item.strip()]
    return [int(year) for year in years]


def parse_dates(value: str | None) -> list[date] | None:
    if value is None:
        return None
    dates = [item.strip() for item in value.split(",") if item.strip()]
    return [date.fromisoformat(item) for item in dates]


def parse_csv_items(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def megabytes_to_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    return value * 1024 * 1024
