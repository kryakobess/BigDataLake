from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from deltalake import DeltaTable
from deltalake.writer import write_deltalake

from lakehouse.config import GoldConfig


LOGGER = logging.getLogger(__name__)
ANALYTICS_PARTITIONS = ["aggregation_level", "year", "month"]
FEATURE_PARTITIONS = ["year", "month"]


def run_gold(config: GoldConfig) -> None:
    setup_logging(config.log_path)

    try:
        source_table = DeltaTable(config.source_uri)
    except Exception as exc:
        raise FileNotFoundError(f"Silver Delta table was not found: {config.source_uri}") from exc

    silver_version = source_table.version()
    built_at = datetime.now(UTC)

    analytics_scan = build_analytics_plan(config.source_uri, silver_version, built_at)
    features_scan = build_features_plan(config.source_uri, silver_version, built_at)

    LOGGER.info("Gold analytics plan:\n%s", analytics_scan.explain())
    LOGGER.info("Gold features plan:\n%s", features_scan.explain())

    analytics_frame = analytics_scan.collect()
    features_frame = features_scan.collect()

    if analytics_frame.is_empty():
        raise ValueError("Gold analytics mart is empty.")
    if features_frame.is_empty():
        raise ValueError("Gold feature mart is empty.")

    overwrite_delta_table(
        analytics_frame,
        config.analytics_target_uri,
        partition_by=ANALYTICS_PARTITIONS,
    )
    overwrite_delta_table(
        features_frame,
        config.features_target_uri,
        partition_by=FEATURE_PARTITIONS,
    )

    analytics_table = DeltaTable(config.analytics_target_uri)
    features_table = DeltaTable(config.features_target_uri)
    LOGGER.info(
        "Gold load finished. analytics_version=%s analytics_rows=%s features_version=%s features_rows=%s",
        analytics_table.version(),
        analytics_frame.height,
        features_table.version(),
        features_frame.height,
    )


def build_analytics_plan(source_uri: str, silver_version: int, built_at: datetime) -> pl.LazyFrame:
    base_scan = pl.scan_delta(source_uri)

    origin_airport = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "origin"],
        aggregation_level="origin_airport",
        aggregation_key_column="origin",
    )
    dest_airport = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "dest"],
        aggregation_level="dest_airport",
        aggregation_key_column="dest",
    )
    airline = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "marketing_airline"],
        aggregation_level="airline",
        aggregation_key_column="marketing_airline",
    )
    departure_hour = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "dep_hour"],
        aggregation_level="departure_hour",
        aggregation_key_column="dep_hour",
    )
    day_of_week = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "day_of_week"],
        aggregation_level="day_of_week",
        aggregation_key_column="day_of_week",
    )
    season = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "season"],
        aggregation_level="season",
        aggregation_key_column="season",
    )
    route = build_aggregate_plan(
        base_scan,
        group_columns=["year", "month", "route"],
        aggregation_level="route",
        aggregation_key_column="route",
    )

    return (
        pl.concat(
            [origin_airport, dest_airport, airline, departure_hour, day_of_week, season, route],
            how="diagonal_relaxed",
        )
        .with_columns(
            pl.lit(silver_version).cast(pl.Int64).alias("silver_version"),
            pl.lit(built_at).alias("gold_built_at"),
        )
        .sort(["aggregation_level", "year", "month", "aggregation_key"])
    )


def build_aggregate_plan(
    base_scan: pl.LazyFrame,
    *,
    group_columns: list[str],
    aggregation_level: str,
    aggregation_key_column: str,
) -> pl.LazyFrame:
    return (
        base_scan
        .group_by(group_columns)
        .agg(
            pl.len().alias("flights_count"),
            pl.col("arr_delay").mean().round(2).alias("avg_arr_delay"),
            pl.col("dep_delay").mean().round(2).alias("avg_dep_delay"),
            pl.col("arr_delay").median().round(2).alias("median_arr_delay"),
            pl.col("arr_delay").quantile(0.9).round(2).alias("p90_arr_delay"),
            pl.col("is_delayed").mean().round(4).alias("delayed_rate"),
            pl.col("distance").mean().round(2).alias("avg_distance"),
        )
        .with_columns(
            pl.lit(aggregation_level).alias("aggregation_level"),
            pl.col(aggregation_key_column).cast(pl.Utf8).alias("aggregation_key"),
        )
        .select(
            "aggregation_level",
            "aggregation_key",
            "year",
            "month",
            "flights_count",
            "avg_arr_delay",
            "avg_dep_delay",
            "median_arr_delay",
            "p90_arr_delay",
            "delayed_rate",
            "avg_distance",
        )
    )


def build_features_plan(source_uri: str, silver_version: int, built_at: datetime) -> pl.LazyFrame:
    return (
        pl.scan_delta(source_uri)
        .select(
            "flight_id",
            "flight_date",
            "year",
            "quarter",
            "month",
            "day_of_month",
            "day_of_week",
            "season",
            "marketing_airline",
            "operating_airline",
            "marketing_airline_iata",
            "flight_number",
            "tail_number",
            "origin",
            "origin_state",
            "dest",
            "dest_state",
            "route",
            "crs_dep_time",
            "dep_time",
            "dep_hour",
            "dep_time_block",
            "crs_arr_time",
            "arr_hour",
            "arr_time_block",
            "distance",
            "distance_group",
            "crs_elapsed_time",
            "dep_delay",
            "dep_delay_minutes",
            pl.col("arr_delay").alias("target_arr_delay"),
            pl.col("is_delayed").alias("target_is_delayed"),
            pl.lit(silver_version).cast(pl.Int64).alias("silver_version"),
            pl.lit(built_at).alias("gold_built_at"),
        )
        .sort(["flight_date", "flight_id"])
    )


def overwrite_delta_table(frame: pl.DataFrame, target_uri: str, *, partition_by: list[str]) -> None:
    write_deltalake(
        target_uri,
        frame.to_arrow(),
        mode="overwrite",
        partition_by=partition_by,
        schema_mode="overwrite",
    )


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
