from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from deltalake import DeltaTable
from deltalake.writer import write_deltalake

from lakehouse.config import SilverConfig


LOGGER = logging.getLogger(__name__)
SILVER_PARTITIONS = ["year", "month"]
MIN_DELAY_MINUTES = -120.0
MAX_DELAY_MINUTES = 1440.0
MAX_TAXI_MINUTES = 300.0
MAX_ELAPSED_MINUTES = 1500.0


def run_silver(config: SilverConfig) -> None:
    setup_logging(config.log_path)

    try:
        DeltaTable(config.source_uri)
    except Exception as exc:
        raise FileNotFoundError(f"Bronze Delta table was not found: {config.source_uri}") from exc

    silver_scan = build_silver_plan(config)
    LOGGER.info("Silver transformation plan:\n%s", silver_scan.explain())

    silver_frame = silver_scan.collect()
    if silver_frame.is_empty():
        raise ValueError("Silver pipeline produced an empty dataset after cleaning.")

    LOGGER.info(
        "Silver dataset prepared with rows=%s unique_flights=%s",
        silver_frame.height,
        silver_frame.get_column("flight_id").n_unique(),
    )

    write_or_merge_silver(silver_frame, config.target_uri)

    target_table = DeltaTable(config.target_uri)
    LOGGER.info(
        "Silver load finished. delta_version=%s rows=%s",
        target_table.version(),
        silver_frame.height,
    )


def build_silver_plan(config: SilverConfig) -> pl.LazyFrame:
    bronze_scan = pl.scan_delta(config.source_uri)
    bronze_schema = set(bronze_scan.collect_schema().names())

    flight_date_candidates: list[pl.Expr] = []
    if "flightdate" in bronze_schema:
        flight_date_candidates.append(pl.col("flightdate").cast(pl.Date, strict=False))
    if "source_date" in bronze_schema:
        flight_date_candidates.append(pl.col("source_date").cast(pl.Date, strict=False))
    if not flight_date_candidates:
        raise ValueError("Bronze table does not contain flightdate or source_date columns.")

    flight_date_expr = pl.coalesce(flight_date_candidates)
    month_expr = flight_date_expr.dt.month().cast(pl.Int8)
    dep_hour_expr = extract_hour_expr("crsdeptime")
    arr_hour_expr = extract_hour_expr("crsarrtime")

    silver_scan = (
        bronze_scan
        .with_columns(
            flight_date_expr.alias("flight_date"),
            normalize_code_expr("marketing_airline_network").alias("marketing_airline"),
            normalize_code_expr("operating_airline").alias("operating_airline"),
            normalize_code_expr("iata_code_marketing_airline").alias("marketing_airline_iata"),
            normalize_code_expr("tail_number").alias("tail_number"),
            normalize_code_expr("origin").alias("origin"),
            normalize_code_expr("dest").alias("dest"),
            normalize_text_expr("origincityname").alias("origin_city_name"),
            normalize_text_expr("destcityname").alias("dest_city_name"),
            normalize_code_expr("originstate").alias("origin_state"),
            normalize_code_expr("deststate").alias("dest_state"),
            pl.col("flight_number_marketing_airline").cast(pl.Int32, strict=False).alias("flight_number"),
            pl.col("quarter").cast(pl.Int8, strict=False).alias("quarter"),
            pl.col("deptimeblk").cast(pl.Utf8, strict=False).alias("dep_time_block"),
            pl.col("arrtimeblk").cast(pl.Utf8, strict=False).alias("arr_time_block"),
            pl.col("crsdeptime").cast(pl.Int32, strict=False).alias("crs_dep_time"),
            pl.col("crsarrtime").cast(pl.Int32, strict=False).alias("crs_arr_time"),
            pl.col("deptime").cast(pl.Int32, strict=False).alias("dep_time"),
            pl.col("arrtime").cast(pl.Int32, strict=False).alias("arr_time"),
            pl.col("depdelay").cast(pl.Float32, strict=False).alias("dep_delay"),
            pl.col("depdelayminutes").cast(pl.Float32, strict=False).alias("dep_delay_minutes"),
            pl.col("arrdelay").cast(pl.Float32, strict=False).alias("arr_delay"),
            pl.col("arrdelayminutes").cast(pl.Float32, strict=False).alias("arr_delay_minutes"),
            pl.col("taxiout").cast(pl.Float32, strict=False).alias("taxi_out"),
            pl.col("taxiin").cast(pl.Float32, strict=False).alias("taxi_in"),
            pl.col("airtime").cast(pl.Float32, strict=False).alias("air_time"),
            pl.col("crselapsedtime").cast(pl.Float32, strict=False).alias("crs_elapsed_time"),
            pl.col("actualelapsedtime").cast(pl.Float32, strict=False).alias("actual_elapsed_time"),
            pl.col("distance").cast(pl.Float32, strict=False).alias("distance"),
            pl.col("distancegroup").cast(pl.Int16, strict=False).alias("distance_group"),
            pl.col("carrierdelay").cast(pl.Float32, strict=False).fill_null(0.0).alias("carrier_delay"),
            pl.col("weatherdelay").cast(pl.Float32, strict=False).fill_null(0.0).alias("weather_delay"),
            pl.col("nasdelay").cast(pl.Float32, strict=False).fill_null(0.0).alias("nas_delay"),
            pl.col("securitydelay").cast(pl.Float32, strict=False).fill_null(0.0).alias("security_delay"),
            pl.col("lateaircraftdelay").cast(pl.Float32, strict=False).fill_null(0.0).alias("late_aircraft_delay"),
            pl.col("cancelled").cast(pl.Int8, strict=False).fill_null(0).alias("cancelled"),
            pl.col("diverted").cast(pl.Int8, strict=False).fill_null(0).alias("diverted"),
            pl.col("source_file").cast(pl.Utf8, strict=False),
            pl.col("ingested_at").cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False),
            pl.col("ingestion_batch").cast(pl.Utf8, strict=False),
        )
        .with_columns(
            pl.coalesce([pl.col("operating_airline"), pl.col("marketing_airline")]).alias("operating_airline"),
            pl.coalesce([pl.col("tail_number"), pl.lit("UNKNOWN")]).alias("tail_number"),
            flight_date_expr.dt.year().cast(pl.Int32).alias("year"),
            month_expr.alias("month"),
            flight_date_expr.dt.day().cast(pl.Int8).alias("day_of_month"),
            flight_date_expr.dt.weekday().cast(pl.Int8).alias("day_of_week"),
            dep_hour_expr.alias("dep_hour"),
            arr_hour_expr.alias("arr_hour"),
            season_expr(month_expr).alias("season"),
            pl.concat_str([pl.col("origin"), pl.col("dest")], separator="-").alias("route"),
        )
        .filter(
            pl.col("flight_date").is_not_null()
            & pl.col("marketing_airline").is_not_null()
            & pl.col("origin").is_not_null()
            & pl.col("dest").is_not_null()
            & pl.col("flight_number").is_not_null()
            & pl.col("crs_dep_time").is_not_null()
            & pl.col("crs_arr_time").is_not_null()
            & (pl.col("cancelled") == 0)
            & (pl.col("diverted") == 0)
            & pl.col("dep_delay").is_not_null()
            & pl.col("arr_delay").is_not_null()
            & pl.col("distance").is_not_null()
            & (pl.col("distance") > 0)
            & pl.col("dep_delay").is_between(MIN_DELAY_MINUTES, MAX_DELAY_MINUTES, closed="both")
            & pl.col("arr_delay").is_between(MIN_DELAY_MINUTES, MAX_DELAY_MINUTES, closed="both")
            & (
                pl.col("taxi_out").is_null()
                | pl.col("taxi_out").is_between(0.0, MAX_TAXI_MINUTES, closed="both")
            )
            & (
                pl.col("taxi_in").is_null()
                | pl.col("taxi_in").is_between(0.0, MAX_TAXI_MINUTES, closed="both")
            )
            & (
                pl.col("actual_elapsed_time").is_null()
                | pl.col("actual_elapsed_time").is_between(1.0, MAX_ELAPSED_MINUTES, closed="both")
            )
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("flight_date").cast(pl.Utf8),
                    pl.col("marketing_airline"),
                    pl.col("flight_number").cast(pl.Utf8),
                    pl.col("origin"),
                    pl.col("dest"),
                    pl.col("crs_dep_time").cast(pl.Utf8),
                    pl.col("tail_number"),
                ],
                separator="|",
            ).alias("flight_id"),
            (pl.col("arr_delay") > float(config.delay_threshold_minutes)).cast(pl.Int8).alias("is_delayed"),
        )
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
            "origin_city_name",
            "origin_state",
            "dest",
            "dest_city_name",
            "dest_state",
            "route",
            "crs_dep_time",
            "dep_time",
            "dep_hour",
            "dep_time_block",
            "crs_arr_time",
            "arr_time",
            "arr_hour",
            "arr_time_block",
            "dep_delay",
            "dep_delay_minutes",
            "arr_delay",
            "arr_delay_minutes",
            "taxi_out",
            "taxi_in",
            "air_time",
            "crs_elapsed_time",
            "actual_elapsed_time",
            "distance",
            "distance_group",
            "carrier_delay",
            "weather_delay",
            "nas_delay",
            "security_delay",
            "late_aircraft_delay",
            "is_delayed",
            "source_file",
            "ingested_at",
            "ingestion_batch",
        )
        .unique(subset=["flight_id"], keep="last")
        .sort(["flight_date", "flight_id"])
    )

    return silver_scan


def write_or_merge_silver(silver_frame: pl.DataFrame, target_uri: str) -> None:
    if not delta_table_exists(target_uri):
        write_deltalake(
            target_uri,
            silver_frame.to_arrow(),
            mode="append",
            partition_by=SILVER_PARTITIONS,
            schema_mode="merge",
        )
        return

    target_table = DeltaTable(target_uri)
    (
        target_table.merge(
            silver_frame.to_arrow(),
            predicate="target.flight_id = source.flight_id",
            source_alias="source",
            target_alias="target",
            merge_schema=True,
        )
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .when_not_matched_by_source_delete()
        .execute()
    )


def normalize_code_expr(column_name: str) -> pl.Expr:
    cleaned = pl.col(column_name).cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
    return pl.when(cleaned == "").then(None).otherwise(cleaned)


def normalize_text_expr(column_name: str) -> pl.Expr:
    cleaned = pl.col(column_name).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(cleaned == "").then(None).otherwise(cleaned)


def extract_hour_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Int32, strict=False)
        .mod(2400)
        .floordiv(100)
        .cast(pl.Int8)
    )


def season_expr(month_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(month_expr.is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(month_expr.is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(month_expr.is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("fall"))
    )


def delta_table_exists(target_uri: str) -> bool:
    try:
        DeltaTable(target_uri)
    except Exception:
        return False
    return True


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
