from __future__ import annotations

import csv
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
from deltalake import DeltaTable

from lakehouse.config import BronzeConfig


LOGGER = logging.getLogger(__name__)
NULL_VALUES = ["", "NA", "N/A", "NULL", "null"]


def run_bronze(config: BronzeConfig) -> None:
    setup_logging(config.log_path)

    if not config.source_path.exists():
        raise FileNotFoundError(f"Source CSV was not found: {config.source_path}")

    raw_columns = read_csv_header(config.source_path)
    normalized_columns = normalize_columns(raw_columns)
    year_column = detect_year_column(normalized_columns)
    date_column = detect_date_column(normalized_columns)

    base_scan = (
        pl.scan_csv(
            str(config.source_path),
            has_header=True,
            new_columns=normalized_columns,
            infer_schema_length=config.infer_schema_length,
            try_parse_dates=True,
            null_values=NULL_VALUES,
            truncate_ragged_lines=True,
        )
        .with_columns(
            pl.col(year_column).cast(pl.Int32, strict=False),
            pl.col(date_column).cast(pl.Date, strict=False),
        )
    )

    if config.years:
        base_scan = base_scan.filter(pl.col(year_column).is_in(config.years))

    requested_dates = config.dates or discover_dates(base_scan, date_column)
    if not requested_dates:
        raise ValueError("No flight dates were discovered in the source CSV.")

    already_loaded_dates = set()
    if not config.force and delta_table_exists(config.target_uri):
        already_loaded_dates = fetch_loaded_dates(config.target_uri)

    ingestion_batch = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    source_file = config.source_path.name

    LOGGER.info("Bronze load started for dates=%s into %s", requested_dates, config.target_uri)

    for batch_date in requested_dates:
        if not config.force and batch_date in already_loaded_dates:
            LOGGER.info("Skipping date=%s because it already exists in bronze table.", batch_date)
            continue

        batch_timestamp = datetime.now(UTC)
        day_frame = (
            base_scan
            .filter(pl.col(date_column) == batch_date)
            .with_columns(
                pl.lit(batch_date).cast(pl.Date).alias("source_date"),
                pl.col(year_column).cast(pl.Int32, strict=False).alias("source_year"),
                pl.lit(source_file).alias("source_file"),
                pl.lit(batch_timestamp).alias("ingested_at"),
                pl.lit(ingestion_batch).alias("ingestion_batch"),
            )
            .collect()
        )

        if day_frame.is_empty():
            LOGGER.warning("Date=%s produced an empty batch and was skipped.", batch_date)
            continue

        day_frame.write_delta(
            config.target_uri,
            mode="append",
            delta_write_options={"schema_mode": "merge"},
        )
        version = DeltaTable(config.target_uri).version()
        LOGGER.info(
            "Loaded date=%s rows=%s delta_version=%s batch=%s",
            batch_date,
            day_frame.height,
            version,
            ingestion_batch,
        )

    LOGGER.info("Bronze load finished.")


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


def read_csv_header(source_path: Path) -> list[str]:
    with source_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {source_path}") from exc


def normalize_columns(columns: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: dict[str, int] = {}

    for index, original in enumerate(columns):
        candidate = original.strip().lower()
        candidate = re.sub(r"[^a-z0-9]+", "_", candidate)
        candidate = re.sub(r"_+", "_", candidate).strip("_")
        if not candidate:
            candidate = f"unnamed_column_{index}"

        suffix = seen.get(candidate, 0)
        seen[candidate] = suffix + 1
        if suffix:
            candidate = f"{candidate}_{suffix}"

        normalized.append(candidate)

    return normalized


def detect_year_column(columns: list[str]) -> str:
    if "year" in columns:
        return "year"
    raise ValueError("Could not find a year column after normalization.")


def detect_date_column(columns: list[str]) -> str:
    for candidate in ("flightdate", "flight_date"):
        if candidate in columns:
            return candidate
    raise ValueError("Could not find a flight date column after normalization.")


def discover_dates(base_scan: pl.LazyFrame, date_column: str) -> list[date]:
    dates_frame = (
        base_scan
        .select(pl.col(date_column).drop_nulls().unique().sort())
        .collect()
    )
    return list(dates_frame.get_column(date_column).to_list())


def delta_table_exists(target_uri: str) -> bool:
    try:
        DeltaTable(target_uri)
    except Exception:
        return False
    return True


def fetch_loaded_dates(target_uri: str) -> set[date]:
    try:
        delta_scan = pl.scan_delta(target_uri)
        schema_names = set(delta_scan.collect_schema().names())
    except Exception:
        return set()

    if "source_date" in schema_names:
        column_name = "source_date"
    elif "flightdate" in schema_names:
        column_name = "flightdate"
    elif "flight_date" in schema_names:
        column_name = "flight_date"
    else:
        return set()

    dates_frame = (
        delta_scan
        .select(pl.col(column_name).cast(pl.Date, strict=False).drop_nulls().unique())
        .collect()
    )
    return set(dates_frame.get_column(column_name).to_list())
