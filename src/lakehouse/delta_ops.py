from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from deltalake import DeltaTable

from lakehouse.config import DeltaOptimizeConfig, DeltaTimeTravelConfig, DeltaVacuumConfig


LOGGER = logging.getLogger(__name__)


def run_delta_optimize(config: DeltaOptimizeConfig) -> None:
    setup_logging(config.log_path)

    table = load_delta_table(config.source_uri)
    before_version = table.version()
    before_files = len(table.file_uris())

    compact_metrics = table.optimize.compact(target_size=config.target_size_bytes)
    compacted_table = DeltaTable(config.source_uri)

    LOGGER.info(
        "Delta OPTIMIZE compact finished. source=%s before_version=%s after_version=%s before_files=%s after_files=%s metrics=%s",
        config.source_uri,
        before_version,
        compacted_table.version(),
        before_files,
        len(compacted_table.file_uris()),
        compact_metrics,
    )

    if not config.z_order_columns:
        return

    partition_columns = set(compacted_table.metadata().partition_columns)
    invalid_columns = sorted(set(config.z_order_columns) & partition_columns)
    if invalid_columns:
        raise ValueError(
            f"Z-ORDER columns cannot include partition columns: {invalid_columns}. "
            f"Choose non-partition columns instead."
        )

    before_z_order_version = compacted_table.version()
    z_order_metrics = compacted_table.optimize.z_order(
        config.z_order_columns,
        target_size=config.target_size_bytes,
    )
    z_ordered_table = DeltaTable(config.source_uri)
    LOGGER.info(
        "Delta Z-ORDER finished. source=%s columns=%s before_version=%s after_version=%s files=%s metrics=%s",
        config.source_uri,
        config.z_order_columns,
        before_z_order_version,
        z_ordered_table.version(),
        len(z_ordered_table.file_uris()),
        z_order_metrics,
    )


def run_delta_vacuum(config: DeltaVacuumConfig) -> None:
    setup_logging(config.log_path)

    table = load_delta_table(config.source_uri)
    stale_files = table.vacuum(
        retention_hours=config.retention_hours,
        dry_run=config.dry_run,
        enforce_retention_duration=config.enforce_retention_duration,
        full=config.full,
    )
    preview = stale_files[:10]

    LOGGER.info(
        "Delta VACUUM finished. source=%s version=%s dry_run=%s retention_hours=%s candidates=%s preview=%s",
        config.source_uri,
        table.version(),
        config.dry_run,
        config.retention_hours,
        len(stale_files),
        preview,
    )


def run_delta_time_travel(config: DeltaTimeTravelConfig) -> None:
    setup_logging(config.log_path)

    current_table = load_delta_table(config.source_uri)
    current_version = current_table.version()
    if config.version > current_version:
        raise ValueError(
            f"Requested version {config.version} is newer than current table version {current_version}."
        )

    snapshot_table = DeltaTable(config.source_uri, version=config.version)
    snapshot_scan = pl.scan_delta(config.source_uri, version=config.version)
    sample = snapshot_scan.head(config.limit).collect()
    row_count = (
        snapshot_scan
        .select(pl.len().alias("rows"))
        .collect()
        .item(0, 0)
    )
    history = current_table.history(limit=10)

    LOGGER.info(
        "Delta time travel snapshot loaded. source=%s current_version=%s snapshot_version=%s files=%s rows=%s columns=%s",
        config.source_uri,
        current_version,
        snapshot_table.version(),
        len(snapshot_table.file_uris()),
        row_count,
        sample.columns,
    )
    LOGGER.info("Recent Delta history: %s", history)
    LOGGER.info("Snapshot sample:\n%s", sample)


def load_delta_table(source_uri: str) -> DeltaTable:
    try:
        return DeltaTable(source_uri)
    except Exception as exc:
        raise FileNotFoundError(f"Delta table was not found: {source_uri}") from exc


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
