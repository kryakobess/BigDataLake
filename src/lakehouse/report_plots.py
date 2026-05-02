from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd
import polars as pl
from mlflow import MlflowClient

from lakehouse.config import ReportPlotsConfig


matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
DAY_LABELS = {
    "1": "Mon",
    "2": "Tue",
    "3": "Wed",
    "4": "Thu",
    "5": "Fri",
    "6": "Sat",
    "7": "Sun",
}
PRIMARY_COLOR = "#0f766e"
SECONDARY_COLOR = "#2563eb"
ACCENT_COLOR = "#f59e0b"
HIGHLIGHT_COLOR = "#dc2626"


def run_report_plots(config: ReportPlotsConfig) -> None:
    setup_logging(config.log_path)

    analytics_dir = config.output_dir / "analytics"
    ml_dir = config.output_dir / "ml"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)

    build_analytics_overview(config.analytics_source_uri, analytics_dir / "analytics_overview.png")
    build_ml_plots(config, ml_dir)

    LOGGER.info("Report plots saved to %s", config.output_dir)


def build_analytics_overview(source_uri: str, output_path: Path) -> None:
    analytics = (
        pl.scan_delta(source_uri)
        .select(
            "aggregation_level",
            "aggregation_key",
            "flights_count",
            "avg_arr_delay",
            "delayed_rate",
        )
        .collect()
    )
    if analytics.is_empty():
        raise ValueError("Analytics Delta table is empty. Build gold analytics before generating plots.")

    airlines = (
        analytics
        .filter(pl.col("aggregation_level") == "airline")
        .sort("avg_arr_delay", descending=True)
        .head(10)
        .to_pandas()
    )
    departure_hours = (
        analytics
        .filter(pl.col("aggregation_level") == "departure_hour")
        .with_columns(pl.col("aggregation_key").cast(pl.Int64).alias("dep_hour"))
        .sort("dep_hour")
        .to_pandas()
    )
    days = (
        analytics
        .filter(pl.col("aggregation_level") == "day_of_week")
        .with_columns(pl.col("aggregation_key").cast(pl.Int64).alias("day_number"))
        .sort("day_number")
        .with_columns(pl.col("aggregation_key").replace(DAY_LABELS).alias("day_label"))
        .to_pandas()
    )
    routes = (
        analytics
        .filter(
            (pl.col("aggregation_level") == "route")
            & (pl.col("flights_count") >= 100)
        )
        .sort("avg_arr_delay", descending=True)
        .head(10)
        .to_pandas()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Gold Analytics Overview", fontsize=18, fontweight="bold")

    axes[0, 0].barh(
        airlines["aggregation_key"][::-1],
        airlines["avg_arr_delay"][::-1],
        color=PRIMARY_COLOR,
    )
    axes[0, 0].set_title("Top airlines by avg arrival delay")
    axes[0, 0].set_xlabel("Minutes")

    axes[0, 1].plot(
        departure_hours["dep_hour"],
        departure_hours["avg_arr_delay"],
        color=SECONDARY_COLOR,
        linewidth=2.5,
        marker="o",
    )
    axes[0, 1].set_title("Average arrival delay by departure hour")
    axes[0, 1].set_xlabel("Departure hour")
    axes[0, 1].set_ylabel("Minutes")
    axes[0, 1].set_xticks(range(0, 24, 2))

    axes[1, 0].bar(
        days["day_label"],
        days["delayed_rate"] * 100,
        color=ACCENT_COLOR,
    )
    axes[1, 0].set_title("Delayed flight share by weekday")
    axes[1, 0].set_ylabel("Delayed flights, %")

    axes[1, 1].barh(
        routes["aggregation_key"][::-1],
        routes["avg_arr_delay"][::-1],
        color=HIGHLIGHT_COLOR,
    )
    axes[1, 1].set_title("Top busy routes by avg arrival delay")
    axes[1, 1].set_xlabel("Minutes")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_ml_plots(config: ReportPlotsConfig, output_dir: Path) -> None:
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient()
    run_id = config.run_id or resolve_latest_parent_run_id(client, config.experiment_name)

    comparison = load_comparison_report(run_id)
    regression_importance = load_csv_artifact(run_id, "reports/regression_feature_importance.csv")
    classification_importance = load_csv_artifact(run_id, "reports/classification_feature_importance.csv")

    build_ml_metrics_overview(comparison, output_dir / "ml_metrics_overview.png")
    build_feature_importance_plot(
        regression_importance,
        output_dir / "regression_feature_importance.png",
        title="Regression feature importance",
        color=SECONDARY_COLOR,
    )
    build_feature_importance_plot(
        classification_importance,
        output_dir / "classification_feature_importance.png",
        title="Classification feature importance",
        color=PRIMARY_COLOR,
    )

    LOGGER.info("ML plots were built from run_id=%s", run_id)


def resolve_latest_parent_run_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment was not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.mlflow.runName = 'ml_pipeline'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            "No finished ml_pipeline runs were found in MLflow. Run the ML stage before generating report plots."
        )
    return runs[0].info.run_id


def load_comparison_report(run_id: str) -> dict[str, object]:
    summary_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="reports/model_comparison.json",
    )
    return json.loads(Path(summary_path).read_text(encoding="utf-8"))


def load_csv_artifact(run_id: str, artifact_path: str) -> pd.DataFrame:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
    )
    return pd.read_csv(local_path)


def build_ml_metrics_overview(comparison: dict[str, object], output_path: Path) -> None:
    regression = pd.DataFrame(comparison["regression"])
    classification = pd.DataFrame(comparison["classification"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ML Model Comparison", fontsize=18, fontweight="bold")

    regression_plot = regression.set_index("model_name")[["rmse", "mae"]]
    regression_plot.plot(kind="bar", ax=axes[0], color=[SECONDARY_COLOR, ACCENT_COLOR])
    axes[0].set_title("Regression error metrics")
    axes[0].set_ylabel("Metric value")
    axes[0].tick_params(axis="x", rotation=15)

    regression_r2 = regression[["model_name", "r2"]]
    axes[0].plot(
        range(len(regression_r2)),
        regression_r2["r2"],
        color=HIGHLIGHT_COLOR,
        marker="o",
        linewidth=2,
        label="r2",
    )
    axes[0].legend()

    classification_plot = classification.set_index("model_name")[["accuracy", "f1", "roc_auc"]]
    classification_plot.plot(kind="bar", ax=axes[1], color=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR])
    axes[1].set_title("Classification quality metrics")
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_feature_importance_plot(
    importance: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color: str,
) -> None:
    top_features = importance.head(15).copy()
    top_features = top_features.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top_features["feature_name"], top_features["importance"], color=color)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
