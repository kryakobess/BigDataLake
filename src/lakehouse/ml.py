from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import polars as pl
from deltalake import DeltaTable
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lakehouse.config import MLConfig


LOGGER = logging.getLogger(__name__)
REGRESSION_TARGET = "target_arr_delay"
CLASSIFICATION_TARGET = "target_is_delayed"
DROP_COLUMNS = ["flight_id", "flight_date", "gold_built_at"]
CATEGORICAL_COLUMNS = [
    "season",
    "marketing_airline",
    "operating_airline",
    "marketing_airline_iata",
    "origin",
    "origin_state",
    "dest",
    "dest_state",
    "route",
    "dep_time_block",
    "arr_time_block",
]
NUMERIC_COLUMNS = [
    "year",
    "quarter",
    "month",
    "day_of_month",
    "day_of_week",
    "flight_number",
    "crs_dep_time",
    "dep_time",
    "dep_hour",
    "crs_arr_time",
    "arr_hour",
    "distance",
    "distance_group",
    "crs_elapsed_time",
    "dep_delay",
    "dep_delay_minutes",
    "silver_version",
]


def run_ml(config: MLConfig) -> None:
    setup_logging(config.log_path)

    try:
        feature_table = DeltaTable(config.source_uri)
    except Exception as exc:
        raise FileNotFoundError(f"Gold feature Delta table was not found: {config.source_uri}") from exc

    gold_version = feature_table.version()
    dataset = (
        pl.scan_delta(config.source_uri)
        .sort("flight_date")
        .collect()
        .to_pandas()
    )
    if dataset.empty:
        raise ValueError("ML pipeline received an empty feature table.")

    split_date = choose_split_date(dataset)
    train_df = dataset[dataset["flight_date"] < split_date].copy()
    test_df = dataset[dataset["flight_date"] >= split_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Not enough flight dates for holdout evaluation.")

    LOGGER.info(
        "ML dataset prepared. rows=%s train_rows=%s test_rows=%s split_date=%s gold_version=%s",
        len(dataset),
        len(train_df),
        len(test_df),
        split_date,
        gold_version,
    )

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name="ml_pipeline") as run:
        mlflow.log_params(
            {
                "features_source_uri": config.source_uri,
                "gold_features_version": gold_version,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "split_date": str(split_date),
                "categorical_feature_count": len(CATEGORICAL_COLUMNS),
                "numeric_feature_count": len(NUMERIC_COLUMNS),
            }
        )

        regression_summary = run_regression_models(train_df, test_df)
        classification_summary = run_classification_models(train_df, test_df)

        mlflow.log_metrics(
            {
                "best_regression_rmse": regression_summary["best_metrics"]["rmse"],
                "best_regression_mae": regression_summary["best_metrics"]["mae"],
                "best_regression_r2": regression_summary["best_metrics"]["r2"],
                "best_classification_f1": classification_summary["best_metrics"]["f1"],
                "best_classification_accuracy": classification_summary["best_metrics"]["accuracy"],
                "best_classification_roc_auc": classification_summary["best_metrics"]["roc_auc"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "model_comparison.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "gold_features_version": gold_version,
                        "split_date": str(split_date),
                        "regression": regression_summary["results"],
                        "classification": classification_summary["results"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            reg_importance_path = Path(tmp_dir) / "regression_feature_importance.csv"
            regression_summary["feature_importance"].to_csv(reg_importance_path, index=False)

            clf_importance_path = Path(tmp_dir) / "classification_feature_importance.csv"
            classification_summary["feature_importance"].to_csv(clf_importance_path, index=False)

            mlflow.log_artifact(str(summary_path), artifact_path="reports")
            mlflow.log_artifact(str(reg_importance_path), artifact_path="reports")
            mlflow.log_artifact(str(clf_importance_path), artifact_path="reports")

        LOGGER.info(
            "ML training finished. run_id=%s best_regressor=%s best_classifier=%s",
            run.info.run_id,
            regression_summary["best_model_name"],
            classification_summary["best_model_name"],
        )


def run_regression_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    x_train = build_feature_frame(train_df)
    x_test = build_feature_frame(test_df)
    y_train = train_df[REGRESSION_TARGET]
    y_test = test_df[REGRESSION_TARGET]

    candidates = {
        "ridge_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "decision_tree_regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return evaluate_models(
        task_name="regression",
        candidates=candidates,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        scoring_key="rmse",
        higher_is_better=False,
        metric_fn=regression_metrics,
    )


def run_classification_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    x_train = build_feature_frame(train_df)
    x_test = build_feature_frame(test_df)
    y_train = train_df[CLASSIFICATION_TARGET]
    y_test = test_df[CLASSIFICATION_TARGET]

    candidates = {
        "sgd_classifier": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=0.0001,
                        early_stopping=True,
                        max_iter=200,
                        n_iter_no_change=5,
                        tol=1e-3,
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "decision_tree_classifier": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return evaluate_models(
        task_name="classification",
        candidates=candidates,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        scoring_key="f1",
        higher_is_better=True,
        metric_fn=classification_metrics,
    )


def evaluate_models(
    *,
    task_name: str,
    candidates: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scoring_key: str,
    higher_is_better: bool,
    metric_fn,
) -> dict[str, object]:
    results: list[dict[str, float | str]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] | None = None

    for model_name, pipeline in candidates.items():
        with mlflow.start_run(run_name=f"{task_name}_{model_name}", nested=True):
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            probabilities = None
            if task_name == "classification":
                if hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba(x_test)[:, 1]
                elif hasattr(pipeline, "decision_function"):
                    probabilities = pipeline.decision_function(x_test)

            metrics = metric_fn(y_test, predictions, probabilities)
            mlflow.log_param("task", task_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name="model")

            results.append({"model_name": model_name, **metrics})

            current_score = metrics[scoring_key]
            if best_metrics is None:
                should_replace = True
            else:
                should_replace = current_score > best_metrics[scoring_key] if higher_is_better else current_score < best_metrics[scoring_key]

            if should_replace:
                best_name = model_name
                best_pipeline = pipeline
                best_metrics = metrics

    if best_pipeline is None or best_name is None or best_metrics is None:
        raise RuntimeError(f"No models were evaluated for task={task_name}.")

    feature_importance = extract_feature_importance(best_pipeline)
    mlflow.log_param(f"best_{task_name}_model", best_name)
    for metric_name, metric_value in best_metrics.items():
        mlflow.log_metric(f"best_{task_name}_{metric_name}", metric_value)

    return {
        "results": results,
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "feature_importance": feature_importance,
    }


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET], errors="ignore")


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=100,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def regression_metrics(y_true: pd.Series, y_pred, _probabilities) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: pd.Series, y_pred, probabilities) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficients = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
        importance_values = abs(coefficients)
    else:
        importance_values = [0.0] * len(feature_names)

    importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importance_values,
        }
    )
    return importance.sort_values("importance", ascending=False).head(30)


def choose_split_date(dataset: pd.DataFrame):
    unique_dates = sorted(dataset["flight_date"].dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("At least 5 unique flight dates are required for the ML holdout split.")
    split_index = max(1, int(len(unique_dates) * 0.8))
    if split_index >= len(unique_dates):
        split_index = len(unique_dates) - 1
    return unique_dates[split_index]


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
