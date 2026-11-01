"""End-to-end training run: clean -> EDA -> train both models -> evaluate -> save.

Usage:
    python train.py
"""

import json
import sys

import pandas as pd

from src import train_dropout, train_exam_score
from src.config import (
    DROPOUT_CATEGORICAL,
    DROPOUT_MODEL_PATH,
    DROPOUT_NUMERIC,
    EXAM_CATEGORICAL,
    EXAM_MODEL_PATH,
    EXAM_NUMERIC,
    METRICS_DIR,
    MODELS_DIR,
)
from src.data import clean, load_raw
from src.inference import save_bundle
from src.plots import (
    comparison_figure,
    confusion_matrix_figure,
    eda_figures,
    importance_figure,
    regression_figure,
    roc_figure,
)


def main() -> int:
    print("=" * 70)
    print("Academic Performance Prediction Using Supervised Learning")
    print("=" * 70)

    # ---------------------------------------------------------------- data
    raw = load_raw()
    print(f"\nLoaded {raw.shape[0]} rows x {raw.shape[1]} columns from data/raw/")
    df, stats = clean(raw)
    print(f"Cleaning: {stats['duplicates_removed']} duplicates, "
          f"{stats['outliers_removed']} outliers removed -> {stats['rows_clean']} rows")
    print(f"Missing values kept for pipeline imputation: {stats['missing_values'] or 'none'}")

    # ---------------------------------------------------------------- EDA
    print("\n[1/4] Generating EDA figures ...")
    eda_paths = eda_figures(df)
    print(f"      {len(eda_paths)} figures written to reports/figures/")

    # ---------------------------------------------------------------- dropout
    print("\n[2/4] Dropout risk classification")
    drop = train_dropout.run(df)

    # ---------------------------------------------------------------- exam score
    print("\n[3/4] Exam score regression")
    exam = train_exam_score.run(df)

    # ---------------------------------------------------------------- evaluation figures
    print("\n[4/4] Saving evaluation figures and artifacts")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_figure(
        drop["results"],
        ["accuracy", "precision", "recall", "f1"],
        "Comparison of classification models for dropout prediction",
        "eval_classification_comparison.png",
    )
    confusion_matrix_figure(drop["y_test"], drop["y_pred"])
    roc_figure(drop["curves"], drop["best_name"])
    if drop["importances"]:
        importance_figure(drop["importances"], "eval_dropout_feature_importance.png")

    comparison_figure(
        exam["results"],
        ["r2"],
        "Comparison of regression models - R2 (higher is better)",
        "eval_regression_r2.png",
    )
    comparison_figure(
        exam["results"],
        ["rmse", "mae"],
        "Comparison of regression models - error (lower is better)",
        "eval_regression_errors.png",
    )
    regression_figure(exam["y_test"], exam["y_pred"])
    if exam["importances"]:
        importance_figure(exam["importances"], "eval_exam_feature_importance.png")

    # ---------------------------------------------------------------- save models
    save_bundle(
        DROPOUT_MODEL_PATH,
        drop["best_pipeline"],
        DROPOUT_NUMERIC,
        DROPOUT_CATEGORICAL,
        {"best": drop["best_name"], "all": drop["results"]},
        drop["best_name"],
    )
    save_bundle(
        EXAM_MODEL_PATH,
        exam["best_pipeline"],
        EXAM_NUMERIC,
        EXAM_CATEGORICAL,
        {"best": exam["best_name"], "all": exam["results"]},
        exam["best_name"],
    )
    print(f"      saved {DROPOUT_MODEL_PATH.name} and {EXAM_MODEL_PATH.name}")

    # ---------------------------------------------------------------- metrics
    summary = {
        "data": stats,
        "dropout": {"best": drop["best_name"], "results": drop["results"]},
        "exam_score": {"best": exam["best_name"], "results": exam["results"]},
    }
    (METRICS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(drop["results"]).T.to_csv(METRICS_DIR / "dropout_results.csv")
    pd.DataFrame(exam["results"]).T.to_csv(METRICS_DIR / "exam_results.csv")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 70)
    print("RESULTS - Dropout risk classification")
    print("=" * 70)
    drop_df = pd.DataFrame(drop["results"]).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "train_rows"]
    ]
    print(drop_df.round(4).to_string())
    print(f"\nBest model: {drop['best_name']}")

    print("\n" + "=" * 70)
    print("RESULTS - Exam score regression")
    print("=" * 70)
    exam_df = pd.DataFrame(exam["results"]).T[["r2", "rmse", "mae", "train_rows"]]
    print(exam_df.round(4).to_string())
    print(f"\nBest model: {exam['best_name']}")

    print("\nArtifacts:")
    print("  models/       trained pipelines (joblib)")
    print("  reports/figures/   EDA + evaluation plots")
    print("  reports/metrics/   metrics.json, per-model CSVs")
    print("\nNext: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
