"""Figure generation for EDA and model evaluation (saved under reports/figures)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import FIGURES_DIR

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def eda_figures(df: pd.DataFrame) -> list:
    paths = []

    # Target distributions
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    counts = df["dropout_risk"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax[0], hue=counts.index, legend=False)
    ax[0].set_title("Dropout risk distribution")
    ax[0].set_ylabel("students")
    ax[1].hist(df["exam_score"], bins=40, color="#4c72b0", edgecolor="white")
    ax[1].set_title("Exam score distribution")
    ax[1].set_xlabel("exam_score")
    paths.append(_save(fig, "eda_targets.png"))

    # Correlation heatmap (numeric features)
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ("student_id",) if c in df.columns], errors="ignore"
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, fmt=".2f",
                square=True, linewidths=0.3, ax=ax)
    ax.set_title("Correlation heatmap")
    paths.append(_save(fig, "eda_correlation.png"))

    # Key habit distributions
    habits = ["study_hours_per_day", "sleep_hours", "attendance_percentage",
              "stress_level", "social_media_hours", "motivation_level"]
    habits = [h for h in habits if h in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, habits):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color="#55a868")
        ax.set_title(col)
    fig.suptitle("Distribution of key student habits")
    paths.append(_save(fig, "eda_habits.png"))

    # Behaviour vs performance
    pairs = [("stress_level", "exam_score"), ("attendance_percentage", "exam_score"),
             ("study_hours_per_day", "exam_score"), ("previous_gpa", "exam_score")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (x, y) in zip(axes.flat, pairs):
        if x in df.columns:
            sns.scatterplot(data=df, x=x, y=y, hue="dropout_risk", ax=ax,
                            s=12, alpha=0.4)
            ax.set_title(f"{x} vs {y}")
    paths.append(_save(fig, "eda_performance_drivers.png"))

    # Dropout risk vs stress/anxiety
    if {"stress_level", "exam_anxiety_score"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="dropout_risk", y="stress_level", ax=ax)
        ax.set_title("Stress level by dropout risk")
        paths.append(_save(fig, "eda_stress_vs_risk.png"))

    return paths


def comparison_figure(results: dict, metrics: list, title: str, name: str) -> str:
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, metric in enumerate(metrics):
        values = [results[n].get(metric, np.nan) for n in names]
        bars = ax.bar(x + i * width, values, width, label=metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_title(title)
    ax.legend()
    paths = _save(fig, name)
    return paths


def confusion_matrix_figure(y_test, y_pred) -> str:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - best dropout model")
    return _save(fig, "eval_confusion_matrix.png")


def roc_figure(curves: dict, best_name: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, linewidth=2 if name == best_name else 1,
                alpha=1.0 if name == best_name else 0.5, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - dropout models")
    ax.legend(fontsize=8)
    return _save(fig, "eval_roc_curves.png")


def importance_figure(importances: list, name: str, top: int = 15) -> str:
    pairs = importances[:top]
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, values, color="#8172b3")
    ax.set_title(f"Top {len(pairs)} feature importances")
    ax.set_xlabel("importance")
    return _save(fig, name)


def regression_figure(y_test, y_pred) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].scatter(y_pred, y_test, s=8, alpha=0.4, color="#4c72b0")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax[0].set_xlabel("Predicted exam score")
    ax[0].set_ylabel("Actual exam score")
    ax[0].set_title("Predicted vs actual")

    residuals = np.asarray(y_test) - np.asarray(y_pred)
    ax[1].hist(residuals, bins=40, color="#55a868", edgecolor="white")
    ax[1].axvline(0, color="r", linestyle="--")
    ax[1].set_xlabel("Residual (actual - predicted)")
    ax[1].set_title("Residual distribution")
    return _save(fig, "eval_regression_diagnostics.png")
