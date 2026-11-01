"""Objective 1: dropout risk classification.

Baselines (Logistic Regression, SVM, Random Forest, XGBoost) are compared
against the hybrid RF + XGBoost soft-voting ensemble defined in the capstone
design. All models share one preprocessing pipeline: median imputation +
StandardScaler for numeric features and most-frequent imputation + OneHot
encoding for categorical features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import (
    DROPOUT_CATEGORICAL,
    DROPOUT_NUMERIC,
    RANDOM_STATE,
    SVM_MAX_ROWS,
    TEST_SIZE,
)
from .features import add_features, build_preprocessor
from .selection import select_best


def prepare_data(raw: pd.DataFrame):
    df = add_features(raw)
    y = (df["dropout_risk"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _hybrid_estimators():
    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )
    return rf, xgb


def build_models() -> dict:
    rf, xgb = _hybrid_estimators()
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(DROPOUT_NUMERIC, DROPOUT_CATEGORICAL)),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "SVM (RBF Kernel)": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(DROPOUT_NUMERIC, DROPOUT_CATEGORICAL)),
                # SVC(probability=True) is deprecated in scikit-learn 1.9;
                # calibration gives the same class probabilities for soft voting.
                ("model", CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(DROPOUT_NUMERIC, DROPOUT_CATEGORICAL)),
                ("model", RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
                )),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(DROPOUT_NUMERIC, DROPOUT_CATEGORICAL)),
                ("model", XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    eval_metric="logloss",
                )),
            ]
        ),
        "Hybrid RF-XGBoost (Soft Voting)": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(DROPOUT_NUMERIC, DROPOUT_CATEGORICAL)),
                ("model", VotingClassifier(
                    estimators=[("rf", rf), ("xgb", xgb)], voting="soft", n_jobs=-1
                )),
            ]
        ),
    }
    return models


def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def run(raw: pd.DataFrame, verbose: bool = True) -> dict:
    X_train, X_test, y_train, y_test = prepare_data(raw)
    models = build_models()

    # Baseline SVM training is quadratic in n, so it sees a capped subsample.
    svm_cap = min(SVM_MAX_ROWS, len(X_train))
    X_svm, _, y_svm, _ = train_test_split(
        X_train, y_train, train_size=svm_cap, stratify=y_train, random_state=RANDOM_STATE
    )

    results, fitted, curves = {}, {}, {}
    for name, pipeline in models.items():
        fit_X, fit_y = (X_svm, y_svm) if name.startswith("SVM") else (X_train, y_train)
        if verbose:
            print(f"[dropout] training {name} on {len(fit_X)} rows ...", flush=True)
        pipeline.fit(fit_X, fit_y)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        results[name] = _metrics(y_test, y_pred, y_proba)
        results[name]["train_rows"] = int(len(fit_X))
        fitted[name] = pipeline
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        curves[name] = (fpr, tpr)
        if verbose:
            m = results[name]
            print(
                f"    acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}",
                flush=True,
            )

    # Best model is selected on F1 with the documented equivalence margin.
    best_name = select_best(results, "f1")
    best = fitted[best_name]

    from .features import feature_importance  # local import to avoid cycles

    importances = feature_importance(best, DROPOUT_NUMERIC, DROPOUT_CATEGORICAL)
    y_proba_best = best.predict_proba(X_test)[:, 1]

    return {
        "results": results,
        "best_name": best_name,
        "best_pipeline": best,
        "y_test": y_test,
        "y_pred": best.predict(X_test),
        "y_proba": y_proba_best,
        "curves": curves,
        "importances": importances,
    }
