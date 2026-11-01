"""Objective 2: exam score regression.

Baselines (Random Forest, XGBoost, SVR) are compared against the hybrid
RF + XGBoost stacking ensemble defined in the capstone design. The regression
takes the dropout risk label as an input feature, so the dropout model runs
first at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .config import (
    EXAM_CATEGORICAL,
    EXAM_NUMERIC,
    RANDOM_STATE,
    SVM_MAX_ROWS,
    TEST_SIZE,
)
from .features import add_features, build_preprocessor
from .selection import select_best


def prepare_data(raw: pd.DataFrame):
    df = add_features(raw)
    y = df["exam_score"].astype(float)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def _hybrid():
    rf = RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    hybrid = StackingRegressor(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=final,
        passthrough=False,
        cv=5,
        n_jobs=-1,
    )
    return hybrid


def build_models() -> dict:
    return {
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(EXAM_NUMERIC, EXAM_CATEGORICAL)),
                ("model", RandomForestRegressor(
                    n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
                )),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(EXAM_NUMERIC, EXAM_CATEGORICAL)),
                ("model", XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        ),
        "SVR (RBF Kernel)": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(EXAM_NUMERIC, EXAM_CATEGORICAL)),
                ("model", SVR(kernel="rbf")),
            ]
        ),
        "Hybrid RF-XGBoost (Stacking)": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(EXAM_NUMERIC, EXAM_CATEGORICAL)),
                ("model", _hybrid()),
            ]
        ),
    }


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run(raw: pd.DataFrame, verbose: bool = True) -> dict:
    X_train, X_test, y_train, y_test = prepare_data(raw)
    models = build_models()

    # The SVR baseline is quadratic in n, so it sees a capped subsample.
    svr_cap = min(SVM_MAX_ROWS, len(X_train))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), svr_cap, replace=False)
    X_svr, y_svr = X_train.iloc[idx], y_train.iloc[idx]

    results, fitted, predictions = {}, {}, {}
    for name, pipeline in models.items():
        fit_X, fit_y = (X_svr, y_svr) if name.startswith("SVR") else (X_train, y_train)
        if verbose:
            print(f"[exam] training {name} on {len(fit_X)} rows ...", flush=True)
        pipeline.fit(fit_X, fit_y)
        y_pred = pipeline.predict(X_test)
        results[name] = _metrics(y_test, y_pred)
        results[name]["train_rows"] = int(len(fit_X))
        fitted[name] = pipeline
        predictions[name] = y_pred
        if verbose:
            m = results[name]
            print(f"    R2={m['r2']:.4f} RMSE={m['rmse']:.3f} MAE={m['mae']:.3f}", flush=True)

    # Best model is selected on R2 with the documented equivalence margin.
    best_name = select_best(results, "r2")
    best = fitted[best_name]

    from .features import feature_importance  # local import to avoid cycles

    return {
        "results": results,
        "best_name": best_name,
        "best_pipeline": best,
        "y_test": y_test,
        "y_pred": predictions[best_name],
        "importances": feature_importance(best, EXAM_NUMERIC, EXAM_CATEGORICAL),
    }
