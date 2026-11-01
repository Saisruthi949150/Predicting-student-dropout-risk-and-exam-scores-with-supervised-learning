"""Feature engineering and preprocessing pipelines."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features defined in the capstone design.

    screen_time_total = social media hours + netflix hours + other screen time
    study_sleep_ratio = study hours per day / sleep hours
    stress_anxiety    = stress level + exam anxiety score
    """
    out = df.copy()
    if {"social_media_hours", "netflix_hours", "screen_time"} <= set(out.columns):
        out["screen_time_total"] = (
            out["social_media_hours"] + out["netflix_hours"] + out["screen_time"]
        )
    if {"study_hours_per_day", "sleep_hours"} <= set(out.columns):
        sleep = out["sleep_hours"].replace(0, np.nan)
        out["study_sleep_ratio"] = out["study_hours_per_day"] / sleep
    if {"stress_level", "exam_anxiety_score"} <= set(out.columns):
        out["stress_anxiety"] = out["stress_level"] + out["exam_anxiety_score"]
    return out


def feature_importance(pipeline, numeric: list, categorical: list) -> list:
    """Return (feature, importance) pairs from a fitted pipeline, if the model
    exposes feature importances (tree ensembles). Empty list otherwise."""
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = [str(n) for n in pre.get_feature_names_out()]

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "estimators_"):  # VotingClassifier: average the members
        members = []
        for est in model.estimators_:
            # estimators_ holds bare estimators (name/estimator pairs only in `.estimators`)
            candidate = est[1] if isinstance(est, tuple) else est
            if hasattr(candidate, "feature_importances_"):
                members.append(candidate)
        if members:
            importances = np.mean(
                [np.asarray(e.feature_importances_, dtype=float) for e in members], axis=0
            )
    if importances is None or len(importances) != len(names):
        return []

    pairs = sorted(zip(names, importances), key=lambda t: -t[1])
    return [(n.removeprefix("num__").removeprefix("cat__"), float(i)) for n, i in pairs]


def build_preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    """StandardScaler for numeric features, OneHotEncoder for categorical ones."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )
