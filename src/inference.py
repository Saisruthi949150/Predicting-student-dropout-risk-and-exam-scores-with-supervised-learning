"""Shared helpers for training and inference."""

import joblib
import pandas as pd

from .features import add_features


def save_bundle(path, pipeline, numeric, categorical, metrics, model_name):
    bundle = {
        "pipeline": pipeline,
        "numeric": numeric,
        "categorical": categorical,
        "metrics": metrics,
        "model_name": model_name,
    }
    joblib.dump(bundle, path)
    return bundle


def load_bundle(path):
    return joblib.load(path)


def predict_dropout(bundle, record: dict) -> dict:
    """Return {'label': 'Yes'|'No', 'probability': float} for one student."""
    X = add_features(pd.DataFrame([record]))[bundle["numeric"] + bundle["categorical"]]
    proba = float(bundle["pipeline"].predict_proba(X)[0, 1])
    return {"label": "Yes" if proba >= 0.5 else "No", "probability": proba}


def predict_exam_score(bundle, record: dict) -> float:
    X = add_features(pd.DataFrame([record]))[bundle["numeric"] + bundle["categorical"]]
    return float(bundle["pipeline"].predict(X)[0])
