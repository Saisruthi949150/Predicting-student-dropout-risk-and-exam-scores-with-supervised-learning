"""Project configuration: paths, feature lists and training settings."""

from pathlib import Path

# ---------------------------------------------------------------- Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

DROPOUT_MODEL_PATH = MODELS_DIR / "best_dropout_model.pkl"
EXAM_MODEL_PATH = MODELS_DIR / "best_exam_model.pkl"

# ---------------------------------------------------------------- Training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# The SVM/SVR baselines scale quadratically with the number of samples, so they
# are trained on a capped stratified subsample to keep the full training run
# practical on a laptop. Every other model uses the complete training split.
SVM_MAX_ROWS = 10_000

# Columns checked for outliers (IQR rule) during cleaning.
OUTLIER_COLUMNS = [
    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "previous_gpa",
    "exam_score",
]

# Risk bands for the dropout probability (P(dropout = Yes)).
RISK_BANDS = {"High": 0.66, "Medium": 0.33}

# ---------------------------------------------------------------- Features
# Selected features reported in the capstone decks.
DROPOUT_NUMERIC = [
    "stress_level",
    "motivation_level",
    "exam_anxiety_score",
    "previous_gpa",
    "exam_score",
    "mental_health_rating",
    "sleep_hours",
    "exercise_frequency",
    "attendance_percentage",
]

DROPOUT_CATEGORICAL = [
    "gender",
    "major",
    "study_environment",
    "diet_quality",
    "learning_style",
    "family_income_range",
    "parental_education_level",
]

EXAM_NUMERIC = [
    "previous_gpa",
    "motivation_level",
    "stress_anxiety",
    "study_sleep_ratio",
    "screen_time_total",
    "time_management_score",
    "attendance_percentage",
]

EXAM_CATEGORICAL = [
    "gender",
    "major",
    "learning_style",
    "study_environment",
    "dropout_risk",
    "part_time_job",
    "access_to_tutoring",
]

# Engineered features created for both pipelines.
ENGINEERED = ["screen_time_total", "study_sleep_ratio", "stress_anxiety"]

PROJECT_TITLE = "Academic Performance Prediction Using Supervised Learning"
PROJECT_TEAM = "Batch AIML 13 - Department of Artificial Intelligence & Data Science"
PROJECT_ORG = "GITAM School of CSE, GITAM (Deemed to be University), Bengaluru"
