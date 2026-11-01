"""Streamlit application: Academic Performance Prediction.

Teacher role - enter student details, run dropout risk and exam score
predictions, download the PDF report.
Student role  - view an existing student's report only (no prediction form),
                as required by the reviewer feedback in Final Review 1.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DROPOUT_MODEL_PATH,
    EXAM_MODEL_PATH,
    PROJECT_ORG,
    PROJECT_TEAM,
    PROJECT_TITLE,
    RISK_BANDS,
)
from src.data import load_raw, risk_label
from src.inference import load_bundle, predict_dropout, predict_exam_score
from src.report import build_report, score_band

st.set_page_config(
    page_title="Academic Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

RISK_COLORS = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}

NUMERIC_INPUTS = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "screen_time",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "previous_gpa",
    "mental_health_rating",
    "stress_level",
    "motivation_level",
    "exam_anxiety_score",
    "time_management_score",
]

CATEGORICAL_INPUTS = [
    "gender",
    "major",
    "study_environment",
    "diet_quality",
    "learning_style",
    "family_income_range",
    "parental_education_level",
    "part_time_job",
    "access_to_tutoring",
]

FIELD_LABELS = {
    "study_hours_per_day": "Study hours / day",
    "social_media_hours": "Social media hours / day",
    "netflix_hours": "Netflix hours / day",
    "screen_time": "Other screen time / day",
    "attendance_percentage": "Attendance %",
    "sleep_hours": "Sleep hours / day",
    "exercise_frequency": "Exercise / week",
    "previous_gpa": "Previous GPA",
    "mental_health_rating": "Mental health rating",
    "stress_level": "Stress level",
    "motivation_level": "Motivation level",
    "exam_anxiety_score": "Exam anxiety score",
    "time_management_score": "Time management score",
    "exam_score": "Exam score (if known)",
    "gender": "Gender",
    "major": "Major",
    "study_environment": "Study environment",
    "diet_quality": "Diet quality",
    "learning_style": "Learning style",
    "family_income_range": "Family income range",
    "parental_education_level": "Parental education",
    "part_time_job": "Part-time job",
    "access_to_tutoring": "Access to tutoring",
}


# ---------------------------------------------------------------------- loaders
@st.cache_resource
def get_models():
    if not (DROPOUT_MODEL_PATH.exists() and EXAM_MODEL_PATH.exists()):
        return None, None
    return load_bundle(DROPOUT_MODEL_PATH), load_bundle(EXAM_MODEL_PATH)


@st.cache_data
def get_students() -> pd.DataFrame:
    return load_raw()


def predict_all(record: dict, drop_bundle, exam_bundle):
    dropout = predict_dropout(drop_bundle, record)
    level = risk_label(dropout["probability"], RISK_BANDS)
    exam_input = dict(record)
    exam_input["dropout_risk"] = dropout["label"]
    score = predict_exam_score(exam_bundle, exam_input)
    return dropout, level, float(score)


def risk_banner(level: str, probability: float) -> str:
    color = RISK_COLORS[level]
    return f"""
    <div style="border:1px solid {color};border-left:8px solid {color};
                background:{color}14;padding:14px 18px;border-radius:8px;">
      <div style="font-size:15px;color:#555;">Dropout risk level</div>
      <div style="font-size:34px;font-weight:700;color:{color};">{level}</div>
      <div style="font-size:15px;color:#555;">P(dropout) = {probability:.1%}</div>
    </div>
    """


def score_banner(score: float) -> str:
    band = score_band(score)
    color = "#1e8449" if score >= 70 else ("#d68910" if score >= 50 else "#c0392b")
    return f"""
    <div style="border:1px solid {color};border-left:8px solid {color};
                background:{color}14;padding:14px 18px;border-radius:8px;">
      <div style="font-size:15px;color:#555;">Predicted exam score</div>
      <div style="font-size:34px;font-weight:700;color:{color};">{score:.1f} <span style="font-size:18px;">/ 100</span></div>
      <div style="font-size:15px;color:#555;">Performance band: {band}</div>
    </div>
    """


def show_results(dropout, level, score, record, report_key: str):
    left, right = st.columns(2)
    with left:
        st.markdown(risk_banner(level, dropout["probability"]), unsafe_allow_html=True)
    with right:
        st.markdown(score_banner(score), unsafe_allow_html=True)

    st.progress(min(max(dropout["probability"], 0.0), 1.0))
    st.caption(
        f"Risk bands: High ≥ {RISK_BANDS['High']:.0%}, "
        f"Medium ≥ {RISK_BANDS['Medium']:.0%}, else Low."
    )

    pdf = build_report(record, dropout, level, score)
    st.download_button(
        "📄 Download PDF report",
        data=pdf,
        file_name=f"prediction_report_{report_key}.pdf",
        mime="application/pdf",
        type="primary",
    )


# ---------------------------------------------------------------------- app
dropout_bundle, exam_bundle = get_models()

st.title("🎓 Academic Performance Prediction")
st.caption(f"{PROJECT_TITLE} · {PROJECT_TEAM}")

if dropout_bundle is None:
    st.error(
        "Trained models not found. Run `python train.py` first to train and save "
        f"them into {DROPOUT_MODEL_PATH.parent.as_posix()}/."
    )
    st.stop()

students = get_students()

with st.sidebar:
    st.header("Access")
    role = st.radio("I am a", ["Teacher", "Student"], horizontal=True)
    st.divider()
    st.markdown(
        "**Models**\n"
        f"- Dropout: {dropout_bundle['model_name']}\n"
        f"- Exam score: {exam_bundle['model_name']}"
    )
    st.divider()
    st.caption(PROJECT_ORG)

# ---------------------------------------------------------------- Teacher
if role == "Teacher":
    st.subheader("Enter student details")
    st.caption(
        "Teachers enter student data and generate predictions. "
        "Leave the exam score empty if the exam has not been taken yet - "
        "the pipeline then imputes the dataset median for that one input."
    )

    df = students
    with st.form("prediction_form"):
        cols = st.columns(3)
        record = {}
        for i, col_name in enumerate(NUMERIC_INPUTS):
            with cols[i % 3]:
                series = df[col_name].dropna()
                lo, hi = float(series.min()), float(series.max())
                default = float(series.median())
                is_int = bool(np.allclose(series % 1, 0))
                record[col_name] = st.number_input(
                    FIELD_LABELS.get(col_name, col_name),
                    min_value=lo,
                    max_value=hi,
                    value=default,
                    step=1.0 if is_int else 0.1,
                    key=f"num_{col_name}",
                )
        with cols[(len(NUMERIC_INPUTS)) % 3]:
            record["exam_score"] = st.number_input(
                FIELD_LABELS["exam_score"],
                min_value=0.0,
                max_value=100.0,
                value=None,
                step=1.0,
                key="num_exam_score",
                help="Optional. Leave empty to run dropout prediction without a known score.",
            )

        st.divider()
        cat_cols = st.columns(3)
        for i, col_name in enumerate(CATEGORICAL_INPUTS):
            with cat_cols[i % 3]:
                options = sorted(df[col_name].dropna().astype(str).unique().tolist())
                record[col_name] = st.selectbox(
                    FIELD_LABELS.get(col_name, col_name),
                    options,
                    key=f"cat_{col_name}",
                )

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        if record["exam_score"] is None:
            record["exam_score"] = np.nan
        with st.spinner("Running the models..."):
            dropout, level, score = predict_all(record, dropout_bundle, exam_bundle)
        st.session_state["last_result"] = {
            "record": record,
            "dropout": dropout,
            "level": level,
            "score": score,
        }

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Prediction result")
        show_results(
            result["dropout"],
            result["level"],
            result["score"],
            result["record"],
            report_key="teacher",
        )

# ---------------------------------------------------------------- Student
else:
    st.subheader("Your performance report")
    st.caption("Students can view their report only. Prediction inputs are managed by teachers.")

    ids = sorted(students["student_id"].tolist())
    student_id = st.selectbox("Student ID", ids, index=0)
    row = students[students["student_id"] == student_id].iloc[0]

    record = {k: (np.nan if pd.isna(v) else v) for k, v in row.to_dict().items()}
    with st.spinner("Loading report..."):
        dropout, level, score = predict_all(record, dropout_bundle, exam_bundle)

    st.markdown(risk_banner(level, dropout["probability"]), unsafe_allow_html=True)
    st.markdown(score_banner(score), unsafe_allow_html=True)

    with st.expander("Student details"):
        st.dataframe(
            pd.DataFrame(
                [{"Field": FIELD_LABELS.get(k, k.replace("_", " ").title()),
                  "Value": v} for k, v in record.items()
                 if k not in ("student_id",)]
            ),
            hide_index=True,
        )

    pdf = build_report({"student_id": student_id, **record}, dropout, level, score)
    st.download_button(
        "📄 Download PDF report",
        data=pdf,
        file_name=f"report_{student_id}.pdf",
        mime="application/pdf",
        type="primary",
    )
