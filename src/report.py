"""PDF prediction report generation (reportlab), used by training and the app."""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import PROJECT_ORG, PROJECT_TEAM, PROJECT_TITLE

STYLES = getSampleStyleSheet()

RISK_COLORS = {"High": colors.HexColor("#c0392b"),
               "Medium": colors.HexColor("#d68910"),
               "Low": colors.HexColor("#1e8449")}

FIELD_LABELS = {
    "study_hours_per_day": "Study hours per day",
    "social_media_hours": "Social media hours",
    "netflix_hours": "Netflix hours",
    "part_time_job": "Part-time job",
    "attendance_percentage": "Attendance %",
    "sleep_hours": "Sleep hours",
    "diet_quality": "Diet quality",
    "exercise_frequency": "Exercise frequency (per week)",
    "parental_education_level": "Parental education",
    "mental_health_rating": "Mental health rating",
    "previous_gpa": "Previous GPA",
    "stress_level": "Stress level",
    "motivation_level": "Motivation level",
    "exam_anxiety_score": "Exam anxiety score",
    "time_management_score": "Time management score",
    "exam_score": "Exam score",
    "gender": "Gender",
    "major": "Major",
    "study_environment": "Study environment",
    "learning_style": "Learning style",
    "family_income_range": "Family income range",
    "access_to_tutoring": "Access to tutoring",
    "screen_time": "Screen time (other)",
    "screen_time_total": "Total screen time",
    "study_sleep_ratio": "Study : sleep ratio",
    "stress_anxiety": "Stress + anxiety",
    "dropout_risk": "Dropout risk",
    "student_id": "Student ID",
}


def score_band(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Support"


def _table(rows, header=None, widths=None) -> Table:
    style = TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5d8dc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]
    )
    if header:
        style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50"))
        style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
        style.add("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    table = Table(rows, colWidths=widths)
    table.setStyle(style)
    return table


def build_report(record: dict, dropout: dict, risk_level: str, exam_score: float) -> bytes:
    """Render the prediction report and return it as PDF bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Academic Performance Prediction Report",
    )
    story = []

    story.append(Paragraph(PROJECT_TITLE, STYLES["Title"]))
    story.append(Paragraph(f"{PROJECT_TEAM}<br/>{PROJECT_ORG}", STYLES["Normal"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(_table(
        [["Report generated", datetime.now().strftime("%d %b %Y, %H:%M")]],
        widths=[4.5 * cm, 12 * cm],
    ))
    story.append(Spacer(1, 0.6 * cm))

    # 1. Student details
    story.append(Paragraph("<b>1. Student Details</b>", STYLES["Heading2"]))
    detail_rows = []
    for key, value in record.items():
        if value is None or (isinstance(value, float) and value != value):
            value = "Not provided"
        elif isinstance(value, float):
            value = f"{value:.2f}"
        label = FIELD_LABELS.get(key, key.replace("_", " ").title())
        if isinstance(value, float):
            value = f"{value:.2f}"
        detail_rows.append([label, str(value)])
    story.append(_table(detail_rows, widths=[7 * cm, 9.5 * cm]))
    story.append(Spacer(1, 0.6 * cm))

    # 2. Dropout risk
    story.append(Paragraph("<b>2. Dropout Risk Prediction</b>", STYLES["Heading2"]))
    risk_rows = [
        ["Predicted status", dropout["label"]],
        ["Dropout probability", f"{dropout['probability'] * 100:.1f}%"],
        ["Risk level", risk_level],
        ["Model", "Hybrid Random Forest + XGBoost (soft voting)"],
    ]
    risk_table = _table(risk_rows, widths=[7 * cm, 9.5 * cm])
    risk_table.setStyle(TableStyle(
        [("TEXTCOLOR", (1, 2), (1, 2), RISK_COLORS.get(risk_level, colors.black)),
         ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
         ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]
    ))
    story.append(risk_table)
    story.append(Spacer(1, 0.6 * cm))

    # 3. Exam score
    story.append(Paragraph("<b>3. Exam Score Prediction</b>", STYLES["Heading2"]))
    score_rows = [
        ["Predicted exam score", f"{exam_score:.1f} / 100"],
        ["Performance band", score_band(exam_score)],
        ["Model", "Hybrid Random Forest + XGBoost (stacking)"],
    ]
    score_table = _table(score_rows, widths=[7 * cm, 9.5 * cm])
    score_table.setStyle(TableStyle(
        [("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]
    ))
    story.append(score_table)
    story.append(Spacer(1, 0.6 * cm))

    # 4. Guidance
    story.append(Paragraph("<b>4. Suggested Actions</b>", STYLES["Heading2"]))
    actions = []
    if risk_level == "High":
        actions.append("Schedule a one-on-one counselling session with the student this week.")
    elif risk_level == "Medium":
        actions.append("Monitor progress regularly and share study-planning resources.")
    else:
        actions.append("Continue the current support plan; no immediate intervention needed.")
    if exam_score < 60:
        actions.append("Predicted score is below the pass comfort zone - arrange remedial classes and tutoring.")
    elif exam_score < 75:
        actions.append("Predicted score is average - targeted practice in weak subjects is recommended.")
    actions.append("Review attendance, stress and sleep inputs: these are the strongest predictors in the model.")
    for action in actions:
        story.append(Paragraph(f"&bull; {action}", STYLES["Normal"]))

    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "<i>Generated by the Academic Performance Prediction system. "
        "Predictions are decision support only and should be reviewed by an educator.</i>",
        STYLES["Italic"],
    ))

    doc.build(story)
    return buffer.getvalue()
