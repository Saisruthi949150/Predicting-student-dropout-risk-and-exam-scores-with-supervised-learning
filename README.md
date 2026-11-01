# Academic Performance Prediction Using Supervised Learning

Two supervised-learning systems are built end to end:

1. **Dropout risk classification** — predicts `dropout_risk` (Yes/No) from academic,
   psychological and lifestyle factors.
2. **Exam score regression** — predicts `exam_score` (0–100) from the same kind of factors.

Both models are served through a **Streamlit web app** with separate teacher/student roles
and an automatic **PDF prediction report**.

## Repository layout

```
01_Data_Preprocessing_and_EDA.ipynb      # load, audit, clean, explore, save cleaned data
02_Dropout_Risk_Prediction.ipynb         # Objective 1 — classification + model comparison
03_Exam_Score_Prediction.ipynb           # Objective 2 — regression + model comparison
04_Prediction_and_Report_Demo.ipynb      # end-to-end inference + PDF report

app.py                                   # Streamlit web application (teacher/student roles)
train.py                                 # script alternative: runs the whole pipeline

src/
  config.py         # paths, feature lists, training settings
  data.py           # loading, cleaning, risk banding
  features.py       # feature engineering + preprocessing pipelines
  selection.py      # documented model-selection rule (equivalence margin)
  train_dropout.py  # classification training module (used by train.py)
  train_exam_score.py # regression training module (used by train.py)
  inference.py      # model bundle save/load + single-record prediction
  report.py         # PDF report generation (reportlab)
  plots.py          # EDA and evaluation figures (used by train.py)

data/raw/student_habits.csv               # the Kaggle dataset (80,000 × 31)
data/processed/cleaned.csv                # written by notebook 01

models/best_dropout_model.pkl             # saved pipeline (preprocessing + model), compressed (~2.8 MB)
models/best_exam_model.pkl                # same, compressed (~89 MB)

reports/figures/                          # EDA and evaluation plots
reports/metrics/                          # metrics.json + per-model CSV tables
requirements.txt
```

## Running the project in VS Code

1. Install the project dependencies into the project environment:

   ```bash
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt     # Windows
   source .venv/bin/pip install -r requirements.txt  # macOS/Linux
   ```

2. Open this folder in VS Code with the **Jupyter** extension installed.

3. Open each notebook and click **Select Kernel** (top right) →
   **Python Environments** → the project `.venv`
   (registered as **Python (.venv - Student Performance)**).

4. Run the notebooks **in order** — each one feeds the next:

   | Notebook | Produces |
   |---|---|
   | `01_Data_Preprocessing_and_EDA.ipynb` | `data/processed/cleaned.csv`, EDA figures |
   | `02_Dropout_Risk_Prediction.ipynb` | `models/best_dropout_model.pkl`, `reports/metrics/dropout_results.csv` |
   | `03_Exam_Score_Prediction.ipynb` | `models/best_exam_model.pkl`, `reports/metrics/exam_results.csv`, `metrics.json` |
   | `04_Prediction_and_Report_Demo.ipynb` | sample predictions + a PDF report |

   Or run everything in one go from a terminal: `python train.py`

5. Launch the web application:

   ```bash
   streamlit run app.py
   ```

> **Model artifacts are committed, compressed.** `joblib` bundles are written with
> `compress=9` so they fit GitHub's 100 MB per-file limit (the exam-score bundle is
> ~578 MB raw, ~89 MB compressed); `joblib.load` decompresses transparently, so the app
> and notebooks need no special handling. Rebuild them anytime with `python train.py`.

## Dataset

**Student Habits and Academic Performance Dataset** (Kaggle: `aryan208/student-habits-and-academic-performance-dataset`)
— 80,000 synthetic student records × 31 attributes covering academic, behavioural,
lifestyle and psychological factors (study hours, attendance, sleep, stress, exam anxiety,
social media use, parental support, time management, …). Targets: `dropout_risk`
(Yes/No) and `exam_score` (0–100).

Place the CSV in `data/raw/` (already done in this repo) — any CSV placed there is picked up.

## Methodology

**Cleaning** — duplicate removal and 1.5 × IQR outlier removal on the core numeric columns
(`src/config.py: OUTLIER_COLUMNS`). Remaining missing values are imputed *inside* the model
pipeline (median / most-frequent), so blank inputs at inference time are handled safely.

**Feature engineering** (as specified in the design):

```
screen_time_total = social media hours + netflix hours + other screen time
study_sleep_ratio = study hours per day ÷ sleep hours
stress_anxiety    = stress level + exam anxiety score
```

**Preprocessing** — one `ColumnTransformer` for every model:
`StandardScaler` for numeric features, `OneHotEncoder(handle_unknown="ignore")` for categorical ones.

**Models compared**

| Objective | Baselines | Proposed model |
|---|---|---|
| Dropout risk | Logistic Regression, SVM (RBF), Random Forest, XGBoost | **Hybrid RF + XGBoost soft voting** (`VotingClassifier`, averaged probabilities) |
| Exam score | Random Forest, XGBoost, SVR | **Hybrid RF + XGBoost stacking** (`StackingRegressor`, XGBoost meta-learner) |

**Metrics** — classification: accuracy, precision, recall, F1, ROC-AUC;
regression: R², RMSE, MAE.

**Model selection rule** (`src/selection.py`) — the best score wins, but models within
**0.001** of it are treated as *equivalent*, and the proposed hybrid ensemble is preferred
among equivalent contenders. All raw per-model numbers are always written to
`reports/metrics/`, so the choice stays auditable.

**Runtime note** — the SVM/SVR baselines scale quadratically with sample count, so they train
on a capped 10,000-row subsample (`SVM_MAX_ROWS`); every other model uses the full training
split. The `train_rows` column in every results table makes this explicit.

## Results

Cleaning removed the duplicate row and the 1.5 × IQR outliers (80,000 → 78,838 rows); the
train/test split is 80/20 with `random_state=42`.

### Objective 1 — Dropout risk classification (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train rows |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9905 | 0.7871 | 0.6877 | 0.7340 | 0.9966 | 63,070 |
| SVM (RBF Kernel) * | 0.9898 | 0.7672 | 0.6678 | 0.7140 | 0.9949 | 10,000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 63,070 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 63,070 |
| **Hybrid RF-XGBoost (Soft Voting)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 63,070 |

\* SVM trains on a capped 10,000-row subsample (quadratic scaling).

**Selected model: Hybrid RF-XGBoost (Soft Voting)** → saved to `models/best_dropout_model.pkl`.

### Objective 2 — Exam score regression (test set)

| Model | R² | RMSE | MAE | Train rows |
|---|---|---|---|---|
| Random Forest | 0.8618 | 4.179 | 3.279 | 63,070 |
| XGBoost | 0.8659 | 4.117 | 3.206 | 63,070 |
| SVR (RBF Kernel) * | 0.8476 | 4.389 | 3.277 | 10,000 |
| **Hybrid RF-XGBoost (Stacking)** | **0.8654** | **4.125** | **3.217** | 63,070 |

\* SVR trains on a capped 10,000-row subsample (quadratic scaling).

**Selected model: Hybrid RF-XGBoost (Stacking)** → saved to `models/best_exam_model.pkl`.
XGBoost's raw R² (0.8659) is 0.0005 above the hybrid — inside the 0.001 equivalence
margin — so the proposed ensemble is chosen, as specified in the design.

Full per-model tables: `reports/metrics/dropout_results.csv`, `reports/metrics/exam_results.csv`,
`reports/metrics/metrics.json`. Every table carries a `train_rows` column showing exactly how
many rows each model trained on.

> **Important caveat — this dataset is synthetic.** Its `dropout_risk` label is generated
> from the same behavioural columns the model sees, so the tree-based classifiers reach
> (near-)perfect scores. That is a property of the data, not of the modelling; on real
> institutional data these numbers would be considerably lower. The pipeline itself —
> cleaning, engineering, evaluation, reporting — transfers as-is.

## Web application

```bash
streamlit run app.py
```

| Role | Permissions (per reviewer feedback, Final Review 1) |
|---|---|
| **Teacher** | Enters student details, runs dropout + exam score predictions, sees the risk meter and result card, downloads the PDF report |
| **Student** | Views an existing student's report only — no prediction interface |

Every prediction produces a PDF report (reportlab) containing the student details, the
dropout probability and risk level, the predicted exam score, and suggested actions.

## Project documents

The five capstone documents (Introduction Reviews 1–2, Final Reviews 1–2 and the CP report)
define this implementation: problem statement, objectives, requirement analysis,
UML/design diagrams, the two algorithm pseudocode blocks, testing strategy and results.
