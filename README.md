# SmartCare Hospital — AI-Powered 30-Day Patient Readmission Prediction

CCS3440 — Artificial Intelligence Coursework (SLTC, School of Computing & IT)

A decision-support prototype that predicts whether a hospital patient is likely to be
re-admitted within 30 days of discharge, trained on the SmartCare Hospital AI Dataset,
with SHAP-based explainability and a Streamlit front end.

## Repository Contents

| File | Description |
|---|---|
| `SmartCare_Technical_Report.pdf` | Full technical report (all 14 required sections) |
| `SmartCare_Readmission_Prediction.ipynb` | End-to-end notebook: preprocessing, EDA, model training, evaluation, SHAP |
| `SmartCare_Readmission_Prediction.py` | Notebook exported as a plain Python script |
| `app.py` | Streamlit prototype — loads the trained pipeline and serves predictions |
| `models/readmission_model.pkl` | Trained scikit-learn pipeline (Logistic Regression, best ROC-AUC) |
| `models/model_metadata.json` | Feature lists, category options, numeric ranges, test metrics |
| `models/model_comparison_results.csv` | Accuracy / Precision / Recall / F1 / ROC-AUC for all 4 models |
| `data/smartcare_ai_dataset_1000.csv` | Source dataset (1,000 records) |
| `data/smartcare_ai_dataset_data_dictionary.csv` | Attribute definitions |
| `SmartCare_Presentation.pptx` | Presentation slides |

## Running the Prototype

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `models/readmission_model.pkl` and `models/model_metadata.json`, so run it
from the repository root (or adjust the paths in `app.py` if you restructure folders).

## Model Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.636 | 0.829 | 0.667 | 0.739 | **0.637** |
| XGBoost | 0.621 | 0.810 | 0.667 | 0.731 | 0.616 |
| SVM (RBF) | 0.682 | 0.826 | 0.745 | 0.784 | 0.593 |
| Random Forest | 0.788 | 0.785 | 1.000 | 0.879 | 0.536 |

Logistic Regression was selected on ROC-AUC (threshold-independent ranking ability),
not raw accuracy — see Section 8.4 of the technical report for the full justification.

## Disclaimer

This is a coursework proof-of-concept trained on a 330-record synthetic dataset. It is
**not clinically validated** and must not be used to inform real patient care decisions.

## Team

| Name | Student ID | Contribution |
|---|---|---|
| [Name] | [ID] | [e.g. Literature review & technical report] |
| [Name] | [ID] | [e.g. Data preprocessing & feature engineering] |
| [Name] | [ID] | [e.g. Model development & evaluation] |
| [Name] | [ID] | [e.g. Explainable AI & prototype] |
