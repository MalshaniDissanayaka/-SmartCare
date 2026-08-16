"""
SmartCare Hospital — 30-Day Readmission Risk Decision-Support Prototype
=========================================================================
CCS3440 Artificial Intelligence Coursework — Task 08 (AI Prototype Development)

Run with:
    streamlit run app.py

Loads the trained pipeline exported by the notebook (models/readmission_model.pkl)
and lets a clinician / ward administrator enter a patient's details to obtain a
30-day readmission risk score, together with a plain-language explanation of the
main drivers behind that specific prediction (SHAP-based local explanation).
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartCare — Readmission Risk", page_icon="🏥", layout="wide")

# ---------------------------------------------------------------------------
# Load model + metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("models/readmission_model.pkl")
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_model()
numeric_features = meta["numeric_features"]
categorical_features = meta["categorical_features"]
category_options = meta["category_options"]
numeric_ranges = meta["numeric_ranges"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏥 SmartCare Hospital — Patient Readmission Risk Prototype")
st.caption(
    f"Decision-support tool for 30-day readmission risk. "
    f"Model: **{meta['best_model_name']}** · Test ROC-AUC: **{meta['test_roc_auc']:.3f}** · "
    f"Test F1: **{meta['test_f1']:.3f}**"
)
st.info(
    "⚠️ This prototype is trained on a 330-record synthetic teaching dataset. "
    "It is a coursework decision-support demonstration, **not** a clinically validated tool, "
    "and predictions should never replace clinical judgement.",
    icon="⚠️",
)

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Patient Information")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 0, 100, 45)
        gender = st.selectbox("Gender", category_options["gender"])
        department = st.selectbox("Department", category_options["department"])
        room_type = st.selectbox("Room Type", category_options["room_type"])

    with col2:
        st.markdown("**Clinical Measurements**")
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 130)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
        blood_sugar = st.slider("Blood Sugar (mg/dL)", 60, 300, 110)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 350, 200)
        bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)

    with col3:
        st.markdown("**Admission & History**")
        length_of_stay = st.slider("Length of Stay (days)", 0, 30, 4)
        previous_admissions = st.slider("Previous Admissions", 0, 10, 1)
        previous_appointments = st.slider("Previous Appointments", 0, 15, 3)
        missed_appointments = st.slider("Missed Previous Appointments", 0, 10, 0)
        waiting_days = st.slider("Waiting Days (booking to appointment)", 0, 60, 14)

    st.markdown("**Diagnosis & Treatment**")
    col4, col5, col6 = st.columns(3)
    with col4:
        diagnosis_group = st.selectbox("Diagnosis Group", category_options["diagnosis_group"])
        lab_tests_count = st.slider("Lab Tests Count", 0, 15, 3)
    with col5:
        treatments_count = st.slider("Treatments Count", 0, 15, 3)
        payment_status = st.selectbox("Payment Status", category_options["payment_status"])
    with col6:
        st.markdown("**Charges (LKR)**")
        consultation_fee = st.number_input("Consultation Fee", 0, 10000, 2000)
        room_charge = st.number_input("Room Charge", 0, 200000, 5000)
        lab_charge = st.number_input("Lab Charge", 0, 50000, 5000)
        medicine_charge = st.number_input("Medicine Charge", 0, 50000, 8000)

    submitted = st.form_submit_button("🔍 Predict Readmission Risk", use_container_width=True)

# ---------------------------------------------------------------------------
# Derived / engineered features (mirrors notebook Section 3.5)
# ---------------------------------------------------------------------------
def bmi_category(b):
    if b < 18.5:
        return "Underweight"
    elif b < 25:
        return "Normal"
    elif b < 30:
        return "Overweight"
    else:
        return "Obese"


def bp_category(systolic, diastolic):
    if systolic >= 140 or diastolic >= 90:
        return "Hypertensive"
    elif systolic >= 130 or diastolic >= 80:
        return "Elevated"
    else:
        return "Normal"


if submitted:
    total_bill = consultation_fee + room_charge + lab_charge + medicine_charge
    cost_per_day = total_bill / (length_of_stay + 1)
    prior_utilization = previous_admissions + previous_appointments

    patient_row = pd.DataFrame([{
        "age": age,
        "waiting_days": waiting_days,
        "previous_appointments": previous_appointments,
        "missed_previous_appointments": missed_appointments,
        "length_of_stay_days": length_of_stay,
        "previous_admissions": previous_admissions,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "blood_sugar_mg_dl": blood_sugar,
        "cholesterol_mg_dl": cholesterol,
        "bmi": bmi,
        "lab_tests_count": lab_tests_count,
        "treatments_count": treatments_count,
        "consultation_fee_lkr": consultation_fee,
        "room_charge_lkr": room_charge,
        "lab_charge_lkr": lab_charge,
        "medicine_charge_lkr": medicine_charge,
        "prior_utilization": prior_utilization,
        "cost_per_day": cost_per_day,
        "gender": gender,
        "department": department,
        "room_type": room_type,
        "payment_status": payment_status,
        "bmi_category": bmi_category(bmi),
        "bp_category": bp_category(systolic_bp, diastolic_bp),
        "diagnosis_group": diagnosis_group,
    }])

    proba = pipeline.predict_proba(patient_row)[0, 1]
    prediction = pipeline.predict(patient_row)[0]

    st.divider()
    st.subheader("Prediction Result")

    r1, r2 = st.columns([1, 2])
    with r1:
        if prediction == 1:
            st.error(f"### ⚠️ HIGH RISK\n**{proba:.1%}** probability of 30-day readmission")
        else:
            st.success(f"### ✅ LOWER RISK\n**{proba:.1%}** probability of 30-day readmission")
        st.progress(min(max(proba, 0.0), 1.0))

    with r2:
        st.markdown("**Recommended action**")
        if proba >= 0.7:
            st.write(
                "- Flag for structured discharge planning and a follow-up call within 7 days.\n"
                "- Consider a post-discharge home-visit or telehealth check-in.\n"
                "- Review medication adherence plan before discharge."
            )
        elif proba >= 0.4:
            st.write(
                "- Standard discharge planning with a follow-up appointment reminder.\n"
                "- Monitor at next scheduled visit."
            )
        else:
            st.write("- Standard discharge process; no additional intervention indicated by the model.")

    # -----------------------------------------------------------------
    # Local explanation (SHAP) for this specific patient
    # -----------------------------------------------------------------
    st.subheader("Why this prediction? (Explainable AI)")
    with st.spinner("Computing explanation..."):
        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        transformed = prep.transform(patient_row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = prep.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        try:
            if hasattr(clf, "feature_importances_") or type(clf).__name__ in ("RandomForestClassifier", "XGBClassifier"):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(transformed_df)
                if isinstance(sv, list):
                    sv = sv[1]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    sv = sv[:, :, 1]
            else:
                explainer = shap.LinearExplainer(clf, transformed_df)
                sv = explainer.shap_values(transformed_df)

            contrib = pd.Series(sv[0], index=feature_names).sort_values(key=abs, ascending=False).head(8)
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#DD8452" if v > 0 else "#4C72B0" for v in contrib.values]
            ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
            ax.set_xlabel("SHAP value (impact on readmission probability)")
            ax.set_title("Top factors behind this prediction")
            st.pyplot(fig)
            st.caption("Orange bars push the prediction toward *higher* readmission risk; blue bars push it lower.")
        except Exception as e:
            st.warning(f"Explanation unavailable for this configuration ({e}).")

st.divider()
st.caption(
    "SmartCare Hospital AI Coursework (CCS3440) — Prototype built with Streamlit. "
    "Model trained on the SmartCare Hospital AI Dataset (n=1000; admitted-patient subset n=330)."
)
