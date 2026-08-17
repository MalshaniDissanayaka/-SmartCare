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
import plotly.graph_objects as go

st.set_page_config(
    page_title="SmartCare — Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Brand palette (matches the project's report / slide deck)
# ---------------------------------------------------------------------------
NAVY_DARK = "#16283F"
NAVY = "#1F3A5F"
TEAL = "#2E8B8B"
ORANGE = "#D97B4F"
CARD_BG = "#1B2E4A"
INK = "#E8EDF2"
MUTED = "#9FB1C4"
GOOD = "#3FA46A"
BAD = "#D9534F"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at top left, #142238 0%, #0F1E33 45%, #0B1729 100%);
    }}
    #MainMenu, footer {{visibility: hidden;}}

    /* Hero banner */
    .hero {{
        background: linear-gradient(120deg, {NAVY_DARK} 0%, {NAVY} 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(255,255,255,0.06);
    }}
    .hero-kicker {{
        color: {TEAL}; font-size: 0.8rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.3rem;
    }}
    .hero-title {{
        color: {INK}; font-size: 2rem; font-weight: 800; margin: 0 0 0.4rem 0; line-height: 1.15;
    }}
    .hero-sub {{ color: {MUTED}; font-size: 0.98rem; margin-bottom: 0.9rem; }}
    .badge-row {{ display: flex; gap: 0.6rem; flex-wrap: wrap; }}
    .badge {{
        background: rgba(46,139,139,0.15); border: 1px solid rgba(46,139,139,0.4);
        color: {INK}; padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.82rem;
    }}
    .badge b {{ color: {TEAL}; }}

    /* Disclaimer strip */
    .disclaimer {{
        background: rgba(217,123,79,0.12); border-left: 4px solid {ORANGE};
        border-radius: 8px; padding: 0.75rem 1rem; color: {INK}; font-size: 0.88rem; margin-bottom: 1.4rem;
    }}

    /* Section headers */
    .section-head {{
        display: flex; align-items: center; gap: 0.5rem;
        color: {INK}; font-size: 1.1rem; font-weight: 700; margin: 0.2rem 0 0.8rem 0;
    }}
    .section-head .dot {{
        width: 10px; height: 10px; border-radius: 50%; background: {TEAL}; display: inline-block;
    }}

    /* Cards (form groups) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {CARD_BG} !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }}

    /* Buttons */
    .stButton>button, .stFormSubmitButton>button {{
        background: linear-gradient(120deg, {TEAL}, #257575) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; padding: 0.7rem 1rem !important;
        box-shadow: 0 6px 16px rgba(46,139,139,0.35);
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{ filter: brightness(1.08); }}

    /* Risk result card */
    .risk-card {{
        border-radius: 16px; padding: 1.4rem 1.6rem; text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .risk-card.high {{ background: rgba(217,83,79,0.14); border-color: rgba(217,83,79,0.45); }}
    .risk-card.low {{ background: rgba(63,164,106,0.14); border-color: rgba(63,164,106,0.45); }}
    .risk-label {{ font-size: 1.3rem; font-weight: 800; margin-bottom: 0.2rem; }}
    .risk-label.high {{ color: {BAD}; }}
    .risk-label.low {{ color: {GOOD}; }}
    .risk-sub {{ color: {MUTED}; font-size: 0.9rem; }}

    /* Recommendation card */
    .reco-card {{
        background: {CARD_BG}; border-radius: 14px; padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.06); height: 100%;
    }}
    .reco-card h4 {{ color: {INK}; margin-top: 0; }}
    .reco-card li {{ color: {MUTED}; margin-bottom: 0.35rem; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {NAVY_DARK} !important; border-right: 1px solid rgba(255,255,255,0.06);
    }}
    .sb-title {{ color: {INK}; font-size: 1.15rem; font-weight: 800; margin-bottom: 0.1rem; }}
    .sb-sub {{ color: {MUTED}; font-size: 0.82rem; margin-bottom: 1rem; }}
    .sb-metric-label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .sb-metric-value {{ color: {TEAL}; font-size: 1.35rem; font-weight: 800; margin-bottom: 0.7rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model + metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("models/readmission_model.pkl")
    with open("models/model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, meta = load_model()
category_options = meta["category_options"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-title">🏥 SmartCare AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">30-Day Readmission Risk Prototype</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-metric-label">Model</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-metric-value" style="font-size:1.05rem;">{meta["best_model_name"]}</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sb-metric-label">ROC-AUC</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-metric-value">{meta["test_roc_auc"]:.3f}</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown('<div class="sb-metric-label">F1 Score</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-metric-value">{meta["test_f1"]:.3f}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**How to use**")
    st.caption(
        "Fill in the patient's details on the right, then click **Predict "
        "Readmission Risk**. You'll get a risk score, a recommended action, "
        "and a chart explaining which factors drove that specific prediction."
    )
    st.markdown("---")
    st.warning(
        "Trained on a 330-record synthetic teaching dataset. Not a clinically "
        "validated tool — predictions should never replace clinical judgement.",
        icon="⚠️",
    )
    st.caption("CCS3440 — Artificial Intelligence Coursework · SLTC")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Smartcare Hospital · Decision Support</div>
        <div class="hero-title">🏥 Patient Readmission Risk Prototype</div>
        <div class="hero-sub">Enter a patient's clinical and operational details to estimate 30-day readmission risk, with a transparent, per-patient explanation of the result.</div>
        <div class="badge-row">
            <span class="badge">Model: <b>{meta['best_model_name']}</b></span>
            <span class="badge">Test ROC-AUC: <b>{meta['test_roc_auc']:.3f}</b></span>
            <span class="badge">Test F1: <b>{meta['test_f1']:.3f}</b></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer">⚠️ This prototype is a coursework decision-support '
    'demonstration on a small synthetic dataset — <b>not</b> a clinically validated '
    'tool, and predictions should never replace clinical judgement.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-head"><span class="dot"></span> Patient Information</div>',
    unsafe_allow_html=True
)

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", 0, 100, 45)
            gender = st.selectbox("Gender", category_options["gender"])
            department = st.selectbox("Department", category_options["department"])
            room_type = st.selectbox("Room Type", category_options["room_type"])

    with col2:
        with st.container(border=True):
            st.markdown("**🩺 Clinical Measurements**")
            systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 130)
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
            blood_sugar = st.slider("Blood Sugar (mg/dL)", 60, 300, 110)
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 350, 200)
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)

    with col3:
        with st.container(border=True):
            st.markdown("**📋 Admission & History**")
            length_of_stay = st.slider("Length of Stay (days)", 0, 30, 4)
            previous_admissions = st.slider("Previous Admissions", 0, 10, 1)
            previous_appointments = st.slider("Previous Appointments", 0, 15, 3)
            missed_appointments = st.slider("Missed Previous Appointments", 0, 10, 0)
            waiting_days = st.slider("Waiting Days (booking to appointment)", 0, 60, 14)

    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):
            st.markdown("**🩹 Diagnosis & Treatment**")
            diagnosis_group = st.selectbox(
                "Diagnosis Group",
                category_options["diagnosis_group"]
            )
            lab_tests_count = st.slider("Lab Tests Count", 0, 15, 3)

    with col5:
        with st.container(border=True):
            st.markdown("**💊 Treatment & Payment**")
            treatments_count = st.slider("Treatments Count", 0, 15, 3)
            payment_status = st.selectbox(
                "Payment Status",
                category_options["payment_status"]
            )

    with col6:
        with st.container(border=True):
            st.markdown("**💰 Charges (LKR)**")
            consultation_fee = st.number_input(
                "Consultation Fee", 0, 10000, 2000
            )
            room_charge = st.number_input(
                "Room Charge", 0, 200000, 5000
            )
            lab_charge = st.number_input(
                "Lab Charge", 0, 50000, 5000
            )
            medicine_charge = st.number_input(
                "Medicine Charge", 0, 50000, 8000
            )

    st.write("")
    submitted = st.form_submit_button(
        "🔍  Predict Readmission Risk",
        width="stretch"
    )

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


def make_gauge(proba: float) -> go.Figure:
    color = BAD if proba >= 0.5 else GOOD

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"color": INK, "size": 40}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": MUTED,
                    "tickfont": {"color": MUTED}
                },
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(63,164,106,0.18)"},
                    {"range": [40, 70], "color": "rgba(217,123,79,0.18)"},
                    {"range": [70, 100], "color": "rgba(217,83,79,0.18)"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 2},
                    "thickness": 0.8,
                    "value": proba * 100
                },
            },
        )
    )

    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": INK},
    )

    return fig


if submitted:
    total_bill = (
        consultation_fee
        + room_charge
        + lab_charge
        + medicine_charge
    )

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

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------
    proba = pipeline.predict_proba(patient_row)[0, 1]
    prediction = pipeline.predict(patient_row)[0]

    st.write("")
    st.markdown(
        '<div class="section-head"><span class="dot"></span> Prediction Result</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns([1, 1, 1.4])

    with r1:
        risk_class = "high" if prediction == 1 else "low"
        risk_text = "⚠️ HIGH RISK" if prediction == 1 else "✅ LOWER RISK"

        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
                <div class="risk-label {risk_class}">{risk_text}</div>
                <div class="risk-sub">Predicted probability of 30-day readmission</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        st.plotly_chart(
            make_gauge(proba),
            width="stretch",
            config={"displayModeBar": False}
        )

    with r3:
        if proba >= 0.7:
            actions = [
                "Flag for structured discharge planning and a follow-up call within 7 days.",
                "Consider a post-discharge home-visit or telehealth check-in.",
                "Review medication adherence plan before discharge.",
            ]
        elif proba >= 0.4:
            actions = [
                "Standard discharge planning with a follow-up appointment reminder.",
                "Monitor at next scheduled visit.",
            ]
        else:
            actions = [
                "Standard discharge process; no additional intervention indicated by the model."
            ]

        items = "".join(f"<li>{a}</li>" for a in actions)

        st.markdown(
            f'<div class="reco-card"><h4>Recommended action</h4><ul>{items}</ul></div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------------------------
    # Local explanation (SHAP) for this specific patient
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown(
        '<div class="section-head"><span class="dot"></span> Why This Prediction? (Explainable AI)</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Computing explanation..."):
        try:
            prep = pipeline.named_steps["prep"]
            clf = pipeline.named_steps["clf"]

            transformed = prep.transform(patient_row)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = prep.get_feature_names_out()
            transformed_df = pd.DataFrame(
                transformed,
                columns=feature_names
            )

            if hasattr(clf, "feature_importances_") or type(clf).__name__ in (
                "RandomForestClassifier",
                "XGBClassifier"
            ):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(transformed_df)

                if isinstance(sv, list):
                    sv = sv[1]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    sv = sv[:, :, 1]

            else:
                explainer = shap.LinearExplainer(clf, transformed_df)
                sv = explainer.shap_values(transformed_df)

            contrib = pd.Series(
                sv[0],
                index=feature_names
            ).sort_values(
                key=abs,
                ascending=False
            ).head(8)

            plt.rcParams.update({
                "figure.facecolor": "none",
                "axes.facecolor": "none",
                "text.color": INK,
                "axes.labelcolor": MUTED,
                "xtick.color": MUTED,
                "ytick.color": INK,
                "axes.edgecolor": "#33465F",
            })

            fig, ax = plt.subplots(figsize=(8, 4))

            colors = [
                ORANGE if v > 0 else TEAL
                for v in contrib.values
            ]

            ax.barh(
                contrib.index[::-1],
                contrib.values[::-1],
                color=colors[::-1]
            )

            ax.set_xlabel(
                "SHAP value (impact on readmission probability)"
            )

            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_alpha(0.0)

            with st.container(border=True):
                st.pyplot(fig, transparent=True)

                st.caption(
                    f'🟧 Orange bars push risk higher · 🟦 Teal bars push risk lower — '
                    f'these are the top factors for this specific patient.'
                )

        except Exception as e:
            st.warning(
                f"Explanation unavailable for this configuration ({e})."
            )

st.write("")

st.markdown(
    f'<div style="text-align:center; color:{MUTED}; font-size:0.82rem; padding:1rem 0;">'
    "SmartCare Hospital AI Coursework (CCS3440) — Prototype built with Streamlit. "
    "Model trained on the SmartCare Hospital AI Dataset (n=1000; admitted-patient subset n=330)."
    "</div>",
    unsafe_allow_html=True,
)
