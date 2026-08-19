import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Task 3 & Task 4 - SmartCare",
    page_icon="📊",
    layout="wide",
)

st.title("Task 3 & Task 4: Preprocessing & Visualizations")

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
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2E8B8B"},
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=10))
    return fig

st.subheader("Task 3: Feature Engineering & Preprocessing")

col1, col2 = st.columns(2)
with col1:
    test_bmi = st.slider("Select BMI for Testing", 15.0, 40.0, 24.0)
    st.write("BMI Category Result:", bmi_category(test_bmi))

with col2:
    sys = st.slider("Systolic BP", 90, 200, 120)
    dia = st.slider("Diastolic BP", 50, 130, 80)
    st.write("Blood Pressure Category Result:", bp_category(sys, dia))

st.subheader("Task 4: Exploratory Data Analysis & Visualizations")

sample_proba = st.slider("Test Visualization Metric", 0.0, 1.0, 0.65)
st.plotly_chart(make_gauge(sample_proba), use_container_width=True)
