import streamlit as st
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model 
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError 

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ===================================================================
# Fuzzy Logic
# ===================================================================
def predict_fuzzy_salary(age_input, rating_input):
    age = ctrl.Antecedent(np.arange(20, 71, 1), 'age')
    rating = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'rating')
    salary = ctrl.Consequent(np.arange(40, 200, 1), 'salary')

    age['young'] = fuzz.trimf(age.universe, [20, 20, 35])
    age['mid'] = fuzz.trimf(age.universe, [30, 45, 60])
    age['old'] = fuzz.trimf(age.universe, [55, 70, 70])

    rating['low'] = fuzz.trimf(rating.universe, [0, 0, 2.5])
    rating['medium'] = fuzz.trimf(rating.universe, [2, 3.5, 4.5])
    rating['high'] = fuzz.trimf(rating.universe, [4, 5, 5])
    
    salary['low'] = fuzz.trimf(salary.universe, [40, 40, 80])
    salary['medium'] = fuzz.trimf(salary.universe, [70, 110, 150])
    salary['high'] = fuzz.trimf(salary.universe, [140, 200, 200])

    rules = [
        ctrl.Rule(age['young'] & rating['low'], salary['low']),
        ctrl.Rule(age['mid'] & rating['medium'], salary['medium']),
        ctrl.Rule(age['old'] & rating['high'], salary['high']),
        ctrl.Rule(age['young'] & rating['high'], salary['medium']),
        ctrl.Rule(age['old'] & rating['low'], salary['medium']),
        ctrl.Rule(age['mid'] & rating['low'], salary['low']),
        ctrl.Rule(age['mid'] & rating['high'], salary['high'])
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    try:
        sim.input['age'] = age_input
        sim.input['rating'] = rating_input
        sim.compute()
        return sim.output['salary']
    except:
        return np.nan


# ===================================================================
# Load ANN Model
# ===================================================================
@st.cache_resource
def load_all_assets():
    try:
        custom = {'mse': MeanSquaredError(), 'mae': MeanAbsoluteError()}
        model = load_model("ann_salary_model.h5", custom_objects=custom)
        scaler = joblib.load("scaler_for_ann.pkl")
        features = joblib.load("ann_model_features.pkl")
        return model, scaler, features
    except Exception as e:
        st.error(f"Load error: {e}")
        return None, None, None


ann_model, scaler, ann_features = load_all_assets()


# ===================================================================
# PAGE CONFIG + CSS DASHBOARD PERUSAHAAN
# ===================================================================
st.set_page_config(page_title="Salary Prediction Dashboard", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #f5f7fa;
    }
    .header-title {
        font-size: 34px;
        font-weight: 700;
        color: #003366;
        padding-bottom: 5px;
    }
    .subtext {
        font-size: 15px;
        color: #555;
        margin-bottom: 20px;
    }
    .card {
        background: white;
        padding: 25px 30px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 700;
        color: #003366;
    }
    .metric-label {
        color: #666;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# HEADER
# ===================================================================
st.markdown("<div class='header-title'>Salary Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Hybrid Model: ANN Regression + Fuzzy Logic Rules</div>", unsafe_allow_html=True)

st.write("")


# ===================================================================
# SIDEBAR INPUT
# ===================================================================
st.sidebar.markdown("### Input Employee Data")

age = st.sidebar.slider("Age", 18, 70, 30)
rating = st.sidebar.slider("Company Rating", 0.0, 5.0, 4.0, 0.1)
python_raw = st.sidebar.selectbox("Python Skill", ["Yes", "No"])
ny_raw = st.sidebar.selectbox("Located in New York", ["No", "Yes"])

python_yn = 1.0 if python_raw == "Yes" else 0.0
loc_ny = 1.0 if ny_raw == "Yes" else 0.0

predict_btn = st.sidebar.button("Predict Salary")


# ===================================================================
# MAIN OUTPUT
# ===================================================================
if predict_btn:

    # ANN Input
    df = pd.DataFrame(0, index=[0], columns=ann_features)
    df["age"] = age
    df["Rating"] = rating
    df["python_yn"] = python_yn
    if "Location_New York" in df.columns:
        df["Location_New York"] = loc_ny

    scaled = scaler.transform(df)
    ann_pred = ann_model.predict(scaled)[0][0]

    fuzzy_pred = predict_fuzzy_salary(age, rating)

    # ===================== DASHBOARD LAYOUT =====================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Prediction Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='metric-label'>ANN Salary Prediction</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-val'>US$ {ann_pred:,.2f}K</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-label'>Fuzzy Logic Prediction</div>", unsafe_allow_html=True)
        if not np.isnan(fuzzy_pred):
            st.markdown(f"<div class='metric-val'>US$ {fuzzy_pred:,.2f}K</div>", unsafe_allow_html=True)
        else:
            st.warning("Fuzzy cannot calculate for this input.")

    st.markdown("</div>", unsafe_allow_html=True)
