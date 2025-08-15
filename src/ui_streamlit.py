"""Streamlit dashboard for Healthcare Predictive Analytics Platform.

Run locally after training the model:
    streamlit run src/ui_streamlit.py

Features:
- Load processed features and show a table of encounters
- Select an encounter_id to view risk score and SHAP-like bar chart
- Displays top drivers and recommendations (static rules)
"""
import streamlit as st
import pandas as pd
import os, joblib, numpy as np

DATA_PROC = "data/processed"

st.set_page_config(page_title="HPAP Dashboard", layout="wide")

st.title("Healthcare Predictive Analytics — Demo Dashboard")
st.caption("GenAI-augmented readmission risk (demo)")

# Load processed features if available
features_path = os.path.join(DATA_PROC, "features.parquet")
model_path = os.path.join(DATA_PROC, "model.joblib")

if not os.path.exists(features_path):
    st.warning("Processed features not found. Run feature engineering first. For demo, sample screenshots are included.")
    st.image('docs/screenshots/risk_dashboard.png', caption='Demo risk dashboard screenshot', use_column_width=True)
    st.image('docs/screenshots/shap_bar.png', caption='Demo SHAP bar chart', use_column_width=True)
    st.stop()

df = pd.read_parquet(features_path)
st.sidebar.header("Controls")
sample_n = st.sidebar.slider("Sample size", min_value=10, max_value=500, value=100)
sample_df = df.sample(n=min(sample_n, len(df)), random_state=42)
encounter_id = st.sidebar.selectbox("Select encounter_id", options=sample_df['encounter_id'].tolist())

st.subheader("Selected encounter details")
row = df[df['encounter_id']==encounter_id].iloc[0]
st.write(row[['patient_id','age','gender','chronic_conditions_count','social_risk_score']])

# Risk display (if model exists)
if os.path.exists(model_path):
    clf = joblib.load(model_path)
    X = df[df['encounter_id']==encounter_id].drop(columns=['encounter_id','patient_id','admit_date','discharge_date','readmitted_30d'], errors='ignore')
    proba = float(clf.predict_proba(X)[:,1][0]) if hasattr(clf, 'predict_proba') else float(clf.predict(X)[0])
    st.metric(label="Predicted 30-day readmission risk", value=f"{proba:.2%}")
else:
    st.info("Model artifact not found — displaying demo risk (15.4%).")
    proba = 0.154
    st.metric(label="Predicted 30-day readmission risk", value=f"{proba:.2%}")


# Top drivers (demo/shap)
st.subheader("Top contributing features (approx.)")
# For demo, use random importance
np.random.seed(0)
features = ['age','social_risk_score','chronic_conditions_count','hb','creatinine','glucose']
importances = np.abs(np.random.randn(len(features)))
imp_df = pd.DataFrame({'feature': features, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
st.bar_chart(imp_df.set_index('feature'))

st.subheader("Recommended preventive actions")
st.markdown("- Schedule primary care follow-up within 7 days\n- Medication reconciliation before discharge\n- Social work consult for patients with limited support\n- Home health referral for high frailty indicators")