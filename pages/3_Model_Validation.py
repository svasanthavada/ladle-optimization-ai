import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.multioutput_regression import run_multioutput_regression
from src.xgboost_regression import run_xgboost_regression
from src.model_xgb_ranked import run_xgboost_regression_ranked
from src.tabtransformer import run_tabtransformer_model

st.set_page_config(layout="wide")
st.title("🔍 Alloy Model Training & Validation")

# Use uploaded file from session state
if "uploaded_file" not in st.session_state:
    st.error("❗ Please upload a dataset from the Home page.")
    st.stop()

uploaded_file = st.session_state["uploaded_file"]

if not uploaded_file:
    st.info("Please upload a file to enable model training.")
    st.stop()

# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = {}

# --- XGBOOST ---
st.header("⚡ XGBoost Regression")
if st.button("Run XGBoost Model"):
    st.session_state.results["xgb"] = run_xgboost_regression(uploaded_file)

if "xgb" in st.session_state.results:
    results = st.session_state.results["xgb"]
    st.success("✅ XGBoost model trained.")
    st.write(f"📉 RMSE: {results['rmse']}")
    st.write(f"📈 R² Score: {results['r2']}")
    with open(results["model_path"], "rb") as f:
        st.download_button("📦 Download XGBoost Model", data=f, file_name="xgb_model.pkl")

# --- RANKED XGBOOST ---
st.header("🏅 Ranked XGBoost (Top Alloys)")
if st.button("Run Ranked XGBoost"):
    st.session_state.results["ranked_xgb"] = run_xgboost_regression_ranked(uploaded_file)

if "ranked_xgb" in st.session_state.results:
    results = st.session_state.results["ranked_xgb"]
    st.success("✅ Ranked XGBoost model trained.")
    st.write(f"📉 RMSE: {results['rmse']}")
    st.write(f"📈 R² Score: {results['r2']}")
    with open(results["model_path"], "rb") as f:
        st.download_button("📦 Download Ranked XGBoost Model", data=f, file_name="ranked_xgb_model.pkl")

# --- TABTRANSFORMER ---
st.header("📘 TabTransformer Model")
if st.button("Run TabTransformer"):
    st.session_state.results["tabtransformer"] = run_tabtransformer_model(uploaded_file)

if "tabtransformer" in st.session_state.results:
    results = st.session_state.results["tabtransformer"]
    st.success("✅ TabTransformer model trained.")
    st.write(f"📉 RMSE: {results['rmse']}")
    st.write(f"📈 R² Score: {results['r2']}")
    with open(results["model_path"], "rb") as f:
        st.download_button("📦 Download TabTransformer Model", data=f, file_name="tabtransformer_model.pth")
