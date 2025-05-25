import streamlit as st
import os
import sys
import joblib
import numpy as np

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

# --- Hyperparameter Controls ---
with st.sidebar:
    st.header("XGBoost Hyperparameters")
    xgb_n_estimators = st.number_input("XGB n_estimators", min_value=10, max_value=1000, value=100, step=10, key="xgb_n_estimators")
    xgb_max_depth = st.number_input("XGB max_depth", min_value=1, max_value=20, value=6, step=1, key="xgb_max_depth")
    xgb_learning_rate = st.slider("XGB learning_rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, key="xgb_learning_rate")

    st.header("Ranked XGBoost Hyperparameters")
    ranked_xgb_n_estimators = st.number_input("Ranked XGB n_estimators", min_value=10, max_value=1000, value=100, step=10, key="ranked_xgb_n_estimators")
    ranked_xgb_max_depth = st.number_input("Ranked XGB max_depth", min_value=1, max_value=20, value=6, step=1, key="ranked_xgb_max_depth")
    ranked_xgb_learning_rate = st.slider("Ranked XGB learning_rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, key="ranked_xgb_learning_rate")
    ranked_xgb_top_n = st.number_input("Ranked XGB Top N Alloys", min_value=5, max_value=30, value=10, step=1, key="ranked_xgb_top_n")

    st.header("TabTransformer Hyperparameters")
    tabtransformer_epochs = st.number_input("TabTransformer epochs", min_value=10, max_value=500, value=200, step=10, key="tabtransformer_epochs")
    tabtransformer_lr = st.slider(
        "TabTransformer learning_rate",
        min_value=0.001,
        max_value=0.1,
        value=0.001,
        step=0.001,
        format="%.3f",
        key="tabtransformer_lr"
    )
    
# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = {}

# --- XGBOOST ---
st.header("⚡ XGBoost Regression")
if st.button("Run XGBoost Model"):
    st.session_state.results["xgb"] = run_xgboost_regression(
        uploaded_file,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate
    )

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
    st.session_state.results["ranked_xgb"] = run_xgboost_regression_ranked(
        uploaded_file,
        top_n_alloys=ranked_xgb_top_n,
        n_estimators=ranked_xgb_n_estimators,
        max_depth=ranked_xgb_max_depth,
        learning_rate=ranked_xgb_learning_rate
    )

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
    st.session_state.results["tabtransformer"] = run_tabtransformer_model(
        uploaded_file,
        epochs=tabtransformer_epochs,
        learning_rate=tabtransformer_lr
        # Removed batch_size argument
    )

if "tabtransformer" in st.session_state.results:
    results = st.session_state.results["tabtransformer"]
    st.success("✅ TabTransformer model trained.")
    st.write(f"📉 RMSE: {results['rmse']}")
    st.write(f"📈 R² Score: {results['r2']}")
    with open(results["model_path"], "rb") as f:
        st.download_button("📦 Download TabTransformer Model", data=f, file_name="tabtransformer_model.pth")
