import os
os.environ["STREAMLIT_WATCHER_IGNORE_MODULES"] = "torch"

import streamlit as st
import pandas as pd
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import random

from src.preprocessing import preprocess_pipeline
from src.tabtransformer import NumericalTabTransformer
from src.optimization import run_ga_optimization, run_pso_optimization

st.set_page_config(layout="wide")
st.title("🧬 Alloy Optimization using GA and PSO for All Models")

with st.sidebar:
    st.header("GA Hyperparameters")
    if "ga_population_size" not in st.session_state:
        st.session_state.ga_population_size = 50
    if "ga_generations" not in st.session_state:
        st.session_state.ga_generations = 50
    if "ga_crossover_prob" not in st.session_state:
        st.session_state.ga_crossover_prob = 0.5
    if "ga_mutation_prob" not in st.session_state:
        st.session_state.ga_mutation_prob = 0.2

    st.header("PSO Hyperparameters")
    if "pso_particles" not in st.session_state:
        st.session_state.pso_particles = 50
    if "pso_iterations" not in st.session_state:
        st.session_state.pso_iterations = 50
    if "pso_c1" not in st.session_state:
        st.session_state.pso_c1 = 0.5
    if "pso_c2" not in st.session_state:
        st.session_state.pso_c2 = 0.3
    if "pso_w" not in st.session_state:
        st.session_state.pso_w = 0.9

    reset_hyper = st.button("Reset Hyperparameters")

    if reset_hyper:
        st.session_state.ga_population_size = 50
        st.session_state.ga_generations = 50
        st.session_state.ga_crossover_prob = 0.5
        st.session_state.ga_mutation_prob = 0.2
        st.session_state.pso_particles = 50
        st.session_state.pso_iterations = 50
        st.session_state.pso_c1 = 0.5
        st.session_state.pso_c2 = 0.3
        st.session_state.pso_w = 0.9
        st.rerun()

    ga_population_size = st.number_input("GA Population Size", min_value=10, max_value=500, step=1, key="ga_population_size")
    ga_generations = st.number_input("GA Generations", min_value=10, max_value=500, step=1, key="ga_generations")
    ga_crossover_prob = st.slider("GA Crossover Probability", min_value=0.0, max_value=1.0, key="ga_crossover_prob")
    ga_mutation_prob = st.slider("GA Mutation Probability", min_value=0.0, max_value=1.0, key="ga_mutation_prob")
    pso_c1 = st.slider("PSO c1 (cognitive)", min_value=0.0, max_value=5.0, key="pso_c1")
    pso_c2 = st.slider("PSO c2 (social)", min_value=0.0, max_value=5.0, key="pso_c2")
    pso_w = st.slider("PSO inertia weight", min_value=0.0, max_value=1.0, key="pso_w")
    pso_particles = st.number_input("PSO Particles", min_value=10, max_value=500, step=1, key="pso_particles")
    pso_iterations = st.number_input("PSO Iterations", min_value=10, max_value=500, step=1, key="pso_iterations")
   

if "uploaded_file" not in st.session_state:
    st.error("❗ Please upload a dataset from the Home page.")
    st.stop()

uploaded_file = st.session_state["uploaded_file"]
if not uploaded_file:
    st.warning("Upload a dataset to start optimization.")
    st.stop()

# Rerun button
if "rerun_optimization" not in st.session_state:
    st.session_state.rerun_optimization = True

if not st.session_state.get("optimization_results") or st.button("🔁 Rerun Optimization"):
    st.session_state.rerun_optimization = True

if st.session_state.rerun_optimization:
    df, summary_df = preprocess_pipeline(uploaded_file)

    alloy_cols = [
        "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
        "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
        "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
        "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
        "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
    ]
    process_cols = [
        'Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm', 'LRF Holding Time-mm', 'LRF Lime',
        'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
    ]
    delta_cols = [f"Delta_{el.replace('%','')}" for el in process_cols if f"Delta_{el.replace('%','')}" in df.columns]
    features = [col for col in alloy_cols + process_cols + delta_cols if col in df.columns]
    target = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                                   'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'] if f"F-{el}" in df.columns]

    df_model = df.dropna(subset=features + target)

    def clean_datetime_columns(df):
        for col in df.columns:
            if df[col].dtype == 'O' or df[col].apply(lambda x: isinstance(x, datetime.time)).any():
                df[col] = df[col].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second if isinstance(t, datetime.time) else pd.to_numeric(t, errors='coerce'))
        return df.fillna(0)

    cleaned_df = clean_datetime_columns(df_model[features].copy())
    X = cleaned_df.copy()
    y = df_model[target].apply(pd.to_numeric, errors='coerce').fillna(0)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    aim_row = summary_df.iloc[3].fillna(0)
    chem_target = {f"F-{k.strip()}": float(v) for k, v in aim_row.items() if k.strip().endswith("%") and f"F-{k.strip()}" in target}
    base_inputs = cleaned_df.median(numeric_only=True).to_dict()

    models = {
        "TabTransformer": {
            "model": NumericalTabTransformer(input_dim=X_scaled.shape[1], output_dim=y_scaled.shape[1]),
            "path": "models/tabtransformer_model.pth",
            "type": "TabTransformer"
        },
        "XGBoost": {
            "model": joblib.load("models/xgboost_multioutput.pkl"),
            "type": "XGBoost"
        },
        "Ranked XGBoost": {
            "model": joblib.load("models/xgboost_multioutput_ranked.pkl"),
            "type": "Ranked XGBoost"
        }
    }

    results = {}
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    progress = st.progress(0)
    status = st.status("Running model optimizations...", expanded=True)
    total_steps = 2 * len(models)
    step = 0

    with status:
        st.write("Running GA optimization for all models...")
        for i, (name, entry) in enumerate(models.items()):
            st.write(f"GA: Optimizing with {name}...")
            model = entry["model"]
            model_type = entry["type"]

            if name == "TabTransformer":
                model.load_state_dict(torch.load(entry["path"]))
                model.eval()
            optimized_alloys, chem_final = run_pso_optimization(
                model=model,
                model_type=model_type,
                features=features,
                target=target,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                base_inputs=base_inputs,
                target_chemistry_dict=chem_target,
                alloy_cols=alloy_cols,
                df_successful=df_model,
                particles=pso_particles,
                iterations=pso_iterations,
                c1=pso_c1,
                c2=pso_c2,
                w=pso_w,
                seed=SEED
            )

            results[f"{name} + GA"] = {
                "alloys": optimized_alloys,
                "chem": chem_final
            }
            step += 1
            progress.progress(step / total_steps)

        st.write("Running PSO optimization for all models...")
        for j, (name, entry) in enumerate(models.items()):
            st.write(f"PSO: Optimizing with {name}...")
            model = entry["model"]
            model_type = entry["type"]

            optimized_alloys, chem_final = run_pso_optimization(
                model=model,
                model_type=model_type,
                features=features,
                target=target,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                base_inputs=base_inputs,
                target_chemistry_dict=chem_target,
                alloy_cols=alloy_cols,
                df_successful=df_model,
                seed=SEED
            )

            results[f"{name} + PSO"] = {
                "alloys": optimized_alloys,
                "chem": chem_final
            }
            step += 1
            progress.progress(step / total_steps)

    # Store everything needed in session state
    st.session_state["optimization_results"] = results
    st.session_state["summary_df"] = summary_df
    st.session_state["target_df"] = df
    st.session_state.rerun_optimization = False

# Rehydrate variables
results = st.session_state["optimization_results"]
summary_df = st.session_state["summary_df"]
df = st.session_state["target_df"]

summary_row = summary_df.iloc[2].fillna(0)
aim_row = summary_df.iloc[3].fillna(0)

process_cols = [
    'Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm', 'LRF Holding Time-mm', 'LRF Lime',
    'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
]
delta_cols = [f"Delta_{el.replace('%','')}" for el in process_cols if f"Delta_{el.replace('%','')}" in df.columns]
target = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                               'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'] if f"F-{el}" in df.columns]
chem_target = {f"F-{k.strip()}": float(v) for k, v in aim_row.items()
               if k.strip().endswith("%") and f"F-{k.strip()}" in target}

# --- DISPLAY OUTPUTS ---
st.subheader("🧪 Optimized Alloy Additions (Descending Order)")
alloy_df = pd.DataFrame({k: pd.Series(v["alloys"]) for k, v in results.items()}).fillna(0)
sorted_alloy_df = alloy_df.loc[alloy_df.mean(axis=1).sort_values(ascending=False).index]
st.dataframe(sorted_alloy_df.round(4))

st.subheader("📊 Final Chemistry Predictions vs Aim")
chem_df = pd.DataFrame()
for key, val in results.items():
    chem_df[key] = val["chem"][:len(target)]
chem_df["Aim"] = [chem_target.get(col, 0) for col in target]
chem_df.index = target
st.dataframe(chem_df.round(4))

st.subheader("📉 Variance from Aim (Prediction - Aim)")
for model_key in results.keys():
    variance = chem_df[model_key] - chem_df["Aim"]
    fig, ax = plt.subplots(figsize=(12, 4))
    variance.plot(kind="bar", ax=ax)
    ax.set_ylabel("Delta from Aim")
    ax.set_title(model_key)
    st.pyplot(fig)