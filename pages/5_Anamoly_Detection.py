import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import preprocess_pipeline

st.set_page_config(layout="wide")
st.title("🚨 Anomaly Detection on Ladle Chemistry & Process Data")

# Use uploaded file from session state
if "uploaded_file" not in st.session_state:
    st.error("❗ Please upload a dataset from the Home page.")
    st.stop()

uploaded_file = st.session_state["uploaded_file"]

# Preprocess
df, summary_df = preprocess_pipeline(uploaded_file)

# Define features
alloy_cols = [
    "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
    "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
    "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
    "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
    "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
]
process_cols = [
    'Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm',
    'LRF Holding Time-mm', 'LRF Lime',
    'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
    'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
]
delta_cols = [f"Delta_{el.replace('%','')}" for el in process_cols if f"Delta_{el.replace('%','')}" in df.columns]
features = [col for col in alloy_cols + process_cols + delta_cols if col in df.columns]

df_model = df.dropna(subset=features).copy()

# Convert object and time columns
def convert_time_columns(df):
    for col in df.columns:
        if df[col].dtype == 'O' or df[col].apply(lambda x: isinstance(x, datetime.time)).any():
            df[col] = df[col].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second
                                    if isinstance(t, datetime.time)
                                    else pd.to_numeric(t, errors='coerce'))
    return df.fillna(0)

df_model[features] = convert_time_columns(df_model[features])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[features])

# --- Anomaly Models ---
iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
svm_model = OneClassSVM(kernel="rbf", nu=0.05, gamma='scale')
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

df_model["IsolationForest"] = iso_model.fit_predict(X_scaled)
df_model["OneClassSVM"] = svm_model.fit_predict(X_scaled)
df_model["LOF"] = lof_model.fit_predict(X_scaled)

# Voting and Labeling
df_model["Votes"] = df_model[["IsolationForest", "OneClassSVM", "LOF"]].sum(axis=1)
df_model["Final Anomaly Label"] = df_model["Votes"].apply(lambda x: "Anomalous" if x < 0 else "Normal")

# Human-readable conversion
summary_df = df_model[features + ["IsolationForest", "OneClassSVM", "LOF", "Final Anomaly Label"]].copy()
summary_df["IsolationForest"] = summary_df["IsolationForest"].map({-1: "Anomalous", 1: "Normal"})
summary_df["OneClassSVM"] = summary_df["OneClassSVM"].map({-1: "Anomalous", 1: "Normal"})
summary_df["LOF"] = summary_df["LOF"].map({-1: "Anomalous", 1: "Normal"})

# --- Display Tables ---
st.subheader("📋 Anomaly Detection Results (Top 100 rows)")
st.dataframe(summary_df.head(100))

csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Full Anomaly Detection Results", csv, "anomaly_detection_results.csv", "text/csv")

# --- Counts ---
st.subheader("📊 Anomaly Count Comparison")
counts = pd.DataFrame({
    "IsolationForest": df_model["IsolationForest"].map({-1: "Anomalous", 1: "Normal"}).value_counts(),
    "OneClassSVM": df_model["OneClassSVM"].map({-1: "Anomalous", 1: "Normal"}).value_counts(),
    "LOF": df_model["LOF"].map({-1: "Anomalous", 1: "Normal"}).value_counts(),
}).T

st.dataframe(counts)
st.bar_chart(counts.T)

# --- Final Label Chart ---
st.subheader("📦 Final Anomaly Label Distribution")
final_counts = df_model["Final Anomaly Label"].value_counts()
st.bar_chart(final_counts)

# --- Heatmap ---
st.subheader("🔍 Heatmap of Top 50 Records by Final Anomaly Label")
top50 = df_model.sort_values("Final Anomaly Label").head(50)
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(top50[features].astype(float), cmap="coolwarm", cbar=True, ax=ax)
st.pyplot(fig)
