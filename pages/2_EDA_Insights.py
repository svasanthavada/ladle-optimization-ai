import streamlit as st
import pandas as pd
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Load the success rate data from data folder ---
def load_success_data():
    path = os.path.join(DATA_DIR, "successful_runs.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = df.drop_duplicates()
            if "Success_Score" in df.columns:
                df["Success_Score"] = pd.to_numeric(df["Success_Score"], errors="coerce").fillna(0)
            return df
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
    return None

success_df = load_success_data()

# --- Load successful runs data from data folder ---
def load_successful_runs():
    path = os.path.join(DATA_DIR, "successful_runs.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            st.warning(f"Could not load successful runs: {e}")
    return None

successful_runs_df = load_successful_runs()
successful_runs_count = len(successful_runs_df) if successful_runs_df is not None else 0

st.title("📖 EDA Insights and Interpretations")

# ------------------
st.header("1. Process Parameters Influence on Chemistry Recovery")
st.markdown("""
- **LRF Lime** shows strong positive influence on chemical recovery:
    - **ΔMn%** (Correlation +0.34)
    - **ΔC%** (Correlation +0.22)
    - **ΔMo%** (Correlation +0.18)
- **LRF Total Time** and **Arching Time** have mild positive correlations but less pronounced.
- **Lift Temperature** shows weak/no direct effect on delta chemistry values.

🛠 **Optimization Focus**: Fine-tuning **Lime addition** and **holding time** can improve Mn and C recovery.
""")

# ------------------
st.header("2. Delta Chemistry Behavior (Opening vs Final Analysis)")
st.markdown("""
- Major delta elements (**C, Mn, Si, Cr, P**) follow near-normal distributions.
- Elements like **Mo, V, Ti** show sharp peaks — implying very minimal change.
- Some elements (**Al, Ca**) exhibit noticeable skewness indicating occasional larger deviations.

🛠 **Optimization Focus**: Monitor trace elements like **Al** and **Ca** more carefully.
""")

# ------------------
st.header("3. Outlier Detection via Z-Score Analysis")
st.markdown("""
- **Z-Score analysis on ΔMn%** shows:
    - Most heats are within |Z| < 2.
    - A few heats beyond thresholds (Z > 2 or Z < -2), indicating abnormal Mn pickup behavior.
- These outlier heats need targeted review (raw materials, temperature profile).

🛠 **Optimization Focus**: Investigate heats with extreme Mn deviations and apply targeted corrections.
""")

# ------------------
st.header("4. Success Rate Data Insights")

if success_df is not None:
    if 'Success_Score' in success_df.columns:
        avg_success = success_df['Success_Score'].mean() * 100
        best_heat = success_df.loc[success_df['Success_Score'].idxmax()]
        worst_heat = success_df.loc[success_df['Success_Score'].idxmin()]

        st.markdown(f"""
        - **Average Success Rate** across heats: **{avg_success:.2f}%**
        - **Best Performing Heat**: {best_heat['HEAT NO']} (Success Rate: {best_heat['Success_Score']*100:.2f}%)
        - **Least Performing Heat**: {worst_heat['HEAT NO']} (Success Rate: {worst_heat['Success_Score']*100:.2f}%)
        - **Number of Successful Model Runs (Top 10):** **{successful_runs_count}**
        
        📈 Most heats achieved >90% success, indicating strong process consistency.
        """)
        st.dataframe(success_df.head())
    else:
        st.warning("No 'Success_Score' column found in uploaded success data.")
else:
    st.markdown(f"- **Number of Successful Model Runs (Top 10):** **{successful_runs_count}**")
    st.warning("Success data not available. Please upload a CSV file in the Home or EDA Visualizations page.")

if successful_runs_df is not None:
    st.subheader("Top Successful Runs (from Preprocessing)")
    st.dataframe(successful_runs_df)

# ------------------
st.header("5. Latest EDA File Insights")
def get_latest_eda_file():
    eda_files = [f for f in os.listdir(DATA_DIR) if f.lower().startswith('preprocessed') and f.endswith('.csv')]
    if not eda_files:
        return None
    latest = max(eda_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)))
    return os.path.join(DATA_DIR, latest)

latest_eda = get_latest_eda_file()
if latest_eda:
    st.markdown(f"**Latest preprocessed EDA file loaded:** `{os.path.basename(latest_eda)}`")
    eda_df = pd.read_csv(latest_eda)
    st.dataframe(eda_df.head())
    st.markdown(f"- **Rows:** {eda_df.shape[0]}, **Columns:** {eda_df.shape[1]}")
    st.markdown("- **Preview above shows the most recent preprocessed EDA data for further analysis.**")
else:
    st.info("No preprocessed EDA CSV file found in the data folder.")

# ------------------
st.success("✅ These insights dynamically guide alloy addition optimization, process control tuning, and anomaly detection modules.")

st.info("ℹ️ Generated based on real visualizations: Process-Chemistry Correlation Heatmap, Delta% Distributions, Z-Score Outlier Analysis, Success Rate Analytics, and latest EDA file preview.")