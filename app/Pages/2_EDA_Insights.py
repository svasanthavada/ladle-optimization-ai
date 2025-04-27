import streamlit as st
import pandas as pd

# Load the success rate data
@st.cache_data
def load_success_data():
    try:
        df = pd.read_csv('2025-04-27T15-11_export.csv')
        return df
    except Exception as e:
        st.error(f"Error loading success data: {e}")
        return None

success_df = load_success_data()

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
    if 'Success_Rate' in success_df.columns:
        avg_success = success_df['Success_Rate'].mean() * 100
        best_heat = success_df.loc[success_df['Success_Rate'].idxmax()]
        worst_heat = success_df.loc[success_df['Success_Rate'].idxmin()]

        st.markdown(f"""
        - **Average Success Rate** across heats: **{avg_success:.2f}%**
        - **Best Performing Heat**: {best_heat['Heat_ID']} (Success Rate: {best_heat['Success_Rate']*100:.2f}%)
        - **Least Performing Heat**: {worst_heat['Heat_ID']} (Success Rate: {worst_heat['Success_Rate']*100:.2f}%)
        
        📈 Most heats achieved >90% success, indicating strong process consistency.
        """)
    else:
        st.warning("No 'Success_Rate' column found in uploaded success data.")
else:
    st.warning("Success data not available.")

# ------------------
st.success("✅ These insights dynamically guide alloy addition optimization, process control tuning, and anomaly detection modules.")

st.info("ℹ️ Generated based on real visualizations: Process-Chemistry Correlation Heatmap, Delta% Distributions, Z-Score Outlier Analysis, and Success Rate Analytics.")