import streamlit as st

st.set_page_config(layout="wide", page_title="Ladle Refining Optimization AI Platform")
st.title("🧪 Ladle Refining Optimization AI Platform")
st.markdown("---")

# Upload file and store in session_state
uploaded_file = st.sidebar.file_uploader("📂 Upload FE Alloying Excel File", type=["xlsx"])
if uploaded_file:
    st.session_state["uploaded_file"] = uploaded_file
    st.success("✅ File uploaded successfully and ready for all modules.")
else:
    if "uploaded_file" not in st.session_state:
        st.info("Please upload a file to enable downstream modules.")

st.markdown("""
Welcome to the **Ladle Refining Optimization Platform**, an interactive AI-powered dashboard for steelmaking process intelligence.

---

### 🔍 1. Exploratory Data Analysis (EDA)
- Analyze temperature, time, chemistry trends.
- Understand alloy effectiveness and element pickup.
- Evaluate heat quality via Success Score.

---

### 🧠 2. Model Training & Validation
- Models supported:
  - **TabTransformer (Deep Learning)**
  - **XGBoost Multi-output Regression**
  - **Ranked XGBoost with Top Alloys**
- Returns RMSE, R², and downloadable trained models.

---

### ⚙️ 3. Alloy Optimization
- Optimize alloy inputs using:
  - **Genetic Algorithm (GA)**
  - **Particle Swarm Optimization (PSO)**
- Runs optimization across all models in one click.
- Compare predicted chemistries and alloy recommendations side-by-side.

---

### 🚨 4. Anomaly Detection
- Detect outlier heats using:
  - **Isolation Forest**
  - **One-Class SVM**
  - **Local Outlier Factor (LOF)**
- Visualize anomalies, inspect heat chemistry, and export results.

---

### 📊 5. Prediction Analysis
- Compare **Predicted vs Aim/Max Chemistry**.
- Visualize **variance from Aim** per model.
- Export chemistry prediction data.

---

### 📝 Instructions
- 📁 Upload your **ladle refining data (Excel)** using the uploader above.
- Navigate to analysis sections from the sidebar.

""")
