# app/eda_dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import matplotlib.dates as mdates

st.set_page_config(layout="wide")

# --- Load Data ---
st.title("🔬 Advanced EDA Dashboard - Ladle Refining AI")
uploaded = st.file_uploader("Upload Preprocessed Excel File", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded, sheet_name="Heats")
    df.columns = df.columns.str.strip()

    # Identify Features
    delta_chem = [col for col in df.columns if col.startswith("Delta_")]
    final_chem = [col for col in df.columns if col.startswith("F-")]
    open_chem = [col for col in df.columns if "%" in col and not col.startswith("F-") and not col.startswith("Delta_")]
    process_cols = ["Lift Temp", "Liquidus temp (° C)", "Arching Time-mm", "LRF Holding Time-mm", "LRF Lime", "Power Consumption"]

    st.sidebar.header("Navigation")
    analysis = st.sidebar.radio("Choose Analysis", [
        "1. Process vs Δ Chemistry Correlation",
        "2. Delta Chemistry Distributions",
        "3. Lift Temp vs Delta Chemistry",
        "4. Arching Time vs Delta Chemistry",
        "5. Anomaly Detection (Z-Score)",
        "6. Final Chemistry Boxplots",
        "7. Opening vs Final Chemistry Correlation",
        "8. Alloy vs Final Chemistry Correlation",
        "9. Heat Success Score Over Time",
        "10. Top Successful Heats"
    ])

    if analysis == "1. Process vs Δ Chemistry Correlation":
        st.subheader("🔥 Process Parameters vs Δ Chemistry")
        heatmap_df = df[delta_chem + process_cols].corr().loc[process_cols, delta_chem]
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

    elif analysis == "2. Delta Chemistry Distributions":
        st.subheader("📊 Distributions of Delta Chemistry")
        fig, axs = plt.subplots(3, 4, figsize=(18, 12))
        axs = axs.flatten()
        for i, col in enumerate(delta_chem[:12]):
            sns.histplot(df[col], kde=True, bins=30, color="cornflowerblue", ax=axs[i])
            axs[i].set_title(col)
            axs[i].grid(True)
        st.pyplot(fig)

    elif analysis == "3. Lift Temp vs Delta Chemistry":
        st.subheader("🌡️ Lift Temp vs Delta Chemistry")
        for delta in delta_chem[:6]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x="Lift Temp", y=delta, alpha=0.6, ax=ax)
            ax.grid(True)
            ax.set_title(delta)
            st.pyplot(fig)

    elif analysis == "4. Arching Time vs Delta Chemistry":
        st.subheader("🕒 Arching Time vs Delta Chemistry")
        for delta in delta_chem[:6]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x="Arching Time-mm", y=delta, alpha=0.6, ax=ax)
            ax.grid(True)
            ax.set_title(delta)
            st.pyplot(fig)

    elif analysis == "5. Anomaly Detection (Z-Score)":
        st.subheader("🚨 Anomaly Detection: Δ_Mn")
        if "Delta_Mn" in df.columns:
            df['Z_Δ_Mn'] = zscore(df["Delta_Mn"])
            fig, ax = plt.subplots()
            sns.histplot(df['Z_Δ_Mn'], bins=50, kde=True, ax=ax)
            ax.axvline(x=2, color='red', linestyle='--')
            ax.axvline(x=-2, color='red', linestyle='--')
            ax.grid(True)
            st.pyplot(fig)
            st.dataframe(df[df['Z_Δ_Mn'].abs() > 2][["HEAT_NO", "Delta_Mn", "Z_Δ_Mn"]].head())

    elif analysis == "6. Final Chemistry Boxplots":
        st.subheader("📦 Final Chemistry Distributions")
        fig, axs = plt.subplots(2, 4, figsize=(18, 10))
        axs = axs.flatten()
        for i, col in enumerate(final_chem[:8]):
            sns.boxplot(y=df[col], ax=axs[i], color="seagreen")
            axs[i].grid(True)
            axs[i].set_title(col)
        st.pyplot(fig)

    elif analysis == "7. Opening vs Final Chemistry Correlation":
        st.subheader("🔗 Opening vs Final Chemistry Correlation")
        chem_open_final = {col: f"F-{col}" for col in ['C%', 'Mn%', 'Cr%', 'Mo%', 'Ni%', 'Al%'] if f"F-{col}" in df.columns and col in df.columns}
        if chem_open_final:
            corr_df = df[list(chem_open_final.keys()) + list(chem_open_final.values())].corr()
            heatmap = corr_df.loc[chem_open_final.keys(), chem_open_final.values()]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap, annot=True, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)

    elif analysis == "8. Alloy vs Final Chemistry Correlation":
        st.subheader("🔧 Alloy Additions vs Final Chemistry")
        alloy_cols = ['Mn HC', 'Mn MC', 'FeCr HC', 'FeMo Metal', 'FeSi']
        final_chem_cols = [col for col in df.columns if col.startswith("F-")][:6]
        valid_alloy_cols = [c for c in alloy_cols if c in df.columns]
        if valid_alloy_cols:
            corr_df = df[valid_alloy_cols + final_chem_cols].corr()
            heatmap = corr_df.loc[valid_alloy_cols, final_chem_cols]
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(heatmap, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    elif analysis == "9. Heat Success Score Over Time":
        st.subheader("📈 Heat Quality Over Time")
        if 'Success_Score' in df.columns:
            if 'Heat_Timestamp' not in df.columns:
                df['Heat_Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='min')
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df['Heat_Timestamp'], df['Success_Score'], color="mediumseagreen")
            ax.set_title("Success Score Over Time")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            st.pyplot(fig)

    elif analysis == "10. Top Successful Heats":
        st.subheader("🏆 Top 10 Successful Heats")
        if 'Success_Score' in df.columns:
            top10 = df.sort_values(by="Success_Score", ascending=False).head(10)
            st.dataframe(top10[['HEAT_NO', 'Success_Score'] + final_chem[:6]])

    elif analysis == "11. Opening vs Final Chemistry Overall":
    st.subheader("📊 Average Opening vs Final Chemistry Comparison")

    # Calculate means
    mean_open = df[[col for col in open_chem if col in df.columns]].mean().sort_index()
    mean_final = df[[col for col in final_chem if col in df.columns]].mean().sort_index()

    # Align elements
    open_base = [c.replace("%", "") for c in mean_open.index]
    final_base = [c.replace("F-", "") for c in mean_final.index]
    common_elements = sorted(set(open_base).intersection(final_base))

    # Plot
    import numpy as np
    x = np.arange(len(common_elements))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, [mean_open[f"{el}%"] for el in common_elements], width, label='Opening Chemistry')
    ax.bar(x + width/2, [mean_final[f"F-{el}%"] for el in common_elements], width, label='Final Chemistry')
    ax.set_xticks(x)
    ax.set_xticklabels(common_elements, rotation=45)
    ax.set_ylabel("Average %")
    ax.set_title("Opening Chemistry vs Final Chemistry (Average Comparison)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

