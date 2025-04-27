import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

# Setup path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing import preprocess_pipeline

st.set_page_config(page_title="EDA Visualizations - Ladle Refining", layout="wide")
st.title("📊 Advanced EDA Visualizations - Ladle Refining Optimization")

uploaded = st.file_uploader("📤 Upload your Raw Ladle Excel file", type=["xlsx"])

if uploaded:
    with st.spinner('⏳ Running Preprocessing on uploaded file...'):
        # Load and preprocess the data
        df, summary_df = preprocess_pipeline(uploaded)


    st.success("✅ Data Preprocessing Completed!")

    st.download_button(
        label="📥 Download Preprocessed CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='preprocessed_ladle_data.csv',
        mime='text/csv',
    )

    delta_cols = [col for col in df.columns if col.startswith('Delta_')]
    final_cols = [col for col in df.columns if col.startswith('F-')]
    open_chem = [col for col in df.columns if '%' in col and not col.startswith('F-')]
    alloy_cols = ['Mn HC', 'Mn MC', 'FeCr HC', 'FeMo Metal', 'FeSi']
    ladle_features = ['Liquidus temp (° C)', 'LRF Total Time (Min)', 'Lift Temp', 'Arching Time-mm', 'LRF Lime']

    # --- Tabs ---
    tabs = st.tabs([
        "🔥 Correlation Heatmap",
        "📊 Delta Chemistry Distributions",
        "🌡️ Lift Temp vs Δ Chemistry",
        "🕒 Arching Time vs Δ Chemistry",
        "🚨 Z-Score Anomaly Detection",
        "📦 Final Chemistry Boxplots",
        "📈 Heat Quality Over Time",
        "🔗 Opening vs Final Chemistry Correlation",
        "🔧 Alloy vs Final Chemistry",
        "🧩 3D Alloy Influence",
        "📈 Pairplots of Alloys vs Chemistry"
    ])

    with tabs[0]:
        st.subheader("🔥 Correlation Heatmap: Process vs Δ Chemistry")
        available = [col for col in delta_cols + ladle_features if col in df.columns]
        if available:
            heatmap_df = df[available].corr().loc[[c for c in ladle_features if c in df.columns], [c for c in delta_cols if c in df.columns]]
            if not heatmap_df.empty:
                fig, ax = plt.subplots(figsize=(16, 6))
                sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Not enough data to plot correlation heatmap.")
        else:
            st.warning("⚠️ Required columns not found for correlation.")

    with tabs[1]:
        st.subheader("📊 Distribution of Δ Chemistry Elements")
        available_deltas = [col for col in delta_cols if col in df.columns]
        if available_deltas:
            fig, axs = plt.subplots(3, 4, figsize=(18, 12))
            axs = axs.flatten()
            for i, col in enumerate(available_deltas[:12]):
                sns.histplot(df[col], kde=True, bins=30, color="cornflowerblue", ax=axs[i])
                axs[i].set_title(col)
                axs[i].grid(True)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No delta chemistry columns found.")

    with tabs[2]:
        st.subheader("🌡️ Lift Temp vs Δ Chemistry Scatterplots")
        if "Lift Temp" in df.columns:
            for delta in available_deltas[:6]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df, x="Lift Temp", y=delta, alpha=0.6, ax=ax)
                ax.grid(True)
                ax.set_title(delta)
                st.pyplot(fig)
        else:
            st.warning("⚠️ Lift Temp column missing.")

    with tabs[3]:
        st.subheader("🕒 Arching Time vs Δ Chemistry Scatterplots")
        if "Arching Time-mm" in df.columns:
            for delta in available_deltas[:6]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df, x="Arching Time-mm", y=delta, alpha=0.6, ax=ax)
                ax.grid(True)
                ax.set_title(delta)
                st.pyplot(fig)
        else:
            st.warning("⚠️ Arching Time-mm column missing.")

    with tabs[4]:
        st.subheader("🚨 Anomaly Detection: Z-Score on ΔMn")
        if "Delta_Mn" in df.columns:
            df['Z_Δ_Mn'] = zscore(df["Delta_Mn"])
            fig, ax = plt.subplots()
            sns.histplot(df['Z_Δ_Mn'], bins=50, kde=True, ax=ax)
            ax.axvline(x=2, color='red', linestyle='--')
            ax.axvline(x=-2, color='red', linestyle='--')
            ax.grid(True)
            st.pyplot(fig)
            st.dataframe(df[df['Z_Δ_Mn'].abs() > 2][["HEAT NO", "Delta_Mn", "Z_Δ_Mn"]])
        else:
            st.warning("⚠️ Delta_Mn column missing for anomaly detection.")

    with tabs[5]:
        st.subheader("📦 Final Chemistry Boxplots")
        available_final = [col for col in final_cols if col in df.columns]
        if available_final:
            fig, axs = plt.subplots(2, 4, figsize=(18, 10))
            axs = axs.flatten()
            for i, col in enumerate(available_final[:8]):
                sns.boxplot(y=df[col], ax=axs[i], color="seagreen")
                axs[i].grid(True)
                axs[i].set_title(col)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Final chemistry columns missing.")

    with tabs[6]:
        st.subheader("🏆 Top 10 Successful Heats (Based on Target Chemistry)")

        try:
            if summary_df is not None:
                grade_row = summary_df.iloc[0]  # First row = target chemistry

                target_chem = {}
                for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']:
                    val = grade_row.get(el)
                    if pd.notnull(val) and isinstance(val, (int, float)):
                        target_chem[f"F-{el}"] = float(val)

                if target_chem:
                    from sklearn.metrics.pairwise import euclidean_distances

                    final_cols_present = [col for col in target_chem if col in df.columns]
                    final_df = df[final_cols_present].fillna(0).copy()

                    target_vector = np.array([target_chem[col] for col in final_cols_present]).reshape(1, -1)
                    distance = euclidean_distances(final_df, target_vector).flatten()

                    df['Success_Score'] = -distance

                    if 'HEAT NO' in df.columns:
                        df['HEAT NO'] = df['HEAT NO'].astype(str)
                    else:
                        df['HEAT NO'] = df.index.astype(str)

                    top_10_heats = df.sort_values(by="Success_Score", ascending=False).head(10)

                    top_10_summary = top_10_heats[["HEAT NO", "Success_Score"] + list(final_cols_present)].copy()
                    top_10_summary.reset_index(drop=True, inplace=True)

                    st.dataframe(top_10_summary.style.format({"Success_Score": "{:.4f}"}))
                else:
                    st.warning("⚠️ Target chemistry not found in Summary data.")
            else:
                st.warning("⚠️ Summary sheet not available from uploaded file.")
        except Exception as e:
            st.error(f"⚠️ Error calculating top heats: {str(e)}")


    with tabs[7]:
        st.subheader("🔗 Opening Chemistry vs Final Chemistry Correlation")
        chem_pairs = {col: f"F-{col}" for col in ['C%', 'Mn%', 'Cr%', 'Mo%', 'Ni%', 'Al%'] if col in df.columns and f"F-{col}" in df.columns}
        if chem_pairs:
            corr_df = df[list(chem_pairs.keys()) + list(chem_pairs.values())].corr()
            if not corr_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr_df.loc[list(chem_pairs.keys()), list(chem_pairs.values())], annot=True, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Not enough chemistry columns for correlation.")
        else:
            st.warning("⚠️ Opening and Final Chemistry columns not matched.")

    with tabs[8]:
        st.subheader("🔧 Alloy vs Final Chemistry Correlation")
        valid_alloy_cols = [c for c in alloy_cols if c in df.columns]
        valid_final_cols = [c for c in final_cols if c in df.columns]
        if valid_alloy_cols and valid_final_cols:
            corr_df = df[valid_alloy_cols + valid_final_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_df.loc[valid_alloy_cols, valid_final_cols], annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Alloy or Final Chemistry columns missing.")

    with tabs[9]:
        st.subheader("🧩 3D Alloy Influence (Mn HC, FeSi → F-Mn%)")
        if all(c in df.columns for c in ["Mn HC", "FeSi", "F-Mn%"]):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df["Mn HC"], df["FeSi"], df["F-Mn%"], c=df["F-Mn%"], cmap='viridis')
            ax.set_xlabel("Mn HC")
            ax.set_ylabel("FeSi")
            ax.set_zlabel("F-Mn%")
            st.pyplot(fig)
        else:
            st.warning("⚠️ 3D Plot columns missing.")

    with tabs[10]:
        st.subheader("📈 Pairplots of Key Alloys vs Chemistry")
        important_cols = ["Mn HC", "FeSi", "F-Mn%", "F-Si%", "F-Cr%"]
        available_imp_cols = [col for col in important_cols if col in df.columns]
        if len(available_imp_cols) >= 3:
            fig = sns.pairplot(df[available_imp_cols])
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough columns for pairplot.")