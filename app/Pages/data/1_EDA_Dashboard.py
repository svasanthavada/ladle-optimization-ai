import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# --- Load Data ---
df = pd.read_excel("app\Pages\data\Preprocessed_FE_Alloying_Data.xlsx", sheet_name="Heats")

# --- Page Title ---
st.title("Exploratory Data Analysis: Ladle Refining")

# --- Key Metrics ---
st.subheader("Key Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Heats", len(df))
col2.metric("Available Elements", len([c for c in df.columns if 'F-' in c]))
col3.metric("Missing Values", df.isnull().sum().sum())

# --- Final Chemistry Boxplots ---
st.subheader("Final Chemistry Boxplots")
final_cols = [c for c in df.columns if c.startswith("F-")][:6]
fig, axs = plt.subplots(1, len(final_cols), figsize=(18, 5))
for i, col in enumerate(final_cols):
    sns.boxplot(y=df[col], ax=axs[i])
    axs[i].set_title(col)
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("Final Chemistry vs Opening Chemistry Correlation")
chem_open_final = {col: f"F-{col}" for col in ['C%', 'Mn%', 'Cr%', 'Mo%', 'Ni%', 'Al%'] if f"F-{col}" in df.columns and col in df.columns}
if chem_open_final:
    corr_df = df[list(chem_open_final.keys()) + list(chem_open_final.values())].corr()
    heatmap = corr_df.loc[chem_open_final.keys(), chem_open_final.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)

# --- Alloy vs Chemistry Correlation ---
st.subheader("Alloy Addition vs Final Chemistry")
alloy_cols = ['Mn HC', 'Mn MC', 'FeCr HC', 'FeMo Metal', 'FeSi']
final_chem_cols = [col for col in df.columns if col.startswith("F-")][:6]
valid_alloy_cols = [c for c in alloy_cols if c in df.columns]
if valid_alloy_cols:
    corr_df = df[valid_alloy_cols + final_chem_cols].corr()
    heatmap = corr_df.loc[valid_alloy_cols, final_chem_cols]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(heatmap, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)