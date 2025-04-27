import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Heats") if filepath.name.endswith('.xlsx') else pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if df.iloc[0].isnull().sum() < 5:
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.strip()
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    return df

def load_summary(filepath):
    summary_df = pd.read_excel(filepath, sheet_name="Summary") if filepath.name.endswith('.xlsx') else None
    if summary_df is not None:
        summary_df.columns = summary_df.columns.str.strip()
    return summary_df

def create_delta_columns(df):
    open_chem = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    final_chem = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]

    for open_col, final_col in zip(open_chem, final_chem):
        if open_col in df.columns and final_col in df.columns:
            delta_col = f"Delta_{open_col.replace('%', '')}"
            df[delta_col] = df[final_col] - df[open_col]
    
    return df

def handle_missing(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def clip_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)
    return df

def preprocess_pipeline(filepath):
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    df = handle_missing(df)
    df = create_delta_columns(df)
    df = clip_outliers(df)
    return df, summary_df