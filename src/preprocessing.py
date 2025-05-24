import pandas as pd
import numpy as np

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
    if filepath.name.endswith('.xlsx'):
        summary_df = pd.read_excel(filepath, sheet_name="Summary")
        summary_df.columns = summary_df.columns.str.strip()
        summary_df = summary_df.dropna(how='all')
        return summary_df
    return None

def get_target_chemistry(summary_df):
    target_row = summary_df.iloc[0]
    required_elements = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                         'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    return {
        f"F-{el}": float(target_row[el])
        for el in required_elements
        if el in target_row and pd.notnull(target_row[el])
    }

def create_delta_columns(df):
    open_chem = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    final_chem = [f"F-{el}" for el in open_chem]
    for open_col, final_col in zip(open_chem, final_chem):
        if open_col in df.columns and final_col in df.columns:
            delta_col = f"Delta_{open_col.replace('%', '')}"
            df[delta_col] = pd.to_numeric(df[final_col], errors='coerce') - pd.to_numeric(df[open_col], errors='coerce')
    return df

def handle_missing(df):
    df.ffill(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def clip_outliers(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df

def preprocess_pipeline(filepath):
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    df = handle_missing(df)
    df = create_delta_columns(df)
    df = clip_outliers(df)
    return df, summary_df

def preprocess_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_datetime(df[col]).dt.total_seconds()
            except Exception:
                df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
    df.fillna(0, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df