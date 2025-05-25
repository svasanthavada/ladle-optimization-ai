import pandas as pd
import numpy as np
import os
import joblib
import random
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from src.preprocessing import create_delta_columns
from src.preprocessing import load_data, load_summary, preprocess_data

def run_xgboost_regression_ranked(
    filepath,
    top_n_alloys=10,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.01
):
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    df = load_data(filepath)
    summary_df = load_summary(filepath)
    df = create_delta_columns(df)  # Make sure delta columns are created

    target_row = summary_df.iloc[0]
    required_elements = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                         'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    target_chemistry = {f"F-{el}": float(target_row[el]) for el in required_elements if el in target_row and pd.notnull(target_row[el])}

    alloy_features = [
        "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
        "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
        "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
        "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
        "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
    ]

    # Rank alloys by usage and select top N for optimization (not for training)
    alloy_usage_counts = (df[alloy_features] != 0).sum().sort_values(ascending=False)
    selected_alloy_features = alloy_usage_counts.head(top_n_alloys).index.tolist()

    open_chemistry = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                      'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    process_features = ['Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm',
                        'LRF Holding Time-mm', 'LRF Lime']

    # Generate delta columns dynamically
    delta_cols = [f"Delta_{el.replace('%', '')}" for el in open_chemistry if f"Delta_{el.replace('%', '')}" in df.columns]

    # Use all alloys for training, but only top N for optimization
    input_features = process_features + alloy_features + open_chemistry + delta_cols
    target_columns = list(target_chemistry.keys())

    # Filter features and targets to ensure they exist in the dataset
    input_features = [col for col in input_features if col in df.columns]
    target_columns = [col for col in target_columns if col in df.columns]

    if not input_features or not target_columns:
        raise ValueError("Missing required input or target columns in dataset.")

    X = df[input_features]
    y = df[target_columns]

    # Split the data into training and temporary sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    y_train = preprocess_data(y_train)
    y_val = preprocess_data(y_val)
    y_test = preprocess_data(y_test)

    base_model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=SEED
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_multioutput_ranked.pkl"
    joblib.dump(model, model_path)

    return {
        "model_path": model_path,
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "y_pred": y_pred,
        "y_test": y_test
    }