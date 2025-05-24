import pandas as pd
import numpy as np
import os
import joblib

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#from src.preprocessing import load_data, load_summary, preprocess_data

def run_xgboost_regression(filepath):
    # Load and preprocess
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    df = preprocess_data(df)

    # Get target chemistry values from summary
    target_row = summary_df.iloc[0]
    required_elements = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    target_chemistry = {f"F-{el}": float(target_row[el]) for el in required_elements if el in target_row and pd.notnull(target_row[el])}

    # Define full list of alloy and open chemistry input features
    alloy_features = [
        "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
        "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
        "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
        "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
        "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
    ]
    open_chemistry = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    process_features = ['Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm', 'LRF Holding Time-mm', 'LRF Lime']

    input_features = process_features + alloy_features + open_chemistry
    target_columns = list(target_chemistry.keys())

    input_features = [col for col in input_features if col in df.columns]
    target_columns = [col for col in target_columns if col in df.columns]

    if not input_features or not target_columns:
        raise ValueError("Missing required input or target columns in dataset.")

    # Split
    X = df[input_features]
    y = df[target_columns]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Preprocess
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    y_train = preprocess_data(y_train)
    y_val = preprocess_data(y_val)
    y_test = preprocess_data(y_test)

    # Train model
    base_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_multioutput.pkl"
    joblib.dump(model, model_path)

    return model_path, round(rmse, 4), round(r2, 4)