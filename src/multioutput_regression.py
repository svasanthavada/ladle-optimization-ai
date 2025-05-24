import os
import joblib
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, load_summary, get_target_chemistry, preprocess_data

def run_multioutput_regression(filepath):
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    target_chem = get_target_chemistry(summary_df)

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
    target_columns = list(target_chem.keys())

    if not input_features or not target_columns:
        raise ValueError("Missing required input or target columns in dataset.")

    X = df[input_features]
    y = df[target_columns]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    y_train = preprocess_data(y_train)
    y_val = preprocess_data(y_val)
    y_test = preprocess_data(y_test)

    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/multioutput_regressor.pkl")

    return {
        "model_path": "models/multioutput_regressor.pkl",
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "y_pred": y_pred,
        "y_test": y_test
    }