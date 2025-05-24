import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    load_data,
    load_summary,
    get_target_chemistry,
    preprocess_data,
    create_delta_columns
)

class NumericalTabTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super(NumericalTabTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.layers(x)
        return self.output_layer(x)

def run_tabtransformer_model(filepath):
    # Set random seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    df = preprocess_data(df)
    df = create_delta_columns(df)  # Make sure delta columns are created
    target_chem = get_target_chemistry(summary_df)

    # Feature definitions
    alloy_cols = [
        "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
        "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
        "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
        "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
        "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
    ]
    process_chem_cols_base = [
        'Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm',
        'LRF Holding Time-mm', 'LRF Lime',
        'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
        'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
    ]
    delta_cols = [f"Delta_{el.replace('%','')}" for el in process_chem_cols_base if f"Delta_{el.replace('%','')}" in df.columns]

    features = [col for col in alloy_cols + process_chem_cols_base + delta_cols if col in df.columns]
    target_cols = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%',
                                       'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
    target = [col for col in target_cols if col in df.columns]

    if not features or not target:
        raise ValueError("Missing required input or target columns in dataset.")

    print(f"✅ Using {len(features)} features: {features}")

    # Prepare data
    X = df[features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    y_train = preprocess_data(y_train)
    y_val = preprocess_data(y_val)
    y_test = preprocess_data(y_test)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    input_dim = X_train_scaled.shape[1]
    output_dim = y_train_scaled.shape[1]

    model = NumericalTabTransformer(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor)
        loss = criterion(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Save model and scalers
    os.makedirs("models", exist_ok=True)
    model_path = "models/tabtransformer_model.pth"
    torch.save(model.state_dict(), model_path)
    joblib.dump(feature_scaler, "models/tabtransformer_feature_scaler.pkl")
    joblib.dump(target_scaler, "models/tabtransformer_target_scaler.pkl")

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_test_scaled)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_manual = 1 - (ss_residual / ss_total)

    print("SS_total:", ss_total)
    print("SS_residual:", ss_residual)
    print("Manual R²:", r2_manual)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model_path": model_path,
        "rmse": round(rmse, 4),
        "r2": round(r2_manual, 4),
        "y_pred": y_pred,
        "y_test": y_true
    }
