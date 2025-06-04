import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.preprocessing import preprocess_pipeline, clean_datetime_columns
from src.tabtransformer import NumericalTabTransformer
from src.optimization import run_ga_optimization, run_pso_optimization

def run_full_optimization(df: pd.DataFrame, summary_df: pd.DataFrame):
    SEED = 42
    alloy_cols = [
        "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
        "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
        "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
        "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
        "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
    ]
    process_cols = [
        'Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm', 'LRF Holding Time-mm', 'LRF Lime',
        'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
    ]
    delta_cols = [f"Delta_{el.replace('%','')}" for el in process_cols if f"Delta_{el.replace('%','')}" in df.columns]
    features = [col for col in alloy_cols + process_cols + delta_cols if col in df.columns]
    target = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'] if f"F-{el}" in df.columns]

    df_model = df.dropna(subset=features + target)
    cleaned_df = clean_datetime_columns(df_model[features].copy())
    X = cleaned_df.copy()
    y = df_model[target].apply(pd.to_numeric, errors='coerce').fillna(0)

    feature_scaler = StandardScaler().fit(X)
    target_scaler = StandardScaler().fit(y)

    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y)

    # Get chemistry target from summary
    aim_row = summary_df.iloc[3].fillna(0)
    chem_target = {f"F-{k.strip()}": float(v) for k, v in aim_row.items() if k.strip().endswith("%") and f"F-{k.strip()}" in target}
    base_inputs = cleaned_df.median(numeric_only=True).to_dict()

    models = {
        "TabTransformer": {
            "model": NumericalTabTransformer(input_dim=X_scaled.shape[1], output_dim=y_scaled.shape[1]),
            "path": "models/tabtransformer_model.pth",
            "type": "TabTransformer"
        },
        "XGBoost": {
            "model": joblib.load("models/xgboost_multioutput.pkl"),
            "type": "XGBoost"
        },
        "Ranked XGBoost": {
            "model": joblib.load("models/xgboost_multioutput_ranked.pkl"),
            "type": "Ranked XGBoost"
        }
    }

    results = {}

    for name, entry in models.items():
        model = entry["model"]
        model_type = entry["type"]

        if model_type == "TabTransformer":
            model.load_state_dict(torch.load(entry["path"]))
            model.eval()

        # Run GA
        alloys_ga, chem_ga = run_ga_optimization(
            model=model,
            model_type=model_type,
            features=features,
            target=target,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            base_inputs=base_inputs,
            target_chemistry_dict=chem_target,
            alloy_cols=alloy_cols,
            df=df_model,
            seed=SEED
        )
        results[f"{name} + GA"] = {"alloys": alloys_ga, "chem": chem_ga}

        # Run PSO
        alloys_pso, chem_pso = run_pso_optimization(
            model=model,
            model_type=model_type,
            features=features,
            target=target,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            base_inputs=base_inputs,
            target_chemistry_dict=chem_target,
            alloy_cols=alloy_cols,
            df_successful=df_model,
            seed=SEED
        )
        results[f"{name} + PSO"] = {"alloys": alloys_pso, "chem": chem_pso}

    return {
        "results": results,
        "target_chemistry": chem_target
    }
