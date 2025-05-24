import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms
import random
import joblib
import os
import datetime
from IPython.display import display



def load_data(filepath):
    # Use os.path.basename to get the filename for checking the extension
    df = pd.read_excel(filepath, sheet_name="Heats") if os.path.basename(filepath).endswith('.xlsx') else pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if df.iloc[0].isnull().sum() < 5:
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.strip()
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    return df

def load_summary(filepath):
    # Use os.path.basename to extract filename from filepath
    if os.path.basename(filepath).endswith('.xlsx'): # Use os.path.basename to extract filename
        summary_df = pd.read_excel(filepath, sheet_name="Summary")
        summary_df.columns = summary_df.columns.str.strip()
        summary_df = summary_df.dropna(how='all')
        return summary_df
    return None

def handle_missing(df):
    # Use fillna(method='ffill') first for forward fill
    df = df.ffill()
    # Then fill remaining NaNs (e.g., at the beginning) with the median
    # Specify numeric_only=True to avoid errors on non-numeric columns
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def create_delta_columns(df):
    open_chem = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    final_chem = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
    # Ensure columns exist before attempting calculation
    for open_col, final_col in zip(open_chem, final_chem):
        if open_col in df.columns and final_col in df.columns:
            delta_col = f"Delta_{open_col.replace('%', '')}"
            # Ensure columns are numeric before subtraction
            # Use errors='coerce' to turn non-numeric values into NaN, then fill NaN
            df[open_col] = pd.to_numeric(df[open_col], errors='coerce')
            df[final_col] = pd.to_numeric(df[final_col], errors='coerce')
            df[delta_col] = df[final_col] - df[open_col]
            # Fill NaNs that might result from coercion or original NaNs
            df[delta_col].fillna(df[delta_col].median(), inplace=True) # Fill with median of the delta column
    return df

def clip_outliers(df):
    # Select only numeric columns for clipping
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Calculate quantiles and IQR, ignoring NaNs
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clip values in the column
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def preprocess_pipeline(filepath):
    df = load_data(filepath)
    summary_df = load_summary(filepath) # summary_df is loaded here but not used in subsequent steps within this function
    df = handle_missing(df)
    df = create_delta_columns(df)
    df = clip_outliers(df)
    # Note: This pipeline returns the processed df and the summary_df separately
    return df, summary_df
# --- End Re-defining Preprocessing Functions ---


# 1. Preprocessing and Scaling
filepath = "FE Alloying.xlsx"
df, summary_df = preprocess_pipeline(filepath)

# Define alloy and process features
alloy_cols = [
    "CSP-SiMn", "Mn HC", "Mn MC", "Mn LC", "Mn Metal", "FeSi", "Ladle Cov",
    "FeMo Metal", "FeV", "FeNb lumps", "FeTi lumps", "FeTi Wire", "FeB", "FeAl",
    "Cal Carb", "Al bar", "Al  wire", "FeP", "Sul Stick", "Al mix", "CaSi wire",
    "Cal Wire", "CaFeAl Wire", "S Wire", "Ni Plate", "FeCr LC", "FeCr HC",
    "Al Shot", "Lead Wire", "Mo Metal", "Syn Slag"
]
# Add Delta columns to the potential list of process/chemistry features
process_chem_cols_base = [
    'Lift Temp', 'Liquidus temp (° C)', 'Arching Time-mm',
    'LRF Holding Time-mm', 'LRF Lime',
    'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
]
delta_cols = [f"Delta_{el.replace('%','')}" for el in process_chem_cols_base if f"Delta_{el.replace('%','')}" in df.columns]

# The features used for the model should include original process/open chem AND delta columns
features = [col for col in alloy_cols + process_chem_cols_base + delta_cols if col in df.columns]

target_cols = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
# Filter target_cols to only include those present in the dataframe
target = [col for col in target_cols if col in df.columns]


# Select data using the filtered features and target lists
X = df[features]
y = df[target]

# Handle datetime.time columns and other non-numeric types by coercing to numeric
# This step needs to be applied consistently to X and any future input dataframes
for col in X.select_dtypes(include=['object']).columns:
    try:
        # Ensure this conversion method matches the one used later in evaluate_alloy_additions
        # Convert to string first to handle mixed types or datetime.time
        X[col] = pd.to_numeric(X[col].astype(str), errors='coerce').fillna(0)
    except Exception as e:
        print(f"Could not convert column {col} to numeric in X: {e}")


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
# Fit scaler ONLY on the training features (X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale target variables separately
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train) # Fit the target scaler on original y_train
y_test_scaled = target_scaler.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. PyTorch TabTransformer Model (adapted for numerical features)
class NumericalTabTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, num_heads=2):
        super(NumericalTabTransformer, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Simulate Transformer layers with Dense layers and potential feature mixing
        # Use LayerNorm before residual connection as is common in Transformers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), # Another linear layer for transformation
                nn.LayerNorm(hidden_dim) # Add Layer Normalization
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        # Pass through transformer-like layers
        for layer in self.layers:
            # Apply layer and add residual connection
            x = x + layer(x)
        x = self.output_layer(x)
        return x

input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1] # Output dimension matches the scaled target variables
model = NumericalTabTransformer(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Model Training
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}") # Suppress for cleaner output


# Save the trained PyTorch model
os.makedirs("models", exist_ok=True) # Ensure models directory exists
torch.save(model.state_dict(), "models/numerical_tab_transformer.pth")
print("Numerical TabTransformer model saved.")

# Model Evaluation (RMSE, R²)
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Inverse transform the scaled predictions and actuals using the target scaler
predictions_inv_scaled = target_scaler.inverse_transform(predictions)
actuals_inv_scaled = target_scaler.inverse_transform(actuals)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions_inv_scaled - actuals_inv_scaled)**2))

# Calculate R²
from sklearn.metrics import r2_score
r2 = r2_score(actuals_inv_scaled, predictions_inv_scaled)

print(f"Test RMSE (Inverse Scaled): {rmse:.4f}")
print(f"Test R² (Inverse Scaled): {r2:.4f}")

# 4. Genetic Algorithm Optimization using DEAP

# Load the trained PyTorch model for optimization
loaded_model = NumericalTabTransformer(input_dim, output_dim)
loaded_model.load_state_dict(torch.load("models/numerical_tab_transformer.pth"))
loaded_model.eval() # Set to evaluation mode

# Get base inputs from median of successful heats (reusing previous logic)
# This should use the SAME list of feature columns ('features') that the model was trained on
max_summary_row = summary_df.iloc[2].fillna(0)
aim_summary_row = summary_df.iloc[3].fillna(0)
min_summary_row = summary_df.iloc[1].fillna(0)

target_keys_summary = [k.strip() for k in max_summary_row.index if k.strip().endswith("%")]
target_vector_summary = max_summary_row[target_keys_summary].dropna().astype(float)
# Ensure 'F-' columns exist in the dataframe before using them for Success_Score calculation
present_f_cols_summary = [f"F-{k}" for k in target_keys_summary if f"F-{k}" in df.columns]

# Calculate Success_Score only if relevant 'F-' columns are present
if present_f_cols_summary:
     df['Success_Score'] = -((df[present_f_cols_summary] - target_vector_summary.loc[[k for k in target_keys_summary if f"F-{k}" in df.columns]].values) ** 2).sum(axis=1) ** 0.3
     df_successful = df[df['Success_Score'] >= df['Success_Score'].quantile(0.5)]
else:
     print("Warning: No relevant 'F-%' columns found for success score calculation. Using entire df for base inputs/bounds.")
     df_successful = df.copy() # Fallback to using entire df if no F-% columns


# Select base inputs from median of successful heats using the 'features' list
# This ensures base_inputs_opt starts with a dictionary containing keys from 'features'
# Calculate median only for columns present in df_successful and 'features'
cols_for_base_inputs = [col for col in features if col in df_successful.columns]
base_inputs_opt_df = df_successful[cols_for_base_inputs].median(numeric_only=True)
base_inputs_opt = base_inputs_opt_df.to_dict()


# Extract target chemistry for optimization (using max_chem or aim_chem)
# Ensure the order of target_chem_opt matches the order of target_cols for RMSE calculation later
max_chem = {
    f"F-{k.strip()}": float(v) for k, v in max_summary_row.items()
    if k.strip().endswith("%") and pd.notnull(v) and f"F-{k.strip()}" in target
}
aim_chem = {
    f"F-{k.strip()}": float(v) for k, v in aim_summary_row.items()
    if k.strip().endswith("%") and pd.notnull(v) and f"F-{k.strip()}" in target
}
min_chem = {
    f"F-{k.strip()}": float(v) for k, v in min_summary_row.items()
    if k.strip().endswith("%") and pd.notnull(v) and f"F-{k.strip()}" in target
}


# Use max_chem for optimization target values, ensuring they are in the order of target_cols
# This dictionary will be used to create the target vector for RMSE calculation
target_chem_opt_dict = {col: max_chem.get(col, 0) for col in target} # Use 'target' which is the filtered target_cols


# Bounds for alloy elements for optimization
# Only use bounds for the alloy_cols that are actually included in the model features
alloy_features_in_model = [col for col in alloy_cols if col in features]

# Calculate bounds based on df_successful, but enforce a minimum of 0 for alloy additions
bounds = [(max(0, df_successful[el].min()), df_successful[el].max()) for el in alloy_features_in_model]

# Adjust bounds to ensure min < max, add a small epsilon if min == max and min is 0
bounds = [(lower, upper) if lower < upper else (lower, lower + 1e-6 if lower == 0 else lower + abs(lower)*1e-6) for lower, upper in bounds]

# Extract bounds into low and up lists for the custom mutation operator
low_bounds = [b[0] for b in bounds]
up_bounds = [b[1] for b in bounds]

# DEAP setup (re-create if needed in this cell)
try:
    # Attempt to delete existing definitions if running cell multiple times
    del creator.FitnessMin
    del creator.Individual
except AttributeError:
    pass # Ignore if they don't exist

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Register the individual initializer to use the actual bounds directly
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [random.uniform(low, high) for low, high in bounds]) # Initialize within actual bounds
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Define the custom mutation operator that respects bounds
def mutGaussianBounded(individual, mu, sigma, indpb, low, up):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            # Apply Gaussian mutation
            mutated_value = individual[i] + random.gauss(mu, sigma)
            # Clip to bounds
            individual[i] = np.clip(mutated_value, low[i], up[i])
    return individual,

# Register the bounded mutation operator
toolbox.register("mutate", mutGaussianBounded, mu=0, sigma=1, indpb=0.2, low=low_bounds, up=up_bounds)
toolbox.register("select", tools.selTournament, tournsize=3)


# Register the evaluation function using the trained PyTorch model
def evaluate_alloy_additions(individual):
    # The individual already contains alloy values for the features in alloy_features_in_model
    alloy_values = individual # Individual is a list of floats corresponding to alloy_features_in_model

    # Create a dictionary with the alloy values from the individual
    alloy_dict = dict(zip(alloy_features_in_model, alloy_values))

    # Combine base inputs with the current individual's alloy additions
    # Start with a copy of base_inputs_opt (which contains median of 'features')
    input_data_for_scaling = base_inputs_opt.copy()
    # Update with the alloy values from the current individual's alloy additions
    # This overwrites the median alloy values from base_inputs_opt with the current individual's values
    input_data_for_scaling.update(alloy_dict)

    # Create a DataFrame ensuring all and only the 'features' columns are present
    # Use the exact 'features' list which was used to create X_train and fit the scaler
    input_df = pd.DataFrame([input_data_for_scaling])

    # --- DEBUG PRINT STATEMENTS ---
    # print("\n--- Debugging scaler input ---")
    # print("Expected features (from X_train.columns or 'features'):", features)
    # print("Input DataFrame columns:", input_df.columns.tolist())
    # print("Are columns identical and in same order?", features == input_df.columns.tolist())
    # print("Missing features in input_df:", set(features) - set(input_df.columns))
    # print("Extra features in input_df:", set(input_df.columns) - set(features))
    # print("--------------------------")
    # --- END DEBUG PRINT STATEMENTS ---

    # ***FIX***: Ensure the DataFrame columns match the exact features used for training
    # Reindex to the order of 'features' and fill any potentially missing with 0
    # This step is crucial for the scaler
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Handle datetime.time columns and other non-numeric types by coercing to numeric
    # This step must exactly mirror how X_train was handled before scaling
    for col in input_df.select_dtypes(include=['object']).columns:
        try:
            # Use the same conversion method as applied to X before splitting
            input_df[col] = pd.to_numeric(input_df[col].astype(str), errors='coerce').fillna(0)
        except Exception as e:
             # Print error but continue, coercing to numeric should handle most issues
             print(f"Warning: Could not convert column {col} to numeric in evaluate_alloy_additions: {e}")
             input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)


    # Scale the input using the same scaler used during training
    # The scaler was fitted on X_train, which contains the 'features' columns in order
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Make prediction using the loaded PyTorch model
    with torch.no_grad(): # No gradient calculation needed for evaluation
        predicted_scaled = loaded_model(input_tensor).numpy()[0]

    # Inverse transform the scaled prediction to original scale
    # Use the target_scaler which was fitted specifically on the target variables (y_train)
    # predicted_scaled is a 1D array corresponding to the scaled target columns
    predicted_chem = target_scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]


    # Calculate the objective (RMSE against target chemistry)
    # Use the target_chem_opt_dict to get target values in the correct order (matching 'target' list)
    target_values = np.array([target_chem_opt_dict.get(col, 0) for col in target])

    # Ensure prediction and target have the same number of elements for comparison
    # The number of elements in predicted_chem should match the number of 'target' columns
    num_elements_to_compare = len(target) # Use the length of the filtered target columns
    predicted_chem_sliced = predicted_chem[:num_elements_to_compare]
    target_values_sliced = target_values[:num_elements_to_compare]

    # Calculate RMSE
    # Add a small epsilon to avoid log(0) if using other metrics, but RMSE is fine with 0
    rmse = np.sqrt(np.mean((predicted_chem_sliced - target_values_sliced)**2))

    return rmse, # Comma for DEAP tuple

# Register the evaluation function with DEAP
toolbox.register("evaluate", evaluate_alloy_additions)

# Run Genetic Algorithm
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

print("Starting Genetic Algorithm Optimization...")
# Run the GA
# Use HallOfFame as a list so we can access the best individual after the run# Run the GA
logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=False) # Corrected keyword argument to halloffame

print("Genetic Algorithm Optimization Finished.")

# Best solution from GA is in hof[0]
if hof:
    best_individual = hof[0]
else:
    # Handle case where hall of fame might be empty (e.g., ngen=0)
    print("Warning: Hall of Fame is empty. Cannot retrieve best individual.")
    best_individual = None
    optimized_alloy_additions = {}


if best_individual is not None:
    # Map the optimized values back to the alloy column names
    # best_individual is a list of optimized values corresponding to alloy_features_in_model
    optimized_alloy_additions_raw = dict(zip(alloy_features_in_model, best_individual))

    # ***FIX: Ensure all final alloy additions are non-negative by clipping at 0***
    optimized_alloy_additions = {key: max(0.0, value) for key, value in optimized_alloy_additions_raw.items()}


    print("\nOptimized Alloy Additions (TabTransformer + GA):")
    # Display as DataFrame for better readability, including all original alloy_cols
    display_optimized_alloys = {col: optimized_alloy_additions.get(col, 0.0) for col in alloy_cols}
    # Ensure the values are displayed with appropriate precision and handle the index name
    display(pd.DataFrame(display_optimized_alloys, index=["Recommended (kg)"]).T.round(4))


    # Predict the final chemistry for the optimized alloys using the trained model
    # Recreate the input DataFrame using the optimized alloys and base inputs
    optimized_input_dict = base_inputs_opt.copy() # Start with base inputs (median of features)
    # Update with the *clipped* optimized alloy values
    optimized_input_dict.update(optimized_alloy_additions)

    # Create DataFrame with the exact 'features' columns
    optimized_input_df = pd.DataFrame([optimized_input_dict]).reindex(columns=features, fill_value=0)

    # Handle non-numeric types consistently
    for col in optimized_input_df.select_dtypes(include=['object']).columns:
        try:
            optimized_input_df[col] = pd.to_numeric(optimized_input_df[col].astype(str), errors='coerce').fillna(0)
        except Exception as e:
             print(f"Warning: Could not convert column {col} to numeric for final prediction: {e}")
             optimized_input_df[col] = pd.to_numeric(optimized_input_df[col], errors='coerce').fillna(0)


    # Scale the input using the feature scaler
    optimized_input_scaled = scaler.transform(optimized_input_df)
    optimized_input_tensor = torch.tensor(optimized_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        predicted_scaled_opt = loaded_model(optimized_input_tensor).numpy()[0]

    # Inverse transform using the target scaler
    predicted_chem_opt = target_scaler.inverse_transform(predicted_scaled_opt.reshape(1, -1))[0]


    print("\nPredicted Final Chemistry vs Target (using optimized alloys):")
    # Align predicted chemistry with target_cols for comparison
    # Create a dictionary from the predicted_chem_opt array using the target column names
    predicted_chem_dict = dict(zip(target, predicted_chem_opt[:len(target)])) # Use the filtered 'target' list

    # Create comparison DataFrame
    chem_comparison_df = pd.DataFrame({
        "Target (Max)": [max_chem.get(col, 0) for col in target], # Use max_chem and filtered 'target' list
        "Aim": [aim_chem.get(col, 0) for col in target],       # Use aim_chem and filtered 'target' list
        "Min": [min_chem.get(col, 0) for col in target],       # Use min_chem and filtered 'target' list
        "Predicted": [predicted_chem_dict.get(col, 0) for col in target] # Use predicted_chem_dict and filtered 'target' list
    }, index=target) # Use the filtered 'target' list as index

    display(chem_comparison_df.round(5))
else:
    print("Optimization did not produce a valid best individual.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms
import random
import joblib # Make sure joblib is imported
import os
import datetime
from IPython.display import display
from sklearn.metrics import mean_squared_error # Ensure this is imported if needed

# --- Re-defining Preprocessing Functions (Ensure these match your other blocks if needed) ---
# Re-define the preprocessing functions if they are not guaranteed to be in scope
# from src.preprocessing import load_data, load_summary, preprocess_data # If using external file
# Otherwise, include the function definitions directly here if they are modified
def load_data(filepath):
    # Use os.path.basename to get the filename for checking the extension
    df = pd.read_excel(filepath, sheet_name="Heats") if os.path.basename(filepath).endswith('.xlsx') else pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if df.iloc[0].isnull().sum() < 5:
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.strip()
    # Avoid inplace=True and chained assignment warnings by chaining calls and assigning back
    # Also add .copy() after dropping rows/columns to avoid SettingWithCopyWarning later
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all').copy()
    return df

def load_summary(filepath):
    # Use os.path.basename to extract filename from filepath
    if os.path.basename(filepath).endswith('.xlsx'): # Use os.path.basename to extract filename
        summary_df = pd.read_excel(filepath, sheet_name="Summary")
        summary_df.columns = summary_df.columns.str.strip()
        # Avoid inplace=True and add .copy()
        summary_df = summary_df.dropna(how='all').copy()
        return summary_df
    return None

def handle_missing(df):
    # Avoid inplace=True and chained assignment warnings
    # ***FIX: Replace fillna(method='ffill') with ffill()***
    df = df.ffill() # Use ffill() directly
    # Then fill remaining NaNs with the median, specifying numeric_only=True
    # Use the result of fillna directly and assign back to df
    # Add .infer_objects(copy=False) to address downcasting warning
    df = df.fillna(df.median(numeric_only=True)).infer_objects(copy=False)
    return df # Return the modified DataFrame

def create_delta_columns(df):
    # Avoid modifying input df directly with inplace=True and chained assignment warnings
    df_copy = df.copy() # Work on a copy to prevent modifying the original df unexpectedly
    open_chem = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    final_chem = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
    # Ensure columns exist before attempting calculation
    for open_col, final_col in zip(open_chem, final_chem):
        if open_col in df_copy.columns and final_col in df_copy.columns:
            delta_col = f"Delta_{open_col.replace('%', '')}"
            # Ensure columns are numeric before subtraction
            # Use errors='coerce' to turn non-numeric values into NaN
            df_copy[open_col] = pd.to_numeric(df_copy[open_col], errors='coerce')
            df_copy[final_col] = pd.to_numeric(df_copy[final_col], errors='coerce')
            df_copy[delta_col] = df_copy[final_col] - df_copy[open_col]
            # Fill NaNs that might result from coercion or original NaNs
            # Calculate median of the delta column and use fillna on the column directly
            median_delta = df_copy[delta_col].median()
            # Avoid inplace=True
            df_copy[delta_col] = df_copy[delta_col].fillna(median_delta)

    return df_copy # Return the modified copy of the DataFrame

def clip_outliers(df):
    # Avoid modifying input df directly with inplace=True warnings
    df_copy = df.copy() # Work on a copy
    # Select only numeric columns for clipping
    numeric_cols = df_copy.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Calculate quantiles and IQR, ignoring NaNs
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clip values in the column and assign back
        # Avoid inplace=True
        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
    return df_copy # Return the modified copy

def preprocess_pipeline(filepath):
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    # Chain the function calls and reassign the result
    # Each function should return a new or modified DataFrame
    df = handle_missing(df)
    df = create_delta_columns(df)
    df = clip_outliers(df)
    # Note: This pipeline returns the processed df and the summary_df separately
    return df, summary_df

# Assuming preprocess_data is also used elsewhere and might need similar updates
def preprocess_data(df):
    # Avoid modifying input df directly
    df = df.copy()
    # Convert datetime or object to numeric where needed
    for col in df.columns:
        if df[col].dtype == 'O':  # If the column is of object (string) type
            try:
                # Specify the format of your date/time column here (if known)
                # Use errors='coerce' and fillna to handle non-datetime strings gracefully
                # Assign the result back to the column
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.total_seconds().fillna(0)
            except Exception: # Catch other potential errors during conversion
                 try:
                      # Handle cases where the column is not a date/time but might be numeric strings
                      # Assign the result back to the column
                      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                 except Exception as e:
                      print(f"Could not convert column {col} to numeric in preprocess_data: {e}")
                      # Assign a default value or leave as NaN if conversion fails completely
                      df[col] = np.nan # Assign NaN then the fillna(0) below will handle it


    # Fill remaining NaNs with 0 (as done in your original preprocess_data)
    # Avoid inplace=True
    df = df.fillna(0)

    # Outlier clipping
    # Avoid modifying input df directly
    # Select only numeric columns for clipping
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clip values in the column and assign back
        # Avoid inplace=True
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df # Return the modified DataFrame

# --- End Corrected Preprocessing Functions ---


# --- Section where X and y are created (Modify this) ---

# 1. Preprocessing and Scaling
filepath = "FE Alloying.xlsx"
df, summary_df = preprocess_pipeline(filepath)

# Define alloy and process features
# ... (alloy_cols, process_chem_cols_base, delta_cols, features, target definitions remain the same)
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
    'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
]
delta_cols = [f"Delta_{el.replace('%','')}" for el in process_chem_cols_base]
features = [col for col in alloy_cols + process_chem_cols_base + delta_cols if col in df.columns]

target_cols = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
target = [col for col in target_cols if col in df.columns]


# Select data using the filtered features and target lists
# ***FIX: Explicitly create a copy of X to avoid SettingWithCopyWarning***
X = df[features].copy() # Use .copy() here
y = df[target].copy() # Use .copy() here as well for consistency


# Handle datetime.time columns and other non-numeric types by coercing to numeric
# This step needs to be applied consistently to X and any future input dataframes
for col in X.select_dtypes(include=['object']).columns:
    try:
        # Ensure this conversion method matches the one used later in evaluate_alloy_additions
        # Convert to string first to handle mixed types or datetime.time
        # Assign the result back to the column in the copied DataFrame X
        X[col] = pd.to_numeric(X[col].astype(str), errors='coerce').fillna(0)
    except Exception as e:
        print(f"Could not convert column {col} to numeric in X: {e}")


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
# Fit scaler ONLY on the training features (X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale target variables separately
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train) # Fit the target scaler on original y_train
y_test_scaled = target_scaler.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. PyTorch TabTransformer Model (adapted for numerical features)
class NumericalTabTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, num_heads=2):
        super(NumericalTabTransformer, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Simulate Transformer layers with Dense layers and potential feature mixing
        # Use LayerNorm before residual connection as is common in Transformers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), # Another linear layer for transformation
                nn.LayerNorm(hidden_dim) # Add Layer Normalization
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        # Pass through transformer-like layers
        for layer in self.layers:
            # Apply layer and add residual connection
            x = x + layer(x)
        x = self.output_layer(x)
        return x

input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1] # Output dimension matches the scaled target variables
model = NumericalTabTransformer(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Model Training
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}") # Suppress for cleaner output


# Save the trained PyTorch model and the scalers
os.makedirs("models", exist_ok=True) # Ensure models directory exists

# Define the paths where you want to save the model and scalers
model_path = "models/numerical_tab_transformer.pth"
feature_scaler_path = "models/feature_scaler.pkl"
target_scaler_path = "models/target_scaler.pkl"


torch.save(model.state_dict(), model_path)
print(f"Numerical TabTransformer model saved to: {model_path}")

# ***ADD THESE LINES TO SAVE SCALERS***
joblib.dump(scaler, feature_scaler_path) # Save the feature scaler
joblib.dump(target_scaler, target_scaler_path) # Save the target scaler
print(f"Feature scaler saved to: {feature_scaler_path}")
print(f"Target scaler saved to: {target_scaler_path}")
# ************************************


# Model Evaluation (RMSE, R²)
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Inverse transform the scaled predictions and actuals using the target scaler
predictions_inv_scaled = target_scaler.inverse_transform(predictions)
actuals_inv_scaled = target_scaler.inverse_transform(actuals)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions_inv_scaled - actuals_inv_scaled)**2))

# Calculate R²
from sklearn.metrics import r2_score
r2 = r2_score(actuals_inv_scaled, predictions_inv_scaled)

print(f"Test RMSE (Inverse Scaled): {rmse:.4f}")
print(f"Test R² (Inverse Scaled): {r2:.4f}")

# 4. Genetic Algorithm Optimization using DEAP

# Load the trained PyTorch model for optimization
loaded_model = NumericalTabTransformer(input_dim, output_dim)
loaded_model.load_state_dict(torch.load(model_path)) # Load from the defined path
loaded_model.eval() # Set to evaluation mode

# Load the scalers for use in the evaluation function
loaded_feature_scaler = joblib.load(feature_scaler_path) # Load feature scaler
loaded_target_scaler = joblib.load(target_scaler_path) # Load target scaler


# Get base inputs from median of successful heats (reusing previous logic)
# This should use the SAME list of feature columns ('features') that the model was trained on
max_summary_row = summary_df.iloc[2].fillna(0)
aim_summary_row = summary_df.iloc[3].fillna(0)
min_summary_row = summary_df.iloc[1].fillna(0)

target_keys_summary = [k.strip() for k in max_summary_row.index if k.strip().endswith("%")]
target_vector_summary = max_summary_row[target_keys_summary].dropna().astype(float)
# Ensure 'F-' columns exist in the dataframe before using them for Success_Score calculation
present_f_cols_summary = [f"F-{k}" for k in target_keys_summary if f"F-{k}" in df.columns]

# Calculate Success_Score only if relevant 'F-' columns are present
if present_f_cols_summary:
     df['Success_Score'] = -((df[present_f_cols_summary] - target_vector_summary.loc[[k for k in target_keys_summary if f"F-{k}" in df.columns]].values) ** 2).sum(axis=1) ** 0.3
     df_successful = df[df['Success_Score'] >= df['Success_Score'].quantile(0.5)].copy() # Use .copy()
else:
     print("Warning: No relevant 'F-%' columns found for success score calculation. Using entire df for base inputs/bounds.")
     df_successful = df.copy() # Fallback to using entire df if no F-% columns


# Select base inputs from median of successful heats using the 'features' list
# This ensures base_inputs_opt starts with a dictionary containing keys from 'features'
# Calculate median only for columns present in df_successful and 'features'
cols_for_base_inputs = [col for col in features if col in df_successful.columns]
base_inputs_opt_df = df_successful[cols_for_base_inputs].median(numeric_only=True)
base_inputs_opt = base_inputs_opt_df.to_dict()


# Extract target chemistry for optimization (using max_chem or aim_chem)
# Ensure the order of target_chem_opt matches the order of target_cols for RMSE calculation later
max_chem = {
    f"F-{k.strip()}": float(v) for k, v in max_summary_row.items()
    if k.strip().endswith("%") and pd.notnull(v) and f"F-{k.strip()}" in target # Filter to only include targets in the model output
}
aim_chem = {
    f"F-{k.strip()}": float(v) for k, v in aim_summary_row.items()
    if k.strip().endswith("%") and pd.notnull(v) and f"F-{k.strip()}" in target
}
min_chem = {
    f"F-{k.strip()}": float(v) for k, v in min_summary_row.items()
    if k.strip().endswith("%") and pd.notnull(v) and f"F-{k.strip()}" in target
}


# ***CHANGE HERE: Use aim_chem for the optimization target***
target_chem_opt_dict = {col: aim_chem.get(col, 0) for col in target} # Use 'aim_chem' as the target dictionary for optimization


# Bounds for alloy elements for optimization
# Only use bounds for the alloy_cols that are actually included in the model features
alloy_features_in_model = [col for col in alloy_cols if col in features]

# Calculate bounds based on df_successful, but enforce a minimum of 0 for alloy additions
# Ensure the columns exist in df_successful before attempting to get min/max
bounds = [(max(0, df_successful[el].min()) if el in df_successful.columns else 0.0,
           df_successful[el].max() if el in df_successful.columns else 100.0) # Default max if column missing
          for el in alloy_features_in_model]

# Adjust bounds to ensure min < max, add a small epsilon if min == max and min is 0
bounds = [(lower, upper) if lower < upper else (lower, lower + 1e-6 if lower == 0 else lower + abs(lower)*1e-6) for lower, upper in bounds]

# Extract bounds into low and up lists for the custom mutation operator
low_bounds = [b[0] for b in bounds]
up_bounds = [b[1] for b in bounds]

# DEAP setup (re-create if needed in this cell)
try:
    # Attempt to delete existing definitions if running cell multiple times
    del creator.FitnessMin
    del creator.Individual
except AttributeError:
    pass # Ignore if they don't exist

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Register the individual initializer to use the actual bounds directly
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [random.uniform(low, high) for low, high in bounds]) # Initialize within actual bounds
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Define the custom mutation operator that respects bounds
def mutGaussianBounded(individual, mu, sigma, indpb, low, up):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            # Apply Gaussian mutation
            mutated_value = individual[i] + random.gauss(mu, sigma)
            # Clip to bounds
            individual[i] = np.clip(mutated_value, low[i], up[i])
    return individual,

# Register the bounded mutation operator
toolbox.register("mutate", mutGaussianBounded, mu=0, sigma=1, indpb=0.2, low=low_bounds, up=up_bounds)
toolbox.register("select", tools.selTournament, tournsize=3)


# Register the evaluation function using the trained PyTorch model
def evaluate_alloy_additions(individual):
    # The individual already contains alloy values for the features in alloy_features_in_model
    alloy_values = individual # Individual is a list of floats corresponding to alloy_features_in_model

    # Create a dictionary with the alloy values from the individual
    alloy_dict = dict(zip(alloy_features_in_model, alloy_values))

    # Combine base inputs with the current individual's alloy additions
    # Start with a copy of base_inputs_opt (which contains median of 'features')
    input_data_for_scaling = base_inputs_opt.copy()
    # Update with the alloy values from the current individual's alloy additions
    # This overwrites the median alloy values from base_inputs_opt with the current individual's values
    input_data_for_scaling.update(alloy_dict)

    # Create a DataFrame ensuring all and only the 'features' columns are present
    # Use the exact 'features' list which was used to create X_train and fit the scaler
    input_df = pd.DataFrame([input_data_for_scaling])

    # ***FIX***: Ensure the DataFrame columns match the exact features used for training
    # Reindex to the order of 'features' and fill any potentially missing with 0
    # This step is crucial for the scaler
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Handle non-numeric types consistently
    for col in input_df.select_dtypes(include=['object']).columns:
        try:
            # Use the same conversion method as applied to X before splitting
            input_df[col] = pd.to_numeric(input_df[col].astype(str), errors='coerce').fillna(0)
        except Exception as e:
             # Print error but continue, coercing to numeric should handle most issues
             print(f"Warning: Could not convert column {col} to numeric in evaluate_alloy_additions: {e}")
             input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)


    # Scale the input using the loaded feature scaler
    # The scaler was fitted on X_train, which contains the 'features' columns in order
    input_scaled = loaded_feature_scaler.transform(input_df) # Use loaded_feature_scaler
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Make prediction using the loaded PyTorch model
    with torch.no_grad(): # No gradient calculation needed for evaluation
        predicted_scaled = loaded_model(input_tensor).numpy()[0]

    # Inverse transform the scaled prediction to original scale
    # Use the loaded target_scaler
    predicted_chem = loaded_target_scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0] # Use loaded_target_scaler


    # Calculate the objective (RMSE against target chemistry)
    # Use the target_chem_opt_dict to get target values in the correct order (matching 'target' list)
    target_values = np.array([target_chem_opt_dict.get(col, 0) for col in target])

    # Ensure prediction and target have the same number of elements for comparison
    # The number of elements in predicted_chem should match the number of 'target' columns
    num_elements_to_compare = len(target) # Use the length of the filtered target columns
    predicted_chem_sliced = predicted_chem[:num_elements_to_compare]
    target_values_sliced = target_values[:num_elements_to_compare]

    # Calculate RMSE
    # Add a small epsilon to avoid log(0) if using other metrics, but RMSE is fine with 0
    rmse = np.sqrt(np.mean((predicted_chem_sliced - target_values_sliced)**2))

    return rmse, # Comma for DEAP tuple

# Register the evaluation function with DEAP
toolbox.register("evaluate", evaluate_alloy_additions)

# Run Genetic Algorithm
pop = toolbox.population(n=50) # Increase population size? e.g., n=100
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

print("Starting Genetic Algorithm Optimization...")
# Run the GA
# Use HallOfFame as a list so we can access the best individual after the run# Run the GA
# Increase number of generations? e.g., ngen=200
logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=False) # Corrected keyword argument to halloffame

print("Genetic Algorithm Optimization Finished.")

# Best solution from GA is in hof[0]
if hof:
    best_individual = hof[0]
else:
    # Handle case where hall of fame might be empty (e.g., ngen=0)
    print("Warning: Hall of Fame is empty. Cannot retrieve best individual.")
    best_individual = None
    optimized_alloy_additions = {}


if best_individual is not None:
    # Map the optimized values back to the alloy column names
    # best_individual is a list of optimized values corresponding to alloy_features_in_model
    optimized_alloy_additions_raw = dict(zip(alloy_features_in_model, best_individual))

    # ***FIX: Ensure all final alloy additions are non-negative by clipping at 0***
    optimized_alloy_additions = {key: max(0.0, value) for key, value in optimized_alloy_additions_raw.items()}


    print("\nOptimized Alloy Additions (TabTransformer + GA):")
    # Display as DataFrame for better readability, including all original alloy_cols
    display_optimized_alloys = {col: optimized_alloy_additions.get(col, 0.0) for col in alloy_cols}
    # Ensure the values are displayed with appropriate precision and handle the index name
    display(pd.DataFrame(display_optimized_alloys, index=["Recommended (kg)"]).T.round(4))


    # Predict the final chemistry for the optimized alloys using the trained model
    # Recreate the input DataFrame using the optimized alloys and base inputs
    optimized_input_dict = base_inputs_opt.copy() # Start with base inputs (median of features)
    # Update with the *clipped* optimized alloy values
    optimized_input_dict.update(optimized_alloy_additions)

    # Create DataFrame with the exact 'features' columns
    optimized_input_df = pd.DataFrame([optimized_input_dict]).reindex(columns=features, fill_value=0)

    # Handle non-numeric types consistently
    for col in optimized_input_df.select_dtypes(include=['object']).columns:
        try:
            optimized_input_df[col] = pd.to_numeric(optimized_input_df[col].astype(str), errors='coerce').fillna(0)
        except Exception as e:
             print(f"Warning: Could not convert column {col} to numeric for final prediction: {e}")
             optimized_input_df[col] = pd.to_numeric(optimized_input_df[col], errors='coerce').fillna(0)


    # Scale the input using the loaded feature scaler
    optimized_input_scaled = loaded_feature_scaler.transform(optimized_input_df) # Use loaded_feature_scaler
    optimized_input_tensor = torch.tensor(optimized_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        predicted_scaled_opt = loaded_model(optimized_input_tensor).numpy()[0]

    # Inverse transform using the loaded target scaler
    predicted_chem_opt = loaded_target_scaler.inverse_transform(predicted_scaled_opt.reshape(1, -1))[0] # Use loaded_target_scaler


    print("\nPredicted Final Chemistry vs Target (using optimized alloys):")
    # Align predicted chemistry with target_cols for comparison
    # Create a dictionary from the predicted_chem_opt array using the target column names
    predicted_chem_dict = dict(zip(target, predicted_chem_opt[:len(target)])) # Use the filtered 'target' list

    # Create comparison DataFrame
    # ***FIX: Use aim_chem for the "Aim" column in the comparison DataFrame***
    chem_comparison_df = pd.DataFrame({
        "Target (Max)": [max_chem.get(col, 0) for col in target],
        "Aim": [aim_chem.get(col, 0) for col in target], # Use aim_chem here
        "Min": [min_chem.get(col, 0) for col in target],
        "Predicted": [predicted_chem_dict.get(col, 0) for col in target]
    }, index=target)

    # Add Delta to Aim column for easier comparison
    chem_comparison_df['Delta to Aim'] = chem_comparison_df['Predicted'] - chem_comparison_df['Aim']


    display(chem_comparison_df.round(5))

    # Optional: Calculate and print RMSE for the predicted vs Aim chemistry for the optimization result
    # Use the values from the comparison DataFrame for the calculation
    aim_values = chem_comparison_df['Aim'].dropna().values
    predicted_values = chem_comparison_df['Predicted'].dropna().values

    # Only calculate RMSE if there are non-NaN values to compare and lengths match
    if len(aim_values) > 0 and len(predicted_values) == len(aim_values):
        rmse_predicted_to_aim = np.sqrt(mean_squared_error(aim_values, predicted_values))
        print(f"\nRMSE of Predicted vs Aim (TabTransformer + GA Optimization): {rmse_predicted_to_aim:.4f}")
    else:
        print("\nCannot calculate RMSE of Predicted vs Aim: Mismatched or missing data for comparison.")


else:
    print("Optimization did not produce a valid best individual.")

# --- End of TabTransformer + GA Optimization Block ---

import pandas as pd
import numpy as np
import random
import joblib
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
from IPython.display import display
import time # Import time to measure execution

# --- Corrected Preprocessing Functions (as in the previous fix) ---
# Include the corrected definitions of load_data, load_summary, handle_missing,
# create_delta_columns, clip_outliers, preprocess_pipeline, preprocess_data here.
# (Copy-paste the corrected functions from the previous response)

def load_data(filepath):
    # Use os.path.basename to get the filename for checking the extension
    df = pd.read_excel(filepath, sheet_name="Heats") if os.path.basename(filepath).endswith('.xlsx') else pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if df.iloc[0].isnull().sum() < 5:
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.strip()
    # Avoid inplace=True and chained assignment warnings by chaining calls and assigning back
    # Also add .copy() after dropping rows/columns to avoid SettingWithCopyWarning later
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all').copy()
    return df

def load_summary(filepath):
    # Use os.path.basename to extract filename from filepath
    if os.path.basename(filepath).endswith('.xlsx'): # Use os.path.basename to extract filename
        summary_df = pd.read_excel(filepath, sheet_name="Summary")
        summary_df.columns = summary_df.columns.str.strip()
        # Avoid inplace=True and add .copy()
        summary_df = summary_df.dropna(how='all').copy()
        return summary_df
    return None

def handle_missing(df):
    # Avoid inplace=True and chained assignment warnings
    # ***FIX: Replace fillna(method='ffill') with ffill()***
    df = df.ffill() # Use ffill() directly
    # Then fill remaining NaNs with the median, specifying numeric_only=True
    # Use the result of fillna directly and assign back to df
    # Add .infer_objects(copy=False) to address downcasting warning
    df = df.fillna(df.median(numeric_only=True)).infer_objects(copy=False)
    return df # Return the modified DataFrame

def create_delta_columns(df):
    # Avoid modifying input df directly with inplace=True and chained assignment warnings
    df_copy = df.copy() # Work on a copy to prevent modifying the original df unexpectedly
    open_chem = ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']
    final_chem = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
    # Ensure columns exist before attempting calculation
    for open_col, final_col in zip(open_chem, final_chem):
        if open_col in df_copy.columns and final_col in df_copy.columns:
            delta_col = f"Delta_{open_col.replace('%', '')}"
            # Ensure columns are numeric before subtraction
            # Use errors='coerce' to turn non-numeric values into NaN
            df_copy[open_col] = pd.to_numeric(df_copy[open_col], errors='coerce')
            df_copy[final_col] = pd.to_numeric(df_copy[final_col], errors='coerce')
            df_copy[delta_col] = df_copy[final_col] - df_copy[open_col]
            # Fill NaNs that might result from coercion or original NaNs
            # Calculate median of the delta column and use fillna on the column directly
            median_delta = df_copy[delta_col].median()
            # Avoid inplace=True
            df_copy[delta_col] = df_copy[delta_col].fillna(median_delta)

    return df_copy # Return the modified copy of the DataFrame

def clip_outliers(df):
    # Avoid modifying input df directly with inplace=True warnings
    df_copy = df.copy() # Work on a copy
    # Select only numeric columns for clipping
    numeric_cols = df_copy.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Calculate quantiles and IQR, ignoring NaNs
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clip values in the column and assign back
        # Avoid inplace=True
        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
    return df_copy # Return the modified copy

def preprocess_pipeline(filepath):
    df = load_data(filepath)
    summary_df = load_summary(filepath)
    # Chain the function calls and reassign the result
    # Each function should return a new or modified DataFrame
    df = handle_missing(df)
    df = create_delta_columns(df)
    df = clip_outliers(df)
    # Note: This pipeline returns the processed df and the summary_df separately
    return df, summary_df

def preprocess_data(df):
    # Avoid modifying input df directly
    df = df.copy()
    # Convert datetime or object to numeric where needed
    for col in df.columns:
        if df[col].dtype == 'O':  # If the column is of object (string) type
            try:
                # Specify the format of your date/time column here (if known)
                # Use errors='coerce' and fillna to handle non-datetime strings gracefully
                # Assign the result back to the column
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.total_seconds().fillna(0)
            except Exception: # Catch other potential errors during conversion
                 try:
                      # Handle cases where the column is not a date/time but might be numeric strings
                      # Assign the result back to the column
                      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                 except Exception as e:
                      print(f"Could not convert column {col} to numeric in preprocess_data: {e}")
                      # Assign a default value or leave as NaN if conversion fails completely
                      df[col] = np.nan # Assign NaN then the fillna(0) below will handle it


    # Fill remaining NaNs with 0 (as done in your original preprocess_data)
    # Avoid inplace=True
    df = df.fillna(0)

    # Outlier clipping
    # Avoid modifying input df directly
    # Select only numeric columns for clipping
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clip values in the column and assign back
        # Avoid inplace=True
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df # Return the modified DataFrame

# Re-define the TabTransformer model class if it's not in scope
# Assuming this is the model you trained
class NumericalTabTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, num_heads=2):
        super(NumericalTabTransformer, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), # Another linear layer for transformation
                nn.LayerNorm(hidden_dim) # Add Layer Normalization
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.layers:
            x = x + layer(x)
        x = self.output_layer(x)
        return x

# --- End Corrected Preprocessing Functions and Model Class ---


# --- TabTransformer Model Training Block (Ensure this is run first) ---

# 1. Preprocessing and Scaling
filepath = "FE Alloying.xlsx"
df, summary_df = preprocess_pipeline(filepath)

# Define alloy and process features
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
    'C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%'
]
delta_cols = [f"Delta_{el.replace('%','')}" for el in process_chem_cols_base]
features = [col for col in alloy_cols + process_chem_cols_base + delta_cols if col in df.columns]

target_cols = [f"F-{el}" for el in ['C%', 'Mn%', 'S%', 'P%', 'Si%', 'Cr%', 'Ni%', 'Mo%', 'V%', 'Ti%', 'Al%', 'Ca%', 'N%', 'Pb%', 'Nb%']]
target = [col for col in target_cols if col in df.columns]

# Select data using the filtered features and target lists
# ***FIX: Explicitly create a copy of X and y to avoid SettingWithCopyWarning***
X = df[features].copy() # Use .copy() here
y = df[target].copy() # Use .copy() here

# Handle datetime.time columns and other non-numeric types by coercing to numeric
# This step needs to be applied consistently to X and any future input dataframes
for col in X.select_dtypes(include=['object']).columns:
    try:
        # Ensure this conversion method matches the one used later in evaluate_alloy_additions
        # Convert to string first to handle mixed types or datetime.time
        X[col] = pd.to_numeric(X[col].astype(str), errors='coerce').fillna(0)
    except Exception as e:
        print(f"Could not convert column {col} to numeric in X: {e}")


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale target variables separately
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. PyTorch TabTransformer Model (adapted for numerical features)
input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]
model = NumericalTabTransformer(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Model Training
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

# Save the trained PyTorch model and the scalers
os.makedirs("models", exist_ok=True)

model_path = "models/numerical_tab_transformer.pth"
feature_scaler_path = "models/feature_scaler.pkl"
target_scaler_path = "models/target_scaler.pkl"

torch.save(model.state_dict(), model_path)
print(f"Numerical TabTransformer model saved to: {model_path}")

joblib.dump(scaler, feature_scaler_path)
joblib.dump(target_scaler, target_scaler_path)
print(f"Feature scaler saved to: {feature_scaler_path}")
print(f"Target scaler saved to: {target_scaler_path}")


# Model Evaluation (RMSE, R²)
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

predictions_inv_scaled = target_scaler.inverse_transform(predictions)
actuals_inv_scaled = target_scaler.inverse_transform(actuals)

rmse = np.sqrt(np.mean((predictions_inv_scaled - actuals_inv_scaled)**2))
from sklearn.metrics import r2_score
r2 = r2_score(actuals_inv_scaled, predictions_inv_scaled)

print(f"Test RMSE (Inverse Scaled): {rmse:.4f}")
print(f"Test R² (Inverse Scaled): {r2:.4f}")

# --- End TabTransformer Model Training Block ---


# --- TabTransformer + GA Optimization Block (Modified for multiple runs) ---

def run_ga_optimization(model_path, feature_scaler_path, target_scaler_path, base_inputs, target_chemistry_dict,
                        alloy_elements, all_model_features, df_successful,
                        population_size=50, num_generations=100):
    """
    Runs a single Genetic Algorithm optimization run.

    Args:
        model_path, feature_scaler_path, target_scaler_path (str): Paths to model/scalers.
        base_inputs (dict): Base process and open chemistry inputs.
        target_chemistry_dict (dict): Target final chemistry values (original scale).
        alloy_elements (list): List of alloy elements to optimize.
        all_model_features (list): List of all features the model was trained on.
        df_successful (pd.DataFrame): DataFrame for determining alloy bounds.
        population_size (int): Number of individuals in the GA population.
        num_generations (int): Number of generations to run the GA.

    Returns:
        tuple: A dictionary of optimized alloy additions, the predicted final chemistry
               for these additions, and the final RMSE.
    """
    # Load the trained PyTorch model and scalers
    loaded_model = NumericalTabTransformer(len(all_model_features), len(target_chemistry_dict)) # Match dimensions
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    loaded_feature_scaler = joblib.load(feature_scaler_path)
    loaded_target_scaler = joblib.load(target_scaler_path)

    # Get base inputs (must match the features the model was trained on)
    # Ensure base_inputs is reindexed to match all_model_features order
    base_inputs_df = pd.DataFrame([base_inputs]).reindex(columns=all_model_features, fill_value=0)
    # Handle non-numeric types consistently as done during training
    for col in base_inputs_df.select_dtypes(include=['object']).columns:
        try:
            base_inputs_df[col] = pd.to_numeric(base_inputs_df[col].astype(str), errors='coerce').fillna(0)
        except Exception as e:
             print(f"Warning: Could not convert column {col} to numeric in base_inputs for GA: {e}")
             base_inputs_df[col] = pd.to_numeric(base_inputs_df[col], errors='coerce').fillna(0)
    base_inputs_reindexed = base_inputs_df.iloc[0].to_dict()


    # Identify the alloy features included in the model's features
    alloy_features_in_model = [col for col in alloy_elements if col in all_model_features]

    # Bounds for alloy elements for optimization
    bounds = [(max(0, df_successful[el].min()) if el in df_successful.columns else 0.0,
               df_successful[el].max() if el in df_successful.columns else 100.0)
              for el in alloy_features_in_model]
    bounds = [(lower, upper) if lower < upper else (lower, lower + 1e-6 if lower == 0 else lower + abs(lower)*1e-6) for lower, upper in bounds]

    low_bounds = [b[0] for b in bounds]
    up_bounds = [b[1] for b in bounds]

    # DEAP setup
    # Create new types unique to this run to avoid conflicts
    try:
        del creator.FitnessMinGA
        del creator.IndividualGA
    except AttributeError:
        pass
    creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
    creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.IndividualGA,
                     lambda: [random.uniform(low, high) for low, high in bounds])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutGaussianBounded, mu=0, sigma=1, indpb=0.2, low=low_bounds, up=up_bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluation function for DEAP
    def evaluate_alloy_additions(individual):
        alloy_values = individual
        alloy_dict = dict(zip(alloy_features_in_model, alloy_values))

        input_data = base_inputs_reindexed.copy() # Use the reindexed base inputs
        input_data.update(alloy_dict)

        # Create DataFrame with exact 'all_model_features' columns
        input_df = pd.DataFrame([input_data]).reindex(columns=all_model_features, fill_value=0)

        # Handle non-numeric types consistently
        for col in input_df.select_dtypes(include=['object']).columns:
             try:
                 input_df[col] = pd.to_numeric(input_df[col].astype(str), errors='coerce').fillna(0)
             except Exception as e:
                  print(f"Warning: Could not convert column {col} to numeric in evaluate_alloy_additions: {e}")
                  input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Scale input
        input_scaled = loaded_feature_scaler.transform(input_df)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predict (scaled)
        with torch.no_grad():
            predicted_scaled = loaded_model(input_tensor).numpy()[0]

        # Inverse transform prediction
        predicted_chem = loaded_target_scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]

        # Calculate RMSE against target chemistry (original scale)
        # Ensure target values are in the same order as the model's output targets
        target_values = np.array([target_chemistry_dict.get(col, 0) for col in target_chemistry_dict.keys()]) # Use keys from the target dict

        # Ensure prediction and target have same number of elements for comparison
        num_elements_to_compare = min(len(predicted_chem), len(target_values))
        predicted_chem_sliced = predicted_chem[:num_elements_to_compare]
        target_values_sliced = target_values[:num_elements_to_compare]

        rmse = np.sqrt(np.mean((predicted_chem_sliced - target_values_sliced)**2))

        return rmse,

    toolbox.register("evaluate", evaluate_alloy_additions)

    # Run GA
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Run the GA
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=False)

    # Best solution from this run
    if hof:
        best_individual = hof[0]
        final_rmse = best_individual.fitness.values[0]
        # Map optimized values back to alloy column names (including non-optimized as 0)
        optimized_alloy_additions_raw = dict(zip(alloy_features_in_model, best_individual))
        optimized_alloy_additions = {key: max(0.0, value) for key, value in optimized_alloy_additions_raw.items()}
        optimized_alloy_additions_full = {col: optimized_alloy_additions.get(col, 0.0) for col in alloy_elements}

        # Predict final chemistry for the best individual
        optimized_input_dict = base_inputs_reindexed.copy()
        optimized_input_dict.update(optimized_alloy_additions_full) # Use the full dictionary here

        optimized_input_df = pd.DataFrame([optimized_input_dict]).reindex(columns=all_model_features, fill_value=0)
        for col in optimized_input_df.select_dtypes(include=['object']).columns:
             try:
                 optimized_input_df[col] = pd.to_numeric(optimized_input_df[col].astype(str), errors='coerce').fillna(0)
             except Exception as e:
                  print(f"Warning: Could not convert column {col} to numeric for final prediction: {e}")
                  optimized_input_df[col] = pd.to_numeric(optimized_input_df[col], errors='coerce').fillna(0)

        optimized_input_scaled = loaded_feature_scaler.transform(optimized_input_df)
        optimized_input_tensor = torch.tensor(optimized_input_scaled, dtype=torch.float32)

        with torch.no_grad():
            predicted_scaled_opt = loaded_model(optimized_input_tensor).numpy()[0]

        predicted_chem_opt = loaded_target_scaler.inverse_transform(predicted_scaled_opt.reshape(1, -1))[0]

        return optimized_alloy_additions_full, predicted_chem_opt, final_rmse

    else:
        return None, None, float('inf') # Return None and infinity if no solution found


# --- Run Multiple GA Optimizations ---

# Load necessary data and define parameters (ensure these match the training block)
# Use base_inputs_opt and the target chemistry dictionary you chose (e.g., max_chem or aim_chem)
# These variables should be available from the execution of the training block above.

# Assuming base_inputs_opt, max_chem, aim_chem, min_chem, alloy_cols, features, df_successful are defined

# ***CHOOSE YOUR TARGET CHEMISTRY FOR OPTIMIZATION***
# target_chemistry_for_optimization = max_chem # Or
target_chemistry_for_optimization = aim_chem # ***Often you want to optimize towards Aim***


num_ga_runs = 5 # Number of times to run the GA
population_size = 100 # GA population size
num_generations = 200 # Number of GA generations

best_rmse_across_runs = float('inf')
best_alloys_overall = None
best_predicted_chem_overall = None

all_run_results = [] # Store results from each run

print(f"\n--- Running {num_ga_runs} Genetic Algorithm Optimizations ---")

for run in range(num_ga_runs):
    print(f"Starting GA Run {run + 1}/{num_ga_runs}...")
    start_time = time.time()

    optimized_alloys_this_run, predicted_chem_this_run, rmse_this_run = run_ga_optimization(
        model_path=model_path, # Path from training block
        feature_scaler_path=feature_scaler_path, # Path from training block
        target_scaler_path=target_scaler_path, # Path from training block
        base_inputs=base_inputs_opt, # Use the base inputs dictionary
        target_chemistry_dict=target_chemistry_for_optimization, # Use the chosen target chemistry
        alloy_elements=alloy_cols, # Use the full list of possible alloy columns
        all_model_features=features, # Use the exact list of features the model was trained on
        df_successful=df_successful, # Pass df_successful for bounds
        population_size=population_size,
        num_generations=num_generations
    )

    end_time = time.time()
    print(f"Run {run + 1} finished in {end_time - start_time:.2f} seconds with RMSE: {rmse_this_run:.4f}")

    if optimized_alloys_this_run is not None:
         all_run_results.append({
             'run': run + 1,
             'rmse': rmse_this_run,
             'alloys': optimized_alloys_this_run,
             'predicted_chem': predicted_chem_this_run
         })

        # Track the best result across all runs
         if rmse_this_run < best_rmse_across_runs:
             best_rmse_across_runs = rmse_this_run
             best_alloys_overall = optimized_alloys_this_run
             best_predicted_chem_overall = predicted_chem_this_run

print("\n--- Optimization Complete ---")

# --- Display Results ---

if best_alloys_overall is not None:
    print("\n--- Best Optimized Alloy Additions Across All Runs (TabTransformer + GA) ---")
    # Ensure the optimized_alloys_pso dictionary includes all original alloy_cols, setting non-optimized to 0
    best_alloys_df = pd.DataFrame([best_alloys_overall]).T.rename(columns={0: "Recommended kg"})
    display(best_alloys_df.round(4))

    print("\n--- Predicted Final Chemistry (Best Run) vs Target ---")
    # Ensure consistent keys for comparison using the keys from target_chemistry_for_optimization
    all_target_chem_keys = list(target_chemistry_for_optimization.keys())

    predicted_chem_comparison_data = {
        "Target": [target_chemistry_for_optimization.get(k, np.nan) for k in all_target_chem_keys],
        # Create a dictionary from the predicted array using the target keys for alignment
        "Predicted": dict(zip(all_target_chem_keys, best_predicted_chem_overall[:len(all_target_chem_keys)]))
    }

    chem_comparison_df = pd.DataFrame(predicted_chem_comparison_data, index=all_target_chem_keys)

    # Add Delta to Target column for easier comparison
    chem_comparison_df['Delta to Target'] = chem_comparison_df['Predicted'] - chem_comparison_df['Target']

    display(chem_comparison_df.round(5))

    print(f"\nOverall Best RMSE to Target: {best_rmse_across_runs:.4f}")


    # Optional: You could also calculate the average or median of the optimized alloys
    # across all runs with reasonable RMSE if you prefer an ensemble recommendation.
    # Filter successful runs (e.g., RMSE below a threshold or top N)
    # successful_runs = [res for res in all_run_results if res['rmse'] < best_rmse_across_runs * 1.1] # Example threshold
    # if successful_runs:
    #     # Create a DataFrame from alloy dictionaries
    #     alloy_dfs = [pd.DataFrame([run['alloys']]) for run in successful_runs]
    #     combined_alloys = pd.concat(alloy_dfs, ignore_index=True)
    #     # Calculate median recommendations
    #     median_alloys = combined_alloys.median().to_dict()
    #     print("\n--- Median Recommended Alloy Additions (Across Successful Runs) ---")
    #     display(pd.DataFrame([median_alloys]).T.rename(columns={0: "Recommended kg"}).round(4))

else:
    print("No successful optimization runs completed.")