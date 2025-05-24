# alloy_optimizer.py

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import MinMaxScaler
import joblib

# Define the optimization function
def optimize_alloy_additions(model, base_inputs, target_chemistry, alloy_elements, df_successful):
    # Bounds for each alloy based on min/max scaling in successful heats
    bounds = [(df_successful[el].min(), df_successful[el].max()) for el in alloy_elements]

    # Normalize input features for model prediction
    def prepare_input(alloy_values):
        input_dict = base_inputs.copy()
        input_dict.update(dict(zip(alloy_elements, alloy_values)))
        input_df = pd.DataFrame([input_dict])
        return input_df

    # Define the fitness function
    def fitness_function(individual):
        input_df = prepare_input(individual)
        prediction = model.predict(input_df)[0]
        target = np.array([target_chemistry.get(col, 0) for col in model.feature_names_out_])
        return np.sqrt(np.mean((prediction - target) ** 2)),  # Comma for DEAP tuple

    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda i=0: np.random.uniform(*bounds[i]))
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: [np.random.uniform(low, high) for low, high in bounds])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run GA
    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, halloffame=hof, verbose=True)

    # Best solution
    best_individual = hof[0]
    best_prediction = model.predict(prepare_input(best_individual))[0]

    return dict(zip(alloy_elements, best_individual)), best_prediction

# Example usage (Streamlit or script should call this):
# model = joblib.load("models/xgboost_multioutput.pkl")
# optimized_alloys, predicted_chem = optimize_alloy_additions(model, base_inputs, target_chem, alloy_list, df_successful)