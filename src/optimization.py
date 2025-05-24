import numpy as np
import pandas as pd
import torch
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
from pyswarms.single.global_best import GlobalBestPSO

def setup_ga_environment(bounds):
    try:
        del creator.FitnessMin
        del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [np.random.uniform(low, high) for low, high in bounds])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    return toolbox, mutGaussianBounded

def mutGaussianBounded(individual, mu, sigma, indpb, low, up):
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            mutated = individual[i] + np.random.normal(mu, sigma)
            individual[i] = np.clip(mutated, low[i], up[i])
    return individual,

def run_ga_optimization(
    model, model_type, features, target, feature_scaler, target_scaler,
    base_inputs, target_chemistry_dict, alloy_cols, df,
    ngen=50, pop_size=50, seed=42
):
    np.random.seed(seed)
    random.seed(seed)
    alloy_features_in_model = [col for col in alloy_cols if col in features]
    bounds = [(max(0, df[col].min()), df[col].max()) for col in alloy_features_in_model]
    low_bounds, up_bounds = zip(*[(l, u if l < u else l + 1e-6) for l, u in bounds])

    toolbox, mutator = setup_ga_environment(bounds)
    toolbox.register("mutate", mutator, mu=0, sigma=1, indpb=0.2, low=low_bounds, up=up_bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(individual):
        input_data = base_inputs.copy()
        input_data.update(dict(zip(alloy_features_in_model, individual)))
        input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        scaled = feature_scaler.transform(input_df)

        if model_type == "TabTransformer":
            input_tensor = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model(input_tensor).numpy()[0]
        else:
            pred_scaled = model.predict(scaled)[0]

        pred = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        target_vec = np.array([target_chemistry_dict.get(col, 0) for col in target])
        return np.sqrt(np.mean((pred[:len(target)] - target_vec[:len(target)]) ** 2)),

    toolbox.register("evaluate", evaluate)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=False)

    best = hof[0]
    optimized_alloys = dict(zip(alloy_features_in_model, best))
    optimized_alloys = {k: max(0.0, v) for k, v in optimized_alloys.items()}

    final_input = base_inputs.copy()
    final_input.update(optimized_alloys)
    final_df = pd.DataFrame([final_input]).reindex(columns=features, fill_value=0).apply(pd.to_numeric, errors='coerce').fillna(0)
    scaled_final = feature_scaler.transform(final_df)

    if model_type == "TabTransformer":
        input_tensor = torch.tensor(scaled_final, dtype=torch.float32)
        with torch.no_grad():
            pred_final = model(input_tensor).numpy()[0]
    else:
        pred_final = model.predict(scaled_final)[0]

    chem = target_scaler.inverse_transform(pred_final.reshape(1, -1))[0]
    return optimized_alloys, chem

def run_pso_optimization(
    model, model_type, features, target, feature_scaler, target_scaler,
    base_inputs, target_chemistry_dict, alloy_cols, df_successful,
    n_particles=50, iters=100, seed=42
):
    np.random.seed(seed)
    random.seed(seed)
    alloy_features_in_model = [col for col in alloy_cols if col in features]
    bounds = [(max(0, df_successful[col].min()), df_successful[col].max()) for col in alloy_features_in_model]
    low, up = zip(*[(l, u if l < u else l + 1e-6) for l, u in bounds])

    def fitness_function(alloy_matrix):
        fitness = []
        for individual in alloy_matrix:
            input_dict = base_inputs.copy()
            input_dict.update(dict(zip(alloy_features_in_model, individual)))
            input_df = pd.DataFrame([input_dict]).reindex(columns=features, fill_value=0)
            input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            scaled = feature_scaler.transform(input_df)

            if model_type == "TabTransformer":
                input_tensor = torch.tensor(scaled, dtype=torch.float32)
                with torch.no_grad():
                    pred_scaled = model(input_tensor).numpy()[0]
            else:
                pred_scaled = model.predict(scaled)[0]

            pred = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            target_vec = np.array([target_chemistry_dict.get(col, 0) for col in target])
            rmse = np.sqrt(np.mean((pred[:len(target)] - target_vec[:len(target)]) ** 2))
            fitness.append(rmse)
        return np.array(fitness)

    from pyswarms.single.global_best import GlobalBestPSO
    options = {'c1': 0.5, 'c2': 0.7, 'w': 0.4}
    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=len(alloy_features_in_model),
                              options=options, bounds=(low, up))
    cost, pos = optimizer.optimize(fitness_function, iters=iters)

    best_individual = pos
    optimized_alloys = dict(zip(alloy_features_in_model, best_individual))
    optimized_alloys = {k: max(0.0, v) for k, v in optimized_alloys.items()}

    final_input = base_inputs.copy()
    final_input.update(optimized_alloys)
    final_df = pd.DataFrame([final_input]).reindex(columns=features, fill_value=0).apply(pd.to_numeric, errors='coerce').fillna(0)
    scaled_final = feature_scaler.transform(final_df)

    if model_type == "TabTransformer":
        input_tensor = torch.tensor(scaled_final, dtype=torch.float32)
        with torch.no_grad():
            pred_final = model(input_tensor).numpy()[0]
    else:
        pred_final = model.predict(scaled_final)[0]

    chem = target_scaler.inverse_transform(pred_final.reshape(1, -1))[0]
    return optimized_alloys, chem
