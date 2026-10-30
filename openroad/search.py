"""Nested CMA-ES and uniform search with exact lower-level responses."""

from time import perf_counter

import numpy as np

from .oracle import RCCutOracle


def train_search(model, method, seed=0, time_budget=300.0, iterations=1000000,
                 backend="pymaxflow", sigma=0.15):
    if method not in ("cmaes", "random"):
        raise ValueError("Expected cmaes or random")
    if time_budget <= 0 or not np.isfinite(time_budget) or iterations < 1:
        raise ValueError("Positive time budget and evaluation limit required")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Positive finite CMA-ES sigma required")
    if method == "cmaes":
        import cma
    generator = np.random.default_rng(seed)
    lower = model.lower_bound.ravel()
    widths = (model.upper_bound - model.lower_bound).ravel()
    if np.any(widths <= 0):
        raise ValueError("Search requires positive coordinate intervals")
    normalized_start = (model.reference.ravel() - lower) / widths
    history = []
    started = perf_counter()
    oracle = RCCutOracle(model, backend=backend)
    strategy = None
    if method == "cmaes":
        strategy = cma.CMAEvolutionStrategy(normalized_start.tolist(), sigma,
                                            {"bounds": [0.0, 1.0], "seed": seed + 1,
                                             "verbose": -9, "verb_log": 0})
    population, scores = [], []
    stop_reason = "evaluation_limit"
    for iteration in range(iterations):
        if perf_counter() - started >= time_budget:
            stop_reason = "time_budget"
            break
        if iteration == 0:
            positions = model.reference.copy()
        elif method == "random":
            positions = (lower + widths * generator.random(lower.size)).reshape(model.reference.shape)
        else:
            if not population:
                if strategy.stop():
                    stop_reason = str(strategy.stop())
                    break
                population = strategy.ask()
                scores = []
            positions = (lower + widths * np.asarray(population[len(scores)])).reshape(model.reference.shape)
        labels = oracle.solve(positions, "g")
        measured = model.evaluate(positions, labels)
        seconds = perf_counter() - started
        if seconds > time_budget:
            stop_reason = "time_budget"
            break
        history.append({"iteration": iteration, "seconds": seconds, "positions": positions.tolist(),
                        "responses": {"g": {"labels": labels.tolist(),
                                             "f": measured["upper"], "g": measured["lower"]}},
                        "training_cut_calls": oracle.calls})
        if method == "cmaes" and iteration > 0:
            scores.append(measured["upper"])
            if len(scores) == len(population):
                strategy.tell(population, scores)
                population = []
    return {"method": method, "seed": seed, "gamma": 1.0,
            "history": history, "algorithm_output": None,
            "time_budget_seconds": time_budget, "elapsed_training_seconds": perf_counter() - started,
            "cut_calls": oracle.calls, "stop_reason": stop_reason,
            "cma_sigma_normalized": sigma if strategy is not None else None,
            "cma_population_size": strategy.popsize if strategy is not None else None,
            "search_objective": "f(x, exact g response), with backend LL tie rule",
            "clock": "includes optimizer initialization, proposal/update cost, exact LL solves and record generation; excludes deferred audit",
            "budget_policy": "only completed candidates within budget are recorded; final operation can overrun"}
