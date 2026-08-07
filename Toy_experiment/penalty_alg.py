"""Algorithm ``pen:alg`` (Penalty subgradient method), generic over the
application. See bilevel_discret.tex, Sec. "Algorithm and convergence"
(pen:sec:alg), lines 950-965.

An "application" is any object exposing:
    x0                       -- initial point
    project(x)                -- Pi_X(x)
    solve_g(x)                -- SFM on g(x,.); returns (y2, g_val)
    solve_f_plus_gamma_g(x, gamma) -- SFM on f(x,.)+gamma g(x,.);
                                       returns (y1, f_val_at_y1, g_val_at_y1)
    grad_x_f(x, y)             -- gradient of f(x,y) w.r.t. x
    grad_x_g(x, y)             -- gradient of g(x,y) w.r.t. x
    f(x, y), g(x, y)           -- energy evaluation (for logging)

Both SFM calls are minimum s-t cuts (mincut.solve_energy) in both toy
applications, per app:lem:ising.
"""

import numpy as np


def run(app, gamma, num_iters, step_sizes):
    """Run Algorithm pen:alg for num_iters steps.

    step_sizes: array-like of length num_iters, the eta_t.

    Returns a dict of per-iteration logs:
        x_hist       -- x_t for t = 0..num_iters (num_iters+1 entries)
        p_gamma      -- p_gamma(x_t) = f(x_t,y1) + gamma*(g(x_t,y1)-g(x_t,y2))
        exact_gap    -- g(x_t,y1) - g(x_t,y2), i.e. the joint-vs-lower-solve
                        lower-level gap (pen:lem:basic/pen:cor:awayties
                        predict this -> 0 as x_t moves away from tie sets)
        phi_est      -- f(x_t,y2), the running bilevel-objective estimate
        y1_hist, y2_hist -- the two SFM solutions at each step
    """
    x = np.array(app.x0, dtype=float)
    x_hist = [x.copy()]
    p_gamma_hist = []
    gap_hist = []
    phi_hist = []
    y1_hist = []
    y2_hist = []

    for t in range(num_iters):
        y1, f_at_y1, g_at_y1 = app.solve_f_plus_gamma_g(x, gamma)
        y2, g_at_y2 = app.solve_g(x)

        p_gamma_hist.append(f_at_y1 + gamma * (g_at_y1 - g_at_y2))
        gap_hist.append(g_at_y1 - g_at_y2)
        phi_hist.append(app.f(x, y2))
        y1_hist.append(y1)
        y2_hist.append(y2)

        u = app.grad_x_f(x, y1) + gamma * (app.grad_x_g(x, y1) - app.grad_x_g(x, y2))
        x = app.project(x - step_sizes[t] * u)
        x_hist.append(x.copy())

    return {
        "x_hist": x_hist,
        "p_gamma": np.array(p_gamma_hist),
        "exact_gap": np.array(gap_hist),
        "phi_est": np.array(phi_hist),
        "y1_hist": y1_hist,
        "y2_hist": y2_hist,
    }
