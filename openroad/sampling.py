"""Ball sampling and small validation fixtures."""

import numpy as np


def controlled_instance(cell_count, seed):
    if not 1 <= cell_count <= 12:
        raise ValueError("The controlled exact-enumeration tier supports 1–12 cells")
    generator = np.random.default_rng(seed)
    reference = generator.uniform([2.0, 2.0], [8.0, 8.0], size=(cell_count, 2))
    node_count = cell_count + 2
    resistance = np.tile([2.0, 1.2], (node_count, 1))
    input_cap = np.tile([0.8, 1.3], (node_count, 1))
    width = np.tile([0.5, 0.7], (node_count, 1))
    height = np.tile([0.5, 0.7], (node_count, 1))
    intrinsic_delay = np.tile([0.15, 0.12], (node_count, 1))
    leakage = np.tile([0.5, 0.9], (node_count, 1))
    for node in range(cell_count, node_count):
        resistance[node] = [1.0, 1.0]
        input_cap[node] = [0.8, 0.8]
        width[node] = [0, 0]
        height[node] = [0, 0]
        intrinsic_delay[node] = [0, 0]
        leakage[node] = [0, 0]
    arcs = [(cell_count, 0, 0.04, 1.0)]
    arcs.extend((node, node + 1, 0.04, 1.0) for node in range(cell_count - 1))
    arcs.append((cell_count - 1, cell_count + 1, 0.04, 1.0))
    return {
        "origin": "synthetic_diagnostic",
        "die": [10.0, 10.0], "grid": [4, 4],
        "fixed_positions": [[0.5, 5.0], [9.5, 5.0]],
        "reference": reference.tolist(),
        "resistance": resistance.tolist(), "input_cap": input_cap.tolist(),
        "width": width.tolist(), "height": height.tolist(),
        "intrinsic_delay": intrinsic_delay.tolist(), "leakage": leakage.tolist(),
        "wire_cap_base": [0.05] * node_count,
        "lower_timing_weights": [1.0] * cell_count + [1.0, 0.0],
        "upper_timing_weights": [1.0 + 0.4 * (node % 2) for node in range(cell_count)] + [0.5, 0.0],
        "arcs": arcs,
        "nets": [{"nodes": [source, target], "weight": 1.0}
                 for source, target, _, _ in arcs],
        "wire_smoothing": 0.1, "rc_smoothing": 0.05,
        "normalizers": {"wirelength": 10.0 * cell_count,
                        "timing": 1.0 * cell_count,
                        "density": 1.0,
                        "area": 0.25 * cell_count,
                        "leakage": 0.5 * cell_count},
        "lower_weights": {"area": 0.05, "leakage": 0.05, "density": 1.0},
        "upper_weights": {"wirelength": 0.5, "displacement": 0.2,
                          "density": 1.0, "timing": 1.0,
                          "area": 0.8, "leakage": 0.4},
    }


def ball_sample(generator, shape):
    direction = generator.normal(size=shape)
    direction /= np.linalg.norm(direction)
    return direction * generator.random() ** (1.0 / direction.size)

