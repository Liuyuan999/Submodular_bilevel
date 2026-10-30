"""Sparse exact RC cuts with separate persistent g and F graphs."""

from __future__ import annotations

import numpy as np

from dreamplace.cut_oracle import CutOracle


class RCCutOracle:
    def __init__(self, model, gamma=1.0, backend="auto", reuse_trees=True):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("Gamma must be finite and positive")
        self.model = model
        self.gamma = float(gamma)
        self.scale = self.gamma
        self.pairs = {
            "g": model.lower_pairs,
            "F": (model.upper_pairs + self.gamma * model.lower_pairs) / self.scale,
        }
        self.graphs = {name: CutOracle(model.movable_count, model.pair_first, model.pair_second,
                                      -pairs / 2, [], [], backend=backend, reuse_trees=reuse_trees)
                       for name, pairs in self.pairs.items()}

    @property
    def calls(self):
        return sum(graph.calls for graph in self.graphs.values())

    def solve(self, positions, response="g"):
        if response not in self.graphs:
            raise ValueError("Response must be g or F")
        lower, upper = self.model.unaries(positions)
        unary = lower if response == "g" else (upper + self.gamma * lower) / self.scale
        unary = unary.copy()
        np.add.at(unary, self.model.pair_first, self.pairs[response] / 2)
        np.add.at(unary, self.model.pair_second, self.pairs[response] / 2)
        return self.graphs[response].solve(np.zeros(self.model.movable_count), unary)

    def responses(self, positions):
        penalty_labels = self.solve(positions, "F")
        return self.solve(positions, "g"), penalty_labels

    def direction(self, positions):
        lower_labels, upper_labels = self.responses(positions)
        lower = self.model.evaluate(positions, lower_labels, True)
        upper = self.model.evaluate(positions, upper_labels, True)
        direction = (upper["upper_gradient"] + self.gamma *
                     (upper["lower_gradient"] - lower["lower_gradient"]))
        return direction, lower_labels, upper_labels
