"""Shared placement geometry and binary cell-sizing model."""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


class PlacementSizingModel:
    def __init__(self, data):
        self.data = data
        self.die = np.asarray(data["die"], dtype=float)
        self.grid = tuple(int(value) for value in data["grid"])
        self.fixed_positions = np.asarray(data.get("fixed_positions", []), dtype=float).reshape(-1, 2)
        self.resistance = np.asarray(data["resistance"], dtype=float)
        self.input_cap = np.asarray(data["input_cap"], dtype=float)
        self.intrinsic_delay = np.asarray(data["intrinsic_delay"], dtype=float)
        self.width = np.asarray(data["width"], dtype=float)
        self.height = np.asarray(data["height"], dtype=float)
        self.leakage = np.asarray(data["leakage"], dtype=float)
        self.node_count = len(self.resistance)
        self.movable_count = self.node_count - len(self.fixed_positions)
        self.reference = np.asarray(data["reference"], dtype=float)
        self.wire_cap_base = np.asarray(data["wire_cap_base"], dtype=float)
        self.lower_timing_weights = np.asarray(data["lower_timing_weights"], dtype=float)
        self.upper_timing_weights = np.asarray(data["upper_timing_weights"], dtype=float)
        self.arcs = [tuple(arc) for arc in data["arcs"]]
        self.nets = [(tuple(net["nodes"]), float(net["weight"])) for net in data["nets"]]
        self.normalizers = data["normalizers"]
        self.lower_weights = data["lower_weights"]
        self.upper_weights = data["upper_weights"]
        self.wire_smoothing = float(data["wire_smoothing"])
        self.rc_smoothing = float(data["rc_smoothing"])
        self.fixed_charge_factor = float(data.get("fixed_charge_factor", 1.0))
        self.area = self.width * self.height
        self.bin_width = self.die[0] / self.grid[0]
        self.bin_height = self.die[1] / self.grid[1]
        self.bin_area = self.bin_width * self.bin_height
        frequency_x = np.arange(self.grid[0])[:, None]
        frequency_y = np.arange(self.grid[1])[None, :]
        denominator = (np.pi * frequency_x / self.die[0]) ** 2 + (np.pi * frequency_y / self.die[1]) ** 2
        self.inverse_frequency = np.zeros(self.grid, dtype=float)
        np.divide(1.0, denominator, out=self.inverse_frequency, where=denominator > 0)
        self.lower_bound = np.column_stack((np.max(self.width[:self.movable_count], axis=1) / 2,
                                             np.max(self.height[:self.movable_count], axis=1) / 2))
        self.upper_bound = self.die - self.lower_bound
        self._validate()

    def _validate(self):
        if self.movable_count < 1 or self.node_count < self.movable_count:
            raise ValueError("At least one movable cell is required")
        if self.die.shape != (2,) or np.any(self.die <= 0) or min(self.grid) < 1:
            raise ValueError("Invalid die or bin grid")
        for array in (self.resistance, self.input_cap, self.intrinsic_delay,
                      self.width, self.height, self.leakage):
            if array.shape != (self.node_count, 2) or not np.all(np.isfinite(array)):
                raise ValueError("Each library property needs two finite values per node")
            if np.any(array < 0):
                raise ValueError("Library properties must be nonnegative")
            if np.any(array[self.movable_count:, 0] != array[self.movable_count:, 1]):
                raise ValueError("Fixed nodes must have identical variants")
        if np.any(self.resistance <= 0) or np.any(self.area[:, 1] < self.area[:, 0]):
            raise ValueError("Resistance must be positive and large-cell area cannot decrease")
        if np.any(self.resistance[:, 1] > self.resistance[:, 0]):
            raise ValueError("Large-cell resistance cannot increase")
        if np.any(self.input_cap[:, 1] < self.input_cap[:, 0]) or np.any(self.leakage[:, 1] < self.leakage[:, 0]):
            raise ValueError("Large-cell input capacitance and leakage cannot decrease")
        if self.reference.shape != (self.movable_count, 2) or np.any(self.lower_bound > self.upper_bound):
            raise ValueError("Invalid placement reference or empty placement domain")
        if not np.all(np.isfinite(self.reference)) or not np.all(np.isfinite(self.fixed_positions)):
            raise ValueError("Placement coordinates must be finite")
        if np.any(self.reference < self.lower_bound) or np.any(self.reference > self.upper_bound):
            raise ValueError("The reference placement must be feasible")
        for array in (self.wire_cap_base, self.lower_timing_weights, self.upper_timing_weights):
            if array.shape != (self.node_count,) or np.any(array < 0):
                raise ValueError("Invalid node weights or base wire capacitances")
        if self.wire_smoothing <= 0 or self.rc_smoothing <= 0:
            raise ValueError("Both smoothing lengths must be positive")
        if not 0 <= self.fixed_charge_factor <= 1:
            raise ValueError("Fixed charge factor must lie in [0, 1]")
        if any(float(value) <= 0 for value in self.normalizers.values()):
            raise ValueError("Normalizers must be positive")
        if any(float(value) < 0 for weights in (self.lower_weights, self.upper_weights)
               for value in weights.values()):
            raise ValueError("Objective weights must be nonnegative")
        for arc in self.arcs:
            if len(arc) not in {4, 5}:
                raise ValueError("Timing arcs need four fields or an explicit binary input-capacitance pair")
            source, target, wire_cap, input_scale = arc[:4]
            if source == target or min(source, target) < 0 or max(source, target) >= self.node_count:
                raise ValueError("Invalid timing arc")
            if wire_cap < 0 or input_scale < 0:
                raise ValueError("Timing arc capacitances must be nonnegative")
            if len(arc) == 5:
                input_pair = np.asarray(arc[4], dtype=float)
                if (input_pair.shape != (2,) or not np.all(np.isfinite(input_pair)) or
                        np.any(input_pair < 0) or input_pair[1] < input_pair[0]):
                    raise ValueError("An arc's binary input-capacitance pair must be nonnegative and monotone")
        adjacency = [[] for _ in range(self.node_count)]
        indegree = np.zeros(self.node_count, dtype=int)
        for arc in self.arcs:
            source, target = arc[:2]
            adjacency[source].append(target)
            indegree[target] += 1
        available = list(np.flatnonzero(indegree == 0))
        visited = 0
        while available:
            source = available.pop()
            visited += 1
            for target in adjacency[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)
        if visited != self.node_count:
            raise ValueError("The timing graph must be acyclic")
        for nodes, weight in self.nets:
            if not nodes or min(nodes) < 0 or max(nodes) >= self.node_count or weight < 0:
                raise ValueError("Invalid net")

    def project(self, positions):
        return np.clip(np.asarray(positions, dtype=float), self.lower_bound, self.upper_bound)

    def physical_hpwl(self, positions):
        coordinates = self._coordinates(positions)
        total = 0.0
        for nodes, weight in self.nets:
            selected = coordinates[list(nodes)]
            total += weight * float(np.ptp(selected[:, 0]) + np.ptp(selected[:, 1]))
        return total

    def _coordinates(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.movable_count, 2) or not np.all(np.isfinite(positions)):
            raise ValueError("Invalid movable coordinates")
        return np.vstack((positions, self.fixed_positions))

    def _variants(self, labels):
        labels = np.asarray(labels, dtype=int)
        if labels.shape != (self.movable_count,) or np.any((labels != 0) & (labels != 1)):
            raise ValueError("Labels must be a binary vector for movable cells")
        return np.concatenate((labels, np.zeros(len(self.fixed_positions), dtype=int)))

    @staticmethod
    def _axis_deposition(center, footprint, die_length, bin_count):
        edges = np.linspace(0, die_length, bin_count + 1)
        left = center - footprint / 2
        right = center + footprint / 2
        overlap = np.maximum(0.0, np.minimum(right, edges[1:]) - np.maximum(left, edges[:-1]))
        total = overlap.sum()
        if total <= 0:
            raise ValueError("Density footprint does not intersect the die")
        derivative = np.where(overlap > 0, (right < edges[1:]).astype(float) -
                              (left > edges[:-1]).astype(float), 0.0)
        normalized = overlap / total
        normalized_derivative = (derivative - normalized * derivative.sum()) / total
        return normalized, normalized_derivative

    def _density(self, coordinates, variants, with_grad):
        charges = np.zeros(self.grid, dtype=float)
        deposits = []
        for node in range(self.node_count):
            choice = int(variants[node])
            charged_area = self.area[node, choice]
            if node >= self.movable_count:
                charged_area *= self.fixed_charge_factor
            if charged_area == 0:
                deposits.append(None)
                continue
            horizontal, horizontal_derivative = self._axis_deposition(
                coordinates[node, 0], max(self.width[node, choice], self.bin_width),
                self.die[0], self.grid[0])
            vertical, vertical_derivative = self._axis_deposition(
                coordinates[node, 1], max(self.height[node, choice], self.bin_height),
                self.die[1], self.grid[1])
            charges += charged_area * np.outer(horizontal, vertical)
            deposits.append((charged_area, horizontal, vertical,
                             horizontal_derivative, vertical_derivative))
        potential = idctn(dctn(charges, norm="ortho") * self.inverse_frequency,
                          norm="ortho") / self.bin_area
        energy = 0.5 * float(np.sum(charges * potential))
        if not with_grad:
            return energy, None, charges
        gradient = np.zeros((self.movable_count, 2), dtype=float)
        for node in range(self.movable_count):
            if deposits[node] is None:
                continue
            charged_area, horizontal, vertical, horizontal_derivative, vertical_derivative = deposits[node]
            gradient[node, 0] = charged_area * np.sum(potential * np.outer(horizontal_derivative, vertical))
            gradient[node, 1] = charged_area * np.sum(potential * np.outer(horizontal, vertical_derivative))
        return energy, gradient, charges

    def _wirelength(self, coordinates, with_grad):
        total = 0.0
        gradient = np.zeros((self.movable_count, 2), dtype=float)
        for nodes, weight in self.nets:
            for axis in range(2):
                values = coordinates[list(nodes), axis] / self.wire_smoothing
                positive_peak = np.max(values)
                negative_peak = np.max(-values)
                positive_exp = np.exp(values - positive_peak)
                negative_exp = np.exp(-values - negative_peak)
                total += weight * self.wire_smoothing * (
                    positive_peak + np.log(positive_exp.sum()) +
                    negative_peak + np.log(negative_exp.sum()))
                if with_grad:
                    direction = weight * (positive_exp / positive_exp.sum() -
                                          negative_exp / negative_exp.sum())
                    for node, value in zip(nodes, direction):
                        if node < self.movable_count:
                            gradient[node, axis] += value
        return float(total), gradient

    def _timing(self, coordinates, variants, with_grad):
        selected = np.arange(self.node_count), variants.astype(int)
        resistance = self.resistance[selected]
        input_cap = self.input_cap[selected]
        delay = self.intrinsic_delay[selected] + resistance * self.wire_cap_base
        lower_gradient = np.zeros((self.movable_count, 2), dtype=float)
        upper_gradient = np.zeros_like(lower_gradient)
        for arc in self.arcs:
            source, target, wire_cap, input_scale = arc[:4]
            difference = coordinates[source] - coordinates[target]
            smoothed = np.sqrt(difference ** 2 + self.rc_smoothing ** 2)
            receiver_cap = (arc[4][int(variants[target])] if len(arc) == 5 else
                            input_scale * input_cap[target])
            delay[source] += resistance[source] * (wire_cap * smoothed.sum() +
                                                    receiver_cap)
            if with_grad:
                direction = difference / smoothed
                for weights, gradient in ((self.lower_timing_weights, lower_gradient),
                                          (self.upper_timing_weights, upper_gradient)):
                    contribution = weights[source] * resistance[source] * wire_cap * direction
                    if source < self.movable_count:
                        gradient[source] += contribution
                    if target < self.movable_count:
                        gradient[target] -= contribution
        return (float(self.lower_timing_weights @ delay),
                float(self.upper_timing_weights @ delay), lower_gradient, upper_gradient)

    def evaluate(self, positions, labels, with_grad=False):
        coordinates = self._coordinates(positions)
        variants = self._variants(labels)
        wirelength, wire_gradient = self._wirelength(coordinates, with_grad)
        density, density_gradient, charges = self._density(coordinates, variants, with_grad)
        lower_timing, upper_timing, lower_timing_gradient, upper_timing_gradient = self._timing(
            coordinates, variants, with_grad)
        selected = np.arange(self.movable_count), variants[:self.movable_count].astype(int)
        area = float(self.area[selected].sum())
        leakage = float(self.leakage[selected].sum())
        normal = self.normalizers
        lower = self.lower_weights
        upper = self.upper_weights
        displacement = float(np.sum((coordinates[:self.movable_count] - self.reference) ** 2))
        lower_value = (lower_timing / normal["timing"] +
                       lower["area"] * area / normal["area"] +
                       lower["leakage"] * leakage / normal["leakage"] +
                       lower["density"] * density / normal["density"])
        upper_value = (upper["wirelength"] * wirelength / normal["wirelength"] +
                       upper["displacement"] * displacement / normal["wirelength"] ** 2 +
                       upper["density"] * density / normal["density"] +
                       upper["timing"] * upper_timing / normal["timing"] +
                       upper["area"] * area / normal["area"] +
                       upper["leakage"] * leakage / normal["leakage"])
        result = {"lower": float(lower_value), "upper": float(upper_value),
                  "wirelength": wirelength, "density": density,
                  "lower_timing": lower_timing, "upper_timing": upper_timing,
                  "area": area, "leakage": leakage, "charge": charges}
        if with_grad:
            result["lower_gradient"] = (lower_timing_gradient / normal["timing"] +
                                         lower["density"] * density_gradient / normal["density"])
            result["upper_gradient"] = (
                upper["wirelength"] * wire_gradient / normal["wirelength"] +
                2 * upper["displacement"] * (coordinates[:self.movable_count] - self.reference) /
                normal["wirelength"] ** 2 +
                upper["density"] * density_gradient / normal["density"] +
                upper["timing"] * upper_timing_gradient / normal["timing"])
        return result
