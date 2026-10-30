"""Density-free placement refinement with a submodular binary RC model."""

from __future__ import annotations

import numpy as np

from openroad.placement_model import PlacementSizingModel


class RCModel(PlacementSizingModel):
    def __init__(self, data):
        super().__init__(data)
        if self.lower_weights["density"] != 0 or self.upper_weights["density"] != 0:
            raise ValueError("RCModel requires zero density weights in both objectives")
        values = [*self.normalizers.values(), *self.lower_weights.values(),
                  *self.upper_weights.values(), self.wire_smoothing, self.rc_smoothing]
        if not np.all(np.isfinite(values)):
            raise ValueError("All model weights and normalizers must be finite")
        for values in (self.die, self.wire_cap_base, self.lower_timing_weights,
                       self.upper_timing_weights):
            if not np.all(np.isfinite(values)):
                raise ValueError("Physical data and timing weights must be finite")
        radius = float(data.get("placement_radius", min(self.die)))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Placement radius must be positive and finite")
        self.lower_bound = np.maximum(self.lower_bound, self.reference - radius)
        self.upper_bound = np.minimum(self.upper_bound, self.reference + radius)
        self.arc_source = np.asarray([arc[0] for arc in self.arcs], dtype=int)
        self.arc_target = np.asarray([arc[1] for arc in self.arcs], dtype=int)
        self.arc_wire_cap = np.asarray([arc[2] for arc in self.arcs], dtype=float)
        self.arc_input_cap = np.asarray([
            arc[4] if len(arc) == 5 else np.asarray(self.input_cap[arc[1]]) * arc[3]
            for arc in self.arcs], dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(self.arc_wire_cap)) or not np.all(np.isfinite(self.arc_input_cap)):
            raise ValueError("Arc capacitances must be finite")
        self.pair_first, self.pair_second, self.lower_pairs, self.upper_pairs = self._pairs()

    def _pairs(self):
        coefficients = {}
        for arc_index, (source, target) in enumerate(zip(self.arc_source, self.arc_target)):
            if source >= self.movable_count or target >= self.movable_count:
                continue
            key = tuple(sorted((int(source), int(target))))
            base = ((self.resistance[source, 1] - self.resistance[source, 0]) *
                    (self.arc_input_cap[arc_index, 1] - self.arc_input_cap[arc_index, 0]) /
                    self.normalizers["timing"])
            pair = coefficients.setdefault(key, np.zeros(2))
            pair += base * np.asarray([self.lower_timing_weights[source],
                                      self.upper_weights["timing"] * self.upper_timing_weights[source]])
        keys = sorted(coefficients)
        values = np.asarray([coefficients[key] for key in keys]).reshape(-1, 2)
        if np.any(values > 0) or not np.all(np.isfinite(values)):
            raise ValueError("RC pair coefficients violate submodularity")
        return (np.asarray([key[0] for key in keys], dtype=int),
                np.asarray([key[1] for key in keys], dtype=int), values[:, 0], values[:, 1])

    def _loads(self, coordinates):
        differences = coordinates[self.arc_source] - coordinates[self.arc_target]
        distances = np.sqrt(differences ** 2 + self.rc_smoothing ** 2)
        return self.arc_wire_cap * distances.sum(axis=1), differences / distances

    def _timing(self, coordinates, variants, with_grad):
        selected = np.arange(self.node_count), variants
        resistance = self.resistance[selected]
        wire_load, direction = self._loads(coordinates)
        delay = self.intrinsic_delay[selected] + resistance * self.wire_cap_base
        receiver_load = self.arc_input_cap[np.arange(len(self.arcs)), variants[self.arc_target]]
        np.add.at(delay, self.arc_source, resistance[self.arc_source] * (wire_load + receiver_load))
        gradients = []
        for weights in (self.lower_timing_weights, self.upper_timing_weights):
            gradient = np.zeros((self.node_count, 2))
            if with_grad:
                contribution = (weights[self.arc_source] * resistance[self.arc_source] *
                                self.arc_wire_cap)[:, None] * direction
                np.add.at(gradient, self.arc_source, contribution)
                np.add.at(gradient, self.arc_target, -contribution)
            gradients.append(gradient[:self.movable_count])
        return (float(self.lower_timing_weights @ delay), float(self.upper_timing_weights @ delay),
                *gradients)

    def evaluate(self, positions, labels, with_grad=False):
        label_values = np.asarray(labels)
        if np.any((label_values != 0) & (label_values != 1)):
            raise ValueError("Labels must be binary")
        coordinates = self._coordinates(positions)
        variants = self._variants(labels)
        wirelength, wire_gradient = self._wirelength(coordinates, with_grad)
        lower_timing, upper_timing, lower_gradient, upper_gradient = self._timing(
            coordinates, variants, with_grad)
        selected = np.arange(self.movable_count), variants[:self.movable_count]
        area = float(self.area[selected].sum())
        leakage = float(self.leakage[selected].sum())
        normal, lower, upper = self.normalizers, self.lower_weights, self.upper_weights
        displacement = float(np.sum((np.asarray(positions) - self.reference) ** 2))
        result = {
            "lower": float(lower_timing / normal["timing"] + lower["area"] * area / normal["area"] +
                           lower["leakage"] * leakage / normal["leakage"]),
            "upper": float(upper["wirelength"] * wirelength / normal["wirelength"] +
                           upper["displacement"] * displacement / normal["wirelength"] ** 2 +
                           upper["timing"] * upper_timing / normal["timing"] +
                           upper["area"] * area / normal["area"] +
                           upper["leakage"] * leakage / normal["leakage"]),
            "wirelength": wirelength, "lower_timing": lower_timing, "upper_timing": upper_timing,
            "area": area, "leakage": leakage, "displacement": displacement,
        }
        if with_grad:
            result["lower_gradient"] = lower_gradient / normal["timing"]
            result["upper_gradient"] = (
                upper["wirelength"] * wire_gradient / normal["wirelength"] +
                2 * upper["displacement"] * (np.asarray(positions) - self.reference) /
                normal["wirelength"] ** 2 + upper["timing"] * upper_gradient / normal["timing"])
        return result

    def unaries(self, positions):
        coordinates = self._coordinates(positions)
        wire_load, _ = self._loads(coordinates)
        loads = self.wire_cap_base.copy()
        np.add.at(loads, self.arc_source, wire_load + self.arc_input_cap[:, 0])
        resistance_change = self.resistance[:, 1] - self.resistance[:, 0]
        intrinsic_change = self.intrinsic_delay[:, 1] - self.intrinsic_delay[:, 0]
        input_change = self.arc_input_cap[:, 1] - self.arc_input_cap[:, 0]
        unaries = []
        for timing_weights, timing_scale, resource_weights in (
                (self.lower_timing_weights, 1.0, self.lower_weights),
                (self.upper_timing_weights, self.upper_weights["timing"], self.upper_weights)):
            unary = timing_weights * (intrinsic_change + resistance_change * loads)
            np.add.at(unary, self.arc_target,
                      timing_weights[self.arc_source] * self.resistance[self.arc_source, 0] * input_change)
            unary = unary[:self.movable_count] * timing_scale / self.normalizers["timing"]
            unary += (resource_weights["area"] * np.diff(self.area[:self.movable_count], axis=1)[:, 0] /
                      self.normalizers["area"])
            unary += (resource_weights["leakage"] * np.diff(self.leakage[:self.movable_count], axis=1)[:, 0] /
                      self.normalizers["leakage"])
            unaries.append(unary)
        return tuple(unaries)

    def certificate(self):
        return {
            "A1": {"status": "verified_for_declared_model", "domain": "compact nonempty placement box",
                   "extension": "smooth LSE and square-root RC lengths on all real coordinates"},
            "A2": {"status": "verified_for_declared_model", "beta": 0.0,
                   "scope": "all placements, not just sampled queries",
                   "lower_max_positive_pair": float(np.max(self.lower_pairs, initial=0)),
                   "upper_max_positive_pair": float(np.max(self.upper_pairs, initial=0)),
                   "strictly_negative_lower_pairs": int(np.count_nonzero(self.lower_pairs < 0)),
                   "pair_count": len(self.lower_pairs)},
            "A3": {"status": "vacuous_unconstrained_follower", "constraint_count": 0,
                   "multipliers": "empty vectors; exact binary graph cuts replace dual iterations"},
            "A4": {"status": "not_verified", "claim": "no A4-dependent convergence claim"},
        }
