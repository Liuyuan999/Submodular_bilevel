"""Exact binary graph cuts, with a small dependency-free validation backend."""

from __future__ import annotations

from collections import deque

import numpy as np


class _Dinic:
    def __init__(self, count):
        self.adjacency = [[] for _ in range(count)]

    def add_edge(self, start, end, capacity):
        forward = [end, float(capacity), len(self.adjacency[end])]
        backward = [start, 0.0, len(self.adjacency[start])]
        self.adjacency[start].append(forward)
        self.adjacency[end].append(backward)

    def solve(self, source, sink):
        tolerance = 1e-10
        while True:
            level = [-1] * len(self.adjacency)
            level[source] = 0
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor, capacity, _ in self.adjacency[current]:
                    if capacity > tolerance and level[neighbor] < 0:
                        level[neighbor] = level[current] + 1
                        queue.append(neighbor)
            if level[sink] < 0:
                break
            cursor = [0] * len(self.adjacency)

            def augment(current, available):
                if current == sink:
                    return available
                while cursor[current] < len(self.adjacency[current]):
                    edge = self.adjacency[current][cursor[current]]
                    neighbor = edge[0]
                    if edge[1] > tolerance and level[neighbor] == level[current] + 1:
                        pushed = augment(neighbor, min(available, edge[1]))
                        if pushed > tolerance:
                            edge[1] -= pushed
                            self.adjacency[neighbor][edge[2]][1] += pushed
                            return pushed
                    cursor[current] += 1
                return 0.0

            while augment(source, float("inf")) > tolerance:
                pass
        reachable = [False] * len(self.adjacency)
        reachable[source] = True
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor, capacity, _ in self.adjacency[current]:
                if capacity > tolerance and not reachable[neighbor]:
                    reachable[neighbor] = True
                    queue.append(neighbor)
        return reachable


class CutOracle:
    def __init__(self, node_count, edge_first, edge_second, edge_weight,
                 anchor_cell, anchor_macro, backend="auto", reuse_trees=False):
        self.node_count = int(node_count)
        self.edge_first = np.asarray(edge_first, dtype=np.int64)
        self.edge_second = np.asarray(edge_second, dtype=np.int64)
        self.edge_weight = np.asarray(edge_weight, dtype=np.float64)
        self.anchor_cell = np.asarray(anchor_cell, dtype=np.int64)
        self.anchor_macro = np.asarray(anchor_macro, dtype=np.int64)
        if not (len(self.edge_first) == len(self.edge_second) == len(self.edge_weight)):
            raise ValueError("Pair-edge arrays must have equal length")
        if len(self.anchor_cell) != len(self.anchor_macro):
            raise ValueError("Anchor arrays must have equal length")
        if np.any(self.edge_weight < 0) or not np.all(np.isfinite(self.edge_weight)):
            raise ValueError("Pair weights must be finite and nonnegative")
        if any(np.any(indices < 0) or np.any(indices >= node_count)
               for indices in (self.edge_first, self.edge_second, self.anchor_cell, self.anchor_macro)):
            raise ValueError("Cut graph contains an invalid node index")
        if backend == "auto":
            try:
                import maxflow
                backend = "pymaxflow"
            except ImportError:
                backend = "dinic"
        if backend not in {"pymaxflow", "dinic"}:
            raise ValueError(f"Unknown cut backend: {backend}")
        if backend == "dinic" and node_count > 128:
            raise RuntimeError("The validation cut backend is limited to 128 nodes; install PyMaxflow")
        self.backend = backend
        self.reuse_trees = bool(reuse_trees and backend == "pymaxflow")
        self.calls = 0
        self._cached_graph = None
        self._cached_nodes = None
        self._cached_zero = None
        self._cached_one = None
        self._cached_scale = None

    def solve(self, unary_zero, unary_one, pair_scale=1.0, hard_constraints=False):
        unary_zero = np.asarray(unary_zero, dtype=np.float64)
        unary_one = np.asarray(unary_one, dtype=np.float64)
        if unary_zero.shape != (self.node_count,) or unary_one.shape != (self.node_count,):
            raise ValueError("Unary vectors have the wrong length")
        if not np.all(np.isfinite(unary_zero)) or not np.all(np.isfinite(unary_one)):
            raise ValueError("Unary vectors must be finite")
        if pair_scale < 0 or not np.isfinite(pair_scale):
            raise ValueError("Pair scale must be finite and nonnegative")
        shifted = np.minimum(unary_zero, unary_one)
        cost_zero = np.maximum(unary_zero - shifted, 0.0)
        cost_one = np.maximum(unary_one - shifted, 0.0)
        pair_weights = pair_scale * self.edge_weight
        barrier = 1.0 + float(np.sum(np.abs(unary_one - unary_zero))) + float(np.sum(pair_weights))
        if self.backend == "pymaxflow":
            labels = self._solve_pymaxflow(cost_zero, cost_one, pair_weights,
                                           barrier if hard_constraints else None, pair_scale)
        else:
            labels = self._solve_dinic(cost_zero, cost_one, pair_weights,
                                       barrier if hard_constraints else None)
        if hard_constraints and np.any(labels[self.anchor_cell] > labels[self.anchor_macro]):
            raise ArithmeticError("Graph cut returned an infeasible implication assignment")
        self.calls += 1
        return labels

    def _solve_pymaxflow(self, cost_zero, cost_one, pair_weights, barrier, pair_scale):
        import maxflow

        if (self.reuse_trees and barrier is None and self._cached_graph is not None
                and self._cached_scale == pair_scale):
            difference_one = cost_one - self._cached_one
            difference_zero = cost_zero - self._cached_zero
            if np.any(difference_one) or np.any(difference_zero):
                self._cached_graph.add_grid_tedges(self._cached_nodes, difference_one, difference_zero)
                self._cached_graph.mark_grid_nodes(self._cached_nodes)
            self._cached_graph.maxflow(reuse_trees=True)
            self._cached_zero = cost_zero.copy()
            self._cached_one = cost_one.copy()
            return self._cached_graph.get_grid_segments(self._cached_nodes).astype(np.uint8)

        edge_capacity = len(self.edge_first) + (len(self.anchor_cell) if barrier is not None else 0)
        graph = maxflow.Graph[float](self.node_count, edge_capacity)
        nodes = graph.add_nodes(self.node_count)
        graph.add_grid_tedges(nodes, cost_one, cost_zero)
        if len(self.edge_first):
            graph.add_edges(self.edge_first, self.edge_second, pair_weights, pair_weights)
        if barrier is not None and len(self.anchor_cell):
            capacities = np.full(len(self.anchor_cell), barrier, dtype=np.float64)
            graph.add_edges(self.anchor_macro, self.anchor_cell, capacities,
                            np.zeros(len(self.anchor_cell), dtype=np.float64))
        graph.maxflow()
        if self.reuse_trees and barrier is None:
            self._cached_graph = graph
            self._cached_nodes = nodes
            self._cached_zero = cost_zero.copy()
            self._cached_one = cost_one.copy()
            self._cached_scale = pair_scale
        return graph.get_grid_segments(nodes).astype(np.uint8)

    def _solve_dinic(self, cost_zero, cost_one, pair_weights, barrier):
        source, sink = self.node_count, self.node_count + 1
        graph = _Dinic(self.node_count + 2)
        for node in range(self.node_count):
            graph.add_edge(source, node, cost_one[node])
            graph.add_edge(node, sink, cost_zero[node])
        for first, second, weight in zip(self.edge_first, self.edge_second, pair_weights):
            graph.add_edge(int(first), int(second), weight)
            graph.add_edge(int(second), int(first), weight)
        if barrier is not None:
            for cell, macro in zip(self.anchor_cell, self.anchor_macro):
                graph.add_edge(int(macro), int(cell), barrier)
        reachable = graph.solve(source, sink)
        return np.asarray([not reachable[node] for node in range(self.node_count)], dtype=np.uint8)


class WarmBlockCutOracle:
    """One exact conditional block cut per call, warm-started across calls."""

    def __init__(self, node_count, edge_first, edge_second, edge_weight,
                 block_assignment, backend="auto", reuse_trees=False):
        self.node_count = int(node_count)
        self.edge_first = np.asarray(edge_first, dtype=np.int64)
        self.edge_second = np.asarray(edge_second, dtype=np.int64)
        self.edge_weight = np.asarray(edge_weight, dtype=np.float64)
        self.block_assignment = np.asarray(block_assignment, dtype=np.int64)
        if self.block_assignment.shape != (self.node_count,):
            raise ValueError("Block assignment must cover every node")
        self.block_count = int(self.block_assignment.max(initial=-1)) + 1
        if self.block_count < 2 or self.block_count > self.node_count:
            raise ValueError("Inexact block cutting requires at least two nonempty blocks")
        if np.any(self.block_assignment < 0) or not np.array_equal(
                np.unique(self.block_assignment), np.arange(self.block_count)):
            raise ValueError("Block identifiers must be consecutive and nonempty")
        self.labels = np.zeros(self.node_count, dtype=np.uint8)
        self.blocks = []
        self.cursor = 0
        self.calls = 0
        for block_index in range(self.block_count):
            active_nodes = np.flatnonzero(self.block_assignment == block_index)
            local_index = np.full(self.node_count, -1, dtype=np.int64)
            local_index[active_nodes] = np.arange(len(active_nodes))
            first_active = self.block_assignment[self.edge_first] == block_index
            second_active = self.block_assignment[self.edge_second] == block_index
            internal = first_active & second_active
            first_boundary = first_active & ~second_active
            second_boundary = second_active & ~first_active
            boundary_local = np.concatenate((local_index[self.edge_first[first_boundary]],
                                             local_index[self.edge_second[second_boundary]]))
            boundary_other = np.concatenate((self.edge_second[first_boundary],
                                             self.edge_first[second_boundary]))
            boundary_weight = np.concatenate((self.edge_weight[first_boundary],
                                              self.edge_weight[second_boundary]))
            oracle = CutOracle(len(active_nodes), local_index[self.edge_first[internal]],
                               local_index[self.edge_second[internal]],
                               self.edge_weight[internal], [], [], backend=backend,
                               reuse_trees=reuse_trees)
            self.blocks.append((active_nodes, boundary_local, boundary_other,
                                boundary_weight, oracle))

    def solve(self, unary_zero, unary_one, pair_scale=1.0, hard_constraints=False):
        if hard_constraints:
            raise ValueError("The block oracle only solves Lagrangian subproblems")
        unary_zero = np.asarray(unary_zero, dtype=np.float64)
        unary_one = np.asarray(unary_one, dtype=np.float64)
        if unary_zero.shape != (self.node_count,) or unary_one.shape != (self.node_count,):
            raise ValueError("Unary vectors have the wrong length")
        if pair_scale < 0 or not np.isfinite(pair_scale):
            raise ValueError("Pair scale must be finite and nonnegative")
        active_nodes, boundary_local, boundary_other, boundary_weight, oracle = (
            self.blocks[self.cursor])
        cost_zero = unary_zero[active_nodes].copy()
        cost_one = unary_one[active_nodes].copy()
        scaled_weight = pair_scale * boundary_weight
        other_labels = self.labels[boundary_other]
        np.add.at(cost_zero, boundary_local, scaled_weight * other_labels)
        np.add.at(cost_one, boundary_local, scaled_weight * (1 - other_labels))
        self.labels[active_nodes] = oracle.solve(cost_zero, cost_one, pair_scale=pair_scale)
        self.cursor = (self.cursor + 1) % self.block_count
        self.calls += 1
        return self.labels.copy()
