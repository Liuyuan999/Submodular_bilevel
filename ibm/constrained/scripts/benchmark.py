"""Constrained experiment with persistent inner states and deferred auditing.

The shared constrained model supplies graph cuts and baseline runners.
The optional affine F-chain multiplier stepsize is

    theta_lambda(gamma) = theta_F * (1 + gamma).

The scaling follows the manuscript multiplier bound
R = (Delta_f + gamma Delta_g + ...)/sigma, while the g-chain keeps its
independent theta_g.  The side constraint used by this experiment is still a
cut budget and therefore is not claimed to satisfy exact binary dual support.
"""
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


_requested_report_response = os.environ.get(
    'V3_REPORT_RESPONSE', os.environ.get('REPORT_RESPONSE', 'g')
)
if _requested_report_response == 'conditional':
    os.environ['REPORT_RESPONSE'] = 'F'
BASE_PATH = Path(__file__).resolve().parent / 'constrained_core.py'
_spec = importlib.util.spec_from_file_location('constrained_exp4_base', BASE_PATH)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
os.environ['REPORT_RESPONSE'] = _requested_report_response

M = B.M
E = os.environ.get
_base_phi_hat = B.phi_hat
cut2 = B.cut2
F_THETA_RULE = E('F_THETA_RULE', 'affine_gamma')
assert F_THETA_RULE in ('constant', 'affine_gamma', 'linear_gamma')
V3_GAMMA_CLOCK = E('V3_GAMMA_CLOCK', 'iteration')
V3_GAMMA_STEPS = int(E('V3_GAMMA_STEPS', '500'))
V3_EVAL_EVERY = int(E('V3_EVAL_EVERY', '1'))
V3_WARMUP_SECONDS = float(E('V3_WARMUP_SECONDS', '0'))
V3_K_START = int(E('V3_K_START', str(B.K)))
V3_K_END = int(E('V3_K_END', str(V3_K_START)))
V3_K_SCHEDULE = E('V3_K_SCHEDULE', 'linear')
V3_REPORT_RESPONSE = _requested_report_response
V3_PHI_TOL = float(E('V3_PHI_TOL', '1e-5'))
V3_CONDITIONAL_RULE = E('V3_CONDITIONAL_RULE', 'lower_g')
V3_WARMSTART_DUAL = E('V3_WARMSTART_DUAL', '1') == '1'
V3_WARMSTART_RESPONSE = E('V3_WARMSTART_RESPONSE', '1') == '1'
V3_RECORD_BOTH_PHI = E('V3_RECORD_BOTH_PHI', '0') == '1'
V3_CERTIFIED_AUDIT = E('V3_CERTIFIED_AUDIT', '0') == '1'
V3_EXCLUDE_AUDIT_TIME = E('V3_EXCLUDE_AUDIT_TIME', '0') == '1'
V3_DEFER_AUDIT = E('V3_DEFER_AUDIT', '0') == '1'
V3_RECORD_INTERVAL_SECONDS = float(E('V3_RECORD_INTERVAL_SECONDS', '1'))
assert V3_GAMMA_CLOCK in ('iteration', 'wall') and V3_GAMMA_STEPS >= 1 and V3_EVAL_EVERY >= 1
assert 0 <= V3_WARMUP_SECONDS < B.TB
assert V3_K_START >= 1 and V3_K_END >= 1 and V3_K_SCHEDULE in ('linear', 'geometric')
assert V3_REPORT_RESPONSE in ('g', 'F', 'conditional') and V3_PHI_TOL >= 0
assert V3_CONDITIONAL_RULE in ('lower_g', 'legacy_tolerance')
assert V3_RECORD_INTERVAL_SECONDS > 0
assert not V3_DEFER_AUDIT or V3_CERTIFIED_AUDIT
START_FILE = E('START_FILE', '').strip()
START_FIELD = E('START_FIELD', 'x_best')
if START_FILE:
    warm_data = json.load(open(START_FILE))
    warm_groups = warm_data['runs']
    warm_records = []
    for group in warm_groups:
        matches = [record for record in group['runs'] if record['method'] == 'bigap_pd']
        if len(matches) != 1:
            raise ValueError('START_FILE must contain exactly one bigap_pd record per start')
        warm_records.append(matches[0])
    B.STARTS = np.asarray([record[START_FIELD] for record in warm_records], dtype=float)
    B.INIT_SEEDS = [group.get('init_seed') for group in warm_groups]
    B.NSTART = len(B.STARTS)


def theta_F_at(gamma):
    if F_THETA_RULE == 'constant':
        return B.THETA_F
    if F_THETA_RULE == 'linear_gamma':
        return B.THETA_F * gamma
    return B.THETA_F * (1.0 + gamma)


def dualgrad_v3(x, which, K_inner=None, theta=None, state=None, gamma=None):
    """Run projected dual ascent with independent, gamma-aware steps."""
    K_inner = B.K if K_inner is None else K_inner
    state = B.DualState() if state is None else state
    gamma = B.GAMMA if gamma is None else gamma
    if theta is None:
        theta = B.THETA_G if which == 'g' else theta_F_at(gamma)
    c0, c1, dc0, dc1 = M.unary(x)
    if which == 'g':
        u0, u1, pairwise, scale = c0, c1, M.w, 1.0
        objective = lambda y: float(M.g_val(c0, c1, y))
    else:
        u0 = c0 + M.yref / gamma
        u1 = c1 + (1 - M.yref) / gamma
        pairwise, scale = M.w - M.a / gamma, gamma
        objective = lambda y: float(M.f_val(y) + gamma * M.g_val(c0, c1, y))

    multiplier = getattr(state, 'multiplier', 0.0) if V3_WARMSTART_DUAL else 0.0
    initial_multiplier = multiplier
    dual = -np.inf
    best_response = state.feasible_response if V3_WARMSTART_RESPONSE else None
    primal = objective(best_response) if best_response is not None else np.inf
    gradient_sum = np.zeros(M.P)
    multiplier_sum = 0.0
    feasible_count = 0
    for _ in range(K_inner):
        response = state.cut.solve(u0, u1, pairwise, multiplier / scale)
        state.last_response = response.copy()
        constraint = B.cval(response)
        value = objective(response)
        dual = max(dual, value + multiplier * constraint)
        if constraint <= 0:
            feasible_count += 1
            if value < primal:
                primal, best_response = value, response
        gradient_sum += M.gy(dc0, dc1, response)
        multiplier_sum += multiplier
        multiplier = max(0.0, multiplier + theta * constraint)

    if V3_WARMSTART_DUAL:
        state.multiplier = multiplier
    if V3_WARMSTART_RESPONSE and best_response is not None:
        state.feasible_response = best_response.copy()
    multiplier_average = multiplier_sum / K_inner
    return gradient_sum / K_inner, multiplier_average, {
        'dual': dual,
        'primal': primal,
        'gap': (primal - dual) if best_response is not None else None,
        'feasible_frac': feasible_count / K_inner,
        'y_best': best_response,
        'nu_bar': multiplier_average,
        'nu_initial': initial_multiplier,
        'nu_next': multiplier,
        'theta_used': theta,
    }


B.dualgrad = dualgrad_v3


def phi_hat_v3(x, state_g=None, state_F=None, gamma=None):
    if V3_REPORT_RESPONSE != 'conditional' and not (
        V3_CERTIFIED_AUDIT and V3_REPORT_RESPONSE == 'g'
    ):
        return _base_phi_hat(x, state_g, state_F, gamma)
    state_g = B.DualState() if state_g is None else state_g
    state_F = B.DualState() if state_F is None else state_F
    gamma = B.GAMMA if gamma is None else gamma
    _, _, info_F = dualgrad_v3(x, 'F', K_inner=B.K_EVAL, state=state_F, gamma=gamma)
    _, _, info_g = dualgrad_v3(x, 'g', K_inner=B.K_EVAL, state=state_g)
    y_F, y_g = info_F['y_best'], info_g['y_best']
    if y_F is None and y_g is None:
        return float('nan'), {
            'y_best': None, 'y_F': None, 'y_g': None, 'selected_response': None,
            'gap': None, 'll_cert_gap': None, 'll_recovered_gap': None,
            'support_gap_F': info_F['gap'], 'support_gap_g': info_g['gap'],
            'feasible_frac': 0.0, 'gamma': gamma,
        }
    c0, c1, _, _ = M.unary(x)
    g_F = np.inf if y_F is None else float(M.g_val(c0, c1, y_F))
    g_g = np.inf if y_g is None else float(M.g_val(c0, c1, y_g))
    if V3_CERTIFIED_AUDIT and V3_REPORT_RESPONSE == 'g':
        f_F = np.nan if y_F is None else float(M.f_val(y_F))
        f_g = np.nan if y_g is None else float(M.f_val(y_g))
        ll_gap_F = np.nan if y_F is None else max(0.0, g_F - info_g['dual'])
        ll_gap_g = np.nan if y_g is None else max(0.0, g_g - info_g['dual'])
        constraint_F = np.nan if y_F is None else float(B.cval(y_F))
        constraint_g = np.nan if y_g is None else float(B.cval(y_g))
        violation_F = np.nan if y_F is None else max(0.0, constraint_F)
        violation_g = np.nan if y_g is None else max(0.0, constraint_g)
        score_F = np.nan if y_F is None else f_F + ll_gap_F + violation_F
        score_g = np.nan if y_g is None else f_g + ll_gap_g + violation_g
        return score_g, {
            'y_best': y_g, 'y_F': y_F, 'y_g': y_g,
            'selected_response': 'y_g', 'gap': ll_gap_g,
            'll_cert_gap': ll_gap_g, 'll_recovered_gap': (
                None if y_F is None or y_g is None else g_F - g_g
            ),
            'support_gap_F': info_F['gap'], 'support_gap_g': info_g['gap'],
            'feasible_frac': info_g['feasible_frac'],
            'g_F': g_F, 'g_g': g_g, 'phi_F': f_F, 'phi_g': f_g,
            'll_gap_F': ll_gap_F, 'll_gap_g': ll_gap_g,
            'constraint_F': constraint_F, 'constraint_g': constraint_g,
            'constraint_violation_F': violation_F,
            'constraint_violation_g': violation_g,
            'score_F': score_F, 'score_g': score_g, 'gamma': gamma,
        }
    if V3_CONDITIONAL_RULE == 'lower_g':
        use_F = y_F is not None and (y_g is None or g_F <= g_g)
        selected_response = 'y_F' if use_F else 'y_g'
        y_best = y_F if use_F else y_g
        g_best = g_F if use_F else g_g
    else:
        use_g = y_g is not None and (y_F is None or g_g < g_F + V3_PHI_TOL)
        selected_response = 'y_g' if use_g else 'y_F'
        y_best = y_g if use_g else y_F
        g_best = g_g if use_g else g_F
    ll_cert_gap = g_best - info_g['dual']
    ll_recovered_gap = None if y_F is None or y_g is None else g_F - g_g
    return float(M.f_val(y_best)), {
        'y_best': y_best, 'y_F': y_F, 'y_g': y_g,
        'selected_response': selected_response, 'gap': ll_cert_gap,
        'll_cert_gap': ll_cert_gap, 'll_recovered_gap': ll_recovered_gap,
        'support_gap_F': info_F['gap'], 'support_gap_g': info_g['gap'],
        'feasible_frac': (info_F['feasible_frac'] if selected_response == 'y_F'
                          else info_g['feasible_frac']),
        'g_F': g_F, 'g_g': g_g,
        'phi_g': None if y_g is None else float(M.f_val(y_g)),
        'phi_F': None if y_F is None else float(M.f_val(y_F)),
        'conditional_rule': V3_CONDITIONAL_RULE,
        'phi_tolerance': V3_PHI_TOL if V3_CONDITIONAL_RULE == 'legacy_tolerance' else None,
        'gamma': gamma,
    }


phi_hat = phi_hat_v3
B.phi_hat = phi_hat_v3


def gamma_v3(iteration, elapsed):
    if B.GAMMA_SCHEDULE == 'constant':
        return B.GAMMA
    if V3_GAMMA_CLOCK == 'wall':
        if elapsed <= V3_WARMUP_SECONDS:
            return B.GAMMA_START
        duration = max(B.TB - V3_WARMUP_SECONDS, 1e-12)
        progress = min(max((elapsed - V3_WARMUP_SECONDS) / duration, 0.0), 1.0)
        if B.GAMMA_SCHEDULE == 'geometric':
            return B.GAMMA_START * (B.GAMMA_END / B.GAMMA_START) ** progress
        return B.GAMMA_START + progress * (B.GAMMA_END - B.GAMMA_START)
    progress = min(max(iteration / V3_GAMMA_STEPS, 0.0), 1.0)
    if B.GAMMA_SCHEDULE == 'geometric':
        return B.GAMMA_START * (B.GAMMA_END / B.GAMMA_START) ** progress
    return B.GAMMA_START + progress * (B.GAMMA_END - B.GAMMA_START)


if V3_GAMMA_CLOCK == 'wall':
    B.gamma_at = lambda elapsed: gamma_v3(0, elapsed)


def k_v3(elapsed):
    if V3_K_START == V3_K_END or elapsed <= V3_WARMUP_SECONDS:
        return V3_K_START
    duration = max(B.TB - V3_WARMUP_SECONDS, 1e-12)
    progress = min(max((elapsed - V3_WARMUP_SECONDS) / duration, 0.0), 1.0)
    if V3_K_SCHEDULE == 'geometric':
        value = V3_K_START * (V3_K_END / V3_K_START) ** progress
    else:
        value = V3_K_START + progress * (V3_K_END - V3_K_START)
    return max(1, int(np.ceil(value)))


class V3Clock(B.Clock):
    def __init__(self, x0):
        self.audit_seconds = 0.0
        self.snapshots = []
        self.next_snapshot_second = 0.0
        self.current_gamma = gamma_v3(0, 0.0)
        self.current_K = V3_K_START
        self.best_info = None
        self.last_info = None
        self.best_F = (np.inf, x0.copy())
        self.best_F_info = None
        if V3_DEFER_AUDIT:
            M.NCUT[0] = 0
            self.t0 = time.perf_counter()
            self.best = (np.inf, x0.copy())
            self.last = x0.copy()
            self.eval_state_g, self.eval_state_F = B.DualState(), B.DualState()
            self.best_gamma = self.last_gamma = self.current_gamma
            self.hist, self.traj, self.diag, self.stop_reason = [], [], [], 'iteration_cap'
            self.record_snapshot(x0, force=True)
        else:
            super().__init__(x0)

    def evaluate(self, x):
        started = time.perf_counter()
        try:
            return B.phi_hat(x, self.eval_state_g, self.eval_state_F, self.current_gamma)
        finally:
            if V3_EXCLUDE_AUDIT_TIME:
                self.audit_seconds += time.perf_counter() - started

    def elapsed(self):
        return max(0.0, super().elapsed() - self.audit_seconds)

    def record_snapshot(self, x, force=False):
        seconds = min(self.elapsed(), B.TB)
        if not force and seconds + 1e-12 < self.next_snapshot_second:
            return True
        snapshot = {
            'calls': int(M.NCUT[0]), 'seconds': float(seconds),
            'x': np.asarray(x, dtype=float).tolist(),
            'gamma': float(self.current_gamma), 'K_inner': int(self.current_K),
        }
        if self.snapshots and force and np.array_equal(
                np.asarray(snapshot['x']), np.asarray(self.snapshots[-1]['x'])):
            self.snapshots[-1] = snapshot
        else:
            self.snapshots.append(snapshot)
        self.last = np.asarray(x, dtype=float).copy()
        self.last_gamma = self.current_gamma
        self.next_snapshot_second = (
            np.floor(seconds / V3_RECORD_INTERVAL_SECONDS) + 1
        ) * V3_RECORD_INTERVAL_SECONDS
        return True

    def observe(self, x, value, info):
        previous_best = self.best[0]
        observed = super().observe(x, value, info)
        if not observed:
            return observed
        self.last_info = info
        if np.isfinite(value) and value < previous_best:
            self.best_info = info
        if V3_CERTIFIED_AUDIT:
            score_F = info.get('score_F')
            if score_F is not None and np.isfinite(score_F) and score_F < self.best_F[0]:
                self.best_F = (float(score_F), x.copy())
                self.best_F_info = info
            self.hist[-1].extend([
                info.get('phi_g'), info.get('phi_F'), info.get('ll_gap_F'),
                info.get('constraint_g'),
                info.get('constraint_F'), info.get('constraint_violation_g'),
                info.get('constraint_violation_F'), info.get('score_F'),
                info.get('support_gap_F'), info.get('support_gap_g'),
                self.current_gamma, self.current_K,
            ])
        elif V3_RECORD_BOTH_PHI:
            self.hist[-1].append(info.get('phi_g'))
        return observed


def bigap_pd_v3(x0, random_state):
    clock = V3Clock(x0)
    x = x0.copy()
    state_F, state_g = B.DualState(), B.DualState()
    for iteration in range(10 ** 6):
        if not clock.active():
            break
        direction = random_state.normal(size=M.P)
        direction /= np.linalg.norm(direction)
        direction *= random_state.random() ** (1.0 / M.P)
        query = x + B.RHO * direction
        gamma = gamma_v3(iteration, clock.elapsed())
        k_inner = k_v3(clock.elapsed())
        clock.current_gamma = gamma
        clock.current_K = k_inner
        gradient_F, lambda_average, info_F = dualgrad_v3(
            query, 'F', K_inner=k_inner, state=state_F, gamma=gamma
        )
        gradient_g, mu_average, info_g = dualgrad_v3(
            query, 'g', K_inner=k_inner, state=state_g
        )
        gradient = gamma * (gradient_F - gradient_g)
        if B.UPDATE == 'fixed':
            step = (B.ETA / gamma if B.ETA_SCALING == 'inverse_gamma'
                    else min(B.ETA, B.ETA / gamma) if B.ETA_SCALING == 'capped_inverse_gamma'
                    else B.ETA)
        else:
            step = M.steplen(iteration) / max(np.linalg.norm(gradient), 1e-12)
        x_new = M.proj(x - step * gradient)
        diagnostic = [
            iteration, info_F['gap'], info_g['gap'], info_F['feasible_frac'],
            info_g['feasible_frac'], lambda_average, mu_average,
            float(np.linalg.norm(gradient)), float(np.linalg.norm(x_new - x)),
        ]
        if B.GAMMA_SCHEDULE != 'constant':
            diagnostic.append(gamma)
        if V3_K_START != V3_K_END:
            diagnostic.append(k_inner)
        if B.ETA_SCALING != 'constant':
            diagnostic.append(step)
        clock.diag.append(diagnostic)
        x = x_new
        if not clock.active():
            break
        if (iteration + 1) % V3_EVAL_EVERY == 0:
            if V3_DEFER_AUDIT:
                clock.record_snapshot(x)
            else:
                value, info = clock.evaluate(x)
                if not clock.observe(x, value, info):
                    break
    if V3_DEFER_AUDIT:
        clock.record_snapshot(x, force=True)
    return clock


B.FUNS['bigap_pd'] = bigap_pd_v3


_base_record = B.record


def _audit_fields(info):
    if info is None:
        return None
    fields = (
        'phi_g', 'phi_F', 'll_gap_g', 'll_gap_F', 'constraint_g',
        'constraint_F', 'constraint_violation_g', 'constraint_violation_F',
        'score_g', 'score_F', 'support_gap_g', 'support_gap_F', 'gamma',
    )
    return {field: info.get(field) for field in fields}


def _run_deferred_audit(clock):
    """Audit recorded candidates after optimization without charging its clock."""
    started = time.perf_counter()
    state_g, state_F = B.DualState(), B.DualState()
    clock.hist, clock.traj = [], []
    first_x = np.asarray(clock.snapshots[0]['x'], dtype=float)
    clock.best = (np.inf, first_x.copy())
    clock.best_F = (np.inf, first_x.copy())
    clock.best_info = clock.best_F_info = clock.last_info = None
    for snapshot in clock.snapshots:
        x = np.asarray(snapshot['x'], dtype=float)
        gamma = snapshot['gamma']
        value, info = phi_hat_v3(x, state_g, state_F, gamma)
        clock.hist.append([
            snapshot['calls'], value, info.get('ll_gap_g'), snapshot['seconds'],
            info.get('phi_g'), info.get('phi_F'), info.get('ll_gap_F'),
            info.get('constraint_g'), info.get('constraint_F'),
            info.get('constraint_violation_g'), info.get('constraint_violation_F'),
            info.get('score_F'), info.get('support_gap_F'), info.get('support_gap_g'),
            gamma, snapshot['K_inner'],
        ])
        clock.traj.append(snapshot['x'])
        clock.last = x.copy()
        clock.last_gamma = gamma
        clock.last_info = info
        if np.isfinite(value) and value < clock.best[0]:
            clock.best = (float(value), x.copy())
            clock.best_gamma = gamma
            clock.best_info = info
        score_F = info.get('score_F')
        if score_F is not None and np.isfinite(score_F) and score_F < clock.best_F[0]:
            clock.best_F = (float(score_F), x.copy())
            clock.best_F_info = info
    clock.audit_seconds += time.perf_counter() - started


def record_v3(name, seed, clock):
    if not V3_CERTIFIED_AUDIT or name != 'bigap_pd':
        return _base_record(name, seed, clock)
    if V3_DEFER_AUDIT:
        _run_deferred_audit(clock)
    best_info, last_info, best_F_info = clock.best_info, clock.last_info, clock.best_F_info
    record = {
        'method': name, 'seed': seed,
        'x': clock.last.tolist(), 'x_best': clock.best[1].tolist(),
        'x_best_F': clock.best_F[1].tolist(),
        'hist': clock.hist, 'traj': clock.traj,
        'best_phi': float(clock.best[0]),
        'best_score_g': float(clock.best[0]),
        'best_score_F': float(clock.best_F[0]),
        'best': (B.stats(best_info['y_g'])
                 if best_info is not None and best_info.get('y_g') is not None else None),
        'best_F': (B.stats(best_F_info['y_F'])
                   if best_F_info is not None and best_F_info.get('y_F') is not None else None),
        'last': (B.stats(last_info['y_g'])
                 if last_info is not None and last_info.get('y_g') is not None else None),
        'last_F': (B.stats(last_info['y_F'])
                   if last_info is not None and last_info.get('y_F') is not None else None),
        'best_gap': None if best_info is None else best_info.get('ll_gap_g'),
        'last_gap': None if last_info is None else last_info.get('ll_gap_g'),
        'best_gamma': clock.best_gamma, 'last_gamma': clock.last_gamma,
        'best_metrics': _audit_fields(best_info),
        'best_F_metrics': _audit_fields(best_F_info),
        'last_metrics': _audit_fields(last_info),
        'elapsed_seconds': clock.elapsed(),
        'audit_seconds': clock.audit_seconds,
        'total_wall_seconds': clock.elapsed() + clock.audit_seconds,
        'stop_reason': clock.stop_reason,
    }
    if clock.diag:
        record['diag'] = clock.diag
    return record


B.record = record_v3


def run_task_v3(task):
    return B.run_task(task)


def main():
    print(f"constraint: cut' on nets with degree >= {B.NET_DEG} ({len(B.I2)} edges), "
          f"C = {B.C:g}; cut'(y_ref) = {B.cut2(M.yref):g}")
    config = {
        'C': B.C, 'net_deg': B.NET_DEG,
        'K': B.K if V3_K_START == V3_K_END else f'{V3_K_START}-to-{V3_K_END}',
        'K_start': V3_K_START, 'K_end': V3_K_END, 'K_schedule': V3_K_SCHEDULE,
        'K_eval': B.K_EVAL,
        'theta': B.THETA, 'theta_F': B.THETA_F, 'theta_g': B.THETA_G,
        'theta_F_rule': F_THETA_RULE,
        'warmstart_dual': V3_WARMSTART_DUAL,
        'warmstart_response': V3_WARMSTART_RESPONSE,
        'warmstart_graph': True,
        'record_both_phi': V3_RECORD_BOTH_PHI,
        'certified_audit': V3_CERTIFIED_AUDIT,
        'audit_time_excluded': V3_EXCLUDE_AUDIT_TIME,
        'audit_mode': 'deferred' if V3_DEFER_AUDIT else 'inline',
        'record_interval_seconds': V3_RECORD_INTERVAL_SECONDS if V3_DEFER_AUDIT else None,
        'selection_metric': ('f_plus_ll_gap_plus_constraint_violation'
                             if V3_CERTIFIED_AUDIT else None),
        'gamma_clock': V3_GAMMA_CLOCK, 'gamma_steps': V3_GAMMA_STEPS,
        'eval_every': V3_EVAL_EVERY, 'warmup_seconds': V3_WARMUP_SECONDS,
        'stage_boundary_seconds': V3_WARMUP_SECONDS if V3_WARMUP_SECONDS > 0 else None,
        'start_file': Path(START_FILE).name if START_FILE else None, 'start_field': START_FIELD if START_FILE else None,
        'gamma': B.GAMMA if B.GAMMA_SCHEDULE == 'constant' else B.GAMMA_START,
        'gamma_schedule': B.GAMMA_SCHEDULE, 'gamma_start': B.GAMMA_START,
        'gamma_end': B.GAMMA_END, 'report_response': V3_REPORT_RESPONSE,
        'report_metric': ('f_plus_ll_gap_plus_constraint_violation_at_y_g'
                          if V3_CERTIFIED_AUDIT
                          else 'conditional_f_at_lower_g_response'
                          if V3_REPORT_RESPONSE == 'conditional'
                          else 'f_at_F_chain' if V3_REPORT_RESPONSE == 'F'
                          else 'legacy_f_at_g_chain'),
        'conditional_rule': V3_CONDITIONAL_RULE if V3_REPORT_RESPONSE == 'conditional' else None,
        'phi_tolerance': (V3_PHI_TOL if V3_REPORT_RESPONSE == 'conditional'
                          and V3_CONDITIONAL_RULE == 'legacy_tolerance' else None),
        'eta': B.ETA, 'eta_scaling': B.ETA_SCALING, 'rho': B.RHO,
        'update': B.UPDATE, 'time_budget': B.TB, 'mu': M.MU, 'beta': M.beta,
        'methods': B.METHODS, 'nstart': len(B.STARTS), 'start_offset': B.START_OFFSET,
        'start_seeds': B.INIT_SEEDS, 'nseed': B.NSEED, 'seed': B.SEED,
        'nworkers': B.NWORKERS,
    }
    diagnostic_columns = [
        't', 'gap_F', 'gap_g', 'feasible_frac_F', 'feasible_frac_g',
        'lambda_bar', 'mu_bar', 'G_norm', 'dx_norm',
    ]
    if B.GAMMA_SCHEDULE != 'constant':
        diagnostic_columns.append('gamma')
    if V3_K_START != V3_K_END:
        diagnostic_columns.append('K_inner')
    if B.ETA_SCALING != 'constant':
        diagnostic_columns.append('eta_t')
    out = {
        'schema': 'constrained-v3-gamma-aware-dual',
        'config': config,
        'hist_columns': (
            [
                'calls', 'score_y_g', 'll_gap_y_g', 'seconds',
                'f_y_g', 'f_y_F', 'll_gap_y_F',
                'constraint_y_g', 'constraint_y_F',
                'constraint_violation_y_g', 'constraint_violation_y_F',
                'score_y_F', 'support_gap_F', 'support_gap_g',
                'gamma', 'K_inner',
            ] if V3_CERTIFIED_AUDIT else [
                'calls', ('conditional_phi' if V3_REPORT_RESPONSE == 'conditional'
                          else 'f_at_F_chain' if V3_REPORT_RESPONSE == 'F' else 'phi_hat'),
                'll_cert_gap' if V3_REPORT_RESPONSE in ('F', 'conditional') else 'gap_g',
                'seconds',
            ] + (['phi_y_g'] if V3_RECORD_BOTH_PHI else [])
        ),
        'diag_columns': diagnostic_columns,
        'runs': [
            {'x0': start.tolist(), 'init_seed': init_seed, 'runs': []}
            for start, init_seed in zip(
                B.STARTS,
                B.INIT_SEEDS if B.INIT_SEEDS is not None else [None] * len(B.STARTS),
            )
        ],
    }
    if B.RESUME and Path(B.OUT).exists():
        previous = json.load(open(B.OUT))
        if previous.get('schema') != out['schema'] or previous.get('config') != config:
            raise ValueError(f'Cannot resume {B.OUT}: v3 configuration differs')
        out = previous

    completed = {
        (start_index, record['method'], record['seed'])
        for start_index, group in enumerate(out['runs'])
        for record in group['runs']
    }
    init_seeds = B.INIT_SEEDS if B.INIT_SEEDS is not None else [None] * len(B.STARTS)
    tasks = [
        (start_index, start.tolist(), init_seeds[start_index], method, seed)
        for start_index, start in enumerate(B.STARTS)
        for method in B.METHODS
        for seed in range(B.NSEED if method in B.RANDOMIZED else 1)
        if (start_index, method, seed) not in completed
    ]
    timer = time.perf_counter()
    M.NCUT[0] = 0
    B.phi_hat(B.STARTS[0], gamma=B.GAMMA_START)
    evaluation_flows = B.K_EVAL * (2 if V3_CERTIFIED_AUDIT or V3_REPORT_RESPONSE in ('F', 'conditional') else 1)
    query_cost = (f'{2 * V3_K_START}->{2 * V3_K_END}'
                  if V3_K_START != V3_K_END else str(2 * V3_K_START))
    print(f'one BiGAP query costs {query_cost} max-flows; one reported score {evaluation_flows}; '
          f'~{(time.perf_counter() - timer) / evaluation_flows * 1000:.1f} ms per max-flow')
    gamma_label = (f'{B.GAMMA:g}' if B.GAMMA_SCHEDULE == 'constant'
                   else f'{B.GAMMA_SCHEDULE} {B.GAMMA_START:g}->{B.GAMMA_END:g}')
    print(f'{len(tasks)} pending runs; {len(completed)} completed; '
          f'{min(B.NWORKERS, max(1, len(tasks)))} workers; gamma {gamma_label}; '
          f'F-theta {F_THETA_RULE}', flush=True)
    if B.NWORKERS == 1 or not tasks:
        iterator, pool = map(run_task_v3, tasks), None
    else:
        pool = mp.get_context('fork').Pool(min(B.NWORKERS, len(tasks)))
        iterator = pool.imap_unordered(run_task_v3, tasks)
    try:
        for done, (start_index, record) in enumerate(iterator, 1):
            out['runs'][start_index]['runs'].append(record)
            best = record['best'] or {}
            gap = record['best_gap']
            print(f"[{done}/{len(tasks)}] start {B.START_OFFSET + start_index:3d} "
                  f"{record['method']:11s} seed {record['seed']}  best {record['best_phi']:7.0f}  "
                  f"last {record['hist'][-1][1]:7.0f}  cut' {best.get('cut2', float('nan')):5.0f}  "
                  f"gap {gap if gap is not None else float('nan'):6.1f}  "
                  f"iters {len(record['hist']) - 1:4d}  {record['elapsed_seconds']:.1f}s", flush=True)
            if done % B.CHECKPOINT_EVERY == 0:
                B.save_output(out)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        B.save_output(out)
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    for group in out['runs']:
        group['runs'].sort(key=lambda record: (B.METHODS.index(record['method']), record['seed']))
    B.save_output(out)
    print('saved', B.OUT)


if __name__ == '__main__':
    main()
