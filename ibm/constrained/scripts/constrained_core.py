"""ibm01 constrained model, graph-cut oracle, and baseline implementations.

Lower level: min_y g(x,y) subject to c(y) := cut'(y)-C <= 0. Each perturbed outer
query runs independent projected-dual F=f+gamma*g and g chains, resets their multipliers
to zero, averages their K response gradients, and uses
    G = gamma * (mean_k grad_x g(x,y_F^k) - mean_k grad_x g(x,y_g^k)).
The reported legacy phi_hat is f at the best feasible response recovered by the g-chain.

DynamicCut retains exact residual graph/search-tree state whenever capacities permit reuse.
It changes wall-clock cost only: multipliers, oracle directions, reporting, result schema,
and task seeding remain constrained-v2 compatible.

Env: C [1300] NET_DEG [5] KDUAL [10] KEVAL [KDUAL] THETA/THETA_F/THETA_G [5e-4]
     GAMMA [0.01] GAMMA_SCHEDULE [constant|geometric|linear]
     GAMMA_START [GAMMA] GAMMA_END [GAMMA_START] REPORT_RESPONSE [g|F]
     ETA [1e-5] ETA_SCALING [constant|inverse_gamma|capped_inverse_gamma]
     RHO [0.01] UPDATE [fixed|normalized]
     METHODS [bigap_pd,cmaes] TIME_BUDGET NSTART START_OFFSET START_SEEDS NSEED SEED NWORKERS RESUME OUT
     START_SEEDS="0 1 ..." makes each listed seed generate one reproducible x0 shared by all methods.
     EVAL=<results file> scores an unconstrained run's x_best under the constraint.
All other model settings come from unconstrained_benchmark.py (loaded with DRY=1)."""
import os, sys, json, time, importlib.util, multiprocessing as mp, numpy as np, maxflow
from pathlib import Path
os.environ['DRY'] = '1'
spec = importlib.util.spec_from_file_location('unconstrained_benchmark', Path(__file__).resolve().parent.parent.parent / 'unconstrained' / 'scripts' / 'unconstrained_benchmark.py'); M = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(M)
except SystemExit: pass
E = os.environ.get
C = float(E('C', '1300')); NET_DEG = int(E('NET_DEG', '5')); K = int(E('KDUAL', '10')); K_EVAL = int(E('KEVAL', str(K)))
THETA = float(E('THETA', '5e-4')); THETA_F = float(E('THETA_F', str(THETA))); THETA_G = float(E('THETA_G', str(THETA)))
GAMMA = float(E('GAMMA', '0.01')); GAMMA_SCHEDULE = E('GAMMA_SCHEDULE', 'constant')
GAMMA_START = float(E('GAMMA_START', str(GAMMA))); GAMMA_END = float(E('GAMMA_END', str(GAMMA_START)))
ETA = float(E('ETA', '1e-5')); RHO = float(E('RHO', '0.01')); UPDATE = E('UPDATE', 'fixed')
ETA_SCALING = E('ETA_SCALING', 'constant')
REPORT_RESPONSE = E('REPORT_RESPONSE', 'g')
METHODS = E('METHODS', 'bigap_pd,cmaes').split(','); NSEED = int(E('NSEED', '1')); SEED = int(E('SEED', '0'))
NSTART = int(E('NSTART', '20')); START_OFFSET = int(E('START_OFFSET', '0')); NWORKERS = int(E('NWORKERS', '1'))
_start_seed_text = E('START_SEEDS', '').strip()
INIT_SEEDS = [int(value) for value in _start_seed_text.replace(',', ' ').split()] if _start_seed_text else None
CHECKPOINT_EVERY = int(E('CHECKPOINT_EVERY', '4')); RESUME = E('RESUME', '0') == '1'
TB = M.TIME_BUDGET; OUT = E('OUT', str(Path(__file__).parent.parent / 'results' / f'constrained_C{C:g}_K{K}.json'))
assert GAMMA_SCHEDULE in ('constant', 'geometric', 'linear')
assert GAMMA_START > 0 and GAMMA_END > 0
assert min(GAMMA_START, GAMMA_END) >= M.beta, f'gamma schedule [{GAMMA_START}, {GAMMA_END}] falls below beta {M.beta}'
assert K >= 1 and K_EVAL >= 1 and NSTART >= 1 and NWORKERS >= 1
assert UPDATE in ('fixed', 'normalized')
assert ETA_SCALING in ('constant', 'inverse_gamma', 'capped_inverse_gamma')
assert ETA_SCALING == 'constant' or UPDATE == 'fixed'
assert REPORT_RESPONSE in ('g', 'F')

def gamma_at(elapsed):
    if GAMMA_SCHEDULE == 'constant': return GAMMA
    progress = min(max(elapsed / TB, 0.0), 1.0)
    if GAMMA_SCHEDULE == 'geometric': return GAMMA_START * (GAMMA_END / GAMMA_START) ** progress
    return GAMMA_START + progress * (GAMMA_END - GAMMA_START)
if INIT_SEEDS is None:
    _start_rng = np.random.default_rng(SEED)
    STARTS = _start_rng.uniform(M.XLO * 0.8, M.XHI * 0.8, size=(START_OFFSET + NSTART, M.P))[START_OFFSET:]
else:
    if not INIT_SEEDS or min(INIT_SEEDS) < 0 or len(set(INIT_SEEDS)) != len(INIT_SEEDS):
        raise ValueError('START_SEEDS must contain distinct nonnegative integers')
    STARTS = np.array([np.random.default_rng(init_seed).uniform(M.XLO * 0.8, M.XHI * 0.8) for init_seed in INIT_SEEDS])
    NSTART = len(INIT_SEEDS)

# ---------------------------------------------------------------- constraint: second cut function
lines = open(Path(M.DATA) / 'ibm01.hgr').read().split('\n'); m_nets, n = map(int, lines[0].split())
I2, J2 = [], []
for l in lines[1:1 + m_nets]:
    pins = [int(t) - 1 for t in l.split()]
    if len(pins) >= NET_DEG:
        for p_ in range(len(pins)):
            for q_ in range(p_ + 1, len(pins)): I2.append(pins[p_]); J2.append(pins[q_])
I2, J2 = np.array(I2), np.array(J2)
def cut2(y): return float(np.sum(y[I2] != y[J2]))
def cval(y): return cut2(y) - C
EI = np.concatenate([M.ei, I2]); EJ = np.concatenate([M.ej, J2]); NB = len(M.ei)

def graphcut(c0, c1, pw_base, nu):
    """one exact max-flow on the shared graph: base pairwise weights on E, nu on the constraint edges"""
    M.NCUT[0] += 1
    G = maxflow.Graph[float](n, len(EI)); nd = G.add_nodes(n)
    G.add_grid_tedges(nd, c1, c0)
    pw = np.concatenate([pw_base, np.full(len(I2), nu)]); G.add_edges(EI, EJ, pw, pw); G.maxflow()
    return G.get_grid_segments(nd).astype(float)

class DynamicCut:
    def __init__(self):
        self.graph = self.nodes = self.c0 = self.c1 = self.pw_base = None
        self.nu = None

    def build(self, c0, c1, pw_base, nu):
        self.graph = maxflow.Graph[float](n, len(EI)); self.nodes = self.graph.add_nodes(n)
        self.graph.add_grid_tedges(self.nodes, c1, c0)
        pw = np.concatenate([pw_base, np.full(len(I2), nu)])
        self.graph.add_edges(EI, EJ, pw, pw); self.graph.maxflow()
        self.c0 = c0.copy(); self.c1 = c1.copy(); self.pw_base = pw_base.copy(); self.nu = float(nu)

    def solve(self, c0, c1, pw_base, nu):
        M.NCUT[0] += 1
        rebuild = (self.graph is None or not np.array_equal(self.pw_base, pw_base)
                   or abs(nu - self.nu) > 1e-15)
        if rebuild:
            self.build(c0, c1, pw_base, nu)
        else:
            dc0, dc1 = c0 - self.c0, c1 - self.c1
            if np.any(dc0) or np.any(dc1):
                self.graph.add_grid_tedges(self.nodes, dc1, dc0); self.graph.mark_grid_nodes(self.nodes)
            self.graph.maxflow(reuse_trees=True)
            self.c0 = c0.copy(); self.c1 = c1.copy(); self.nu = float(nu)
        return self.graph.get_grid_segments(self.nodes).astype(float)

class DualState:
    def __init__(self):
        self.cut = DynamicCut()
        self.multiplier = 0.0
        self.last_response = None
        self.feasible_response = None

# ---------------------------------------------------------------- v2 projected-dual oracle
def dualgrad(x, which, K_inner=None, theta=None, state=None, gamma=None):
    """Return the v2 averaged grad_x g direction and primal/dual bookkeeping."""
    K_inner = K if K_inner is None else K_inner
    state = DualState() if state is None else state
    gamma = GAMMA if gamma is None else gamma
    theta = (THETA_G if which == 'g' else THETA_F) if theta is None else theta
    c0, c1, dc0, dc1 = M.unary(x)
    if which == 'g':
        u0, u1, pw, sc = c0, c1, M.w, 1.0
        hval = lambda y: float(M.g_val(c0, c1, y))
    else:
        u0, u1, pw, sc = c0 + M.yref / gamma, c1 + (1 - M.yref) / gamma, M.w - M.a / gamma, gamma
        hval = lambda y: float(M.f_val(y) + gamma * M.g_val(c0, c1, y))
    nu = 0.0; dual = -np.inf; primal = np.inf; ybest = None
    grad_sum = np.zeros(M.P); nu_sum = 0.0; nfeas = 0
    for _ in range(K_inner):
        y = state.cut.solve(u0, u1, pw, nu / sc)
        ck = cval(y); hk = hval(y)
        dual = max(dual, hk + nu * ck)
        if ck <= 0:
            nfeas += 1
            if hk < primal: primal, ybest = hk, y
        grad_sum += M.gy(dc0, dc1, y); nu_sum += nu
        nu = max(0.0, nu + theta * ck)
    nu_bar = nu_sum / K_inner
    return grad_sum / K_inner, nu_bar, {'dual': dual, 'primal': primal,
        'gap': (primal - dual) if ybest is not None else None,
        'feasible_frac': nfeas / K_inner, 'y_best': ybest, 'nu_bar': nu_bar, 'nu_next': nu}

def phi_hat(x, state_g=None, state_F=None, gamma=None):
    """Report f at the configured response and, for y_F, certify LL optimality."""
    state_g = DualState() if state_g is None else state_g
    gamma = GAMMA if gamma is None else gamma
    if REPORT_RESPONSE == 'g':
        _, _, info_g = dualgrad(x, 'g', K_inner=K_EVAL, state=state_g)
        y = info_g['y_best']
        return (float('nan') if y is None else float(M.f_val(y))), {
            'y_best': y, 'gap': info_g['gap'], 'feasible_frac': info_g['feasible_frac'], 'gamma': gamma}
    state_F = DualState() if state_F is None else state_F
    _, _, info_F = dualgrad(x, 'F', K_inner=K_EVAL, state=state_F, gamma=gamma)
    _, _, info_g = dualgrad(x, 'g', K_inner=K_EVAL, state=state_g)
    y_F, y_g = info_F['y_best'], info_g['y_best']
    if y_F is None:
        return float('nan'), {'y_best': None, 'y_g': y_g, 'gap': None,
            'll_cert_gap': None, 'll_recovered_gap': None,
            'support_gap_F': info_F['gap'], 'support_gap_g': info_g['gap'],
            'feasible_frac': info_F['feasible_frac'], 'gamma': gamma}
    c0, c1, _, _ = M.unary(x)
    g_F = float(M.g_val(c0, c1, y_F))
    ll_cert_gap = g_F - info_g['dual']
    ll_recovered_gap = None if y_g is None else g_F - float(M.g_val(c0, c1, y_g))
    return float(M.f_val(y_F)), {'y_best': y_F, 'y_g': y_g, 'gap': ll_cert_gap,
        'll_cert_gap': ll_cert_gap, 'll_recovered_gap': ll_recovered_gap,
        'support_gap_F': info_F['gap'], 'support_gap_g': info_g['gap'],
        'feasible_frac': info_F['feasible_frac'], 'gamma': gamma}

# ---------------------------------------------------------------- outer loop (Algorithm 3, M = 1)
class Clock(M.RunClock):
    def __init__(self, x0):
        M.NCUT[0] = 0; self.t0 = time.perf_counter(); self.best = (np.inf, x0.copy()); self.last = x0.copy()
        self.eval_state_g, self.eval_state_F = DualState(), DualState()
        self.best_gamma = self.last_gamma = gamma_at(0.0)
        self.hist, self.traj, self.diag, self.stop_reason = [], [], [], 'iteration_cap'
        fv, info = self.evaluate(x0); self.observe(x0, fv, info)
    def evaluate(self, x):
        gamma = gamma_at(self.elapsed())
        return phi_hat(x, self.eval_state_g, self.eval_state_F, gamma)
    def active(self):
        if self.elapsed() >= TB: self.stop_reason = 'time_budget'; return False
        return True
    def observe(self, x, fv, info):
        secs = self.elapsed()
        if secs > TB: self.stop_reason = 'time_budget'; return False
        self.last = x.copy(); self.last_gamma = info['gamma']
        if fv < self.best[0]: self.best = (float(fv), x.copy()); self.best_gamma = info['gamma']
        self.hist.append([int(M.NCUT[0]), float(fv), info['gap'], secs]); self.traj.append(x.tolist()); return True

def bigap_pd(x0, rs):
    clock = Clock(x0); x = x0.copy(); state_F, state_g = DualState(), DualState()
    for t in range(10 ** 6):
        if not clock.active(): break
        wd = rs.normal(size=M.P); wd /= np.linalg.norm(wd); wd *= rs.random() ** (1.0 / M.P); z = x + RHO * wd
        gamma = gamma_at(clock.elapsed())
        gF, lamb, iF = dualgrad(z, 'F', state=state_F, gamma=gamma)
        gg, mu_, ig = dualgrad(z, 'g', state=state_g)
        G = gamma * (gF - gg)
        if UPDATE == 'fixed':
            step = ETA / gamma if ETA_SCALING == 'inverse_gamma' else min(ETA, ETA / gamma) if ETA_SCALING == 'capped_inverse_gamma' else ETA
        else:
            step = M.steplen(t) / max(np.linalg.norm(G), 1e-12)
        x_new = M.proj(x - step * G)
        diag = [t, iF['gap'], ig['gap'], iF['feasible_frac'], ig['feasible_frac'],
                lamb, mu_, float(np.linalg.norm(G)), float(np.linalg.norm(x_new - x))]
        if GAMMA_SCHEDULE != 'constant': diag.append(gamma)
        if ETA_SCALING != 'constant': diag.append(step)
        clock.diag.append(diag)
        x = x_new
        if not clock.active(): break
        fv, info = clock.evaluate(x)
        if not clock.observe(x, fv, info): break
    return clock

def weighted_pd(x0, rs):
    """Constrained weighted-joint baseline using only the f + gamma*g dual chain."""
    clock = Clock(x0); x = x0.copy(); state = DualState()
    for t in range(10 ** 6):
        if not clock.active(): break
        gamma = gamma_at(clock.elapsed())
        gF, _, _ = dualgrad(x, 'F', state=state, gamma=gamma); G = gamma * gF
        step = M.steplen(t) / max(np.linalg.norm(G), 1e-12)
        x = M.proj(x - step * G)
        if not clock.active(): break
        fv, info = clock.evaluate(x)
        if not clock.observe(x, fv, info): break
    return clock

def zeroth_order(x0, rs):
    """Two-point black-box descent on phi_hat with a normalized update."""
    clock = Clock(x0); x = x0.copy(); smooth = float(E('ZO_RHO', '0.01'))
    for t in range(10 ** 6):
        if not clock.active(): break
        direction = rs.normal(size=M.P)
        fp, _ = clock.evaluate(M.proj(x + smooth * direction))
        if not clock.active(): break
        fm, _ = clock.evaluate(M.proj(x - smooth * direction))
        if not clock.active(): break
        if not (np.isfinite(fp) and np.isfinite(fm)): continue
        G = (fp - fm) * direction / (2 * smooth)
        step = M.steplen(t) / max(np.linalg.norm(G), 1e-12)
        x = M.proj(x - step * G)
        fv, info = clock.evaluate(x)
        if not clock.observe(x, fv, info): break
    return clock

def random_search(x0, rs):
    """Uniform random-search baseline over the same parameter box."""
    clock = Clock(x0)
    while clock.active():
        x = rs.uniform(M.XLO, M.XHI)
        fv, info = clock.evaluate(x)
        if not clock.observe(x, fv, info): break
    return clock

def cmaes(x0, rs):
    if not hasattr(np, 'Inf'): np.Inf = np.inf
    import cma
    def cma_array_compat(value, *args, **kwargs):
        if kwargs.get('copy') is False:
            kwargs.pop('copy')
        return np.array(value, *args, **kwargs)
    cma.evolution_strategy.array = cma_array_compat
    from cma.constraints_handler import BoundNone
    clock = Clock(x0)
    es = cma.CMAEvolutionStrategy(x0.tolist(), M.CMA_SIGMA, {'BoundaryHandler': BoundNone(), 'seed': int(rs.integers(1 << 30)), 'verbose': -9,
                                      'tolflatfitness': 10 ** 9, 'tolfun': 0, 'tolfunhist': 0, 'tolx': 0})
    while clock.active():
        X = es.ask(); F = []
        for raw_x in X:
            if not clock.active(): break
            xx = M.proj(np.asarray(raw_x)); fv, info = clock.evaluate(xx); F.append(fv if np.isfinite(fv) else 1e9)
            if not clock.observe(xx, fv, info): break
        if len(F) == len(X): es.tell(X, F)
    return clock

FUNS = {'bigap_pd': bigap_pd, 'weighted_pd': weighted_pd, 'zo': zeroth_order, 'random': random_search, 'cmaes': cmaes}
RANDOMIZED = {'bigap_pd', 'zo', 'random', 'cmaes'}

def stats(y): return {**M.stats(y), 'cut2': cut2(y), 'c': cval(y)}
def record(name, seed, clock):
    run_elapsed = clock.elapsed(); xf, xb = clock.last, clock.best[1]
    _, ib = phi_hat(xb, gamma=clock.best_gamma); _, il = phi_hat(xf, gamma=clock.last_gamma)
    M.NCUT[0] -= (4 if REPORT_RESPONSE == 'F' else 2) * K_EVAL
    rec = {'method': name, 'seed': seed, 'x': xf.tolist(), 'x_best': xb.tolist(), 'hist': clock.hist, 'traj': clock.traj,
           'best_phi': float(clock.best[0]), 'best': stats(ib['y_best']) if ib['y_best'] is not None else None,
           'last': stats(il['y_best']) if il['y_best'] is not None else None, 'last_gap': il['gap'], 'best_gap': ib['gap'],
           'best_gamma': clock.best_gamma, 'last_gamma': clock.last_gamma,
           'elapsed_seconds': run_elapsed, 'stop_reason': clock.stop_reason}
    if REPORT_RESPONSE == 'F':
        rec.update({'best_ll_cert_gap': ib['ll_cert_gap'], 'last_ll_cert_gap': il['ll_cert_gap'],
                    'best_ll_recovered_gap': ib['ll_recovered_gap'], 'last_ll_recovered_gap': il['ll_recovered_gap'],
                    'best_support_gap_F': ib['support_gap_F'], 'last_support_gap_F': il['support_gap_F'],
                    'best_support_gap_g': ib['support_gap_g'], 'last_support_gap_g': il['support_gap_g']})
    if clock.diag: rec['diag'] = clock.diag
    return rec

def run_task(task):
    start_index, start, init_seed, name, seed = task
    rng_seed = np.random.SeedSequence([init_seed, seed]) if init_seed is not None else 1000 * SEED + 10 * (START_OFFSET + start_index) + seed
    rs = np.random.default_rng(rng_seed)
    return start_index, record(name, seed, FUNS[name](np.asarray(start), rs))

def save_output(out):
    path = Path(OUT); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as handle: json.dump(out, handle)
    os.replace(tmp, path)

# ---------------------------------------------------------------- main
if __name__ == '__main__':
    print(f'constraint: cut\' on nets with degree >= {NET_DEG} ({len(I2)} edges), C = {C:g}; cut\'(y_ref) = {cut2(M.yref):g}')
    if E('EVAL'):   # score an unconstrained run's x_best under the constraint
        d = json.load(open(E('EVAL'))); rows = []
        for r in d['runs']:
            for u in r['runs']:
                if u['method'] != 'bigap': continue
                x = np.array(u['x_best']); y_unc, _ = M.lower(x); fv, info = phi_hat(x)
                rows.append((u['best_phi'], cut2(y_unc), fv, info['gap'], info['feasible_frac']))
        rows = np.array(rows, dtype=float)
        print(f"unconstrained x_best: f = {np.median(rows[:,0]):.0f}, cut' = {np.median(rows[:,1]):.0f} (C = {C:g}; infeasible in {np.mean(rows[:,1] > C):.0%})")
        print(f"same x under the constraint: phi_hat = {np.nanmedian(rows[:,2]):.0f}, gap median {np.nanmedian(rows[:,3]):.1f}, feasible responses {np.mean(rows[:,4]):.2f}")
        sys.exit()
    config = {'C': C, 'net_deg': NET_DEG, 'K': K, 'K_eval': K_EVAL, 'theta': THETA, 'theta_F': THETA_F, 'theta_g': THETA_G,
              'gamma': GAMMA if GAMMA_SCHEDULE == 'constant' else GAMMA_START,
              'gamma_schedule': GAMMA_SCHEDULE, 'gamma_start': GAMMA_START, 'gamma_end': GAMMA_END,
              'report_response': REPORT_RESPONSE,
              'report_metric': 'f_at_F_chain' if REPORT_RESPONSE == 'F' else 'legacy_f_at_g_chain',
              'eta': ETA, 'eta_scaling': ETA_SCALING, 'rho': RHO, 'update': UPDATE, 'time_budget': TB, 'mu': M.MU, 'beta': M.beta,
              'methods': METHODS, 'nstart': len(STARTS), 'start_offset': START_OFFSET, 'start_seeds': INIT_SEEDS,
              'nseed': NSEED, 'seed': SEED, 'nworkers': NWORKERS}
    diag_columns = ['t', 'gap_F', 'gap_g', 'feasible_frac_F', 'feasible_frac_g', 'lambda_bar', 'mu_bar', 'G_norm', 'dx_norm']
    if GAMMA_SCHEDULE != 'constant': diag_columns.append('gamma')
    if ETA_SCALING != 'constant': diag_columns.append('eta_t')
    schema = 'constrained-v2' if GAMMA_SCHEDULE == 'constant' and REPORT_RESPONSE == 'g' else 'constrained-v2-gamma-schedule-F'
    out = {'schema': schema, 'config': config,
           'hist_columns': ['calls', 'f_at_F_chain' if REPORT_RESPONSE == 'F' else 'phi_hat',
                            'll_cert_gap' if REPORT_RESPONSE == 'F' else 'gap_g', 'seconds'],
           'diag_columns': diag_columns,
           'runs': [{'x0': start.tolist(), 'init_seed': init_seed, 'runs': []}
                    for start, init_seed in zip(STARTS, INIT_SEEDS if INIT_SEEDS is not None else [None] * len(STARTS))]}
    if RESUME and Path(OUT).exists():
        previous = json.load(open(OUT))
        keys = ('C', 'net_deg', 'K', 'K_eval', 'theta_F', 'theta_g', 'gamma', 'eta', 'rho', 'time_budget', 'methods',
                'nstart', 'start_offset', 'start_seeds', 'nseed', 'seed')
        if GAMMA_SCHEDULE != 'constant': keys += ('gamma_schedule', 'gamma_start', 'gamma_end')
        if REPORT_RESPONSE != 'g': keys += ('report_response', 'report_metric')
        if ETA_SCALING != 'constant': keys += ('eta_scaling',)
        if any(previous.get('config', {}).get(key) != config.get(key) for key in keys):
            raise ValueError(f'Cannot resume {OUT}: configuration differs')
        out = previous
    completed = {(si, rec['method'], rec['seed']) for si, group in enumerate(out['runs']) for rec in group['runs']}
    init_seeds = INIT_SEEDS if INIT_SEEDS is not None else [None] * len(STARTS)
    tasks = [(si, s.tolist(), init_seeds[si], name, seed) for si, s in enumerate(STARTS) for name in METHODS
             for seed in range(NSEED if name in RANDOMIZED else 1) if (si, name, seed) not in completed]
    t0 = time.perf_counter(); M.NCUT[0] = 0; phi_hat(STARTS[0], gamma=GAMMA_START)
    eval_flows = K_EVAL * (2 if REPORT_RESPONSE == 'F' else 1)
    print(f'one BiGAP query costs {2*K} max-flows; one reported score {eval_flows}; ~{(time.perf_counter()-t0)/eval_flows*1000:.1f} ms per max-flow')
    gamma_label = f'{GAMMA:g}' if GAMMA_SCHEDULE == 'constant' else f'{GAMMA_SCHEDULE} {GAMMA_START:g}->{GAMMA_END:g}'
    print(f'{len(tasks)} pending runs; {len(completed)} completed; {min(NWORKERS, max(1, len(tasks)))} workers; gamma {gamma_label}', flush=True)
    if NWORKERS == 1 or not tasks: iterator = map(run_task, tasks); pool = None
    else: pool = mp.get_context('fork').Pool(min(NWORKERS, len(tasks))); iterator = pool.imap_unordered(run_task, tasks)
    try:
        for done, (si, rec) in enumerate(iterator, 1):
            out['runs'][si]['runs'].append(rec)
            b = rec['best'] or {}; g = rec['best_gap']; last = rec['hist'][-1][1]
            print(f"[{done}/{len(tasks)}] start {START_OFFSET+si:3d} {rec['method']:11s} seed {rec['seed']}  best {rec['best_phi']:7.0f}  "
                  f"last {last:7.0f}  cut' {b.get('cut2', float('nan')):5.0f}  gap {g if g is not None else float('nan'):6.1f}  "
                  f"iters {len(rec['hist'])-1:4d}  {rec['elapsed_seconds']:.1f}s", flush=True)
            if done % CHECKPOINT_EVERY == 0: save_output(out)
    except BaseException:
        if pool is not None:
            pool.terminate(); pool.join()
        save_output(out)
        raise
    else:
        if pool is not None:
            pool.close(); pool.join()
    for group in out['runs']: group['runs'].sort(key=lambda rec: (METHODS.index(rec['method']), rec['seed']))
    save_output(out)
    print('saved', OUT)
