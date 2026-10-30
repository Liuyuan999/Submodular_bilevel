"""Wall-clock benchmark for the EDA graph-cut experiments."""
import os, json, time, platform
from pathlib import Path
# Set before importing NumPy/SciPy. Run methods sequentially on the same machine.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = os.environ.get('BENCH_THREADS', '1')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import maxflow
from scipy.optimize import minimize

E = os.environ.get
NSTART = int(E('NSTART', '20')); T = int(E('T', '1000000'))
MU = float(E('MU', '0')); WMIN = float(E('WMIN', '0.05'))
SEED = int(E('SEED', '0')); NSEED = int(E('NSEED', '1'))
METHODS = E('METHODS', 'bigap,ssvm,bbdiff,cmaes' if E('WARM') else 'bigap,weighted,ssvm,bbdiff,zo,cmaes,relaxed').split(',')
GRID = E('GRID', '0') == '1'; GRID_TH = int(E('GRID_TH', '49')); GRID_B = int(E('GRID_B', '41'))
GRID_GAMMAS = [float(t) for t in E('GRID_GAMMAS', '0.02,1').split(',')]
GAMMA_FIXED = float(E('GAMMA')) if E('GAMMA') else None
GAMMA_SWEEP = [float(t) for t in E('GAMMA_SWEEP', '').split(',') if t]
NSWEEP = int(E('NSWEEP', '5'))
SSVM_LAM = float(E('SSVM_LAM', '1'))
WARM = E('WARM', '')
BOUNDARY = E('BOUNDARY', 'line')                 # line (p=3) | quad (p=6) | poly3 (p=10) | poly4 (p=15)
OUT = E('OUT', str(Path(__file__).resolve().parent.parent / 'results' / (f'walltime_mu{MU:g}_{BOUNDARY}' + ('_warm' if WARM else '') + '.json')))
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
DATA = E('DATA', str(Path(__file__).resolve().parent.parent.parent / 'data'))
TIME_BUDGET = float(E('TIME_BUDGET', '30'))
assert TIME_BUDGET > 0
BATCH = int(E('BATCH', '1')); assert BATCH >= 1
UPDATE = E('UPDATE', 'normalized')
assert UPDATE in ('normalized', 'fixed', 'decay')
ETA = float(E('ETA', '1e-5')); assert ETA > 0
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data
lines = open(f'{DATA}/ibm01.hgr').read().split('\n')
m, n = map(int, lines[0].split())
I, J, W, net_pins, net_ptr = [], [], [], [], [0]
for l in lines[1:1 + m]:
    pins = [int(t) - 1 for t in l.split()]; k = len(pins)
    net_pins.extend(pins); net_ptr.append(len(net_pins))
    if k < 2: continue
    for a in range(k):
        for b in range(a + 1, k):
            I.append(pins[a]); J.append(pins[b]); W.append(1.0 / (k - 1))
net_pins = np.array(net_pins); net_ptr = np.array(net_ptr)
net_sizes = np.diff(net_ptr); net_starts = net_ptr[:-1][net_sizes > 1]
A = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr(); A = A + A.T
A = A.tocoo(); mask = A.row < A.col
ei, ej, w0 = A.row[mask], A.col[mask], A.data[mask]

# placement: data/placement.npy (spectral by default; drop in DREAMPlace coordinates with the same name)
z = np.load(f'{DATA}/placement.npy')
z = (z - z.min(0)) / (z.max(0) - z.min(0)) - 0.5
sigma = float(E('SIGMA', '0.1'))
w = w0 * np.exp(-np.sum((z[ei] - z[ej]) ** 2, 1) / sigma ** 2)
keep = w > 1e-6; ei, ej, w = ei[keep], ej[keep], w[keep]
NE = len(ei)

yref = np.loadtxt(f'{DATA}/gold.txt')
Ed = (yref[ei] != yref[ej]) & (w >= WMIN)               # reference-cut, well-connected pairs
a = np.where(Ed, MU, 0.0)                               # agreement weights of f
beta = float((a / w).max()) if MU > 0 else 0.0          # Prop. 2: Delta f = 2a, Delta g = -2w  -> beta* = max a/w
Wm = sp.coo_matrix((w, (ei, ej)), shape=(n, n)); Wm = (Wm + Wm.T).tocsr()
Lw = sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm

def hcut(y):
    """hypergraph cut size (unit net weights) of a binary labeling"""
    yp = y[net_pins]
    mn = np.minimum.reduceat(yp, net_ptr[:-1]); mx = np.maximum.reduceat(yp, net_ptr[:-1])
    return int(np.sum((mn != mx) & (net_sizes > 1)))

print(f'cells {n}  edges {NE}  |E_d| {int(Ed.sum())}  mu {MU}  beta {beta:.3f}')
print(f'reference: hypergraph cut {hcut(yref)}  balance {yref.mean():.4f}')

# ---------------------------------------------------------------- model
alpha, kappa = float(E('ALPHA', '3')), float(E('KAPPA', '30'))
def softplus(t): return np.logaddexp(0, t)
def sigm(t): return 1 / (1 + np.exp(-t))
STEP0 = float(E('STEP0', '0.005' if WARM else '0.02'))
CMA_SIGMA = float(E('CMA_SIGMA', '0.05' if WARM else '0.3'))
cols = [z[:, 0], z[:, 1]]
if BOUNDARY.startswith('grid'):
    mgrid = int(BOUNDARY[4:])
    cell = np.clip(((z + 0.5) * mgrid).astype(int), 0, mgrid - 1); rid = cell[:, 0] * mgrid + cell[:, 1]
    for r_ in range(mgrid * mgrid): cols.append((rid == r_).astype(float))          # one offset per spatial cell
else:
    DEG = {'line': 1, 'quad': 2, 'poly3': 3, 'poly4': 4}[BOUNDARY]
    for dtot in range(2, DEG + 1):
        for k in range(dtot + 1): cols.append(z[:, 0] ** (dtot - k) * z[:, 1] ** k)  # all monomials of degree dtot
FEAT = np.stack(cols, 1); mx_ = np.abs(FEAT).max(0); FEAT = FEAT * (0.5 / np.where(mx_ > 0, mx_, 1))   # max|.| = 0.5 per feature
P = FEAT.shape[1] + 1
print(f'boundary {BOUNDARY}: p = {P}')
XLO = np.array([-1.5] * (P - 1) + [-0.8]); XHI = np.array([1.5] * (P - 1) + [0.8])
def proj(x): return np.clip(x, XLO, XHI)
def steplen(t): return STEP0 / (1 + t / 40)

def sdist(x):
    s = FEAT @ x[:-1] - x[-1]
    ds = np.concatenate([FEAT, -np.ones((n, 1))], 1)
    return s, ds

def unary(x):
    s, ds = sdist(x)
    c1 = alpha * softplus(-kappa * s); c0 = alpha * softplus(kappa * s)
    dc1 = -alpha * kappa * sigm(-kappa * s)[:, None] * ds
    dc0 = alpha * kappa * sigm(kappa * s)[:, None] * ds
    return c0, c1, dc0, dc1

NCUT = [0]
def graphcut(c0, c1, pw):
    """argmin_y sum c1 y + c0 (1-y) + sum pw 1{y_i != y_j}   (one max-flow)"""
    NCUT[0] += 1
    G = maxflow.Graph[float](n, len(pw)); nodes = G.add_nodes(n)
    G.add_grid_tedges(nodes, c1, c0); G.add_edges(ei, ej, pw, pw); G.maxflow()
    return G.get_grid_segments(nodes).astype(float)

def g_val(c0, c1, y): return np.sum(c1 * y + c0 * (1 - y)) + np.sum(w * (y[ei] != y[ej]))
def f_val(y): return float(np.sum(y != yref) + np.sum(a * (y[ei] == y[ej])))
def gy(dc0, dc1, y): return (dc1 * y[:, None] + dc0 * (1 - y)[:, None]).sum(0)   # grad_x g(x, y)

def lower(x):
    c0, c1, dc0, dc1 = unary(x); return graphcut(c0, c1, w), (c0, c1, dc0, dc1)

def phi(x):
    y, _ = lower(x); return f_val(y)

def stats(y):
    return {'hamming': int(np.sum(y != yref)), 'hcut': hcut(y), 'balance': float(y.mean()), 'f': f_val(y)}

# ---- the two-cut oracle
def penalty_oracle(x, gamma):
    c0, c1, dc0, dc1 = unary(x)
    ystar = graphcut(c0, c1, w)
    pw = w - a / gamma; assert pw.min() >= 0, 'gamma < beta'
    ygam = graphcut(c0 + yref / gamma, c1 + (1 - yref) / gamma, pw)      # argmin g + f/gamma
    p = gamma * ((f_val(ygam) / gamma + g_val(c0, c1, ygam)) - g_val(c0, c1, ystar))
    u = gamma * (gy(dc0, dc1, ygam) - gy(dc0, dc1, ystar))
    return p, u, ystar, ygam, int(np.sum(ygam != ystar))

def gamma_sched(t, gamma_fixed=None):
    if gamma_fixed is not None:
        if gamma_fixed <= 0 or gamma_fixed < beta: raise ValueError('GAMMA must be positive and >= beta')
        return gamma_fixed
    g0 = max(beta, 0.005); return min(g0 * 1.014 ** t, max(5.0, beta))

# Hist rows: [actual cumulative max-flow calls, f, oracle disagreement (1{y_gamma != y*} count) or null, seconds].
# BiGAP additionally keeps a lightweight per-iteration diagnostic (see run_loop): [t, disagreement, ||u(z)||, ||u_bar||, ||dx||].
# All optimization work and objective evaluation for best selection are charged.
class RunClock:
    def __init__(self, x0):
        NCUT[0] = 0
        self.t0 = time.perf_counter()
        self.best = (np.inf, x0.copy())
        self.last = x0.copy()
        self.hist = []
        self.traj = []
        self.diag = []
        self.stop_reason = 'iteration_cap'
        self.observe(x0, phi(x0), None)

    def elapsed(self):
        return time.perf_counter() - self.t0

    def active(self):
        if self.elapsed() >= TIME_BUDGET:
            self.stop_reason = 'time_budget'
            return False
        return True

    def observe(self, x, fv, extra):
        secs = self.elapsed()
        if secs > TIME_BUDGET:
            self.stop_reason = 'time_budget'
            return False  # Never credit an objective evaluation completed after deadline.
        self.last = x.copy()
        if fv < self.best[0]: self.best = (float(fv), x.copy())
        self.hist.append([int(NCUT[0]), float(fv), extra, secs])
        self.traj.append(x.tolist())
        return True

    def finish(self):
        global LAST_META
        LAST_META = {'elapsed_seconds': self.elapsed(), 'total_calls_including_overrun': int(NCUT[0]),
                     'stop_reason': self.stop_reason}
        if not self.hist:
            raise RuntimeError('TIME_BUDGET too small even for initial evaluation; increase it.')
        return self.last, self.hist, self.best[1], self.traj, self.diag

LAST_META = {}
# Import optional solver before benchmarking to avoid first-run import overhead.
if 'cmaes' in METHODS:
    import cma
def run_loop(x0, direction, update='normalized'):
    clock = RunClock(x0); x = x0.copy()
    for t in range(T):
        if not clock.active(): break
        d = direction(x, t); u, extra = d[0], d[1]
        if update == 'fixed': step = ETA
        elif update == 'decay': step = ETA / (1 + t / 40)
        else: step = steplen(t) / max(np.linalg.norm(u), 1e-12)
        x_new = proj(x - step * u)
        if len(d) > 2:   # BiGAP: oracle disagreement vs. actual update, so 'no signal' and 'signal but bad step' can be told apart
            clock.diag.append([t, extra, float(np.mean(d[2])), float(np.linalg.norm(u)), float(np.linalg.norm(x_new - x))])
        x = x_new
        if not clock.active(): break
        if not clock.observe(x, phi(x), extra): break
    return clock.finish()

# ---------------------------------------------------------------- methods
def bigap(x0, rs, gamma_fixed=GAMMA_FIXED, rho=float(E('RHO', '0.01'))):
    def direction(x, t):
        us, dis = [], []
        for _ in range(BATCH):
            wd = rs.normal(size=P); wd /= np.linalg.norm(wd)
            wd *= rs.random() ** (1.0 / P)
            _, u, _, _, ndis = penalty_oracle(x + rho * wd, gamma_sched(t, gamma_fixed))
            us.append(u); dis.append(ndis)
        return np.mean(us, axis=0), float(np.mean(dis)), [float(np.linalg.norm(u)) for u in us]
    return run_loop(x0, direction, update=UPDATE)


def weighted(x0, rs):                       # joint objective f + gamma g: uses y_1 only, drops -gamma grad v
    def direction(x, t):
        gamma = gamma_sched(t); c0, c1, dc0, dc1 = unary(x)
        y1 = graphcut(c0 + yref / gamma, c1 + (1 - yref) / gamma, w - a / gamma)
        return gamma * gy(dc0, dc1, y1), None
    return run_loop(x0, direction)

def ssvm(x0, rs, lam=SSVM_LAM):             # Szummer-Kohli-Hoiem: loss-augmented cut, hinge subgradient
    def direction(x, t):
        c0, c1, dc0, dc1 = unary(x)
        yhat = graphcut(c0 - lam * yref, c1 - lam * (1 - yref), w + lam * a)   # argmin g - lam f
        return gy(dc0, dc1, yref) - gy(dc0, dc1, yhat), None
    return run_loop(x0, direction)

def bbdiff(x0, rs):                         # Vlastelica et al.: f linearized at y* by single flips, lam = 1/gamma
    def direction(x, t):
        gamma = gamma_sched(t); lam = 1 / gamma
        c0, c1, dc0, dc1 = unary(x); ystar = graphcut(c0, c1, w)
        # l_i = f(flip_i y*) - f(y*);  linear model  f~(y) = const + sum_i l_i (1-2y*_i) y_i
        l = np.where(ystar == yref, 1.0, -1.0)
        same = (ystar[ei] == ystar[ej])
        contrib = np.where(same, -a, a)              # flipping i separates (-a) or merges (+a) each E_d pair
        np.add.at(l, ei, contrib); np.add.at(l, ej, contrib)
        c = l * (1 - 2 * ystar)
        ylam = graphcut(c0, c1 + lam * c, w)
        return (gy(dc0, dc1, ylam) - gy(dc0, dc1, ystar)) / lam, int(np.sum(ylam != ystar))
    return run_loop(x0, direction)

def zo(x0, rs, mu=0.01):                    # two-point Gaussian estimate of grad phi (nested / zeroth-order)
    def direction(x, t):
        d = rs.normal(size=P)
        return (phi(x + mu * d) - phi(x - mu * d)) / (2 * mu) * d, None
    return run_loop(x0, direction)

def cmaes(x0, rs, budget=None):
    clock = RunClock(x0)  # Includes CMA-specific setup and the initial evaluation.
    es = cma.CMAEvolutionStrategy(x0, CMA_SIGMA, {
        'bounds': [list(XLO), list(XHI)], 'maxiter': T,
        'seed': int(rs.integers(1, 2 ** 31)), 'verbose': -9})
    while clock.active() and not es.stop():
        X = es.ask(); vals = []
        for xx in X:
            if not clock.active(): break
            xx = np.asarray(xx); fv = phi(xx)
            if not clock.observe(xx, fv, None): break
            vals.append(fv)
        if len(vals) != len(X): break
        es.tell(X, vals)
    if es.stop() and clock.active(): clock.stop_reason = 'solver_stop'
    return clock.finish()


def relaxed(x0, rs, T_rel=int(E('T_REL', str(T)))):   # Ochs-style: quadratic relaxation + implicit differentiation
    clock = RunClock(x0); x = x0.copy(); y = np.full(n, 0.5)
    for t in range(T_rel):
        if not clock.active(): break
        c0, c1, dc0, dc1 = unary(x); c = c1 - c0
        obj = lambda yy: (c @ yy + yy @ (Lw @ yy), c + 2 * Lw @ yy)
        y = minimize(obj, y, jac=True, method='L-BFGS-B', bounds=[(0, 1)] * n, options={'maxiter': 200}).x
        free = (y > 1e-6) & (y < 1 - 1e-6)
        dfy = 1 - 2 * yref; dif = y[ei] - y[ej]                      # relaxed f: sum a (1 - (yi-yj)^2)
        np.add.at(dfy, ei, -2 * a * dif); np.add.at(dfy, ej, 2 * a * dif)
        grad = np.zeros(P)
        if free.sum() > 0:
            Lff = Lw[free][:, free]
            lam = spla.spsolve((2 * Lff + 1e-8 * sp.eye(free.sum())).tocsc(), dfy[free])
            grad = -(lam @ (dc1 - dc0)[free])
        x = proj(x - grad * (steplen(t) / (np.linalg.norm(grad) + 1e-12)))
        if not clock.active(): break
        if not clock.observe(x, phi(x), None): break
    return clock.finish()


FUNS = {'bigap': bigap, 'weighted': weighted, 'ssvm': ssvm, 'bbdiff': bbdiff, 'zo': zo, 'cmaes': cmaes, 'relaxed': relaxed}
RANDOMIZED = {'bigap', 'zo', 'cmaes'}

out = {'config': {'boundary': BOUNDARY, 'p': P, 'n': n, 'edges': NE, 'Ed': int(Ed.sum()), 'mu': MU, 'wmin': WMIN, 'beta': beta, 'T': T,
                  'alpha': alpha, 'kappa': kappa, 'sigma': sigma, 'ssvm_lam': SSVM_LAM, 'time_budget': TIME_BUDGET, 'batch': BATCH, 'update': UPDATE, 'eta': ETA,
                  'gamma_fixed': GAMMA_FIXED, 'rho': float(E('RHO', '0.01')),
                  'step0': STEP0, 'threads': E('BENCH_THREADS', '1'), 'seed': SEED, 'nseed': NSEED,
                  'python': platform.python_version(), 'platform': platform.platform(),
                  'ref': stats(yref)}, 'runs': [], 'sweep': []}

# ---------------------------------------------------------------- landscape + disagreement grid
if GRID:
    tg = time.time()
    TH = np.linspace(-np.pi, np.pi, GRID_TH); B = np.linspace(-0.8, 0.8, GRID_B)
    land = np.zeros((GRID_TH, GRID_B)); maps = {}
    for i, th in enumerate(TH):
        for j, b in enumerate(B):
            land[i, j] = phi(np.array([np.cos(th), np.sin(th)] + [0.0] * (P - 3) + [b]))
    for gam0 in GRID_GAMMAS:
        gam = max(gam0, beta); pen = np.zeros_like(land); sig = np.zeros_like(land); unorm = np.zeros_like(land)
        for i, th in enumerate(TH):
            for j, b in enumerate(B):
                p, u, ystar, ygam, nsig = penalty_oracle(np.array([np.cos(th), np.sin(th)] + [0.0] * (P - 3) + [b]), gam)
                pen[i, j] = p; sig[i, j] = nsig; unorm[i, j] = np.linalg.norm(u)
        maps[f'{gam:g}'] = {'p': pen.tolist(), 'disagreement': sig.tolist(), 'unorm': unorm.tolist()}
    out['grid'] = {'TH': TH.tolist(), 'B': B.tolist(), 'phi': land.tolist(), 'maps': maps}
    print(f'grid done {time.time() - tg:.0f}s  min phi {land.min():.0f}')

# ---------------------------------------------------------------- main runs
if E('DRY'):   # DRY=1: build data + model only (used by reeval_phase1.py and diagnostics); no runs
    raise SystemExit(0)
starts = [rng.uniform(XLO * 0.8, XHI * 0.8) for _ in range(NSTART)]
if BOUNDARY.startswith('grid'):
    for s_ in starts: s_[2:-1] = 0.0                                                # straight-line starts, offsets zero
if E('STARTS'): starts = list(np.load(E('STARTS')))
if WARM:
    prev = json.load(open(WARM)); assert prev['config'].get('p', 3) == P, 'WARM file has a different boundary parameterization'
    starts = [np.array(next(u for u in r_['runs'] if u['method'] == 'bigap')['x_best']) for r_ in prev['runs']]
    out['config']['warm_from'] = WARM; print(f'warm start: {len(starts)} phase-1 solutions from {WARM}')
out['schema'] = 'walltime-v4'
out['hist_columns'] = ['calls', 'f', 'disagreement', 'seconds']
out['diag_columns'] = ['t', 'disagreement', 'u_norm_mean', 'ubar_norm', 'dx_norm']
out['timing_policy'] = 'Includes initialization, graph construction, solves, gradients, updates, and best-selection evaluation; excludes shared data preparation and post-run diagnostics. In-flight work may overrun, but late observations are discarded.'
out['starts'] = [s.tolist() for s in starts]
# Keep bbdiff at MU=0: the oracle formulas match at the same query, but bigap perturbs its query.

def pack(y): return np.packbits(y.astype(np.uint8)).tobytes().hex()

def record(name, s, seed, xf, h, secs, xb, traj, diag=None):
    yf, _ = lower(xf); yb, _ = lower(xb); NCUT[0] -= 2
    rec = {'method': name, 'seed': seed, 'x': xf.tolist(), 'x_best': xb.tolist(), 'hist': h, 'time': secs,
           'last': stats(yf), 'best': stats(yb), 'best_phi': float(min(r[1] for r in h)),
           'y_last': pack(yf), 'y_best': pack(yb), 'traj': traj, **LAST_META}
    if diag:
        rec['diag'] = diag
        D = np.asarray(diag, dtype=float)
        rec['diagnostic_summary'] = {
            'disagreement_rate': float(np.mean(D[:, 1] > 0)),
            'disagreement_count_median': float(np.median(D[:, 1])),
            'sample_direction_norm_mean': float(np.mean(D[:, 2])),
            'batch_direction_norm_mean': float(np.mean(D[:, 3])),
            'displacement_norm_mean': float(np.mean(D[:, 4])),
            'zero_batch_direction_rate': float(np.mean(D[:, 3] == 0)),
        }
    return rec

for si, s in enumerate(starts):
    r = {'start': s.tolist(), 'f0': phi(s), 'runs': []}
    for name in METHODS:
        for seed in range(NSEED if name in RANDOMIZED else 1):
            rs = np.random.default_rng(1000 * SEED + 10 * si + seed)
            t0 = time.perf_counter(); xf, h, xb, traj, diag = FUNS[name](s, rs); secs = time.perf_counter() - t0
            rec = record(name, s, seed, xf, h, secs, xb, traj, diag); r['runs'].append(rec)
            print(f'[{si + 1}/{len(starts)}] {name:9s} seed {seed}  best {rec["best_phi"]:7.0f}  last {rec["last"]["hamming"]:6d}'
                  f'  hcut {rec["last"]["hcut"]:4d}  bal {rec["last"]["balance"]:.3f}  {secs:5.1f}s', flush=True)
    out['runs'].append(r); json.dump(out, open(OUT, 'w'))

# ---------------------------------------------------------------- fixed-gamma sweep (bigap only)
for gam in GAMMA_SWEEP:
    if gam <= 0 or gam < beta: print(f'skip gamma {gam} < beta {beta:.2f}'); continue
    for si, s in enumerate(starts[:NSWEEP]):
        rs = np.random.default_rng(1000 * SEED + 10 * si + 7)
        xf, h, xb, traj, diag = bigap(s, rs, gamma_fixed=gam)
        rec = record('bigap', s, 0, xf, h, LAST_META['elapsed_seconds'], xb, traj, diag); rec['gamma'] = gam; rec['start_index'] = si
        out['sweep'].append(rec)
        print(f'sweep gamma {gam:6.2f} start {si + 1}: best {rec["best_phi"]:.0f}  disagreement rate {np.mean([r[2] > 0 for r in h]):.2f}', flush=True)
    json.dump(out, open(OUT, 'w'))

json.dump(out, open(OUT, 'w'))
print('saved', OUT)
