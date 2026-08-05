"""Real-data instance distributions for the logistic regression certification
experiment.

A problem instance is an UNREGULARIZED (delta = 0, smooth convex, mu = 0)
logistic regression on a uniform random subsample of m_sub rows of a LIBSVM
binary classification dataset. Subsampling a fixed real dataset induces a
natural distribution over problem instances (minibatch / federated
heterogeneity).

Datasets are downloaded once from the LIBSVM site and cached. Cluster compute
nodes have no internet: run `python -m experiment_classes.logreg_data <name>`
on a login node (or locally) first, with DRO_PEP_DATA pointing at the cache.

With delta = 0 a linearly separable instance has no finite minimizer, so
sampling guards against separability: instances whose solver fails or whose
solution norm exceeds cfg.xopt_norm_max are rejected and resampled.
"""
import logging
import os
import pathlib
import urllib.request

import cvxpy as cp
import numpy as np
from sklearn.datasets import load_svmlight_file

log = logging.getLogger(__name__)

LIBSVM_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'

# name -> (remote filename, feature count). n_features fixes the column count
# (LIBSVM sparse files omit trailing all-zero features).
DATASETS = {
    'a9a': ('a9a', 123),
    'ijcnn1': ('ijcnn1.bz2', 22),
    'german.numer': ('german.numer', 24),
}


def data_cache_dir():
    """Cache directory: $DRO_PEP_DATA or <repo>/data/libsvm."""
    env = os.environ.get('DRO_PEP_DATA')
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[2] / 'data' / 'libsvm'


def fetch_dataset_file(name):
    """Return the local path of the raw LIBSVM file, downloading if missing."""
    fname, _ = DATASETS[name]
    cache = data_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / fname
    if not path.is_file():
        url = f'{LIBSVM_URL}/{fname}'
        log.info(f'downloading {url} -> {path}')
        urllib.request.urlretrieve(url, path)
    return path


def load_dataset(name, intercept=True, standardize=False):
    """Load a LIBSVM binary dataset as (A, b) with b in {0, 1}.

    A is dense float64 (m_total, d); an intercept column of ones is appended
    when intercept=True (matching the synthetic LogReg convention).

    standardize=True z-scores each feature over the FULL dataset before
    subsampling (global reparametrization: subsample separability and f*
    are unchanged, conditioning improves). The intercept is appended after
    and left untouched.
    """
    fname, n_features = DATASETS[name]
    path = fetch_dataset_file(name)
    X, y = load_svmlight_file(str(path), n_features=n_features)
    A = np.asarray(X.todense(), dtype=np.float64)
    b = np.where(np.asarray(y) > 0, 1.0, 0.0)
    if standardize:
        mu = A.mean(axis=0)
        sd = A.std(axis=0)
        sd[sd == 0] = 1.0
        A = (A - mu) / sd
    if intercept:
        A = np.hstack([A, np.ones((A.shape[0], 1))])
    log.info(f'{name}: loaded {A.shape[0]} rows, {A.shape[1]} features '
             f'(intercept={intercept}), positives {b.mean():.3f}')
    return A, b


def logreg_instance_L(A, delta):
    """Smoothness constant of one instance: lambda_max(A^T A)/(4 m) + delta."""
    m = A.shape[0]
    lam_max = float(np.max(np.linalg.eigvalsh(A.T @ A)))
    return lam_max / (4 * m) + delta


def solve_instance(A, b, delta):
    """Solve one logistic regression instance with CVXPY/CLARABEL.

    Returns (x_opt, f_opt) or (None, None) if the solver fails (e.g., the
    unregularized problem is unbounded because the instance is separable).
    """
    m, n = A.shape
    x = cp.Variable(n)
    log_likelihood = cp.sum(cp.multiply(b, A @ x) - cp.logistic(A @ x))
    obj = -1 / m * log_likelihood
    if delta > 0:
        obj = obj + 0.5 * delta * cp.sum_squares(x)
    problem = cp.Problem(cp.Minimize(obj))
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None, None
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        return None, None
    return np.asarray(x.value), float(problem.value)


def sample_instance(rng, A_full, b_full, cfg):
    """Sample one non-separable subsample instance.

    Returns (A, b, x_opt, f_opt, L, n_rejected). Rejects (and resamples)
    instances whose solve fails or whose ||x_opt|| exceeds cfg.xopt_norm_max
    (the separability guard for delta = 0).
    """
    m_total = A_full.shape[0]
    for attempt in range(int(cfg.max_resample)):
        idx = rng.choice(m_total, size=int(cfg.m_sub), replace=False)
        A, b = A_full[idx], b_full[idx]
        x_opt, f_opt = solve_instance(A, b, cfg.delta)
        if x_opt is None or np.linalg.norm(x_opt) > cfg.xopt_norm_max:
            continue
        L = logreg_instance_L(A, cfg.delta)
        return A, b, x_opt, f_opt, L, attempt
    raise RuntimeError(
        f'no non-separable instance found in {cfg.max_resample} attempts; '
        f'increase m_sub (currently {cfg.m_sub}) or xopt_norm_max')


def sample_instance_batch(rng, A_full, b_full, cfg, N, desc=''):
    """Sample N instances; returns (list of (A, b, x_opt, f_opt, L), n_rejected)."""
    from tqdm import trange
    out, rejected = [], 0
    iterator = trange(N, desc=desc) if N >= 25 else range(N)
    for _ in iterator:
        A, b, x_opt, f_opt, L, n_rej = sample_instance(rng, A_full, b_full, cfg)
        rejected += n_rej
        out.append((A, b, x_opt, f_opt, L))
    if rejected:
        log.info(f'{desc}: rejected {rejected} separable/failed instances '
                 f'while sampling {N}')
    return out, rejected


def compute_L_R(cfg, A_full, b_full):
    """Calibrate the family constants from cfg.N reference instances.

    L = max instance smoothness constant, R = max ||x_opt|| (the initial
    iterate is x0 = 0, so ||x0 - x*|| = ||x*|| <= R). Deterministic
    (cfg.seed.in_sample) so the samples/pep/dro stages agree. Mutates
    cfg.L and cfg.R; returns (L, R, instances) so the DRO stage can reuse
    the reference instances as its in-sample data.
    """
    rng = np.random.default_rng(cfg.seed.in_sample)
    instances, _ = sample_instance_batch(
        rng, A_full, b_full, cfg, int(cfg.N), desc='calibration instances')
    L = float(max(inst[4] for inst in instances))
    R = float(max(np.linalg.norm(inst[2]) for inst in instances))
    cfg.L = L
    cfg.R = R
    log.info(f'computed L={L}, R={R} from {cfg.N} instances of {cfg.dataset}')
    return L, R, instances


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    for name in (sys.argv[1:] or list(DATASETS)):
        load_dataset(name)
