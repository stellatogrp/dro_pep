"""Plain-numpy loaders for the instance sets the paper figures evaluate on.

Defaults point at the archived della sets for Quad / Lasso (the paper
figures, see iclr_data_outputs/archive/<exp>/create_test_plots.py) and at the
pinned ICLR sample-creation bundle for LogReg (iclr_data_outputs/plotting/
logreg_figures.py). Each loader also returns the *default* step size the
learned sequences are initialised from, with its provenance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2]
ICLR_DIR = SRC_DIR / 'iclr_data_outputs'
ARCHIVE_DIR = ICLR_DIR / 'archive'

DEFAULT_DATA_DIRS = {
    'quad': ARCHIVE_DIR / 'quad' / 'problem_instances',
    'lasso': ARCHIVE_DIR / 'lasso' / 'problem_instances',
    'logreg': ICLR_DIR / 'sample_creation_outputs' / 'LogReg' / '2026-09-18' / '08-30-44',
}

# PDLP: tau_0 = sigma_0 of the archived training runs, i.e. 0.5 / M with the
# pooled operator-norm bound M = 1.2 * max_train ||K||_2
# (archive/pdlp/ldro-pep_timings/K_10/progress.csv, row 0).
PDLP_TAU0_ARCHIVED = 0.10126272219280587
PDLP_M_ARCHIVED = 0.5 / PDLP_TAU0_ARCHIVED


def _meta(path):
    d = np.load(path, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def load_quad(data_dir=None):
    """Quad GD sets: f(z) = 0.5 z^T Q z, x* = 0, f* = 0.

    Default step 1.5 / (mu + L) ("fixed" init in
    learning_experiment_classes/quad.py::get_initial_stepsizes).
    """
    d = Path(data_dir or DEFAULT_DATA_DIRS['quad'])
    meta = _meta(d / 'out_of_sample_metadata.npz')
    mu, L = float(meta['mu']), float(meta['L'])
    splits = {}
    for split, qf, zf in (('test', 'Q_test_samples.npz', 'z0_test_samples.npz'),
                          ('ood', 'Q_out_of_dist_samples.npz', 'z0_out_of_dist_samples.npz')):
        Q = np.load(d / qf)['Q']
        z0 = np.load(d / zf)['z0']
        splits[split] = dict(Q=Q, z0=z0, f_opt=np.zeros(Q.shape[0]))
    t_default = 1.5 / (mu + L) if mu > 0 else 1.5 / L
    return dict(splits=splits, mu=mu, L=L, t_default=t_default, data_dir=str(d))


def load_lasso(data_dir=None):
    """Lasso ISTA sets: shared A per split, b / x_opt / f_opt per instance.

    L = lambda_max(A^T A) of the in-distribution A (compute_lasso_params in
    learning_experiment_classes/lasso.py); default step 1.5 / L ("fixed" init
    in lasso.py::get_initial_stepsizes). x0 = 0 in original coordinates.
    """
    d = Path(data_dir or DEFAULT_DATA_DIRS['lasso'])
    meta = _meta(d / 'out_of_sample_metadata.npz')
    lambd = float(meta['lambd'])
    splits = {}
    for split, af, tag in (('test', 'A_in_dist.npz', 'test'),
                           ('ood', 'A_out_of_dist.npz', 'out_of_dist')):
        A = np.load(d / af)['A']
        b = np.load(d / f'b_{tag}_samples.npz')['b']
        x_opt = np.load(d / f'x_opt_{tag}_samples.npz')['x_opt']
        f_opt = np.load(d / f'f_opt_{tag}_samples.npz')['f_opt']
        splits[split] = dict(A=A, b=b, x_opt=x_opt, f_opt=f_opt)
    A_in = splits['test']['A']
    L = float(np.linalg.eigvalsh(A_in.T @ A_in).max())
    return dict(splits=splits, lambd=lambd, L=L, t_default=1.5 / L, data_dir=str(d))


def load_logreg(data_dir=None):
    """LogReg sets (german.numer): per-instance A (m, n), labels b in {0,1}.

    L = training_L_max from the bundle metadata (what the trainer pins);
    default step 1 / L (mu = 0; logreg.py::get_initial_stepsizes). x0 = 0.
    """
    d = Path(data_dir or DEFAULT_DATA_DIRS['logreg'])
    meta = _meta(d / 'out_of_sample_metadata.npz')
    L = float(meta['training_L_max'])
    splits = {}
    for split, fn in (('test', 'test_set.npz'), ('ood', 'ood_set.npz')):
        z = np.load(d / fn)
        splits[split] = dict(A=z['A_batch'], b=z['b_batch'], f_opt=z['f_opt_batch'],
                             x_opt=z['x_opt_batch'])
    return dict(splits=splits, L=L, t_default=1.0 / L, data_dir=str(d))
