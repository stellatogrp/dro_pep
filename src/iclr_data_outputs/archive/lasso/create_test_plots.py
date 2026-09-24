"""
Test Set Plotting for LDRO-PEP, L2O, and LPEP (WC-PEP) Experiments on Lasso.

For each configuration and K value, creates seven plots:
1. stepsize_schedule: Shows the stepsize (and beta for FISTA) over iterations
2. out_of_sample_loss: Evaluates loss on test set for all methods
3. frac_problems_solved: Fraction of problems solved at each tolerance level
4. out_of_distribution_loss: Evaluates loss on out-of-distribution test set
5. out_of_dist_frac_problems_solved: Fraction of OOD problems solved at each tolerance level
6. in_sample_loss: Evaluates loss on training data for all methods
7. in_sample_frac_problems_solved: Fraction of training problems solved at each tolerance level

Usage:
    python create_test_plots.py              # Use cached data if available
    python create_test_plots.py -recompute   # Force recompute all data
"""
import sys
import os
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add parent directories to path for imports
script_dir = Path(__file__).parent
# Consolidated layout: this file lives at
#   iclr_data_outputs/archive/<problem>/ , so src/ is three levels up,
# where it was two in the original experiment_plots_icml/<problem>/.
src_dir = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from learning.trajectories.ista_fista import (
    problem_data_to_ista_trajectories,
    problem_data_to_fista_trajectories,
)
from data_scrape import configs, ALL_LASSO_CONFIGS, get_leaf_path

jax.config.update("jax_enable_x64", True)


# ==================== Configuration ====================

K_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
eta_vals = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
ood_eta_vals = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]

# Mapping from architecture name to CSV filename
ARCH_TO_CSV = {
    'l2o': 'l2o_best_stepsize_schedule.csv',
    'ldro_pep': 'ldro_pep_best_stepsize_schedule.csv',
    'lpep': 'lpep_best_stepsize_schedule.csv',
    'l2o_alista': 'l2o_alista_best_stepsize_schedule.csv',
}

# Display names for legends
ARCH_DISPLAY_NAMES = {
    'l2o': 'L2O',
    'ldro_pep': 'LDRO-PEP',
    'lpep': 'WC-PEP',
    'l2o_alista': 'l2o-alista',
}

# Per-arch colors used by the test-plot panels (keep in sync with the inline
# `colors = {...}` dicts and with create_paper_plots.ARCH_COLORS).
ARCH_COLORS = {
    'l2o': 'tab:blue',
    'ldro_pep': 'tab:orange',
    'lpep': 'tab:green',
    'l2o_alista': 'tab:red',
}


# ==================== Config Parsing ====================

def parse_config_list_to_dict(config_list):
    """Convert config list like ['alg=ista', 'dro_obj=expectation'] to dict."""
    config_dict = {}
    for item in config_list:
        key, value = item.split('=', 1)
        try:
            if '.' not in value:
                config_dict[key] = int(value)
            else:
                config_dict[key] = float(value)
        except ValueError:
            config_dict[key] = value
    return config_dict


def config_option_to_folder_name(option_str):
    """Convert 'alg=ista' to 'alg_ista'."""
    return option_str.replace('=', '_')


# ==================== Data Loading ====================

def load_test_data():
    """Load test data from problem_instances directory."""
    lasso_dir = Path(__file__).parent
    problem_dir = lasso_dir / 'problem_instances'

    # Load A matrix (in-distribution)
    A_data = np.load(problem_dir / 'A_in_dist.npz')
    A_jax = jnp.array(A_data['A'])

    # Load test b samples
    b_data = np.load(problem_dir / 'b_test_samples.npz')
    b_batch_jax = jnp.array(b_data['b'])

    # Load precomputed optimal solutions and values
    x_opt_data = np.load(problem_dir / 'x_opt_test_samples.npz')
    x_opt_batch_jax = jnp.array(x_opt_data['x_opt'])

    f_opt_data = np.load(problem_dir / 'f_opt_test_samples.npz')
    f_opt_batch_jax = jnp.array(f_opt_data['f_opt'])

    # Load metadata
    metadata_data = np.load(problem_dir / 'out_of_sample_metadata.npz')
    metadata = {key: metadata_data[key].item() if metadata_data[key].ndim == 0 else metadata_data[key]
                for key in metadata_data.keys()}

    lambd = metadata['lambd']

    return A_jax, b_batch_jax, x_opt_batch_jax, f_opt_batch_jax, lambd, metadata


def load_ood_test_data():
    """Load out-of-distribution test data from problem_instances directory."""
    lasso_dir = Path(__file__).parent
    problem_dir = lasso_dir / 'problem_instances'

    # Load A matrix (out-of-distribution)
    A_data = np.load(problem_dir / 'A_out_of_dist.npz')
    A_jax = jnp.array(A_data['A'])

    # Load OOD test b samples
    b_data = np.load(problem_dir / 'b_out_of_dist_samples.npz')
    b_batch_jax = jnp.array(b_data['b'])

    # Load precomputed optimal solutions and values
    x_opt_data = np.load(problem_dir / 'x_opt_out_of_dist_samples.npz')
    x_opt_batch_jax = jnp.array(x_opt_data['x_opt'])

    f_opt_data = np.load(problem_dir / 'f_opt_out_of_dist_samples.npz')
    f_opt_batch_jax = jnp.array(f_opt_data['f_opt'])

    # Load metadata
    metadata_data = np.load(problem_dir / 'out_of_sample_metadata.npz')
    metadata = {key: metadata_data[key].item() if metadata_data[key].ndim == 0 else metadata_data[key]
                for key in metadata_data.keys()}

    lambd = metadata['lambd']

    return A_jax, b_batch_jax, x_opt_batch_jax, f_opt_batch_jax, lambd


def load_in_sample_data(source_dir, K):
    """Reconstruct a run's training set from `problem_instances/training_set.npz`.

    The trainer no longer dumps a per-run `training_set.npz`. Instead it loads
    rows from a centralized `problem_instances/training_set.npz` and subsamples
    `training_sample_N` indices via `np.random.default_rng(training_seed)`
    (mirroring `_load_and_subsample` in `learning_experiment_classes/lasso.py`).
    Both keys live in the run's `.hydra/config.yaml`.

    A is fixed across runs and is loaded from `problem_instances/A_in_dist.npz`.
    Initial points are zero in shifted coordinates.
    """
    del K  # training_set is K-independent
    lasso_dir = Path(__file__).parent
    problem_dir = lasso_dir / 'problem_instances'

    cfg_path = lasso_dir / source_dir / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    N = int(cfg.training_sample_N)
    seed = int(cfg.training_seed)

    A_jax = jnp.array(np.load(problem_dir / 'A_in_dist.npz')['A'])

    d = np.load(problem_dir / 'training_set.npz')
    total = int(d['b_batch'].shape[0])
    if N >= total:
        idx = np.arange(total)
    else:
        idx = np.random.default_rng(seed).choice(total, size=N, replace=False)

    b_batch_jax = jnp.asarray(d['b_batch'][idx])
    x_opt_batch_jax = jnp.asarray(d['x_opt_batch'][idx])
    f_opt_batch_jax = jnp.asarray(d['f_opt_batch'][idx])

    x0_batch_jax = jnp.zeros((b_batch_jax.shape[0], A_jax.shape[1]))

    metadata_data = np.load(problem_dir / 'out_of_sample_metadata.npz')
    metadata = {key: metadata_data[key].item() if metadata_data[key].ndim == 0 else metadata_data[key]
                for key in metadata_data.keys()}
    lambd = metadata['lambd']

    return A_jax, b_batch_jax, x0_batch_jax, x_opt_batch_jax, f_opt_batch_jax, lambd


def load_best_stepsize(k_output_dir, arch, alg, stepsize_type, K):
    """Load best stepsize schedule from CSV and extract source directory."""
    csv_path = k_output_dir / ARCH_TO_CSV[arch]
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    row = df.iloc[0]  # Single row in best stepsize file

    # Extract source directory if present
    source_dir = row.get('source_dir', None)

    has_beta = (alg == 'fista')
    is_vector = (stepsize_type == 'vector')

    # Parse gamma
    if is_vector:
        gamma_cols = [f'gamma_{k}' for k in range(K)]
        gamma = jnp.array([row[col] for col in gamma_cols])
    else:
        gamma = jnp.array(row['gamma'])

    # Parse beta if FISTA
    if has_beta:
        beta_cols = [f'beta_{k}' for k in range(K + 1)]  # Beta has K+1 values
        beta = jnp.array([row[col] for col in beta_cols])
        return (gamma, beta), source_dir
    else:
        return (gamma,), source_dir


# ==================== Evaluation ====================

def compute_loss_trajectory_single(stepsizes, A_jax, b, x0, x_opt, f_opt, lambd, K, alg):
    """
    Compute loss at each iteration k = 1 to K for a single problem.
    Returns array of shape (K,) with f(x_k) - f_opt for k = 1, ..., K.
    """
    # Select trajectory function
    if alg == 'ista':
        traj_fn = problem_data_to_ista_trajectories
        traj_stepsizes = stepsizes[0] if isinstance(stepsizes, tuple) else stepsizes

        # ISTA returns: (x_iter, g_iter, h_iter, f1_iter, f2_iter)
        traj_result = traj_fn(
            traj_stepsizes, A_jax, b, x0, x_opt, f_opt, lambd, K,
            return_Gram_representation=False
        )
        x_iter, g_iter, h_iter, f1_iter, f2_iter = traj_result

        # Total loss is f1 + f2 (both are already shifted: f(x+x_opt) - f_opt)
        f_iter = f1_iter + f2_iter

    else:  # fista
        traj_fn = problem_data_to_fista_trajectories
        traj_stepsizes = stepsizes

        # FISTA returns: (x_iter, y_iter, g_y_iter, g_xK, h_iter, f1_y_iter, f1_xK, f2_x_iter)
        traj_result = traj_fn(
            traj_stepsizes, A_jax, b, x0, x_opt, f_opt, lambd, K,
            return_Gram_representation=False
        )
        x_iter, y_iter, g_y_iter, g_xK, h_iter, f1_y_iter, f1_xK, f2_x_iter = traj_result

        # For FISTA, we need f1(x_k) + f2(x_k) for k=0,...,K
        # We have f2_x_iter (shape K+1), but only f1 at y_k and x_K
        # Recompute f1 at all x_k points from x_iter
        def f1_shifted(x):
            residual = A_jax @ (x + x_opt) - b
            return 0.5 * jnp.sum(residual ** 2) - f_opt

        # Compute f1 at each x_k
        f1_x_vals = jnp.array([f1_shifted(x_iter[:, k]) for k in range(K + 1)])
        f_iter = f1_x_vals + f2_x_iter

    # f_iter has indices 0, 1, ..., K corresponding to f(x_0)-f_opt, ..., f(x_K)-f_opt
    # We want iterations 1 to K
    return f_iter[1:K+1]  # Shape (K,)


def compute_loss_trajectory(stepsizes, A_jax, b_batch, x0_batch, x_opt_batch, f_opt_batch, lambd, K, alg):
    """
    Compute loss trajectory at each iteration k = 1 to K for all problems.
    Returns array of shape (N, K) with loss at each iteration.
    """
    N = b_batch.shape[0]
    losses = []
    for i in range(N):
        loss_traj = compute_loss_trajectory_single(
            stepsizes, A_jax, b_batch[i], x0_batch[i], x_opt_batch[i], f_opt_batch[i], lambd, K, alg
        )
        losses.append(loss_traj)
    return jnp.array(losses)  # (N, K)


def compute_per_problem_losses(stepsizes, A_jax, b_batch, x0_batch, x_opt_batch, f_opt_batch, lambd, K, alg, pep_obj):
    """Compute FINAL loss for each problem in the batch (for frac_solved plot)."""
    # Select trajectory function
    if alg == 'ista':
        traj_fn = problem_data_to_ista_trajectories
        traj_stepsizes = stepsizes[0] if isinstance(stepsizes, tuple) else stepsizes
    else:  # fista
        traj_fn = problem_data_to_fista_trajectories
        traj_stepsizes = stepsizes

    N = b_batch.shape[0]
    pep_objs = []

    for i in range(N):
        traj_result = traj_fn(
            traj_stepsizes, A_jax, b_batch[i], x0_batch[i],
            x_opt_batch[i], f_opt_batch[i], lambd, K,
            return_Gram_representation=False
        )
        x_iter = traj_result[0]  # (n, K+1)
        x_K_shifted = x_iter[:, -1]
        x_K = x_K_shifted + x_opt_batch[i]

        # Compute PEP objective
        if pep_obj == 'obj_val':
            f1_xK = 0.5 * jnp.sum((A_jax @ x_K - b_batch[i]) ** 2)
            f2_xK = lambd * jnp.sum(jnp.abs(x_K))
            obj_val = float(f1_xK + f2_xK - f_opt_batch[i])
        elif pep_obj == 'opt_dist_sq_norm':
            obj_val = float(jnp.sum((x_K - x_opt_batch[i]) ** 2))
        elif pep_obj == 'grad_sq_norm':
            g_f1 = A_jax.T @ (A_jax @ x_K - b_batch[i])
            h_f2 = lambd * jnp.sign(x_K)
            obj_val = float(jnp.sum((g_f1 + h_f2) ** 2))
        else:
            raise ValueError(f"Unknown pep_obj: {pep_obj}")

        pep_objs.append(obj_val)

    return jnp.array(pep_objs)


def compute_risk(losses, risk_type, alpha):
    """Compute risk (expectation or CVaR) from per-problem losses."""
    if risk_type == 'expectation':
        return float(jnp.mean(losses))
    elif risk_type == 'cvar':
        N = losses.shape[0]
        k = max(int(np.ceil(alpha * N)), 1)
        sorted_objs = jnp.sort(losses)[::-1]
        return float(jnp.mean(sorted_objs[:k]))
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")


def compute_risk_trajectory(loss_trajectories, risk_type, alpha):
    """
    Compute risk at each iteration from per-problem loss trajectories.

    Args:
        loss_trajectories: (N, K) array of losses
        risk_type: 'expectation' or 'cvar'
        alpha: CVaR level

    Returns:
        (K,) array of risk values at each iteration
    """
    N, K = loss_trajectories.shape
    risks = []
    for k in range(K):
        losses_at_k = loss_trajectories[:, k]
        risk_k = compute_risk(losses_at_k, risk_type, alpha)
        risks.append(risk_k)
    return risks


def compute_frac_solved(losses, eta_vals):
    """Compute fraction of problems solved at each tolerance level."""
    N = losses.shape[0]
    fractions = []
    for eta in eta_vals:
        num_solved = jnp.sum(losses <= eta)
        fractions.append(float(num_solved) / N)
    return fractions


# ==================== Plot 1: Stepsize Schedule ====================

def create_stepsize_plot(k_output_dir, K, alg, stepsize_type, available_archs, stepsizes_dict):
    """Create stepsize schedule plot with separate subplots for each learning scheme."""
    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    # Check if we have beta values (FISTA)
    has_beta = len(stepsizes_dict[available_archs[0]]) > 1
    n_archs = len(available_archs)

    # Create figure with subplots - one column per architecture
    if has_beta:
        # 2 rows (gamma and beta) x n_archs columns
        fig, axes = plt.subplots(2, n_archs, figsize=(5 * n_archs, 8))
        param_names = ['gamma', 'beta']
        # Ensure axes is 2D even if n_archs == 1
        if n_archs == 1:
            axes = axes.reshape(2, 1)
    else:
        # 1 row x n_archs columns
        fig, axes = plt.subplots(1, n_archs, figsize=(5 * n_archs, 4))
        param_names = ['gamma']
        # Ensure axes is 2D for consistent indexing
        axes = axes.reshape(1, n_archs) if n_archs > 1 else np.array([[axes]])

    # Bar positioning
    x = np.arange(1, K + 1)
    bar_width = 0.6

    for param_idx, param_name in enumerate(param_names):
        for arch_idx, arch in enumerate(available_archs):
            ax = axes[param_idx, arch_idx]
            stepsizes = stepsizes_dict[arch]

            if param_name == 'gamma':
                gamma = stepsizes[0]
                # Handle scalar vs vector gamma
                if jnp.ndim(gamma) == 0:
                    vals = np.full(K, float(gamma))
                else:
                    vals = np.array(gamma)
            else:  # beta
                beta = stepsizes[1]
                # Beta has K+1 values, plot first K
                vals = np.array(beta[:K])

            display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
            ax.bar(x, vals, bar_width, label=display_name, color=colors[arch])

            ax.set_xlabel('Iteration k')
            if arch_idx == 0:  # Only leftmost subplot gets y-label
                ax.set_ylabel(f'Stepsize ({param_name})')

            # Title: show architecture name and parameter type
            if has_beta:
                ax.set_title(f'{display_name} - {param_name} (K={K})')
            else:
                ax.set_title(f'{display_name} (K={K})')

            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(x)

    plt.tight_layout()
    return fig


def save_stepsize_data(k_output_dir, K, available_archs, stepsizes_dict):
    """Save stepsize data to CSV."""
    data = {'k': list(range(1, K + 1))}

    for arch in available_archs:
        stepsizes = stepsizes_dict[arch]
        gamma = stepsizes[0]
        has_beta = len(stepsizes) > 1

        if jnp.ndim(gamma) == 0:
            gamma_vals = [float(gamma)] * K
        else:
            gamma_vals = [float(v) for v in gamma]

        data[f'{arch}_gamma'] = gamma_vals

        if has_beta:
            beta = stepsizes[1]
            # Beta has K+1 values, save first K
            data[f'{arch}_beta'] = [float(v) for v in beta[:K]]

    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'stepsize_schedule_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


# ==================== Plot 2: Out-of-Sample Loss Trajectory ====================

def create_loss_plot(k_output_dir, K, available_archs, loss_traj_dict, risk_type):
    """Create out-of-sample loss trajectory plot (line plot over iterations)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, K + 1)

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    for arch in available_archs:
        losses = loss_traj_dict[arch]  # List of K values
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(x, losses, marker='o', label=display_name, color=colors[arch], markersize=4)

    ax.set_xlabel('Iteration k')
    ax.set_ylabel(f'Test Loss ({risk_type})')
    ax.set_title(f'Out-of-Sample Test Loss Over Iterations (K={K})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_loss_data(k_output_dir, K, available_archs, loss_traj_dict):
    """Save loss trajectory data to CSV."""
    data = {'k': list(range(1, K + 1))}
    for arch in available_archs:
        data[arch] = loss_traj_dict[arch]
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'out_of_sample_loss_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def load_loss_data(k_output_dir):
    """Load loss trajectory data from CSV."""
    csv_path = k_output_dir / 'out_of_sample_loss_data.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    archs = [col for col in df.columns if col != 'k']
    return {arch: df[arch].tolist() for arch in archs}


# ==================== Plot 3: Out-of-Distribution Loss Trajectory ====================

def create_ood_loss_plot(k_output_dir, K, available_archs, ood_loss_traj_dict, risk_type):
    """Create out-of-distribution loss trajectory plot (line plot over iterations)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, K + 1)

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    for arch in available_archs:
        losses = ood_loss_traj_dict[arch]  # List of K values
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(x, losses, marker='o', label=display_name, color=colors[arch], markersize=4)

    ax.set_xlabel('Iteration k')
    ax.set_ylabel(f'OOD Test Loss ({risk_type})')
    ax.set_title(f'Out-of-Distribution Test Loss Over Iterations (K={K})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_ood_loss_data(k_output_dir, K, available_archs, ood_loss_traj_dict):
    """Save out-of-distribution loss trajectory data to CSV."""
    data = {'k': list(range(1, K + 1))}
    for arch in available_archs:
        data[arch] = ood_loss_traj_dict[arch]
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'out_of_distribution_loss_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def load_ood_loss_data(k_output_dir):
    """Load out-of-distribution loss trajectory data from CSV."""
    csv_path = k_output_dir / 'out_of_distribution_loss_data.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    archs = [col for col in df.columns if col != 'k']
    return {arch: df[arch].tolist() for arch in archs}


# ==================== Plot 4: Fraction Problems Solved ====================

def create_frac_solved_plot(k_output_dir, K, available_archs, frac_solved_dict):
    """Create fraction of problems solved plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    for arch in available_archs:
        fracs = frac_solved_dict[arch]
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(eta_vals, fracs, marker='o', label=display_name, color=colors[arch], markersize=5)

    ax.set_xlabel('Tolerance (eta)')
    ax.set_ylabel('Fraction of Problems Solved')
    ax.set_title(f'Fraction of Problems Solved (K={K})')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def save_frac_solved_data(k_output_dir, available_archs, frac_solved_dict):
    """Save fraction solved data to CSV."""
    data = {'eta': eta_vals}
    for arch in available_archs:
        data[arch] = frac_solved_dict[arch]
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'frac_problems_solved_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def load_frac_solved_data(k_output_dir):
    """Load fraction solved data from CSV."""
    csv_path = k_output_dir / 'frac_problems_solved_data.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    archs = [col for col in df.columns if col != 'eta']
    return {arch: df[arch].tolist() for arch in archs}


# ==================== Plot 5: OOD Fraction Problems Solved ====================

def create_ood_frac_solved_plot(k_output_dir, K, available_archs, ood_frac_solved_dict):
    """Create out-of-distribution fraction of problems solved plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    for arch in available_archs:
        fracs = ood_frac_solved_dict[arch]
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(ood_eta_vals, fracs, marker='o', label=display_name, color=colors[arch], markersize=5)

    ax.set_xlabel('Tolerance (eta)')
    ax.set_ylabel('Fraction of OOD Problems Solved')
    ax.set_title(f'Fraction of Out-of-Distribution Problems Solved (K={K})')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def save_ood_frac_solved_data(k_output_dir, available_archs, ood_frac_solved_dict):
    """Save OOD fraction solved data to CSV."""
    data = {'eta': ood_eta_vals}
    for arch in available_archs:
        data[arch] = ood_frac_solved_dict[arch]
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'out_of_dist_frac_problems_solved_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def load_ood_frac_solved_data(k_output_dir):
    """Load OOD fraction solved data from CSV."""
    csv_path = k_output_dir / 'out_of_dist_frac_problems_solved_data.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    archs = [col for col in df.columns if col != 'eta']
    return {arch: df[arch].tolist() for arch in archs}


# ==================== Plot 6: In-Sample Loss Trajectory ====================

def create_in_sample_loss_plot(k_output_dir, K, available_archs, in_sample_loss_traj_dict, risk_type):
    """Create in-sample loss trajectory plot (line plot over iterations)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, K + 1)

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    for arch in available_archs:
        losses = in_sample_loss_traj_dict[arch]  # List of K values
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(x, losses, marker='o', label=display_name, color=colors[arch], markersize=4)

    ax.set_xlabel('Iteration k')
    ax.set_ylabel(f'In-Sample Loss ({risk_type})')
    ax.set_title(f'In-Sample Training Loss Over Iterations (K={K})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_in_sample_loss_data(k_output_dir, K, available_archs, in_sample_loss_traj_dict):
    """Save in-sample loss trajectory data to CSV."""
    data = {'k': list(range(1, K + 1))}
    for arch in available_archs:
        data[arch] = in_sample_loss_traj_dict[arch]
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'in_sample_loss_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def load_in_sample_loss_data(k_output_dir):
    """Load in-sample loss trajectory data from CSV."""
    csv_path = k_output_dir / 'in_sample_loss_data.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    archs = [col for col in df.columns if col != 'k']
    return {arch: df[arch].tolist() for arch in archs}


# ==================== Plot 7: In-Sample Fraction Problems Solved ====================

def create_in_sample_frac_solved_plot(k_output_dir, K, available_archs, in_sample_frac_solved_dict):
    """Create in-sample fraction of problems solved plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green', 'l2o_alista': 'tab:red'}

    for arch in available_archs:
        fracs = in_sample_frac_solved_dict[arch]
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(eta_vals, fracs, marker='o', label=display_name, color=colors[arch], markersize=5)

    ax.set_xlabel('Tolerance (eta)')
    ax.set_ylabel('Fraction of Problems Solved')
    ax.set_title(f'In-Sample Fraction of Problems Solved (K={K})')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def save_in_sample_frac_solved_data(k_output_dir, available_archs, in_sample_frac_solved_dict):
    """Save in-sample fraction solved data to CSV."""
    data = {'eta': eta_vals}
    for arch in available_archs:
        data[arch] = in_sample_frac_solved_dict[arch]
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'in_sample_frac_problems_solved_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def load_in_sample_frac_solved_data(k_output_dir):
    """Load in-sample fraction solved data from CSV."""
    csv_path = k_output_dir / 'in_sample_frac_problems_solved_data.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    archs = [col for col in df.columns if col != 'eta']
    return {arch: df[arch].tolist() for arch in archs}


# ==================== Main Processing ====================

def process_k_value(k_output_dir, config_dict, K, A_test_jax, b_test_batch, x_opt_test_batch, f_opt_test_batch,
                    A_ood_jax, b_ood_batch, x_opt_ood_batch, f_opt_ood_batch, lambd, recompute):
    """Process a single K value: create all seven plots."""
    alg = config_dict['alg']
    dro_obj = config_dict.get('dro_obj', 'expectation')
    alpha = config_dict.get('alpha', 0.1)
    stepsize_type = 'vector'  # Always use vector stepsizes
    pep_obj = config_dict.get('pep_obj', 'obj_val')

    lasso_dir = Path(__file__).parent

    # Check which architectures have best stepsize files and extract source directories
    available_archs = []
    stepsizes_dict = {}
    source_dirs_dict = {}

    for arch in ARCH_TO_CSV.keys():
        stepsizes, source_dir = load_best_stepsize(k_output_dir, arch, alg, stepsize_type, K)
        if stepsizes is not None:
            available_archs.append(arch)
            stepsizes_dict[arch] = stepsizes
            source_dirs_dict[arch] = source_dir
        else:
            print(f"        {ARCH_TO_CSV[arch]} not found, skipping {arch}")

    if not available_archs:
        print(f"      No best_stepsize_schedule files found, skipping K={K}")
        return

    print(f"      Available architectures: {available_archs}")

    # Generate x0_batch for test data (initial points at zero in shifted coordinates)
    N_test = b_test_batch.shape[0]
    n = A_test_jax.shape[1]
    x0_test_batch = jnp.zeros((N_test, n))

    # Generate x0_batch for OOD test data
    N_ood = b_ood_batch.shape[0]
    x0_ood_batch = jnp.zeros((N_ood, n))

    # ===== Plot 1: Stepsize Schedule =====
    print(f"      Creating stepsize_schedule plot...")
    save_stepsize_data(k_output_dir, K, available_archs, stepsizes_dict)
    fig1 = create_stepsize_plot(k_output_dir, K, alg, stepsize_type, available_archs, stepsizes_dict)
    fig1.savefig(k_output_dir / 'stepsize_schedule.pdf', bbox_inches='tight')
    plt.close(fig1)

    # ===== Plot 2: Out-of-Sample Loss Trajectory =====
    loss_data_path = k_output_dir / 'out_of_sample_loss_data.csv'
    if not recompute and loss_data_path.exists():
        print(f"      Loading cached out_of_sample_loss data...")
        loss_traj_dict = load_loss_data(k_output_dir)
        # Filter to only available archs
        loss_traj_dict = {k: v for k, v in loss_traj_dict.items() if k in available_archs}
    else:
        print(f"      Computing loss trajectory on test data...")
        loss_traj_dict = {}
        for arch in available_archs:
            stepsizes = stepsizes_dict[arch]
            # Compute loss at each iteration k = 1 to K
            loss_trajectories = compute_loss_trajectory(
                stepsizes, A_test_jax, b_test_batch, x0_test_batch, x_opt_test_batch, f_opt_test_batch, lambd, K, alg
            )  # Shape (N, K)
            # Compute risk (expectation or cvar) at each iteration
            risk_traj = compute_risk_trajectory(loss_trajectories, dro_obj, alpha)
            loss_traj_dict[arch] = risk_traj
            print(f"        {arch}: final={risk_traj[-1]:.6f}")
        save_loss_data(k_output_dir, K, available_archs, loss_traj_dict)

    fig2 = create_loss_plot(k_output_dir, K, available_archs, loss_traj_dict, dro_obj)
    fig2.savefig(k_output_dir / 'out_of_sample_loss.pdf', bbox_inches='tight')
    plt.close(fig2)

    # ===== Plot 3: Fraction Problems Solved =====
    frac_data_path = k_output_dir / 'frac_problems_solved_data.csv'
    if not recompute and frac_data_path.exists():
        print(f"      Loading cached frac_problems_solved data...")
        frac_solved_dict = load_frac_solved_data(k_output_dir)
        frac_solved_dict = {k: v for k, v in frac_solved_dict.items() if k in available_archs}
    else:
        print(f"      Computing frac_problems_solved on test data...")
        frac_solved_dict = {}
        for arch in available_archs:
            stepsizes = stepsizes_dict[arch]
            per_problem_losses = compute_per_problem_losses(
                stepsizes, A_test_jax, b_test_batch, x0_test_batch, x_opt_test_batch, f_opt_test_batch, lambd, K, alg, pep_obj
            )
            fracs = compute_frac_solved(per_problem_losses, eta_vals)
            frac_solved_dict[arch] = fracs
            print(f"        {arch}: {fracs}")
        save_frac_solved_data(k_output_dir, available_archs, frac_solved_dict)

    fig3 = create_frac_solved_plot(k_output_dir, K, available_archs, frac_solved_dict)
    fig3.savefig(k_output_dir / 'frac_problems_solved.pdf', bbox_inches='tight')
    plt.close(fig3)

    # ===== Plot 4: Out-of-Distribution Loss Trajectory =====
    if A_ood_jax is not None:
        ood_loss_data_path = k_output_dir / 'out_of_distribution_loss_data.csv'
        if not recompute and ood_loss_data_path.exists():
            print(f"      Loading cached out_of_distribution_loss data...")
            ood_loss_traj_dict = load_ood_loss_data(k_output_dir)
            # Filter to only available archs
            ood_loss_traj_dict = {k: v for k, v in ood_loss_traj_dict.items() if k in available_archs}
        else:
            print(f"      Computing loss trajectory on OOD test data...")
            ood_loss_traj_dict = {}
            for arch in available_archs:
                stepsizes = stepsizes_dict[arch]
                # Compute loss at each iteration k = 1 to K on OOD data
                ood_loss_trajectories = compute_loss_trajectory(
                    stepsizes, A_ood_jax, b_ood_batch, x0_ood_batch, x_opt_ood_batch, f_opt_ood_batch, lambd, K, alg
                )  # Shape (N, K)
                # Compute risk (expectation or cvar) at each iteration
                ood_risk_traj = compute_risk_trajectory(ood_loss_trajectories, dro_obj, alpha)
                ood_loss_traj_dict[arch] = ood_risk_traj
                print(f"        {arch}: final={ood_risk_traj[-1]:.6f}")
            save_ood_loss_data(k_output_dir, K, available_archs, ood_loss_traj_dict)

        fig4 = create_ood_loss_plot(k_output_dir, K, available_archs, ood_loss_traj_dict, dro_obj)
        fig4.savefig(k_output_dir / 'out_of_distribution_loss.pdf', bbox_inches='tight')
        plt.close(fig4)
    else:
        print(f"      Skipping OOD loss plot (no OOD data available)")

    # ===== Plot 5: Out-of-Distribution Fraction Problems Solved =====
    if A_ood_jax is not None:
        ood_frac_data_path = k_output_dir / 'out_of_dist_frac_problems_solved_data.csv'
        if not recompute and ood_frac_data_path.exists():
            print(f"      Loading cached out_of_dist_frac_problems_solved data...")
            ood_frac_solved_dict = load_ood_frac_solved_data(k_output_dir)
            ood_frac_solved_dict = {k: v for k, v in ood_frac_solved_dict.items() if k in available_archs}
        else:
            print(f"      Computing frac_problems_solved on OOD test data...")
            ood_frac_solved_dict = {}
            for arch in available_archs:
                stepsizes = stepsizes_dict[arch]
                per_problem_losses = compute_per_problem_losses(
                    stepsizes, A_ood_jax, b_ood_batch, x0_ood_batch, x_opt_ood_batch, f_opt_ood_batch, lambd, K, alg, pep_obj
                )
                fracs = compute_frac_solved(per_problem_losses, ood_eta_vals)
                ood_frac_solved_dict[arch] = fracs
                print(f"        {arch}: {fracs}")
            save_ood_frac_solved_data(k_output_dir, available_archs, ood_frac_solved_dict)

        fig5 = create_ood_frac_solved_plot(k_output_dir, K, available_archs, ood_frac_solved_dict)
        fig5.savefig(k_output_dir / 'out_of_dist_frac_problems_solved.pdf', bbox_inches='tight')
        plt.close(fig5)
    else:
        print(f"      Skipping OOD frac_problems_solved plot (no OOD data available)")

    # ===== Plot 6 & 7: In-Sample Loss and Fraction Problems Solved =====
    # Load in-sample data for each architecture from their respective source directories
    # Special case: LPEP uses L2O's training data
    in_sample_data_dict = {}
    for arch in available_archs:
        source_dir = source_dirs_dict.get(arch)

        # LPEP fallback: use L2O's source directory
        if arch == 'lpep' and (not source_dir or source_dir == ''):
            if 'l2o' in source_dirs_dict:
                source_dir = source_dirs_dict['l2o']
                print(f"      [{arch}] Using L2O's training data (source_dir from l2o)")

        if source_dir:
            try:
                A_in, b_in, x0_in, x_opt_in, f_opt_in, lambd_in = load_in_sample_data(source_dir, K)
                in_sample_data_dict[arch] = (A_in, b_in, x0_in, x_opt_in, f_opt_in)
                print(f"      [{arch}] Loaded {b_in.shape[0]} in-sample training examples from {source_dir}")
            except Exception as e:
                print(f"      [{arch}] Warning: Failed to load in-sample data from {source_dir}: {e}")
                # LPEP fallback on load failure
                if arch == 'lpep' and 'l2o' in source_dirs_dict and source_dirs_dict['l2o'] != source_dir:
                    try:
                        print(f"      [{arch}] Attempting to use L2O's training data as fallback...")
                        l2o_source_dir = source_dirs_dict['l2o']
                        A_in, b_in, x0_in, x_opt_in, f_opt_in, lambd_in = load_in_sample_data(l2o_source_dir, K)
                        in_sample_data_dict[arch] = (A_in, b_in, x0_in, x_opt_in, f_opt_in)
                        print(f"      [{arch}] Loaded {b_in.shape[0]} in-sample training examples from {l2o_source_dir} (L2O fallback)")
                    except Exception as e2:
                        print(f"      [{arch}] L2O fallback also failed: {e2}")
        else:
            print(f"      [{arch}] Warning: No source_dir found in CSV")

    if in_sample_data_dict:
        # Plot 6: In-Sample Loss Trajectory
        in_sample_loss_data_path = k_output_dir / 'in_sample_loss_data.csv'
        if not recompute and in_sample_loss_data_path.exists():
            print(f"      Loading cached in_sample_loss data...")
            in_sample_loss_traj_dict = load_in_sample_loss_data(k_output_dir)
            in_sample_loss_traj_dict = {k: v for k, v in in_sample_loss_traj_dict.items() if k in in_sample_data_dict}
        else:
            print(f"      Computing loss trajectory on in-sample training data...")
            in_sample_loss_traj_dict = {}
            for arch in in_sample_data_dict.keys():
                A_in, b_in, x0_in, x_opt_in, f_opt_in = in_sample_data_dict[arch]
                stepsizes = stepsizes_dict[arch]
                # Compute loss at each iteration k = 1 to K
                loss_trajectories = compute_loss_trajectory(
                    stepsizes, A_in, b_in, x0_in, x_opt_in, f_opt_in, lambd, K, alg
                )  # Shape (N, K)
                # Compute risk (expectation or cvar) at each iteration
                risk_traj = compute_risk_trajectory(loss_trajectories, dro_obj, alpha)
                in_sample_loss_traj_dict[arch] = risk_traj
                print(f"        {arch}: final={risk_traj[-1]:.6f}")
            save_in_sample_loss_data(k_output_dir, K, list(in_sample_loss_traj_dict.keys()), in_sample_loss_traj_dict)

        fig6 = create_in_sample_loss_plot(k_output_dir, K, list(in_sample_loss_traj_dict.keys()), in_sample_loss_traj_dict, dro_obj)
        fig6.savefig(k_output_dir / 'in_sample_loss.pdf', bbox_inches='tight')
        plt.close(fig6)

        # Plot 7: In-Sample Fraction Problems Solved
        in_sample_frac_data_path = k_output_dir / 'in_sample_frac_problems_solved_data.csv'
        if not recompute and in_sample_frac_data_path.exists():
            print(f"      Loading cached in_sample_frac_problems_solved data...")
            in_sample_frac_solved_dict = load_in_sample_frac_solved_data(k_output_dir)
            in_sample_frac_solved_dict = {k: v for k, v in in_sample_frac_solved_dict.items() if k in in_sample_data_dict}
        else:
            print(f"      Computing frac_problems_solved on in-sample training data...")
            in_sample_frac_solved_dict = {}
            for arch in in_sample_data_dict.keys():
                A_in, b_in, x0_in, x_opt_in, f_opt_in = in_sample_data_dict[arch]
                stepsizes = stepsizes_dict[arch]
                per_problem_losses = compute_per_problem_losses(
                    stepsizes, A_in, b_in, x0_in, x_opt_in, f_opt_in, lambd, K, alg, pep_obj
                )
                fracs = compute_frac_solved(per_problem_losses, eta_vals)
                in_sample_frac_solved_dict[arch] = fracs
                print(f"        {arch}: {fracs}")
            save_in_sample_frac_solved_data(k_output_dir, list(in_sample_frac_solved_dict.keys()), in_sample_frac_solved_dict)

        fig7 = create_in_sample_frac_solved_plot(k_output_dir, K, list(in_sample_frac_solved_dict.keys()), in_sample_frac_solved_dict)
        fig7.savefig(k_output_dir / 'in_sample_frac_problems_solved.pdf', bbox_inches='tight')
        plt.close(fig7)
    else:
        print(f"      Skipping in-sample plots (no in-sample data available)")

    print(f"      Saved all plots to {k_output_dir.relative_to(lasso_dir)}")


def process_configuration(config_list, recompute):
    """Process a single configuration: create plots for each K."""
    config_dict = parse_config_list_to_dict(config_list)

    alg = config_dict['alg']
    dro_obj = config_dict.get('dro_obj', 'expectation')
    pep_obj = config_dict.get('pep_obj', 'obj_val')
    stepsize_type = 'vector'  # Always use vector stepsizes

    lasso_dir = Path(__file__).parent

    print(f"\n{'='*60}")
    print(f"Config: {config_list}")
    print(f"  alg={alg}, dro_obj={dro_obj}, pep_obj={pep_obj}, stepsize_type={stepsize_type}")

    # Get leaf directory
    leaf_path = get_leaf_path(config_list)

    if not leaf_path.exists():
        print(f"  Leaf directory does not exist, skipping.")
        return

    # Load test data (used for loss and frac_solved plots)
    print(f"  Loading out-of-sample TEST data...")
    try:
        A_test_jax, b_test_batch, x_opt_test_batch, f_opt_test_batch, lambd, metadata = load_test_data()
        print(f"  Loaded {b_test_batch.shape[0]} test samples, A shape={A_test_jax.shape}, lambd={lambd}")
    except FileNotFoundError as e:
        print(f"  Test data not found: {e}")
        print(f"  Skipping this configuration.")
        return

    # Load out-of-distribution test data
    print(f"  Loading out-of-distribution TEST data...")
    try:
        A_ood_jax, b_ood_batch, x_opt_ood_batch, f_opt_ood_batch, lambd_ood = load_ood_test_data()
        print(f"  Loaded {b_ood_batch.shape[0]} OOD test samples, A shape={A_ood_jax.shape}")
    except FileNotFoundError as e:
        print(f"  OOD test data not found: {e}")
        print(f"  Will skip OOD plot for this configuration.")
        A_ood_jax = b_ood_batch = x_opt_ood_batch = f_opt_ood_batch = None

    # Process each K value
    for K in K_vals:
        k_output_dir = leaf_path / f'K_{K}'
        if not k_output_dir.exists():
            print(f"\n    K={K}: Directory does not exist, skipping.")
            continue

        print(f"\n    K={K}:")
        process_k_value(k_output_dir, config_dict, K, A_test_jax, b_test_batch, x_opt_test_batch, f_opt_test_batch,
                       A_ood_jax, b_ood_batch, x_opt_ood_batch, f_opt_ood_batch, lambd, recompute)


def main():
    parser = argparse.ArgumentParser(description='Create test set plots for stepsize comparison.')
    parser.add_argument('-recompute', action='store_true',
                        help='Force recompute all data even if cached CSVs exist')
    args = parser.parse_args()

    print("="*60)
    print("Test Set Plotting (problem_instances/)")
    print("="*60)
    print(f"Total configurations: {len(ALL_LASSO_CONFIGS)}")
    print(f"K values: {K_vals}")
    print(f"Recompute: {args.recompute}")

    for i, config in enumerate(ALL_LASSO_CONFIGS):
        print(f"\n[{i+1}/{len(ALL_LASSO_CONFIGS)}] Processing config...")
        process_configuration(config, args.recompute)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
