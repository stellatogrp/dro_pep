"""
Test Set Plotting for LDRO-PEP, L2O, and LPEP (WC-PEP) Experiments.

For each configuration and K value, creates five plots:
1. stepsize_schedule: Shows the stepsize (and beta for FGM) over iterations
2. out_of_sample_loss: Evaluates loss on test set for all methods
3. frac_problems_solved: Fraction of problems solved at each tolerance level
4. out_of_distribution_loss: Evaluates loss on out-of-distribution test set
5. out_of_dist_frac_problems_solved: Fraction of OOD problems solved at each tolerance level

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

from learning.trajectories.gd_fgm import (
    problem_data_to_gd_trajectories,
    problem_data_to_nesterov_fgm_trajectories,
    problem_data_to_pep_obj,
)
from data_scrape import configs, ALL_QUAD_CONFIGS, get_leaf_path

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
}

# Display names for legends
ARCH_DISPLAY_NAMES = {
    'l2o': 'L2O',
    'ldro_pep': 'LDRO-PEP',
    'lpep': 'WC-PEP',
}


# ==================== Config Parsing ====================

def parse_config_list_to_dict(config_list):
    """Convert config list like ['alg=vanilla_gd', 'mu=0'] to dict."""
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
    """Convert 'alg=vanilla_gd' to 'alg_vanilla_gd'."""
    return option_str.replace('=', '_')


# ==================== Data Loading ====================

def load_test_data():
    """Load Q and z0 samples from TEST data."""
    quad_dir = Path(__file__).parent
    problem_dir = quad_dir / 'problem_instances'

    Q_data = np.load(problem_dir / 'Q_test_samples.npz')
    z0_data = np.load(problem_dir / 'z0_test_samples.npz')

    Q_batch = jnp.array(Q_data['Q'])
    z0_batch = jnp.array(z0_data['z0'])

    # Optimal point is zero, optimal value is zero for quadratics
    N = Q_batch.shape[0]
    M = Q_batch.shape[1]
    zs_batch = jnp.zeros((N, M))
    fs_batch = jnp.zeros(N)

    return Q_batch, z0_batch, zs_batch, fs_batch


def load_ood_test_data():
    """Load Q and z0 samples from OUT-OF-DISTRIBUTION TEST data."""
    quad_dir = Path(__file__).parent
    problem_dir = quad_dir / 'problem_instances'

    Q_data = np.load(problem_dir / 'Q_out_of_dist_samples.npz')
    z0_data = np.load(problem_dir / 'z0_out_of_dist_samples.npz')

    Q_batch = jnp.array(Q_data['Q'])
    z0_batch = jnp.array(z0_data['z0'])

    # Optimal point is zero, optimal value is zero for quadratics
    N = Q_batch.shape[0]
    M = Q_batch.shape[1]
    zs_batch = jnp.zeros((N, M))
    fs_batch = jnp.zeros(N)

    return Q_batch, z0_batch, zs_batch, fs_batch


def load_in_sample_data(source_dir, K):
    """Reconstruct a run's training set from `problem_instances/training_set.npz`.

    The trainer no longer dumps a per-run `training_set.npz`. Instead it loads
    rows from a centralized `problem_instances/training_set.npz` and subsamples
    `training_sample_N` indices via `np.random.default_rng(training_seed)`
    (mirroring `_load_and_subsample_quad` in `learning_experiment_classes/quad.py`).
    Both keys live in the run's `.hydra/config.yaml`.
    """
    del K  # training_set is K-independent
    quad_dir = Path(__file__).parent

    cfg_path = quad_dir / source_dir / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    N = int(cfg.training_sample_N)
    seed = int(cfg.training_seed)

    npz_path = quad_dir / 'problem_instances' / 'training_set.npz'
    d = np.load(npz_path)
    total = int(d['Q_batch'].shape[0])

    if N >= total:
        idx = np.arange(total)
    else:
        idx = np.random.default_rng(seed).choice(total, size=N, replace=False)

    return (
        jnp.asarray(d['Q_batch'][idx]),
        jnp.asarray(d['z0_batch'][idx]),
        jnp.asarray(d['zs_batch'][idx]),
        jnp.asarray(d['fs_batch'][idx]),
    )


def load_best_stepsize(k_output_dir, arch, alg, stepsize_type, K):
    """Load best stepsize schedule from CSV and extract source directory."""
    csv_path = k_output_dir / ARCH_TO_CSV[arch]
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    row = df.iloc[0]  # Single row in best stepsize file

    # Extract source directory if present
    source_dir = row.get('source_dir', None)

    has_beta = (alg == 'nesterov_fgm')
    is_vector = (stepsize_type == 'vector')

    # Parse t
    if is_vector:
        t_cols = [f't{i}' for i in range(K)]
        t = jnp.array([row[col] for col in t_cols])
    else:
        t = jnp.array(row['t'])

    # Parse beta if FGM
    if has_beta:
        beta_cols = [f'beta{i}' for i in range(K)]
        beta = jnp.array([row[col] for col in beta_cols])
        return (t, beta), source_dir
    else:
        return (t,), source_dir


# ==================== Evaluation ====================

def compute_loss_trajectory_single_gd(stepsizes, Q, z0, zs, fs, K):
    """
    Compute loss at each iteration k = 1 to K for GD.
    Returns array of shape (K,) with f(z_k) - fs for k = 1, ..., K.
    """
    z_stack, g_stack, f_stack = problem_data_to_gd_trajectories(
        stepsizes, Q, z0, zs, fs, K, return_Gram_representation=False
    )
    # f_stack has indices 0, 1, ..., K corresponding to f(z_0)-fs, ..., f(z_K)-fs
    # We want iterations 1 to K
    return f_stack[1:K+1]  # Shape (K,)


def compute_loss_trajectory_single_fgm(stepsizes, Q, z0, zs, fs, K):
    """
    Compute loss at each iteration k = 1 to K for FGM, evaluated at x iterates.
    
    The FGM trajectory function stores y iterates and g(y) iterates.
    We need to compute x iterates: x_{k} = y_{k-1} - t_{k-1} * g(y_{k-1})
    Then evaluate f(x_k) = 0.5 * x_k @ Q @ x_k + fs
    
    Returns array of shape (K,) with f(x_k) - fs for k = 1, ..., K.
    """
    t, beta = stepsizes
    y_iter, g_stack, f_stack = problem_data_to_nesterov_fgm_trajectories(
        stepsizes, Q, z0, zs, fs, K, return_Gram_representation=False
    )
    
    # y_iter has shape (d, K) with columns y_0, y_1, ..., y_{K-1}
    # g_stack has shape (d, K+1) with g(y_0), ..., g(y_{K-1}), g(x_K)
    # We need to compute x_k = y_{k-1} - t_{k-1} * g(y_{k-1}) for k = 1, ..., K
    
    losses = jnp.zeros(K)
    
    def compute_x_loss(k, losses):
        # x_{k+1} = y_k - t_k * g(y_k), but we're indexing from k=1
        # So x_k = y_{k-1} - t_{k-1} * g(y_{k-1})
        y_prev = y_iter[:, k-1] if k > 0 else z0
        g_prev = g_stack[:, k-1] if k > 0 else Q @ z0
        tk = t[k-1] if t.ndim > 0 else t
        x_k = y_prev - tk * g_prev
        f_x_k = 0.5 * x_k @ Q @ x_k  # f(x_k) - fs (since fs=0 for quadratics)
        return losses.at[k-1].set(f_x_k)
    
    # For k = 1 to K
    for k in range(1, K + 1):
        y_prev = y_iter[:, k-1]
        g_prev = g_stack[:, k-1]
        tk_idx = k - 1
        tk = t[tk_idx] if jnp.ndim(t) > 0 else t
        x_k = y_prev - tk * g_prev
        f_x_k = 0.5 * x_k @ Q @ x_k  # f(x_k) - fs (since optimal point has zs=0, fs=0)
        losses = losses.at[k-1].set(f_x_k)
    
    return losses


def compute_loss_trajectory(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K, alg):
    """
    Compute loss trajectory at each iteration k = 1 to K for all problems.
    Returns array of shape (N, K) with loss at each iteration.
    """
    if alg == 'vanilla_gd':
        traj_fn = compute_loss_trajectory_single_gd
    else:
        traj_fn = compute_loss_trajectory_single_fgm
    
    batch_fn = jax.vmap(
        lambda Q, z0, zs, fs: traj_fn(stepsizes, Q, z0, zs, fs, K),
        in_axes=(0, 0, 0, 0)
    )
    return batch_fn(Q_batch, z0_batch, zs_batch, fs_batch)  # (N, K)


def compute_per_problem_losses(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K, alg, pep_obj):
    """Compute FINAL loss for each problem in the batch (for frac_solved plot)."""
    if alg == 'vanilla_gd':
        traj_fn = problem_data_to_gd_trajectories
    else:
        traj_fn = problem_data_to_nesterov_fgm_trajectories

    batch_pep_obj_func = jax.vmap(
        lambda Q, z0, zs, fs: problem_data_to_pep_obj(
            stepsizes, Q, z0, zs, fs, K, traj_fn, pep_obj
        ),
        in_axes=(0, 0, 0, 0)
    )
    return batch_pep_obj_func(Q_batch, z0_batch, zs_batch, fs_batch)


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
    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

    # Check if we have beta values (FGM)
    has_beta = len(stepsizes_dict[available_archs[0]]) > 1
    n_archs = len(available_archs)

    # Create figure with subplots - one column per architecture
    if has_beta:
        # 2 rows (t and beta) x n_archs columns
        fig, axes = plt.subplots(2, n_archs, figsize=(5 * n_archs, 8))
        param_names = ['t', 'beta']
        # Ensure axes is 2D even if n_archs == 1
        if n_archs == 1:
            axes = axes.reshape(2, 1)
    else:
        # 1 row x n_archs columns
        fig, axes = plt.subplots(1, n_archs, figsize=(5 * n_archs, 4))
        param_names = ['t']
        # Ensure axes is 2D for consistent indexing
        axes = axes.reshape(1, n_archs) if n_archs > 1 else np.array([[axes]])

    # Bar positioning
    x = np.arange(1, K + 1)
    bar_width = 0.6

    for param_idx, param_name in enumerate(param_names):
        for arch_idx, arch in enumerate(available_archs):
            ax = axes[param_idx, arch_idx]
            stepsizes = stepsizes_dict[arch]

            if param_name == 't':
                t = stepsizes[0]
                # Handle scalar vs vector t
                if jnp.ndim(t) == 0:
                    vals = np.full(K, float(t))
                else:
                    vals = np.array(t)
            else:  # beta
                beta = stepsizes[1]
                vals = np.array(beta)

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
        t = stepsizes[0]
        has_beta = len(stepsizes) > 1
        
        if jnp.ndim(t) == 0:
            t_vals = [float(t)] * K
        else:
            t_vals = [float(v) for v in t]
        
        data[f'{arch}_t'] = t_vals
        
        if has_beta:
            beta = stepsizes[1]
            data[f'{arch}_beta'] = [float(v) for v in beta]
    
    df = pd.DataFrame(data)
    csv_path = k_output_dir / 'stepsize_schedule_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


# ==================== Plot 2: Out-of-Sample Loss Trajectory ====================

def create_loss_plot(k_output_dir, K, available_archs, loss_traj_dict, risk_type):
    """Create out-of-sample loss trajectory plot (line plot over iterations)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, K + 1)

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

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

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

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

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

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

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

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

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

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

    colors = {'l2o': 'tab:blue', 'ldro_pep': 'tab:orange', 'lpep': 'tab:green'}

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

def process_k_value(k_output_dir, config_dict, K, Q_batch, z0_batch, zs_batch, fs_batch,
                    Q_ood_batch, z0_ood_batch, zs_ood_batch, fs_ood_batch, recompute):
    """Process a single K value: create all seven plots."""
    alg = config_dict['alg']
    dro_obj = config_dict.get('dro_obj', 'expectation')
    alpha = config_dict.get('alpha', 0.1)
    stepsize_type = 'vector'  # Always use vector stepsizes
    pep_obj = config_dict.get('pep_obj', 'obj_val')

    quad_dir = Path(__file__).parent

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
                stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K, alg
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
                stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K, alg, pep_obj
            )
            fracs = compute_frac_solved(per_problem_losses, eta_vals)
            frac_solved_dict[arch] = fracs
            print(f"        {arch}: {fracs}")
        save_frac_solved_data(k_output_dir, available_archs, frac_solved_dict)
    
    fig3 = create_frac_solved_plot(k_output_dir, K, available_archs, frac_solved_dict)
    fig3.savefig(k_output_dir / 'frac_problems_solved.pdf', bbox_inches='tight')
    plt.close(fig3)

    # ===== Plot 4: Out-of-Distribution Loss Trajectory =====
    if Q_ood_batch is not None:
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
                    stepsizes, Q_ood_batch, z0_ood_batch, zs_ood_batch, fs_ood_batch, K, alg
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
    if Q_ood_batch is not None:
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
                    stepsizes, Q_ood_batch, z0_ood_batch, zs_ood_batch, fs_ood_batch, K, alg, pep_obj
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
                Q_in, z0_in, zs_in, fs_in = load_in_sample_data(source_dir, K)
                in_sample_data_dict[arch] = (Q_in, z0_in, zs_in, fs_in)
                print(f"      [{arch}] Loaded {Q_in.shape[0]} in-sample training examples from {source_dir}")
            except Exception as e:
                print(f"      [{arch}] Warning: Failed to load in-sample data from {source_dir}: {e}")
                # LPEP fallback on load failure
                if arch == 'lpep' and 'l2o' in source_dirs_dict and source_dirs_dict['l2o'] != source_dir:
                    try:
                        print(f"      [{arch}] Attempting to use L2O's training data as fallback...")
                        l2o_source_dir = source_dirs_dict['l2o']
                        Q_in, z0_in, zs_in, fs_in = load_in_sample_data(l2o_source_dir, K)
                        in_sample_data_dict[arch] = (Q_in, z0_in, zs_in, fs_in)
                        print(f"      [{arch}] Loaded {Q_in.shape[0]} in-sample training examples from {l2o_source_dir} (L2O fallback)")
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
                Q_in, z0_in, zs_in, fs_in = in_sample_data_dict[arch]
                stepsizes = stepsizes_dict[arch]
                # Compute loss at each iteration k = 1 to K
                loss_trajectories = compute_loss_trajectory(
                    stepsizes, Q_in, z0_in, zs_in, fs_in, K, alg
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
                Q_in, z0_in, zs_in, fs_in = in_sample_data_dict[arch]
                stepsizes = stepsizes_dict[arch]
                per_problem_losses = compute_per_problem_losses(
                    stepsizes, Q_in, z0_in, zs_in, fs_in, K, alg, pep_obj
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

    print(f"      Saved all plots to {k_output_dir.relative_to(quad_dir)}")


def process_configuration(config_list, recompute):
    """Process a single configuration: create plots for each K."""
    config_dict = parse_config_list_to_dict(config_list)

    alg = config_dict['alg']
    dro_obj = config_dict.get('dro_obj', 'expectation')
    pep_obj = config_dict.get('pep_obj', 'obj_val')
    stepsize_type = 'vector'  # Always use vector stepsizes

    quad_dir = Path(__file__).parent

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
        Q_batch, z0_batch, zs_batch, fs_batch = load_test_data()
        print(f"  Loaded {Q_batch.shape[0]} test samples, dim={Q_batch.shape[1]}")
    except FileNotFoundError as e:
        print(f"  Test data not found: {e}")
        print(f"  Skipping this configuration.")
        return

    # Load out-of-distribution test data
    print(f"  Loading out-of-distribution TEST data...")
    try:
        Q_ood_batch, z0_ood_batch, zs_ood_batch, fs_ood_batch = load_ood_test_data()
        print(f"  Loaded {Q_ood_batch.shape[0]} OOD test samples, dim={Q_ood_batch.shape[1]}")
    except FileNotFoundError as e:
        print(f"  OOD test data not found: {e}")
        print(f"  Will skip OOD plot for this configuration.")
        Q_ood_batch = z0_ood_batch = zs_ood_batch = fs_ood_batch = None

    # Process each K value
    for K in K_vals:
        k_output_dir = leaf_path / f'K_{K}'
        if not k_output_dir.exists():
            print(f"\n    K={K}: Directory does not exist, skipping.")
            continue

        print(f"\n    K={K}:")
        process_k_value(k_output_dir, config_dict, K, Q_batch, z0_batch, zs_batch, fs_batch,
                       Q_ood_batch, z0_ood_batch, zs_ood_batch, fs_ood_batch, recompute)


def main():
    parser = argparse.ArgumentParser(description='Create test set plots for stepsize comparison.')
    parser.add_argument('-recompute', action='store_true', 
                        help='Force recompute all data even if cached CSVs exist')
    args = parser.parse_args()
    
    print("="*60)
    print("Test Set Plotting (out_of_sample_test_data)")
    print("="*60)
    print(f"Total configurations: {len(ALL_QUAD_CONFIGS)}")
    print(f"K values: {K_vals}")
    print(f"Recompute: {args.recompute}")
    
    for i, config in enumerate(ALL_QUAD_CONFIGS):
        print(f"\n[{i+1}/{len(ALL_QUAD_CONFIGS)}] Processing config...")
        process_configuration(config, args.recompute)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()