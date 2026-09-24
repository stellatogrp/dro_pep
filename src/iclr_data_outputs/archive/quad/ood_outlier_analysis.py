"""One-off: per-instance OOD final-iterate loss distribution for ldro_pep, per K.

For each K in K_VALS, loads:
  - the ldro_pep best stepsize schedule from plots/K_{K}/ldro_pep_best_stepsize_schedule.csv
  - the OOD problem set from problem_instances/{Q,z0}_out_of_dist_samples.npz
runs vanilla_gd for K steps on each problem, then prints distribution stats and
the top outliers (highest final-iterate losses) for that K.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

script_dir = Path(__file__).parent
# Consolidated layout: this file lives at
#   iclr_data_outputs/archive/<problem>/ , so src/ is three levels up,
# where it was two in the original experiment_plots_icml/<problem>/.
src_dir = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from learning.trajectories.gd_fgm import problem_data_to_gd_trajectories  # noqa: E402

jax.config.update("jax_enable_x64", True)


K_VALS = list(range(1, 16))
TOP_K_OUTLIERS = 5
PLOTS_DIR = script_dir / 'plots'
PROBLEM_DIR = script_dir / 'problem_instances'


def load_ood_set():
    Q = jnp.array(np.load(PROBLEM_DIR / 'Q_out_of_dist_samples.npz')['Q'])
    z0 = jnp.array(np.load(PROBLEM_DIR / 'z0_out_of_dist_samples.npz')['z0'])
    N, M = Q.shape[0], Q.shape[1]
    zs = jnp.zeros((N, M))
    fs = jnp.zeros(N)
    return Q, z0, zs, fs


def load_ldro_t(K: int):
    csv_path = PLOTS_DIR / f'K_{K}' / 'ldro_pep_best_stepsize_schedule.csv'
    if not csv_path.exists():
        return None
    row = pd.read_csv(csv_path).iloc[0]
    return jnp.array([row[f't{i}'] for i in range(K)])


def per_problem_final_loss(t, Q, z0, zs, fs, K):
    """Return final-iterate (k=K) loss for each problem in the OOD batch."""
    def single(Q, z0, zs, fs):
        _, _, f_stack = problem_data_to_gd_trajectories(
            (t,), Q, z0, zs, fs, K, return_Gram_representation=False
        )
        return f_stack[K]  # f(z_K) - fs (fs=0 for quad)
    return jax.vmap(single, in_axes=(0, 0, 0, 0))(Q, z0, zs, fs)


def describe(losses_np, K):
    arr = np.asarray(losses_np)
    print(f"\n=== K={K} (N={len(arr)}) ===")
    print(f"  mean   = {arr.mean():.6e}")
    print(f"  median = {np.median(arr):.6e}")
    print(f"  std    = {arr.std():.6e}")
    print(f"  min    = {arr.min():.6e}")
    print(f"  max    = {arr.max():.6e}")
    qs = np.quantile(arr, [0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"  q50/75/90/95/99 = "
          f"{qs[0]:.3e} {qs[1]:.3e} {qs[2]:.3e} {qs[3]:.3e} {qs[4]:.3e}")
    top_idx = np.argsort(arr)[-TOP_K_OUTLIERS:][::-1]
    print(f"  top {TOP_K_OUTLIERS} (idx, loss):")
    for i in top_idx:
        print(f"    idx={int(i):4d}  loss={arr[i]:.6e}")
    # Contribution of the top-1 to the mean.
    top1_share = arr[top_idx[0]] / (arr.sum() + 1e-30)
    print(f"  top-1 share of total = {top1_share*100:.2f}%")
    return top_idx


def main():
    Q, z0, zs, fs = load_ood_set()
    print(f"OOD set: Q={Q.shape}, z0={z0.shape}")

    per_K_outliers = {}
    per_K_losses = {}
    for K in K_VALS:
        t = load_ldro_t(K)
        if t is None:
            print(f"K={K}: missing schedule, skipping.")
            continue
        losses = per_problem_final_loss(t, Q, z0, zs, fs, K)
        losses_np = np.asarray(losses)
        per_K_losses[K] = losses_np
        per_K_outliers[K] = describe(losses_np, K)

    # Cross-K outlier recurrence: which problem indices are in the top-5
    # most often across K?
    from collections import Counter
    counter = Counter()
    for K, idxs in per_K_outliers.items():
        for i in idxs:
            counter[int(i)] += 1
    print("\n=== Cross-K outlier recurrence (top-5 hits per K) ===")
    for idx, cnt in counter.most_common(10):
        Ks = [K for K, idxs in per_K_outliers.items() if idx in idxs]
        print(f"  idx={idx:4d}  appears in {cnt}/{len(per_K_outliers)} K's: {Ks}")

    # Save tidy CSV.
    rows = []
    for K, arr in per_K_losses.items():
        for i, v in enumerate(arr):
            rows.append({'K': K, 'problem_idx': i, 'final_loss': float(v)})
    out_csv = script_dir / 'ood_outlier_analysis.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.relative_to(script_dir)}")


if __name__ == '__main__':
    main()
