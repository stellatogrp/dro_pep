"""Pool-wide PEP interpolation probe for the TV-inpainting LP.

For every instance in `training_set.npz` (or any of the cached splits), this
script:
  1. Reconstructs (K_mat, c, q) from (image_index, mask) using the same
     scipy-sparse path as sample creation.
  2. Computes M_actual = ||K_mat||_op, R_actual = ||z0 - z*||.
  3. Runs reference Chambolle-Pock for K_max steps, builds (G, F).
  4. Constructs the production CP PEP at three (M, R) variants:
       a. (M_actual, R_actual)        — tightest possible
       b. (M_val_pool, R_val_pool)    — what training uses
       c. (1.5 * M_actual, 1.5 * R_actual) — slack to confirm direction
  5. Evaluates every constraint group; records max violation per group.

Outputs:
  - CSV file `tv_pep_probe.csv` next to the input dir.
  - Stdout summary table: pass/fail counts per group per (M, R) variant.

Usage:
  python src/tools/probe_tv_inpainting_pep.py \\
      --data_source_dir <sample_creation_outputs/PDLP/.../...> \\
      --K_max 5 \\
      [--max_instances 20] \\
      [--split training]
"""
import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# pylint: disable=import-error
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from learning.pep_constructions import construct_chambolle_pock_pep_data
from learning.tv_inpainting_test import (
    extract_constraint_matrices,
    _vertical_diff_matrix,
    _horizontal_diff_matrix,
)
from learning_experiment_classes.pdlp import build_strict_interior_init
from sklearn.datasets import fetch_olivetti_faces

# Reuse problem-agnostic CP / Gram / constraint-eval helpers.
from tests.test_chambolle_pock_facility_location import (
    run_cp_on_facility,
    build_gram_and_F_facility,
)
from tests.test_chambolle_pock_interpolation import (
    eval_scalar_constraint,
    eval_psd_block,
)

log = logging.getLogger("probe_tv_inpainting_pep")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)


def _split_violations(violations: np.ndarray, K_max: int):
    n_algo = K_max + 1
    n_interp = n_algo * (n_algo + 1)
    n_value_pin = 4
    n_IC = 1
    f1 = violations[:n_interp]
    h = violations[n_interp:2 * n_interp]
    vp = violations[2 * n_interp:2 * n_interp + n_value_pin]
    rest = violations[2 * n_interp + n_value_pin:]
    adj = rest[:-n_IC]
    IC = rest[-n_IC:]
    return f1, h, vp, adj, IC


def reconstruct_lp(image: np.ndarray, mask_flat: np.ndarray,
                   M_img: int, N_img: int, lp_upper: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build (K_mat dense, q, c, m1) and bounds from one cached instance."""
    known_indices = np.flatnonzero(mask_flat.astype(bool))
    known_values = image.reshape(-1)[known_indices]
    mats = extract_constraint_matrices(known_indices, known_values, M_img, N_img)
    K_csr = sp.vstack([mats.G, mats.A], format="csr")
    K_mat = K_csr.toarray().astype(np.float64)
    K = M_img * N_img
    K_v = (M_img - 1) * N_img
    K_h = M_img * (N_img - 1)
    m1 = 2 * K_v + 2 * K_h
    n_vars = K + K_v + K_h
    c = np.concatenate([np.zeros(K), np.ones(K_v), np.ones(K_h)]).astype(np.float64)
    q = np.concatenate([np.zeros(m1), mats.b])
    return K_mat, q, c, m1, n_vars


def evaluate_pep_at(
    K_mat, q, c, l, u, m1,
    v_s, y_s, vs_iters, ys_iters, gf1, gh, w, z,
    tau, sigma, theta, K_max, M, R,
):
    """Evaluate all constraint groups + PSD blocks at the given (M, R)."""
    G, F, _, _ = build_gram_and_F_facility(
        K_mat, q, c, v_s, y_s, vs_iters, ys_iters, gf1, gh, w, z,
        tau, sigma, theta, K_max,
    )
    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=M, R=R, K_max=K_max,
    )
    (_A_obj, _b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, _PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]
    num_scalar = A_vals.shape[0]
    violations = np.zeros(num_scalar)
    for i in range(num_scalar):
        violations[i] = eval_scalar_constraint(
            A_vals[i], b_vals[i], c_vals[i], G, F,
        )
    f1_v, h_v, vp_v, adj_v, IC_v = _split_violations(violations, K_max)

    psd_min_eigs = []
    for idx in range(len(PSD_A_vals)):
        H = eval_psd_block(
            PSD_A_vals[idx], PSD_b_vals[idx], PSD_c_vals[idx], G, F,
        )
        psd_min_eigs.append(float(np.min(np.linalg.eigvalsh(H))))

    return {
        'f1_max': float(np.max(f1_v)),
        'h_max': float(np.max(h_v)),
        'vp_max': float(np.max(vp_v)),
        'adj_max': float(np.max(adj_v)),
        'IC': float(IC_v[0]),
        'psd0_min': psd_min_eigs[0] if len(psd_min_eigs) > 0 else np.nan,
        'psd1_min': psd_min_eigs[1] if len(psd_min_eigs) > 1 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_source_dir", required=True,
                        help="sample-creation output directory")
    parser.add_argument("--K_max", type=int, default=5)
    parser.add_argument("--split", choices=["training", "validation", "test", "ood"],
                        default="training")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="cap to this many instances (debug)")
    parser.add_argument("--out_csv", default=None,
                        help="output CSV path (default: <data_source_dir>/tv_pep_probe.csv)")
    args = parser.parse_args()

    meta = np.load(os.path.join(args.data_source_dir, "out_of_sample_metadata.npz"),
                   allow_pickle=False)
    M_img = int(meta['M']); N_img = int(meta['N']); lp_upper = float(meta['lp_upper'])
    M_val_pool = float(meta['M_val'])
    R_val_pool = float(meta['R_val'])
    log.info(
        f"Loaded metadata: M={M_img} N={N_img} lp_upper={lp_upper}; "
        f"pool M_val={M_val_pool:.4f} R_val={R_val_pool:.4f}"
    )

    split = np.load(
        os.path.join(args.data_source_dir, f"{args.split}_set.npz"), allow_pickle=False,
    )
    image_index_b = split['image_index_batch']
    mask_b = split['mask_batch']
    x_opt_b = split['x_opt_batch']
    y_opt_b = split['y_opt_batch']
    n_total = image_index_b.shape[0]
    n_run = n_total if args.max_instances is None else min(n_total, args.max_instances)
    log.info(f"Probing {n_run}/{n_total} {args.split} instances at K_max={args.K_max}")

    images = fetch_olivetti_faces().images.astype(np.float64) * lp_upper

    # Box bounds, fixed across instances.
    K = M_img * N_img
    K_v = (M_img - 1) * N_img
    K_h = M_img * (N_img - 1)
    n_vars_total = K + K_v + K_h
    l = np.zeros(n_vars_total)
    u = lp_upper * np.ones(n_vars_total)

    out_csv = args.out_csv or os.path.join(args.data_source_dir, "tv_pep_probe.csv")
    rows = []

    for i in range(n_run):
        image = images[int(image_index_b[i])]
        mask_flat = np.asarray(mask_b[i], dtype=bool)
        K_mat, q, c, m1, n_vars = reconstruct_lp(
            image, mask_flat, M_img, N_img, lp_upper,
        )
        v_s = x_opt_b[i].astype(np.float64)
        y_s = y_opt_b[i].astype(np.float64)

        # Operator norm of K_mat (sparse top SV).
        K_csr = sp.csr_matrix(K_mat)
        _, sv, _ = spla.svds(K_csr, k=1, which="LM")
        M_actual = float(sv[0])

        # Strict-interior init — same one the trainer uses, so R_actual matches
        # the cached R_val_batch[i] from sample creation.
        S_inst = K_mat.shape[0] - m1
        v0, y0 = build_strict_interior_init(n_vars, m1, S_inst, lp_upper)
        R_actual = float(np.linalg.norm(np.concatenate([v0 - v_s, y0 - y_s])))

        # CP iterates (K_max steps with the production initial stepsize choice).
        tau = sigma = 0.9 / M_val_pool
        theta = 1.0
        vs_iters, ys_iters, gf1, gh, w, z = run_cp_on_facility(
            K_mat, q, c, l, u, m1, v0, y0, tau, sigma, theta, args.K_max,
        )

        # Evaluate at three (M, R) variants.
        variants = [
            ('actual', M_actual, R_actual),
            ('pool', M_val_pool, R_val_pool),
            ('slack15', 1.5 * M_actual, 1.5 * R_actual),
        ]
        row = {
            'instance_idx': i,
            'image_index': int(image_index_b[i]),
            'M_actual': M_actual,
            'R_actual': R_actual,
        }
        for label, M_var, R_var in variants:
            stats = evaluate_pep_at(
                K_mat, q, c, l, u, m1, v_s, y_s,
                vs_iters, ys_iters, gf1, gh, w, z,
                tau, sigma, theta, args.K_max, M_var, R_var,
            )
            for k, v in stats.items():
                row[f'{label}_{k}'] = v

        rows.append(row)
        if (i + 1) % 5 == 0 or i == n_run - 1:
            log.info(
                f"[{i+1}/{n_run}] image={row['image_index']} "
                f"M_actual={M_actual:.4f} R_actual={R_actual:.4f} "
                f"pool_f1_max={row['pool_f1_max']:.2e} pool_h_max={row['pool_h_max']:.2e} "
                f"pool_IC={row['pool_IC']:.2e} pool_PSD0={row['pool_psd0_min']:.2e}"
            )

    # CSV write
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]}" for k in fieldnames) + "\n")
    log.info(f"Wrote {out_csv} ({len(rows)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY ({args.split}, K_max={args.K_max}, n={n_run})")
    print("=" * 70)
    eps = 1e-3
    psd_eps = 1e-1
    for label in ('actual', 'pool', 'slack15'):
        f1_max = max(r[f'{label}_f1_max'] for r in rows)
        h_max = max(r[f'{label}_h_max'] for r in rows)
        vp_max = max(r[f'{label}_vp_max'] for r in rows)
        adj_max = max(r[f'{label}_adj_max'] for r in rows)
        IC_max = max(r[f'{label}_IC'] for r in rows)
        psd0_min = min(r[f'{label}_psd0_min'] for r in rows)
        psd1_min = min(r[f'{label}_psd1_min'] for r in rows)
        n_IC_fail = sum(1 for r in rows if r[f'{label}_IC'] > eps)
        n_PSD0_fail = sum(1 for r in rows if r[f'{label}_psd0_min'] < -psd_eps)
        print(f"\n  variant '{label}':")
        print(f"    f1     max viol = {f1_max:.3e}   (eps={eps})")
        print(f"    h      max viol = {h_max:.3e}")
        print(f"    vp     max viol = {vp_max:.3e}")
        print(f"    adj    max viol = {adj_max:.3e}")
        print(f"    IC     max value = {IC_max:.3e}   (n_fail={n_IC_fail}/{n_run})")
        print(f"    PSD0   min eig   = {psd0_min:.3e}  (n_fail={n_PSD0_fail}/{n_run})")
        print(f"    PSD1   min eig   = {psd1_min:.3e}")
    print()


if __name__ == "__main__":
    main()
