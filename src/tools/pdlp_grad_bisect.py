"""Bisect the PDLP ldro-pep gradient at the real training configuration.

The loss is  theta -> (A_data, b) [JAX part: trajectory + PEP + canon]
             (A_data, b) -> p* [SDP layer: scs_solve_wrapper_sparse].
At a fixed schedule and minibatch this reports:
  1. SDP solve status / residuals at the wrapper's tolerance.
  2. SDP layer: wrapper VJP vs envelope formula (y_i x_j, -y) vs FD of p*
     along random (A_data, b) directions.
  3. JAX part: jax.jvp of theta -> (A, b) vs FD of the same map.
  4. Full loss: jax.grad vs envelope-composed gradient vs FD along random
     theta directions and along -grad.

Usage (from src/):
    python tools/pdlp_grad_bisect.py --data-dir <sample dir> [--eps 100] [--row 0]
        [--sched path/to/progress.csv]
"""
import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)

import _jax_setup  # noqa: E402,F401
import diffcp  # noqa: E402
import jax  # noqa: E402
import jax.flatten_util  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as spa  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from learning.jax_scs_layer import get_direct_solve_method, scs_solve_wrapper_sparse  # noqa: E402
from learning.unified_trainer import UnifiedTrainer, _reparam_modes, to_raw_params  # noqa: E402
from learning_experiment_classes.pdlp import PDLPProblemModule  # noqa: E402

WRAPPER_TOL = dict(tol_gap_abs=1e-5, tol_gap_rel=1e-5, tol_feas=1e-5,
                   reduced_tol_gap_abs=5e-5, reduced_tol_gap_rel=5e-5, reduced_tol_feas=1e-4)
TIGHT_TOL = dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9,
                 reduced_tol_gap_abs=1e-8, reduced_tol_gap_rel=1e-8, reduced_tol_feas=1e-8)


def closure_vars(fn):
    return {n: c.cell_contents for n, c in zip(fn.__code__.co_freevars, fn.__closure__)}


def to_csc(A_data, A_idx, shape):
    A_data, A_idx = np.asarray(A_data), np.asarray(A_idx)
    v = (A_idx[:, 0] < shape[0]) & (A_idx[:, 1] < shape[1])
    return spa.csc_matrix((A_data[v], (A_idx[v, 0], A_idx[v, 1])), shape=shape), v


def solve(A_csc, b, c, cones, tol):
    r = diffcp.solve_and_derivative_internal(
        A_csc, np.asarray(b), np.asarray(c), cones, solve_method="CLARABEL",
        direct_solve_method=get_direct_solve_method(), verbose=False, **tol)
    return r


def report_solve(tag, r, A_csc, b, c):
    x, y, s = r["x"], r["y"], r["s"]
    pres = np.linalg.norm(A_csc @ x + s - b) / (1 + np.linalg.norm(b))
    dres = np.linalg.norm(A_csc.T @ y + c) / (1 + np.linalg.norm(c))
    pobj, dobj = c @ x, -b @ y
    print(f"  [{tag}] status={r['info']['status']} p*={pobj:.8g} d*={dobj:.8g} "
          f"gap={abs(pobj - dobj):.3g} pres={pres:.3g} dres={dres:.3g} "
          f"|x|={np.linalg.norm(x):.3g} |y|={np.linalg.norm(y):.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--eps", type=float, default=100.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--sched", default=None, help="progress.csv; default = untrained init")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=0)
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--h-layer", type=float, default=1e-4, help="relative FD step on (A_data, b)")
    ap.add_argument("--h-theta", type=float, default=1e-3)
    ap.add_argument("--value-grad", choices=["diffcp", "envelope"], default=None,
                    help="override cfg sdp_value_grad for the full-loss check")
    args = ap.parse_args()

    cfg = OmegaConf.load(SRC_DIR / "configs_learning/pdlp.yaml")
    cfg.data_source_dir, cfg.learning_framework = args.data_dir, "ldro-pep"
    cfg.N, cfg.eps = args.N, args.eps
    if args.value_grad:
        cfg.sdp_value_grad = args.value_grad
    pm = PDLPProblemModule(cfg)
    tr = UnifiedTrainer(pm, cfg, jax.random.PRNGKey(0))
    tr.prepare_data()
    L, mu, R = pm.compute_L_mu_R()
    tr._modes = _reparam_modes(3, False)
    init = pm.get_initial_stepsizes("cp", args.K, L, mu)
    loss = tr._build_loss_function(args.K, L, mu, R, init)
    cv = closure_vars(loss)
    build, static, shape = cv["_build_inputs"], cv["static_data"], cv["A_shape"]
    cones = static.diffcp_cone_dict
    print(f"L(M)={L:.4g} R={R:.4g} A_shape={shape} cones={ {k: (v if isinstance(v, int) else len(v)) for k, v in cones.items()} }")

    if args.sched:
        df = pd.read_csv(args.sched)
        s = tuple(jnp.array(df.loc[args.row, [f"{p}_{k}" for k in range(args.K)]].to_numpy(float))
                  for p in ("tau", "sigma", "theta"))
    else:
        s = init
    raw = to_raw_params(s, tr._modes)
    batch = tr._get_minibatch(args.minibatch)
    A_data, A_idx, b, c = build(raw, batch)
    A_csc, valid = to_csc(A_data, A_idx, shape)
    print(f"nnz={A_csc.nnz} |A|_max={abs(A_csc).max():.3g} |b|_max={np.abs(np.asarray(b)).max():.3g}")

    # ---- 1. solve quality
    print("\n[1] SDP solve quality")
    r_w = solve(A_csc, b, c, cones, WRAPPER_TOL)
    report_solve("wrapper tol", r_w, A_csc, np.asarray(b), np.asarray(c))
    r_t = solve(A_csc, b, c, cones, TIGHT_TOL)
    report_solve("tight tol", r_t, A_csc, np.asarray(b), np.asarray(c))
    x, y = r_t["x"], r_t["y"]
    rows, cols = np.asarray(A_idx[:, 0])[valid], np.asarray(A_idx[:, 1])[valid]

    # ---- 2. SDP layer
    print("\n[2] SDP layer: d p* along random (A_data, b) directions")
    f_layer = lambda Ad, bb: scs_solve_wrapper_sparse(static, Ad, A_idx, shape, bb, c, value_grad="diffcp")
    gA, gb = jax.grad(f_layer, argnums=(0, 1))(A_data, b)
    gA, gb = np.asarray(gA)[valid], np.asarray(gb)
    eA, eb = y[rows] * x[cols], -y
    print(f"  |vjp|={np.sqrt(gA @ gA + gb @ gb):.4g}  |env|={np.sqrt(eA @ eA + eb @ eb):.4g}  "
          f"cos(vjp, env)={(gA @ eA + gb @ eb) / np.sqrt((gA @ gA + gb @ gb) * (eA @ eA + eb @ eb)):.4f}")
    Ad_np, b_np = np.asarray(A_data)[valid], np.asarray(b)
    rng = np.random.default_rng(1)
    for i in range(args.n_dirs):
        dA = rng.standard_normal(rows.size) * np.abs(Ad_np)   # relative perturbation
        db = rng.standard_normal(b_np.size) * np.abs(b_np)
        h = args.h_layer
        fp = c @ solve(spa.csc_matrix((Ad_np + h * dA, (rows, cols)), shape=shape), b_np + h * db,
                       c, cones, WRAPPER_TOL)["x"]
        fm = c @ solve(spa.csc_matrix((Ad_np - h * dA, (rows, cols)), shape=shape), b_np - h * db,
                       c, cones, WRAPPER_TOL)["x"]
        print(f"  dir {i}: vjp {gA @ dA + gb @ db:+.6g}  env {eA @ dA + eb @ db:+.6g}  "
              f"fd {(fp - fm) / (2 * h):+.6g}", flush=True)

    # ---- 3. JAX part
    print("\n[3] JAX part: jvp of theta -> (A_data, b) vs FD")
    rl, unravel = jax.flatten_util.ravel_pytree(raw)
    dirs = [np.asarray(v) / np.linalg.norm(v) for v in rng.standard_normal((args.n_dirs, rl.size))]

    def ab_map(r):
        Ad, Ai, bb, _ = build(unravel(r), batch)
        M, _ = to_csc(Ad, Ai, shape)
        return M, np.asarray(bb)
    for i, d in enumerate(dirs):
        _, (tA, tb) = jax.jvp(lambda r: build(unravel(r), batch)[0::2], (rl,), (jnp.array(d),))
        tA_csc, _ = to_csc(tA, A_idx, shape)
        h = args.h_theta
        Mp, bp = ab_map(rl + h * d)
        Mm, bm = ab_map(rl - h * d)
        fdA, fdb = (Mp - Mm) / (2 * h), (bp - bm) / (2 * h)
        eA_ = abs(tA_csc - fdA).max() / max(abs(fdA).max(), 1e-12)
        eb_ = np.abs(np.asarray(tb) - fdb).max() / max(np.abs(fdb).max(), 1e-12)
        pat = (abs(Mp) > 0).astype(int) - (abs(Mm) > 0).astype(int)
        print(f"  dir {i}: rel max err A {eA_:.3g}  b {eb_:.3g}  |dA|max {abs(fdA).max():.3g}  "
              f"pattern changes between +-h: {abs(pat).sum()}", flush=True)

    # ---- 4. full loss
    print(f"\n[4] Full loss (sdp_value_grad={tr.sdp_value_grad!r}): jax.grad vs envelope-composed vs FD")
    val, g = jax.value_and_grad(lambda r: loss(unravel(r), batch))(rl)
    g = np.asarray(g)
    # envelope-composed gradient: J^T [env], via vjp of the build map
    _, vjp_build = jax.vjp(lambda r: build(unravel(r), batch)[0::2], rl)
    envA_full = np.zeros(A_data.shape[0]); envA_full[valid] = eA
    (g_env,) = vjp_build((jnp.array(envA_full), jnp.array(eb)))
    g_env = np.asarray(g_env)
    print(f"  f={float(val):.6g}  |grad|={np.linalg.norm(g):.4g}  |grad_env|={np.linalg.norm(g_env):.4g}  "
          f"cos={g @ g_env / (np.linalg.norm(g) * np.linalg.norm(g_env)):.4f}")
    for name, d in [("-grad", -g / np.linalg.norm(g)), ("-grad_env", -g_env / np.linalg.norm(g_env))] + \
                   [(f"rand{i}", d) for i, d in enumerate(dirs)]:
        h = args.h_theta
        fp = float(loss(unravel(rl + h * d), batch))
        fm = float(loss(unravel(rl - h * d), batch))
        print(f"  {name:9s}: grad {g @ d:+.6g}  env {g_env @ d:+.6g}  fd {(fp - fm) / (2 * h):+.6g}", flush=True)


if __name__ == "__main__":
    main()
