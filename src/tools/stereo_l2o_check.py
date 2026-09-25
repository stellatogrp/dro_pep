"""L2O-only stress test for the stereo-disparity LP (learning/stereo_lp.py).

Gray Olivetti faces are split by subject (28/4/8, as in the PDLP pipeline);
each face gets a seeded synthetic disparity. An L2O PDHG schedule (K steps) is
trained on the gray training pairs with a replica of the UnifiedTrainer L2O
path (weighted gap loss w_k ~ 0.9^(K-k), sqrt reparametrization, clip +
adamw with warmup-cosine, N-minibatches with per-epoch shuffle, validation
mean final gap, validation-best selection; init tau = sigma = 0.5 / M_val,
theta = 1, M_val = 1.2 max train ||G||). PDHG starts at the box center and
y0 = dual_init (paper: 1).

The untrained and trained schedules are evaluated (numpy PDHG) on the gray
test split and on color OOD pairs (Tiny ImageNet, Imagenette 64x64):
final Lagrangian gap, disparity RMSE vs the truth and vs the LP optimum, ||G||.

Usage (from src/):
    python tools/stereo_l2o_check.py --out iclr_data_outputs/explore_warmstart/stereo
"""
import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
for _p in (SRC_DIR, SRC_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _jax_setup  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pandas as pd  # noqa: E402
from jax.experimental import sparse as jsparse  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from learning.stereo_lp import grad_x, make_pair, stereo_lp, stereo_x0, stereo_y0, true_disparity  # noqa: E402
from learning.trajectories.cp_lp import problem_data_to_cp_lp_trajectories  # noqa: E402
from learning.tv_averages import run_pdhg_capture_gaps  # noqa: E402
from learning.tv_inpainting_color_averages import load_color_images  # noqa: E402
from learning.tv_inpainting_reduced import reduced_opnorm, solve_reduced  # noqa: E402
from learning_experiment_classes.pdlp import split_persons_by_subject  # noqa: E402

CACHE = SRC_DIR / "iclr_data_outputs/explore_warmstart/lp_cache"


# ----------------------------------------------------------------------------- instances
def instance_builder(left, seed, lam, place):
    """-> (mats, d_true) for an (M, N, C) image in [0, 1]."""
    d = true_disparity(left.shape[0], left.shape[1], np.random.default_rng(seed))
    L, R = make_pair(left, d)
    return stereo_lp(L, R, lam, place), d, L


def solve_cached(key, left, seed, lam, place):
    path = CACHE / key
    if path.exists():
        z = np.load(path)
        return z["raw_x"], z["raw_y"], float(z["f_opt"]), float(z["opnorm"])
    mats, _, _ = instance_builder(left, seed, lam, place)
    s = solve_reduced(mats)
    out = (s["raw_x"], s["raw_y"], float(s["objective_value"]), reduced_opnorm(mats))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw_x=out[0], raw_y=out[1], f_opt=out[2], opnorm=out[3])
    return out


def build_sets(args):
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces().images.astype(np.float64)
    tr_p, va_p, te_p = split_persons_by_subject(12345, 28, 4, 8, 40)
    sets = {}
    for name, persons in (("train", tr_p), ("val", va_p), ("test", te_p)):
        idx = np.concatenate([np.arange(p * 10, (p + 1) * 10) for p in persons])
        sets[name] = [(faces[i][..., None], 1000 + int(i)) for i in idx]
    tiny = load_color_images(args.n_color)
    sets["tiny"] = [(im.astype(np.float64) / 255, 50000 + i) for i, im in enumerate(tiny)]
    from pdlp_warmstart_explore import load_imagenette
    inet = load_imagenette(SRC_DIR.parent / "data" / "imagenette2-160", args.n_color, 64)
    sets["imagenette"] = [(im.astype(np.float64) / 255, 60000 + i) for i, im in enumerate(inet)]
    return sets


def solve_set(name, items, args):
    tag = f"stereo_{args.place}{args.lam:g}_{name}"
    return Parallel(n_jobs=args.n_jobs)(
        delayed(solve_cached)(f"{tag}/{j}.npz", im, seed, args.lam, args.place)
        for j, (im, seed) in enumerate(tqdm(items, desc=f"solve {name}")))


# ----------------------------------------------------------------------------- JAX (gray, fixed pattern)
def gray_arrays(items, sols, args):
    """Per-instance value arrays for the shared gray sparsity pattern."""
    out = {k: [] for k in ("wg", "h", "u", "x_opt", "y_opt")}
    for (im, seed), (rx, ry, _, _) in zip(items, sols):
        mats, _, L = instance_builder(im, seed, args.lam, args.place)
        w = args.lam if args.place == "inside" else 1.0
        out["wg"].append(w * grad_x(L[0]).ravel())
        out["h"].append(mats.h); out["u"].append(mats.u)
        out["x_opt"].append(rx); out["y_opt"].append(ry)
    return pattern_template(im.shape[0], im.shape[1], args), {k: jnp.array(np.stack(v)) for k, v in out.items()}


def pattern_template(M, N, args):
    """Gray stereo LP whose d-block has every diagonal entry structurally present (ramp image)."""
    ramp = np.tile(np.linspace(0.1, 0.9, N), (M, 1))
    return stereo_lp([ramp], [ramp], args.lam, args.place)


def make_gap_traj(template, K_pix, dual_init, slack_init=None, feasible=False, dual_warm=False):
    """Gap trajectory of one gray instance; G pattern from ``template``, d-block values replaced."""
    coo = template.G.tocoo()
    rows, cols = jnp.array(coo.row), jnp.array(coo.col)
    base = jnp.array(coo.data)
    # entries in the d-columns of the first 2K rows are -+wg (row r = pixel r, row K + r = pixel r)
    is_data = np.asarray((coo.col < K_pix) & (coo.row < 2 * K_pix))
    sign = jnp.array(np.where(coo.row < K_pix, -1.0, 1.0))
    pix = jnp.array(np.where(is_data, coo.col, 0))
    is_data = jnp.array(is_data)
    shape = template.G.shape
    c, l = jnp.array(template.c), jnp.array(template.l)
    idx = jnp.stack([rows, cols], 1)

    def gap_traj(stepsizes, wg, h, u, x_opt, y_opt, K):
        data = jnp.where(is_data, sign * wg[pix], base)
        G = jsparse.BCOO((data, idx), shape=shape)
        x0, y0 = 0.5 * (l + u), dual_init * jnp.ones(shape[0])
        if dual_warm:
            top = h[:K_pix]
            w_top = jnp.where(top > 0, 0.99, jnp.where(top < 0, 0.01, 0.5))
            y0 = jnp.concatenate([w_top, 1.0 - w_top, 0.5 * jnp.ones(shape[0] - 2 * K_pix)])
        if feasible:
            x0 = x0.at[K_pix:].set(0.01).at[K_pix:2 * K_pix].set(jnp.abs(h[:K_pix]) + 0.01)
        elif slack_init is not None:
            x0 = x0.at[K_pix:].set(slack_init)
        v, y = problem_data_to_cp_lp_trajectories(
            stepsizes, c, G, h, l, u, x_opt, y_opt, x0, y0, K, shape[0],
            return_Gram_representation=False)[:2]
        L = lambda xx, yy: c @ xx - yy @ (G @ xx) + h @ yy
        return jax.vmap(lambda xk, yk: L(xk, y_opt) - L(x_opt, yk))(v, y)

    return gap_traj


def train_l2o(gap_traj, train, val, K, M_val, args):
    keys = ("wg", "h", "u", "x_opt", "y_opt")
    batched = jax.vmap(lambda s, *a: gap_traj(s, *a, K), in_axes=(None,) + (0,) * len(keys))
    w = 0.9 ** (K - jnp.arange(K + 1)); w = w / w.sum()
    sq = lambda raw: tuple(r ** 2 for r in raw)
    loss = lambda raw, d: jnp.mean(batched(sq(raw), *(d[k] for k in keys)) @ w)
    # Pass the validation arrays as jit ARGUMENTS: closed over as constants, XLA
    # constant-folds the BCOO scatter and returns wrong gaps (seen: -1.2e4 vs +373).
    _val_jit = jax.jit(lambda raw, v: jnp.mean(batched(sq(raw), *(v[k] for k in keys))[:, -1]))
    val_fn = lambda raw: _val_jit(raw, val)
    raw0 = tuple(jnp.sqrt(s) for s in (jnp.full(K, 0.5 / M_val), jnp.full(K, 0.5 / M_val), jnp.ones(K)))
    eager = np.mean([float(gap_traj(sq(raw0), *(val[k][j] for k in keys), K)[-1]) for j in range(val["h"].shape[0])])
    if not np.isclose(float(val_fn(raw0)), eager, rtol=1e-8):
        raise RuntimeError(f"jitted validation {float(val_fn(raw0))} != eager {eager}")
    vg = jax.jit(jax.value_and_grad(loss))
    raw = tuple(jnp.sqrt(s) for s in (jnp.full(K, 0.5 / M_val), jnp.full(K, 0.5 / M_val), jnp.ones(K)))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(
        optax.warmup_cosine_decay_schedule(1e-6, args.eta, int(0.1 * args.iters), args.iters, 1e-6),
        weight_decay=args.wd))
    state = opt.init(raw)
    n = train["h"].shape[0]; n_mb = n // args.N
    rng = np.random.default_rng(args.seed)
    rec = lambda it, tl, r: dict(iteration=it, training_loss=tl, validation_loss=float(val_fn(r)),
                                 **{f"{p}_{k}": float(r[j][k] ** 2) for k in range(K)
                                    for j, p in enumerate(("tau", "sigma", "theta"))})
    rows = [rec(0, float(loss(raw, {k: v[:args.N] for k, v in train.items()})), raw)]
    perm = np.arange(n)
    for it in tqdm(range(args.iters), desc="L2O"):
        if it % n_mb == 0:
            perm = rng.permutation(n)
        sel = perm[(it % n_mb) * args.N:(it % n_mb + 1) * args.N]
        l_, g = vg(raw, {k: v[sel] for k, v in train.items()})
        upd, state = opt.update(g, state, raw)
        raw = optax.apply_updates(raw, upd)
        rows.append(rec(it + 1, float(l_), raw))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- evaluation
def evaluate(name, items, sols, schedules, args):
    out = []
    for j, ((im, seed), (rx, ry, f, opn)) in enumerate(zip(items, sols)):
        mats, d_true, _ = instance_builder(im, seed, args.lam, args.place)
        K_pix = d_true.size
        x0 = stereo_x0(mats.l, mats.u, K_pix, args.slack_init,
                       feasible_h=mats.h if args.feasible_init else None, n_channels=im.shape[2])
        y0 = (stereo_y0(mats.h, K_pix, im.shape[2]) if args.dual_warm
              else args.dual_init * np.ones(mats.G.shape[0]))
        d_lp = rx[:K_pix]
        for label, (tau, sigma, theta) in schedules.items():
            x = x0.copy(); y = y0.copy()
            for k in range(len(tau)):
                xn = np.clip(x - tau[k] * (mats.c - mats.G.T @ y), mats.l, mats.u)
                y = np.maximum(y + sigma[k] * (mats.h - mats.G @ (xn + theta[k] * (xn - x))), 0.0)
                x = xn
            L = lambda xx, yy: mats.c @ xx - yy @ (mats.G @ xx) + mats.h @ yy
            out.append(dict(split=name, idx=j, label=label, gap=L(x, ry) - L(rx, y), f_opt=f, opnorm=opn,
                            rmse_true=float(np.sqrt(np.mean((x[:K_pix] - d_true.ravel()) ** 2))),
                            rmse_lp=float(np.sqrt(np.mean((x[:K_pix] - d_lp) ** 2)))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--place", choices=["inside", "cost"], default="inside")
    ap.add_argument("--dual-init", type=float, default=1.0)
    ap.add_argument("--dual-warm", action="store_true",
                    help="warm dual start: subgradient of |residual| at d = 0 per constraint pair")
    ap.add_argument("--feasible-init", action="store_true",
                    help="start the slacks at their values for d = 0 (feasible start)")
    ap.add_argument("--slack-init", type=float, default=None,
                    help="start the slacks at this value (default: box center)")
    ap.add_argument("--eval-runs", nargs="*", default=[],
                    help="skip training; evaluate label=progress.csv[@row] schedules (plus row 0 of the first)")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--n-color", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sets = build_sets(args)
    sols = {k: solve_set(k, v, args) for k, v in sets.items()}
    Mtr = np.array([s[3] for s in sols["train"]])
    M_val = 1.2 * Mtr.max()
    print(f"train ||G|| median {np.median(Mtr):.3f} max {Mtr.max():.3f} -> M_val {M_val:.3f}")
    for k in ("test", "tiny", "imagenette"):
        v = np.array([s[3] for s in sols[k]])
        print(f"  {k:10s} ||G|| median {np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]  frac > M_val {np.mean(v > M_val):.2f}")

    template, train = gray_arrays(sets["train"], sols["train"], args)
    _, val = gray_arrays(sets["val"], sols["val"], args)
    K_pix = sets["train"][0][0].shape[0] * sets["train"][0][0].shape[1]
    if args.eval_runs:
        from pdlp_warmstart_explore import load_schedules
        S = load_schedules(args.eval_runs)
        first = args.eval_runs[0].split("=", 1)[0]
        schedules = {"untrained": S[f"{first}:untrained"][1:]}
        schedules.update({k: v[1:] for k, v in S.items() if not k.endswith(":untrained")})
        suffix = "_runs"
    else:
        df = train_l2o(make_gap_traj(template, K_pix, args.dual_init, args.slack_init, args.feasible_init,
                                     args.dual_warm),
                       train, val, args.K, M_val, args)
        df.to_csv(args.out / f"progress_K{args.K}.csv", index=False)
        best = int(df.validation_loss.idxmin())
        print(f"validation: row0 {df.validation_loss[0]:.4g} -> best {df.validation_loss[best]:.4g} at {best}")
        row = lambda r: tuple(df.loc[r, [f"{p}_{k}" for k in range(args.K)]].to_numpy(float) for p in ("tau", "sigma", "theta"))
        schedules = {"untrained": row(0), "l2o": row(best)}
        suffix = ""
    for k, v in schedules.items():
        print(f"  {k}: " + " ".join(f"{p}={np.round(a, 3).tolist()}" for p, a in zip(("tau", "sigma", "theta"), v)))

    res = pd.DataFrame(sum((evaluate(k, sets[k], sols[k], schedules, args) for k in ("test", "tiny", "imagenette")), []))
    res.to_csv(args.out / f"eval_K{args.K}{suffix}.csv", index=False)
    if args.eval_runs:
        labs = [k for k in schedules]
        ref = "l2o" if "l2o" in schedules else labs[1]
        for k in ("test", "tiny", "imagenette"):
            p = res[res.split == k].pivot(index="idx", columns="label", values="gap")
            q = res[res.split == k].pivot(index="idx", columns="label", values="rmse_true")
            print(f"--- {k}")
            for l in labs:
                r = p[l] / p[ref]
                print(f"  {l:14s} gap mean {p[l].mean():10.4g} median {p[l].median():10.4g} max {p[l].max():10.4g} | "
                      f"beats {ref} {np.mean(r < 1):.2f} median ratio {r.median():.3f} | rmse_true {q[l].mean():.4f}")
        return
    base = None
    print(f"\n{'split':11s} | {'untr gap':>9s} {'l2o gap':>9s} {'l2o/untr':>8s} {'rel':>5s} | {'l2o>untr':>8s} | "
          f"{'untr rmse':>9s} {'l2o rmse':>9s} (vs true)")
    for k in ("test", "tiny", "imagenette"):
        p = res[res.split == k].pivot(index="idx", columns="label", values="gap")
        q = res[res.split == k].pivot(index="idx", columns="label", values="rmse_true")
        ratio = p.l2o.mean() / p.untrained.mean(); base = ratio if base is None else base
        print(f"{k:11s} | {p.untrained.mean():9.4g} {p.l2o.mean():9.4g} {ratio:8.4f} {ratio / base:5.2f} | "
              f"{np.mean(p.l2o > p.untrained):8.2f} | {q.untrained.mean():9.4f} {q.l2o.mean():9.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
