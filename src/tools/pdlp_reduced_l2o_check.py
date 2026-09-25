"""L2O-only gate for the reduced TV-inpainting LP (learning/tv_inpainting_reduced.py).

Trains an L2O PDHG schedule (K=5) on the reduced LPs of the gray train split of
an existing PDLP sample set (same images and random-pixel masks), mirroring the
UnifiedTrainer L2O path:
  * loss   'weighted' composition, w_k ~ 0.9^(K-k), k = 0..K, of the Lagrangian
           gap L(x_k, y*) - L(x*, y_k), mean over the minibatch;
  * params sqrt reparametrization of (tau, sigma, theta);
  * optim  clip_by_global_norm(1.0) + adamw(warmup-cosine, peak eta), wd;
  * data   minibatches of N with a per-epoch shuffle; validation = mean final
           gap on the validation split; the validation-best step is selected;
  * init   tau = sigma = 0.5 / M_val, theta = 1, M_val = 1.2 * max train ||G||.
Then evaluates the untrained and trained schedules (numpy PDHG) on:
  gray_random (held-out test split), gray_scratch, gray_blocks,
  color_random and color_scratch (Tiny ImageNet).

Reduced LPs have a varying number of edges; for vmap every instance is padded
to E_max edges with dummy variables fixed at 0 (cost 0, bounds [0, 0]), whose
rows stay inactive, so the padded gap equals the unpadded one.

Usage (from src/):
    python tools/pdlp_reduced_l2o_check.py --data-dir <sample dir> --out <dir>
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

from learning.trajectories.cp_lp import problem_data_to_cp_lp_trajectories  # noqa: E402
from learning.tv_averages import run_pdhg_capture_gaps  # noqa: E402
from learning.tv_inpainting_color_averages import load_color_images  # noqa: E402
from learning.tv_inpainting_reduced import (  # noqa: E402
    build_reduced_lp,
    reduced_edges,
    reduced_opnorm,
    solve_reduced,
    stack_channels,
)
from pdlp_stress_explore import scratch_mask  # noqa: E402

CACHE = SRC_DIR / "iclr_data_outputs/explore_warmstart/lp_cache"
Y0 = 0.1


# ----------------------------------------------------------------------------- data
def block_mask(M, N, frac, rng, size=8):
    missing = np.zeros((M, N), bool)
    while missing.mean() < frac:
        r, c = rng.integers(M - size + 1), rng.integers(N - size + 1)
        missing[r:r + size, c:c + size] = True
    return ~missing


def solve_cached(key, build):
    """(raw_x, raw_y, f_opt, ||G||) of one reduced LP (per-channel solves stacked)."""
    path = CACHE / key
    if path.exists():
        d = np.load(path)
        return d["raw_x"], d["raw_y"], float(d["f_opt"]), float(d["opnorm"])
    parts = build()
    xs, ys, f = [], [], 0.0
    for m in parts:
        s = solve_reduced(m)
        xs.append(s["raw_x"]); ys.append(s["raw_y"]); f += s["objective_value"]
    opnorm = reduced_opnorm(parts[0])  # channels share the mask, hence G
    raw_x, raw_y = np.concatenate(xs), np.concatenate(ys)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw_x=raw_x, raw_y=raw_y, f_opt=f, opnorm=opnorm)
    return raw_x, raw_y, f, opnorm


def gray_split(data_dir, split, faces, M, N, n_jobs):
    d = np.load(data_dir / f"{split}_set.npz")
    pix = [faces[int(i)].reshape(-1) for i in d["image_index_batch"]]
    masks = [np.asarray(m, bool).reshape(-1) for m in d["mask_batch"]]
    sols = Parallel(n_jobs=n_jobs)(
        delayed(solve_cached)(f"reduced_gray_{split}/{i}.npz",
                              lambda p=p, m=m: [build_reduced_lp(p, m, M, N)])
        for i, (p, m) in enumerate(tqdm(list(zip(pix, masks)), desc=f"solve {split}")))
    return pix, masks, sols


# ----------------------------------------------------------------------------- padded JAX instances
def pad_instances(pix, masks, sols, M, N, E_max, nnz_max):
    """Stack padded per-instance arrays for vmap."""
    U = int((~masks[0]).sum())
    out = {k: [] for k in ("rows", "cols", "vals", "h", "c", "u", "x_opt", "y_opt")}
    for p, m, (rx, ry, _, _) in zip(pix, masks, sols):
        D_U, D_K = reduced_edges(m, M, N)
        E = D_U.shape[0]
        if D_U.shape[1] != U or E > E_max:
            raise ValueError("instance does not fit the padded shape")
        coo = D_U.tocoo()
        pad = nnz_max - coo.nnz
        out["rows"].append(np.r_[coo.row, np.zeros(pad, int)])
        out["cols"].append(np.r_[coo.col, np.zeros(pad, int)])
        out["vals"].append(np.r_[coo.data, np.zeros(pad)])
        dk = np.r_[D_K @ p[m], np.zeros(E_max - E)]
        out["h"].append(np.r_[dk, -dk])
        real = np.r_[np.ones(E), np.zeros(E_max - E)]
        out["c"].append(np.r_[np.zeros(U), real])
        out["u"].append(np.r_[np.ones(U), real])          # padded t fixed in [0, 0]
        out["x_opt"].append(np.r_[rx[:U + E], np.zeros(E_max - E)])
        yt, yb = ry[:E], ry[E:]
        out["y_opt"].append(np.r_[yt, np.zeros(E_max - E), yb, np.zeros(E_max - E)])
    return U, {k: jnp.array(np.stack(v)) for k, v in out.items()}


def make_instance_fn(U, E_max):
    n, m = U + E_max, 2 * E_max
    eye_r = jnp.arange(E_max)

    def gap_traj(stepsizes, rows, cols, vals, h, c, u, x_opt, y_opt, K):
        idx = jnp.concatenate([
            jnp.stack([rows, cols], 1), jnp.stack([rows + E_max, cols], 1),
            jnp.stack([eye_r, U + eye_r], 1), jnp.stack([E_max + eye_r, U + eye_r], 1)])
        data = jnp.concatenate([-vals, vals, jnp.ones(E_max), jnp.ones(E_max)])
        Kmat = jsparse.BCOO((data, idx), shape=(m, n))
        l = jnp.zeros(n)
        x0, y0 = 0.5 * (l + u), Y0 * jnp.ones(m)
        v, y = problem_data_to_cp_lp_trajectories(
            stepsizes, c, Kmat, h, l, u, x_opt, y_opt, x0, y0, K, m,
            return_Gram_representation=False)[:2]
        L = lambda xx, yy: c @ xx - yy @ (Kmat @ xx) + h @ yy
        return jax.vmap(lambda xk, yk: L(xk, y_opt) - L(x_opt, yk))(v, y)

    return gap_traj


# ----------------------------------------------------------------------------- training
def train_l2o(train, val, U, E_max, K, M_val, args):
    gap_traj = make_instance_fn(U, E_max)
    keys = ("rows", "cols", "vals", "h", "c", "u", "x_opt", "y_opt")
    batched = jax.vmap(lambda s, *a: gap_traj(s, *a, K), in_axes=(None,) + (0,) * len(keys))
    w = 0.9 ** (K - jnp.arange(K + 1)); w = w / w.sum()

    def loss(raw, data):
        gaps = batched(tuple(r ** 2 for r in raw), *(data[k] for k in keys))
        return jnp.mean(gaps @ w)

    # Pass the validation arrays as jit ARGUMENTS (closed-over constants get their
    # BCOO scatter constant-folded by XLA, which can return wrong gaps).
    _val_jit = jax.jit(lambda raw, v: jnp.mean(batched(tuple(r ** 2 for r in raw), *(v[k] for k in keys))[:, -1]))
    val_fn = lambda raw: _val_jit(raw, val)
    vg = jax.jit(jax.value_and_grad(loss))

    s0 = (jnp.full(K, 0.5 / M_val), jnp.full(K, 0.5 / M_val), jnp.ones(K))
    raw = tuple(jnp.sqrt(s) for s in s0)
    sched = optax.warmup_cosine_decay_schedule(1e-6, args.eta, int(0.1 * args.iters), args.iters, 1e-6)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=args.wd))
    state = opt.init(raw)
    n_train = train["h"].shape[0]
    n_mb = n_train // args.N
    rng = np.random.default_rng(args.seed)
    first = {k: v[:args.N] for k, v in train.items()}
    rows = [dict(iteration=0, training_loss=float(loss(raw, first)), validation_loss=float(val_fn(raw)), raw=raw)]
    perm = np.arange(n_train)
    for it in tqdm(range(args.iters), desc="L2O"):
        if it % n_mb == 0:
            perm = rng.permutation(n_train)
        sel = perm[(it % n_mb) * args.N:(it % n_mb + 1) * args.N]
        l, g = vg(raw, {k: v[sel] for k, v in train.items()})
        upd, state = opt.update(g, state, raw)
        raw = optax.apply_updates(raw, upd)
        rows.append(dict(iteration=it + 1, training_loss=float(l), validation_loss=float(val_fn(raw)), raw=raw))
    df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "raw"},
                        **{f"{p}_{k}": float(r["raw"][j][k] ** 2)
                           for k in range(K) for j, p in enumerate(("tau", "sigma", "theta"))}} for r in rows])
    return df


# ----------------------------------------------------------------------------- evaluation
def eval_family(name, instances, schedules, n_jobs):
    """instances: list of (key, builder -> [per-channel mats]). Returns long-form rows."""
    sols = Parallel(n_jobs=n_jobs)(delayed(solve_cached)(k, b) for k, b in tqdm(instances, desc=f"solve {name}"))
    out = []
    for i, ((key, build), (rx, ry, f, opn)) in enumerate(zip(instances, sols)):
        mats = stack_channels(build())
        x0, y0 = 0.5 * (mats.l + mats.u), Y0 * np.ones(mats.G.shape[0])
        R = float(np.linalg.norm(np.r_[x0 - rx, y0 - ry]))
        for label, (tau, sigma, theta) in schedules.items():
            g = run_pdhg_capture_gaps(mats.c, mats.G, mats.h, mats.A, mats.b, mats.l, mats.u,
                                      rx, ry, x0, y0, tau, sigma, theta)[-1]
            out.append(dict(family=name, idx=i, label=label, gap=float(g), f_opt=f, R=R, opnorm=opn))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--n-color", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.1, help="OOD missing fraction")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; reuse <out>/progress.csv (reference solves are cached)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra schedules label=progress.csv[@row] (validation-best by default)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import fetch_olivetti_faces
    meta = np.load(args.data_dir / "out_of_sample_metadata.npz")
    M, N = int(meta["M"]), int(meta["N"])
    if float(meta["lp_upper"]) != 1.0:
        raise ValueError("expects lp_upper = 1")
    faces = fetch_olivetti_faces().images.astype(np.float64)

    if args.eval_only:
        df = pd.read_csv(args.out / "progress.csv")
    else:
        tr = gray_split(args.data_dir, "training", faces, M, N, args.n_jobs)
        va = gray_split(args.data_dir, "validation", faces, M, N, args.n_jobs)
        M_train = np.array([s[3] for s in tr[2]])
        M_val = 1.2 * M_train.max()
        E_max = max(reduced_edges(m, M, N)[0].shape[0] for m in tr[1] + va[1])
        print(f"train ||G||: median {np.median(M_train):.4f} max {M_train.max():.4f} -> M_val {M_val:.4f}; "
              f"U={int((~tr[1][0]).sum())} E_max={E_max}")
        U, train = pad_instances(*tr, M, N, E_max, 2 * E_max)
        _, val = pad_instances(*va, M, N, E_max, 2 * E_max)
        df = train_l2o(train, val, U, E_max, args.K, M_val, args)
        df.to_csv(args.out / "progress.csv", index=False)
    best = int(df.validation_loss.idxmin())
    print(f"validation: row0 {df.validation_loss[0]:.4g} -> best {df.validation_loss[best]:.4g} at {best}")
    sched = lambda r: tuple(df.loc[r, [f"{p}_{k}" for k in range(args.K)]].to_numpy(float)
                            for p in ("tau", "sigma", "theta"))
    schedules = {"untrained": sched(0), "l2o": sched(best)}
    if args.extra:
        from pdlp_warmstart_explore import load_schedules
        extra = load_schedules(args.extra)
        schedules.update({k: v[1:] for k, v in extra.items() if not k.endswith(":untrained")})
    for k, v in schedules.items():
        print(k, [np.round(a, 3).tolist() for a in v])

    # ---- evaluation families
    test = np.load(args.data_dir / "test_set.npz")
    t_pix = [faces[int(i)].reshape(-1) for i in test["image_index_batch"]]
    fam = {}
    fam["gray_random"] = [(f"reduced_gray_test/{i}.npz",
                           lambda p=p, m=np.asarray(m, bool).reshape(-1): [build_reduced_lp(p, m, M, N)])
                          for i, (p, m) in enumerate(zip(t_pix, test["mask_batch"]))]
    for name, mk in (("gray_scratch", lambda r: scratch_mask(M, N, args.frac, r)),
                     ("gray_blocks", lambda r: block_mask(M, N, args.frac, r))):
        fam[name] = [(f"reduced_{name}_{args.frac:g}/{i}.npz",
                      lambda p=p, m=mk(np.random.default_rng(5000 + i)).reshape(-1): [build_reduced_lp(p, m, M, N)])
                     for i, p in enumerate(t_pix)]
    imgs = load_color_images(args.n_color)
    for name, mk in (("color_random", lambda r, Mi, Ni: r.random((Mi, Ni)) >= args.frac),
                     ("color_scratch", lambda r, Mi, Ni: scratch_mask(Mi, Ni, args.frac, r))):
        insts = []
        for i, im in enumerate(imgs):
            Mi, Ni, _ = im.shape
            m = mk(np.random.default_rng(7000 + i), Mi, Ni).reshape(-1)
            chans = [im[:, :, c].astype(np.float64).reshape(-1) / 255.0 for c in range(3)]
            insts.append((f"reduced_{name}_{args.frac:g}/{i}.npz",
                          lambda chans=chans, m=m, Mi=Mi, Ni=Ni: [build_reduced_lp(p, m, Mi, Ni) for p in chans]))
        fam[name] = insts
    rows = []
    for name, insts in fam.items():
        rows += eval_family(name, insts, schedules, args.n_jobs)
    res = pd.DataFrame(rows)
    res.to_csv(args.out / ("eval_extra.csv" if args.extra else "eval.csv"), index=False)
    if args.extra:
        print(f"\n{'family':14s} " + " ".join(f"{k:>11s}" for k in schedules) + "   (mean final gap)")
        for name in fam:
            p = res[res.family == name].pivot(index="idx", columns="label", values="gap")
            print(f"{name:14s} " + " ".join(f"{p[k].mean():11.4g}" for k in schedules))
        print(f"\n{'family':14s} " + " ".join(f"{k:>11s}" for k in schedules if k != 'l2o')
              + "   (fraction of instances beating l2o)")
        for name in fam:
            p = res[res.family == name].pivot(index="idx", columns="label", values="gap")
            print(f"{name:14s} " + " ".join(f"{np.mean(p[k] < p['l2o']):11.2f}" for k in schedules if k != "l2o"))

    base = None
    print(f"\n{'family':14s} {'||G|| med':>9s} {'R med':>7s} | {'untrained':>10s} {'l2o':>9s} {'l2o/untr':>8s} "
          f"{'rel. to in-dist':>15s} | {'l2o>untr':>8s} {'l2o med':>8s} {'l2o max':>8s}")
    for name in fam:
        p = res[res.family == name].pivot(index="idx", columns="label", values="gap")
        g = res[res.family == name].groupby("idx")[["opnorm", "R"]].first()
        ratio = p.l2o.mean() / p.untrained.mean()
        base = ratio if base is None else base
        print(f"{name:14s} {g.opnorm.median():9.4f} {g.R.median():7.1f} | {p.untrained.mean():10.4g} "
              f"{p.l2o.mean():9.4g} {ratio:8.4f} {ratio / base:15.2f} | {np.mean(p.l2o > p.untrained):8.2f} "
              f"{p.l2o.median():8.4g} {p.l2o.max():8.4g}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
