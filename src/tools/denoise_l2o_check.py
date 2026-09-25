"""L2O-only check for TV-L1 denoising (an LP): does a gray-trained schedule transfer to color?

LP per channel c (noisy image f_c, pixels u_c, data slacks s_c, TV slacks t_c):
    min  lam * sum s_c + sum t_c
    s.t. s_c >= +-(u_c - f_c),  t_c >= +-(D u_c),  box [0, 1]
Standard form x = [u_1, s_1, t_1, ..., u_C, s_C, t_C], G block-diagonal over channels
with blocks [[-I, I, 0], [I, I, 0], [-D, 0, I], [D, 0, I]] and h = [-f; f; 0; 0].
G contains no image data, so ||G|| is the same for every image (gray or color).

PDHG warm start (consistent primal-dual pair at the noisy image): u0 = clip(f, d, 1-d),
s0 = |u0 - f| + d, t0 = |D u0| + d; data duals split evenly (residual 0), TV duals get
(1-d, d) on the active row of |D u0| (even split where D u0 = 0), scaled to the costs.

Trains L2O (UnifiedTrainer replica: weighted gap loss, sqrt params, clip + adamw
warmup-cosine, N-minibatches, validation-best) on gray Olivetti (subject split),
evaluates gray test / Tiny ImageNet / Imagenette, renders denoised images.

Usage (from src/):
    python tools/denoise_l2o_check.py --out iclr_data_outputs/explore_warmstart/denoise
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
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from jax.experimental import sparse as jsparse  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from learning.trajectories.cp_lp import problem_data_to_cp_lp_trajectories  # noqa: E402
from learning.tv_inpainting_color_averages import load_color_images  # noqa: E402
from learning.tv_inpainting_reduced import diff_matrix, reduced_opnorm, solve_reduced  # noqa: E402
from learning.tv_inpainting_test import TVInpaintingMatrices  # noqa: E402
from learning_experiment_classes.pdlp import split_persons_by_subject  # noqa: E402

plt.rcParams["text.usetex"] = False
CACHE = SRC_DIR / "iclr_data_outputs/explore_warmstart/lp_cache"
DELTA = 0.01


def noisy(img, frac, rng):
    """Salt-and-pepper noise on a fraction of pixels (independently per channel)."""
    f = img.copy()
    hit = rng.random(img.shape) < frac
    f[hit] = rng.integers(0, 2, hit.sum()).astype(float)
    return f


def tvl1_lp(chans_f, lam):
    M, N = chans_f[0].shape
    D = diff_matrix(M, N)
    E, K = D.shape
    I, Ie = sp.eye(K), sp.eye(E)
    Zke, Zek, Zkk = sp.csr_matrix((K, E)), sp.csr_matrix((E, K)), sp.csr_matrix((K, K))
    block = sp.bmat([[-I, I, Zke], [I, I, Zke], [-D, Zek, Ie], [D, Zek, Ie]], format="csr")
    C = len(chans_f)
    G = sp.block_diag([block] * C, format="csr")
    h = np.concatenate([np.r_[-f.ravel(), f.ravel(), np.zeros(2 * E)] for f in chans_f])
    c = np.tile(np.r_[np.zeros(K), lam * np.ones(K), np.ones(E)], C)
    n = C * (2 * K + E)
    return TVInpaintingMatrices(c=c, A=sp.csr_matrix((0, n)), b=np.zeros(0), G=G, h=h,
                                l=np.zeros(n), u=np.ones(n))


def warm_start(chans_f, lam):
    """Consistent (x0, y0) at the noisy image (see module docstring)."""
    M, N = chans_f[0].shape
    D = diff_matrix(M, N)
    xs, ys = [], []
    for f in chans_f:
        u0 = np.clip(f.ravel(), DELTA, 1 - DELTA)
        du = D @ u0
        xs.append(np.r_[u0, np.abs(u0 - f.ravel()) + DELTA, np.abs(du) + DELTA])
        yd = 0.5 * lam * np.ones(u0.size)
        tt = np.where(du > 0, 1 - DELTA, np.where(du < 0, DELTA, 0.5))   # top row: t >= D u active if D u > 0
        ys.append(np.r_[yd, lam - yd, tt, 1 - tt])
    return np.concatenate(xs), np.concatenate(ys)


def build_sets(n_color, frac):
    from sklearn.datasets import fetch_olivetti_faces
    from pdlp_warmstart_explore import load_imagenette
    faces = fetch_olivetti_faces().images.astype(np.float64)
    tr, va, te = split_persons_by_subject(12345, 28, 4, 8, 40)
    sets = {}
    for name, persons in (("train", tr), ("val", va), ("test", te)):
        idx = np.concatenate([np.arange(p * 10, (p + 1) * 10) for p in persons])
        sets[name] = [(faces[i][..., None], 3000 + int(i)) for i in idx]
    sets["tiny"] = [(im.astype(float) / 255, 70000 + i) for i, im in enumerate(load_color_images(n_color))]
    sets["imagenette"] = [(im.astype(float) / 255, 80000 + i) for i, im in
                          enumerate(load_imagenette(SRC_DIR.parent / "data" / "imagenette2-160", n_color, 64))]
    return {k: [(im, noisy(im, frac, np.random.default_rng(s))) for im, s in v] for k, v in sets.items()}


def solve_cached(key, f_chans, lam):
    path = CACHE / key
    if path.exists():
        z = np.load(path)
        return z["raw_x"], z["raw_y"], float(z["f_opt"]), float(z["opnorm"])
    m = tvl1_lp(f_chans, lam)
    s = solve_reduced(m)
    out = (s["raw_x"], s["raw_y"], float(s["objective_value"]), reduced_opnorm(tvl1_lp(f_chans[:1], lam)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw_x=out[0], raw_y=out[1], f_opt=out[2], opnorm=out[3])
    return out


def chans(img):
    return [img[:, :, c] for c in range(img.shape[2])]


def train_l2o(G, c, l, u, train, val, K, M_val, args):
    """Replica of the UnifiedTrainer L2O path; data are jit arguments (never closed over)."""
    m = G.shape[0]

    def gap_traj(stepsizes, h, x0, y0, xo, yo):
        v, y = problem_data_to_cp_lp_trajectories(stepsizes, c, G, h, l, u, xo, yo, x0, y0, K, m,
                                                  return_Gram_representation=False)[:2]
        L = lambda xx, yy: c @ xx - yy @ (G @ xx) + h @ yy
        return jax.vmap(lambda xk, yk: L(xk, yo) - L(xo, yk))(v, y)

    keys = ("h", "x0", "y0", "x_opt", "y_opt")
    batched = jax.vmap(lambda s, *a: gap_traj(s, *a), in_axes=(None,) + (0,) * 5)
    w = 0.9 ** (K - jnp.arange(K + 1)); w = w / w.sum()
    sq = lambda raw: tuple(r ** 2 for r in raw)
    loss = lambda raw, d: jnp.mean(batched(sq(raw), *(d[k] for k in keys)) @ w)
    vjit = jax.jit(lambda raw, d: jnp.mean(batched(sq(raw), *(d[k] for k in keys))[:, -1]))
    vg = jax.jit(jax.value_and_grad(loss))
    raw = tuple(jnp.sqrt(s) for s in (jnp.full(K, 0.5 / M_val), jnp.full(K, 0.5 / M_val), jnp.ones(K)))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(
        optax.warmup_cosine_decay_schedule(1e-6, args.eta, int(0.1 * args.iters), args.iters, 1e-6),
        weight_decay=0.0))
    state = opt.init(raw)
    n = train["h"].shape[0]; n_mb = n // args.N
    rng = np.random.default_rng(500)
    rec = lambda it, tl, r: dict(iteration=it, training_loss=tl, validation_loss=float(vjit(r, val)),
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


def pdhg(m, x0, y0, tau, sigma, theta):
    x, y = x0.copy(), y0.copy()
    for k in range(len(tau)):
        xn = np.clip(x - tau[k] * (m.c - m.G.T @ y), m.l, m.u)
        y = np.maximum(y + sigma[k] * (m.h - m.G @ (xn + theta[k] * (xn - x))), 0.0)
        x = xn
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--n-color", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"tvl1_l{args.lam:g}_n{args.noise:g}"

    sets = build_sets(args.n_color, args.noise)
    sols = {k: Parallel(n_jobs=args.n_jobs)(delayed(solve_cached)(f"{tag}_{k}/{j}.npz", chans(fn), args.lam)
                                            for j, (_, fn) in enumerate(tqdm(v, desc=f"solve {k}")))
            for k, v in sets.items()}
    M_val = 1.2 * max(s[3] for s in sols["train"])
    print(f"||G|| (identical for every image): {sols['train'][0][3]:.4f} gray; M_val {M_val:.4f}")

    tmpl = tvl1_lp(chans(sets["train"][0][1]), args.lam)
    G = jsparse.BCOO.from_scipy_sparse(tmpl.G)
    c, l, u = jnp.array(tmpl.c), jnp.array(tmpl.l), jnp.array(tmpl.u)

    def arrays(name):
        out = {k: [] for k in ("h", "x0", "y0", "x_opt", "y_opt")}
        for (_, fn), (rx, ry, _, _) in zip(sets[name], sols[name]):
            m = tvl1_lp(chans(fn), args.lam); x0, y0 = warm_start(chans(fn), args.lam)
            out["h"].append(m.h); out["x0"].append(x0); out["y0"].append(y0)
            out["x_opt"].append(rx); out["y_opt"].append(ry)
        return {k: jnp.array(np.stack(v)) for k, v in out.items()}

    df = train_l2o(G, c, l, u, arrays("train"), arrays("val"), args.K, M_val, args)
    df.to_csv(args.out / f"progress_K{args.K}.csv", index=False)
    best = int(df.validation_loss.idxmin())
    print(f"validation {df.validation_loss[0]:.4g} -> best {df.validation_loss[best]:.4g} at {best}")
    row = lambda r: tuple(df.loc[r, [f"{p}_{k}" for k in range(args.K)]].to_numpy(float) for p in ("tau", "sigma", "theta"))
    sched = {"untrained": row(0), "l2o": row(best)}

    res, keep = [], {}
    for name in ("test", "tiny", "imagenette"):
        for j, ((clean, fn), (rx, ry, f, _)) in enumerate(zip(sets[name], sols[name])):
            m = tvl1_lp(chans(fn), args.lam); x0, y0 = warm_start(chans(fn), args.lam)
            C = clean.shape[2]; K2 = clean.shape[0] * clean.shape[1]; nb = m.c.size // C
            img = lambda x: np.stack([x[q * nb:q * nb + K2] for q in range(C)], -1).reshape(clean.shape)
            L = lambda xx, yy: m.c @ xx - yy @ (m.G @ xx) + m.h @ yy
            for lab, s in sched.items():
                x, y = pdhg(m, x0, y0, *s)
                res.append(dict(split=name, idx=j, label=lab, gap=L(x, ry) - L(rx, y),
                                rmse_clean=float(np.sqrt(np.mean((img(x) - clean) ** 2))),
                                rmse_lp=float(np.sqrt(np.mean((img(x) - img(rx)) ** 2)))))
                if j < 3:
                    keep[(name, j, lab)] = img(x)
            if j < 3:
                keep[(name, j, "clean")], keep[(name, j, "noisy")], keep[(name, j, "lp")] = clean, fn, img(rx)
    df_r = pd.DataFrame(res)
    df_r.to_csv(args.out / f"eval_K{args.K}.csv", index=False)
    for name in ("test", "tiny", "imagenette"):
        p = df_r[df_r.split == name].pivot(index="idx", columns="label", values="gap")
        q = df_r[df_r.split == name].pivot(index="idx", columns="label", values="rmse_clean")
        print(f"{name:10s} gap untr {p.untrained.mean():9.3f} l2o {p.l2o.mean():9.3f} (l2o/untr {p.l2o.mean() / p.untrained.mean():.4f}, "
              f"l2o worse on {np.mean(p.l2o > p.untrained):.2f}) | RMSE vs clean untr {q.untrained.mean():.4f} l2o {q.l2o.mean():.4f}")

    rows_fig = [(n, j) for n in ("test", "tiny", "imagenette") for j in (0, 1)]
    fig, ax = plt.subplots(len(rows_fig), 5, figsize=(12, 2.6 * len(rows_fig)))
    for r, (n, j) in enumerate(rows_fig):
        for c_, (k, t) in enumerate((("clean", "clean"), ("noisy", f"noisy ({args.noise:.0%} salt & pepper)"),
                                     ("lp", "LP optimum"), ("untrained", f"untrained, K={args.K}"), ("l2o", f"L2O, K={args.K}"))):
            a = keep[(n, j, k)]
            ax[r, c_].imshow(np.clip(a.squeeze(), 0, 1), cmap="gray" if a.shape[2] == 1 else None, vmin=0, vmax=1)
            ax[r, c_].set_title(t, fontsize=9); ax[r, c_].set_xticks([]); ax[r, c_].set_yticks([])
        ax[r, 0].set_ylabel(f"{n} {j}", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / f"denoise_K{args.K}.png", dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
