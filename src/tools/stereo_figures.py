"""Per-instance stereo reconstruction figures: L2O vs DR-L2O after K PDHG steps.

For chosen instances of the gray test / Tiny ImageNet / Imagenette sets (as in
tools/stereo_l2o_check.py, same warm start), each figure shows
  row 1: left view, right view, anaglyph, true disparity, LP-optimal disparity
  row 2: disparity after K steps (untrained, L2O, DR-L2O) on the true-disparity
         color scale, and |d_K - d_LP| for L2O and DR-L2O on a shared scale.

Usage (from src/):
    python tools/stereo_figures.py --runs l2o=<csv> dr=<csv> --eval <eval_K5_runs.csv> --out <dir>
"""
import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
for _p in (SRC_DIR, SRC_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from learning.stereo_lp import stereo_x0, stereo_y0  # noqa: E402
from pdlp_warmstart_explore import load_schedules  # noqa: E402
from stereo_l2o_check import CACHE, build_sets, instance_builder  # noqa: E402

plt.rcParams["text.usetex"] = False


def pdhg(m, x0, y0, tau, sigma, theta):
    x, y = x0.copy(), y0.copy()
    for k in range(len(tau)):
        xn = np.clip(x - tau[k] * (m.c - m.G.T @ y), m.l, m.u)
        y = np.maximum(y + sigma[k] * (m.h - m.G @ (xn + theta[k] * (xn - x))), 0.0)
        x = xn
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs=2, required=True, help="l2o=<csv> dr=<csv>")
    ap.add_argument("--eval", required=True, type=Path, help="eval_K5_runs.csv from stereo_l2o_check.py")
    ap.add_argument("--dr-label", default="dr10", help="label of the DR run in the eval CSV")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    S = load_schedules(args.runs)
    (l2o_lab, dr_lab) = [r.split("=", 1)[0] for r in args.runs]
    sched = {"untrained": S[f"{l2o_lab}:untrained"][1:], "L2O": S[l2o_lab][1:], "DR-L2O": S[dr_lab][1:]}

    ev = pd.read_csv(args.eval)
    picks = []
    for split in ("test", "tiny", "imagenette"):
        g = ev[ev.split == split].pivot(index="idx", columns="label", values="gap")
        ratio = (g[args.dr_label] / g["l2o"]).sort_values()
        med = int(ratio.index[len(ratio) // 2])
        picks.append((split, med, "median DR/L2O gap ratio"))
        if split != "test":
            picks.append((split, int(ratio.index[0]), "best DR/L2O gap ratio"))

    sets = build_sets(argparse.Namespace(n_color=40))
    for split, j, why in picks:
        im, seed = sets[split][j]
        m, d_true, L = instance_builder(im, seed, args.lam, "inside")
        K_pix = d_true.size
        z = np.load(CACHE / f"stereo_inside{args.lam:g}_{split}/{j}.npz")
        rx, ry = z["raw_x"], z["raw_y"]
        d_lp = rx[:K_pix].reshape(d_true.shape)
        x0 = stereo_x0(m.l, m.u, K_pix, feasible_h=m.h, n_channels=im.shape[2])
        y0 = stereo_y0(m.h, K_pix, im.shape[2])
        Lg = lambda xx, yy: m.c @ xx - yy @ (m.G @ xx) + m.h @ yy
        out = {}
        for lab, (t, s_, th) in sched.items():
            x = pdhg(m, x0, y0, t, s_, th)
            out[lab] = x[:K_pix].reshape(d_true.shape)
        gaps = {lab: float(ev[(ev.split == split) & (ev.idx == j) & (ev.label == key)].gap.iloc[0])
                for lab, key in (("untrained", "untrained"), ("L2O", "l2o"), ("DR-L2O", args.dr_label))}

        R = [np.stack(c, -1) for c in (L,)][0].squeeze()
        from learning.stereo_lp import warp
        right = np.stack([warp(c, d_true) for c in L], -1).squeeze()
        gl, gr = np.mean(L, 0), np.mean([warp(c, d_true) for c in L], 0)
        vmin, vmax = min(d_true.min(), d_lp.min()), max(d_true.max(), d_lp.max())
        errs = {lab: np.abs(out[lab] - d_lp) for lab in ("L2O", "DR-L2O")}
        emax = max(e.max() for e in errs.values())

        fig, ax = plt.subplots(2, 5, figsize=(15, 6.6))
        gray = im.shape[2] == 1
        ax[0, 0].imshow(R, cmap="gray" if gray else None, vmin=0, vmax=1); ax[0, 0].set_title("left view")
        ax[0, 1].imshow(right, cmap="gray" if gray else None, vmin=0, vmax=1); ax[0, 1].set_title("right view")
        ax[0, 2].imshow(np.stack([gl, gr, gr], -1)); ax[0, 2].set_title("anaglyph (L red / R cyan)")
        ax[0, 3].imshow(d_true, cmap="turbo", vmin=vmin, vmax=vmax); ax[0, 3].set_title("true disparity")
        im_lp = ax[0, 4].imshow(d_lp, cmap="turbo", vmin=vmin, vmax=vmax); ax[0, 4].set_title("LP-optimal disparity")
        for c, lab in enumerate(("untrained", "L2O", "DR-L2O")):
            rmse = np.sqrt(np.mean((out[lab] - d_lp) ** 2))
            ax[1, c].imshow(out[lab], cmap="turbo", vmin=vmin, vmax=vmax)
            ax[1, c].set_title(f"{lab}, K={len(sched[lab][0])}\ngap {gaps[lab]:.1f}   RMSE vs LP {rmse:.3f}", fontsize=9)
        for c, lab in zip((3, 4), ("L2O", "DR-L2O")):
            im_e = ax[1, c].imshow(errs[lab], cmap="magma", vmin=0, vmax=emax)
            ax[1, c].set_title(f"|{lab} - LP optimum|\nmean {errs[lab].mean():.3f}", fontsize=9)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im_lp, ax=ax[0, 4], fraction=0.046, pad=0.02, label="disparity (px)")
        fig.colorbar(im_e, ax=ax[1, 4], fraction=0.046, pad=0.02, label="abs error (px)")
        name = {"test": "gray test", "tiny": "Tiny ImageNet", "imagenette": "Imagenette"}[split]
        fig.suptitle(f"{name} image {j} ({why}: {gaps['DR-L2O'] / gaps['L2O']:.3f})", fontsize=12)
        fig.tight_layout()
        path = args.out / f"stereo_{split}_{j}.png"
        fig.savefig(path, dpi=110); plt.close(fig)
        print(f"{path}: gaps untrained {gaps['untrained']:.1f} L2O {gaps['L2O']:.1f} DR {gaps['DR-L2O']:.1f} | "
              f"RMSE vs LP  L2O {np.sqrt(np.mean(errs['L2O'] ** 2)):.3f}  DR {np.sqrt(np.mean(errs['DR-L2O'] ** 2)):.3f} | "
              f"disparity range true [{d_true.min():.2f}, {d_true.max():.2f}] "
              f"L2O [{out['L2O'].min():.2f}, {out['L2O'].max():.2f}] DR [{out['DR-L2O'].min():.2f}, {out['DR-L2O'].max():.2f}]")


if __name__ == "__main__":
    main()
