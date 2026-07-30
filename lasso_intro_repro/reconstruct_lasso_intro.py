"""Reconstruct figures/lasso_intro.pdf of the DR-L2O paper from raw run data.

Pipeline
--------
1. Schedule selection (per framework, per K in 1..15), from the SLURM sweep
   CSVs pulled from della (see README.md for provenance):
     - L2O      : minimum-validation-loss row across all l2o runs
                  (eta sweep, training_sample_N=1000, May 3-4 2026).
     - DR-L2O   : eps = 10 fixed (the robust choice shown in the figure;
                  NOT cross-validated eps), eta = 1e-3, minimum-validation row.
     - OPT-PEP  : eta = 1e-4 (validation-selected over {1e-5, 1e-4}),
                  minimum-validation row.
2. Evaluate each schedule with ISTA (x0 = 0, lambda = 0.4) on
     - the in-distribution test set (250 instances, archived on della), and
     - the out-of-distribution test set (250 instances, x ~ N(0, 3^2),
       regenerated deterministically by regenerate_ood_set.py).
   Per-instance loss: f(x_K) - f*.
3. Statistics per (panel, series, K):
     - in-distribution: the test instances on which the L2O schedules
       diverge (loss >= initial loss; concretely instances 111 and 189)
       are excluded for ALL series, then mean / q10 / q90 over the
       remaining 248 instances;
     - out-of-distribution: all 250 instances, no exclusion.
   (Both conventions verified to 6+ significant digits against values
   extracted from the original PDF; see compare step.)
4. Compare against data/pdf_extracted.csv (ground truth parsed from the
   original figure) and write data/reconstruction_vs_original.csv.
5. Render lasso_intro_reconstructed.pdf with the exact style/geometry
   measured from the original (page 492.48 x 198.37 pt, axes rects, fonts,
   colors #DC3220/#005AB5/#00B32D, marker size 5, band alpha 0.2,
   grid alpha 0.3, LaTeX Computer Modern).

Usage: .venv/bin/python reconstruct_lasso_intro.py
"""

import glob
import os

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBD = 0.4
KS = range(1, 16)

COLORS = {"L2O": "#DC3220", "DR-L2O": "#005AB5", "OPT-PEP": "#00B32D"}
MARKERS = {"L2O": "o", "DR-L2O": "s", "OPT-PEP": "^"}
SERIES = ["L2O", "DR-L2O", "OPT-PEP"]

# ---------------------------------------------------------------- data ----

A = np.load("data/test_sets/A_in_dist.npz")["A"]
TEST = np.load("data/test_sets/test_set.npz")
OOD = np.load("data/test_sets/ood_set_normal3.npz")


def ista_losses(gammas, dataset):
    """f(x_K) - f* on every instance of `dataset`, ISTA from x0 = 0."""
    B, fopt = dataset["b_batch"], dataset["f_opt_batch"]
    X = np.zeros((B.shape[0], A.shape[1]))
    for g in gammas:
        Y = X - g * ((X @ A.T - B) @ A)
        X = np.sign(Y) * np.maximum(np.abs(Y) - g * LAMBD, 0)
    F = 0.5 * np.sum((X @ A.T - B) ** 2, axis=1) + LAMBD * np.sum(np.abs(X), axis=1)
    return F - fopt


def initial_losses(dataset):
    B, fopt = dataset["b_batch"], dataset["f_opt_batch"]
    return 0.5 * np.sum(B ** 2, axis=1) - fopt


L0_TEST = initial_losses(TEST)


def stats(losses, keep=None):
    if keep is not None:
        losses = losses[keep]
    return losses.mean(), np.quantile(losses, 0.1), np.quantile(losses, 0.9)


# ------------------------------------------------------- schedule pick ----

def load_candidates(framework, K, want=None):
    """Yield (val_loss, gammas, meta) for the min-validation row of every run
    covering this K, optionally restricted by config values in `want`."""
    out = []
    for csv in glob.glob(f"data/runs/{framework}/*/*/*/K_{K}/progress.csv"):
        cfg_path = os.path.join(csv.split("/learn_dro_outputs/")[0], ".hydra/config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        if want and any(cfg.get(k) != v for k, v in want.items()):
            continue
        df = pd.read_csv(csv)
        i = int(df["validation_loss"].idxmin())
        gam = df.loc[i, [f"gamma_{k}" for k in range(K)]].to_numpy(float)
        out.append((float(df.loc[i, "validation_loss"]), gam,
                    dict(csv=csv, row=i, eps=cfg.get("eps"), eta=cfg.get("eta_t"))))
    return out


def select_schedule(series, K):
    if series == "L2O":
        cands = load_candidates("l2o", K)
    elif series == "DR-L2O":
        cands = load_candidates("ldro-pep", K, want={"eps": 10.0, "eta_t": 0.001})
    else:  # OPT-PEP
        cands = load_candidates("lpep", K, want={"eta_t": 0.0001})
    val, gam, meta = min(cands, key=lambda c: c[0])
    meta["val_loss"] = val
    return gam, meta


# ------------------------------------------------------------ evaluate ----

schedules = {(series, K): select_schedule(series, K)
             for series in SERIES for K in KS}

# In-dist exclusion set: instances on which any selected L2O schedule
# diverges (loss >= initial loss). Resolves to test instances {111, 189}.
diverged = np.zeros(len(L0_TEST), dtype=bool)
for K in KS:
    gam, _ = schedules[("L2O", K)]
    diverged |= ista_losses(gam, TEST) >= L0_TEST
keep_in = ~diverged
print(f"in-dist panel excludes test instances {sorted(np.where(diverged)[0])}")

rows, manifest = [], []
for series in SERIES:
    for K in KS:
        gam, meta = schedules[(series, K)]
        m, q10, q90 = stats(ista_losses(gam, TEST), keep=keep_in)
        rows.append(dict(panel="in_dist", series=series, K=K, mean=m, q10=q10, q90=q90))
        m, q10, q90 = stats(ista_losses(gam, OOD))
        rows.append(dict(panel="ood", series=series, K=K, mean=m, q10=q10, q90=q90))
        manifest.append(dict(series=series, K=K, **{k: meta[k] for k in
                                                    ("csv", "row", "eps", "eta", "val_loss")}))

res = pd.DataFrame(rows)
pd.DataFrame(manifest).to_csv("data/selected_schedules.csv", index=False)

# ------------------------------------------------------------- compare ----

if os.path.exists("data/pdf_extracted.csv"):
    ref = pd.read_csv("data/pdf_extracted.csv")
    ref["panel"] = ref["panel"].replace({"in_dist": "in_dist", "ood": "ood"})
    cmp = res.merge(ref, on=["panel", "series", "K"], suffixes=("", "_pdf"))
    for c in ["mean", "q10", "q90"]:
        cmp[f"{c}_relerr_pct"] = 100 * (cmp[c] / cmp[f"{c}_pdf"] - 1)
    cmp.to_csv("data/reconstruction_vs_original.csv", index=False)
    err = cmp[["mean_relerr_pct", "q10_relerr_pct", "q90_relerr_pct"]].abs()
    print("=== reconstruction vs original PDF (|relative error| %) ===")
    print(err.describe().loc[["mean", "50%", "max"]].to_string(float_format=lambda v: f"{v:.3f}"))
    worst = cmp.loc[err.max(axis=1).sort_values(ascending=False).index[:6],
                    ["panel", "series", "K", "mean_relerr_pct", "q10_relerr_pct", "q90_relerr_pct"]]
    print("worst rows:")
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------------------- plot ----

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    # the original draws tick marks with zero width (labels keep the
    # default 3.5 pt offset but no tick ink)
    "xtick.major.width": 0, "xtick.minor.width": 0,
    "ytick.major.width": 0, "ytick.minor.width": 0,
})

# exact geometry measured from the original PDF (points)
PAGE_W, PAGE_H = 492.48, 198.3708946431
AX_W, AX_H, AX_Y0 = 198.086038829, 109.2953679075, 67.5733469217
AX_X0_L, AX_X0_R = 48.9873489551, 287.193961171
# y-transforms: gridline anchors (value at y-position, points per decade)
LEFT_ANCHOR, LEFT_PPD = (103.891702, 1e-1), 27.6609910
RIGHT_ANCHOR, RIGHT_PPD = (99.450116, 1e0), 25.8061995


def ylim_from_geometry(anchor, ppd):
    y_anchor, v_anchor = anchor
    lo = v_anchor * 10 ** ((AX_Y0 - y_anchor) / ppd)
    hi = v_anchor * 10 ** ((AX_Y0 + AX_H - y_anchor) / ppd)
    return lo, hi


fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
ax_l = fig.add_axes([AX_X0_L / PAGE_W, AX_Y0 / PAGE_H, AX_W / PAGE_W, AX_H / PAGE_H])
ax_r = fig.add_axes([AX_X0_R / PAGE_W, AX_Y0 / PAGE_H, AX_W / PAGE_W, AX_H / PAGE_H])

for ax, panel, title, (anchor, ppd) in [
    (ax_l, "in_dist", "In-distribution", (LEFT_ANCHOR, LEFT_PPD)),
    (ax_r, "ood", "Out-of-distribution", (RIGHT_ANCHOR, RIGHT_PPD)),
]:
    for series in SERIES:
        d = res[(res.panel == panel) & (res.series == series)].sort_values("K")
        ax.fill_between(d.K, d.q10, d.q90, color=COLORS[series], alpha=0.2, lw=0)
        ax.plot(d.K, d["mean"], color=COLORS[series], marker=MARKERS[series],
                markersize=5, lw=1.5, zorder=3, label=series)
    ax.set_yscale("log")
    ax.set_xlim(0.3, 15.7)
    ax.set_ylim(*ylim_from_geometry(anchor, ppd))
    ax.set_xticks(range(2, 15, 2))
    ax.set_xlabel("$K$")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)

# major y ticks every second decade, as in the original; no minor labels
ax_l.set_yticks([1e-1, 1e1])
ax_r.set_yticks([1e0, 1e2])
for ax in (ax_l, ax_r):
    ax.minorticks_off()

ax_l.set_ylabel("Test loss")

handles, labels = ax_l.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 1.7 / PAGE_H), framealpha=0.8)

fig.savefig("lasso_intro_reconstructed.pdf")
print("wrote lasso_intro_reconstructed.pdf")
