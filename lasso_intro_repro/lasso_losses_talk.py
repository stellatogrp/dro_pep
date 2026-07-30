"""Talk version of lasso_losses_noalista.pdf (for slides).

Same data and conventions as lasso_losses_noalista.py (L2O / DR-L2O /
OPT-PEP, shared y-limits across panels), restyled for a talk:
  - larger canvas and talk-scale fonts,
  - Helvetica Neue for all non-math text (titles, legend, tick numbers),
  - math (axis labels, y tick exponents) still LaTeX / Computer Modern,
    rendered per-text with usetex (pdflatex cannot load the system
    Helvetica Neue, so usetex is off globally and on for math text only),
  - thicker lines and bigger markers.

Usage: .venv/bin/python lasso_losses_talk.py  ->  lasso_losses_talk.pdf
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBD = 0.4
KS = range(1, 16)
SERIES = ["L2O", "DR-L2O", "OPT-PEP"]
LABELS = {"L2O": "L2O", "DR-L2O": "DR-L2O", "OPT-PEP": "OPT-PEP"}
COLORS = {"L2O": "#DC3220", "DR-L2O": "#005AB5", "OPT-PEP": "#00B32D"}
MARKERS = {"L2O": "o", "DR-L2O": "s", "OPT-PEP": "^"}

# ------------------------------------------------------------ evaluate ----

A = np.load("data/test_sets/A_in_dist.npz")["A"]
TEST = np.load("data/test_sets/test_set.npz")
OOD = np.load("data/test_sets/ood_set_normal3.npz")
EXCLUDE_IN_DIST = [111, 189]


def ista_losses(gammas, dataset):
    B, fopt = dataset["b_batch"], dataset["f_opt_batch"]
    X = np.zeros((B.shape[0], A.shape[1]))
    for g in gammas:
        Y = X - g * ((X @ A.T - B) @ A)
        X = np.sign(Y) * np.maximum(np.abs(Y) - g * LAMBD, 0)
    F = 0.5 * np.sum((X @ A.T - B) ** 2, axis=1) + LAMBD * np.sum(np.abs(X), axis=1)
    return F - fopt


manifest = pd.read_csv("data/selected_schedules.csv").set_index(["series", "K"])
keep_in = np.ones(250, dtype=bool)
keep_in[EXCLUDE_IN_DIST] = False

rows = []
for series in SERIES:
    for K in KS:
        r = manifest.loc[(series, K)]
        df = pd.read_csv(r["csv"])
        gam = df.loc[r["row"], [f"gamma_{k}" for k in range(K)]].to_numpy(float)
        for panel, L in [("in_dist", ista_losses(gam, TEST)[keep_in]),
                         ("ood", ista_losses(gam, OOD))]:
            rows.append(dict(panel=panel, series=series, K=K, mean=L.mean(),
                             q10=np.quantile(L, 0.1), q90=np.quantile(L, 0.9)))
res = pd.DataFrame(rows)

# ---------------------------------------------------------------- plot ----

plt.rcParams.update({
    "text.usetex": False,             # usetex turned on per-text for math
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 26,
    "axes.titlesize": 30,
    "axes.labelsize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
    "xtick.major.width": 0, "xtick.minor.width": 0,
    "ytick.major.width": 0, "ytick.minor.width": 0,
})

YLIM = (10 ** (-2.0 + (64.6933469217 - 79.038) / 45.7610), 1e3)
YTICKS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7.8))
fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.26, wspace=0.18)

for ax, panel, title in [(ax_l, "in_dist", "In-distribution"),
                         (ax_r, "ood", "Out-of-distribution")]:
    for series in SERIES:
        d = res[(res.panel == panel) & (res.series == series)].sort_values("K")
        ax.fill_between(d.K, d.q10, d.q90, color=COLORS[series], alpha=0.2, lw=0)
        ax.plot(d.K, d["mean"], color=COLORS[series], marker=MARKERS[series],
                markersize=11, lw=2.8, zorder=3, label=LABELS[series])
    ax.set_yscale("log")
    ax.set_xlim(0.3, 15.7)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(2, 15, 2))
    ax.set_xticklabels([f"${k}$" for k in range(2, 15, 2)], usetex=True)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in YTICKS], usetex=True)
    ax.minorticks_off()
    ax.set_xlabel("$K$", usetex=True)
    ax.set_title(title, pad=12)          # Helvetica Neue
    ax.grid(True, which="major", alpha=0.3)

ax_l.set_ylabel(r"Avg.\ $f(x^K) - f(x^\star)$", usetex=True)

handles, labels = ax_l.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.8,
           handlelength=1.8, columnspacing=1.6)

fig.savefig("lasso_losses_talk.pdf")
print("wrote lasso_losses_talk.pdf")
