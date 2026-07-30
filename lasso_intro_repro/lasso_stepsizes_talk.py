"""Talk-style bar charts of the learned ISTA stepsize schedules (K = 15).

One panel per baseline (L2O, DR-L2O, OPT-PEP), using the exact schedules
behind the reported loss figures (data/selected_schedules.csv). Bars are
filled with the same light tint as the quantile bands (line color at
alpha 0.2) and edged with the full line color. A dashed reference line
marks the ISTA stability threshold 2/L.

Style matches lasso_losses_talk.py: Helvetica Neue for titles, LaTeX
Computer Modern for all math and axis ticks, talk-scale fonts.

Usage: .venv/bin/python lasso_stepsizes_talk.py -> lasso_stepsizes_talk.pdf
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 15
SERIES = ["L2O", "DR-L2O", "OPT-PEP"]
COLORS = {"L2O": "#DC3220", "DR-L2O": "#005AB5", "OPT-PEP": "#00B32D"}

# ---------------------------------------------------------------- data ----

A = np.load("data/test_sets/A_in_dist.npz")["A"]
L_SMOOTH = float(np.linalg.eigvalsh(A.T @ A).max())

manifest = pd.read_csv("data/selected_schedules.csv").set_index(["series", "K"])
schedules = {}
for s in SERIES:
    r = manifest.loc[(s, K)]
    df = pd.read_csv(r["csv"])
    schedules[s] = df.loc[r["row"], [f"gamma_{k}" for k in range(K)]].to_numpy(float)

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
    "xtick.major.width": 0, "xtick.minor.width": 0,
    "ytick.major.width": 0, "ytick.minor.width": 0,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.19, wspace=0.10)

ymax = 1.25
ks = np.arange(1, K + 1)
for ax, s in zip(axes, SERIES):
    ax.bar(ks, schedules[s], width=0.75,
           facecolor=matplotlib.colors.to_rgba(COLORS[s], 0.2),
           edgecolor=COLORS[s], linewidth=2.0, zorder=3)
    ax.axhline(2 / L_SMOOTH, color="0.25", ls="--", lw=2.0, zorder=4)
    ax.set_xlim(0.25, K + 0.75)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(2, K + 1, 2))
    ax.set_xticklabels([f"${k}$" for k in range(2, K + 1, 2)], usetex=True)
    ax.set_yticks(np.arange(0, 1.3, 0.25))
    ax.set_yticklabels([f"${v:g}$" for v in np.arange(0, 1.3, 0.25)], usetex=True)
    ax.set_xlabel("$k$", usetex=True)
    ax.set_title(s, pad=12, color=COLORS[s])   # Helvetica Neue
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Step size")   # Helvetica Neue, no math
axes[0].text(K + 0.4, 2 / L_SMOOTH + 0.03, r"$2/L$", usetex=True,
             ha="right", va="bottom", fontsize=22, color="0.25")

fig.savefig("lasso_stepsizes_talk.pdf")
print("wrote lasso_stepsizes_talk.pdf")
