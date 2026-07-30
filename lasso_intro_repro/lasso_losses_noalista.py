"""Modified version of the paper's figures/lasso_losses.pdf.

Reproduces the appendix figure (Fig. `lasso_losses` of the DR-L2O paper)
with two requested changes:
  1. L2O-ALISTA is removed (leaving L2O-ISTA, DR-L2O, OPT-PEP);
  2. both panels share the same y-axis limits (union of the original
     per-panel ranges), which makes the out-of-distribution divergence of
     L2O-ISTA visually much more dramatic.

Everything else follows the original exactly: page/axes geometry, fonts,
colors, markers, grid, legend placement, and the data conventions verified
for lasso_intro.pdf (see reconstruct_lasso_intro.py):
  - schedules from data/selected_schedules.csv (validation-selected rows of
    the della sweeps; DR-L2O at eps=10),
  - in-distribution panel excludes the two test instances on which L2O
    diverges (111, 189); out-of-distribution panel uses all 250,
  - mean line, 10th-90th quantile band.

Before plotting, the three retained series are verified against the curve
values extracted from the original lasso_losses.pdf (they match to <0.05%;
the original figure uses the same underlying data as lasso_intro.pdf).

Usage: .venv/bin/python lasso_losses_noalista.py
  -> lasso_losses_noalista.pdf, data/pdf_extracted_losses.csv
"""

import re
import zlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBD = 0.4
KS = range(1, 16)
SERIES = ["L2O", "DR-L2O", "OPT-PEP"]          # manifest names
LABELS = {"L2O": "L2O", "DR-L2O": "DR-L2O", "OPT-PEP": "OPT-PEP"}
COLORS = {"L2O": "#DC3220", "DR-L2O": "#005AB5", "OPT-PEP": "#00B32D"}
MARKERS = {"L2O": "o", "DR-L2O": "s", "OPT-PEP": "^"}

LOSSES_PDF = (
    "/Users/bs37/Library/CloudStorage/Dropbox/work/research/papers/2026/"
    "dr-l2o/figures/lasso_losses.pdf"
)

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

# --------------------------- extract + verify against original figure ----

SERIES_BY_COLOR = {
    "0.862745 0.196078 0.125490": "L2O",
    "0.000000 0.352941 0.709804": "DR-L2O",
    "0.000000 0.701961 0.176471": "OPT-PEP",
    "1.000000 0.800000 0.000000": "ALISTA",
}
# geometry of lasso_losses.pdf, measured from its content stream
GEO = {
    "in_dist": dict(x0=51.1340382501, xK1=60.0898, dxK=197.0126941815 / 15.4,
                    y_anchor=170.560, v_anchor=1e0, ppd=45.7610),
    "ood": dict(x0=288.2673058185, xK1=297.2231, dxK=197.0126941815 / 15.4,
                y_anchor=160.376, v_anchor=1e1, ppd=42.8065),
}


def extract_original():
    raw = open(LOSSES_PDF, "rb").read()
    content = b""
    for s in re.findall(rb"stream\r?\n(.*?)endstream", raw, re.S):
        try:
            content += zlib.decompress(s)
        except zlib.error:
            pass
    txt = content.decode("latin1")
    token_re = re.compile(
        r"([\d.\-]+ [\d.\-]+ (?:m|l))|([\d.\-]+ [\d.\-]+ [\d.\-]+ (?:RG|rg))|(\bS\b|\bf\b|\bB\b)")
    paths, cur, stroke, fill = [], [], None, None
    for mv, col, paint in token_re.findall(txt):
        if mv:
            x, y, op = mv.split()
            if op == "m":
                cur = [(float(x), float(y))]
            else:
                cur.append((float(x), float(y)))
        elif col:
            r, g, b, op = col.split()
            key = f"{float(r):.6f} {float(g):.6f} {float(b):.6f}"
            if op == "RG":
                stroke = key
            else:
                fill = key
        elif paint:
            if cur:
                paths.append((paint, stroke, fill, cur))
            cur = []
    records = {}
    for paint, stroke, fill, pts in paths:
        series = SERIES_BY_COLOR.get(stroke if paint == "S" else fill)
        if series is None or len(pts) < 10:
            continue
        panel = "in_dist" if np.mean([p[0] for p in pts]) < 265 else "ood"
        g = GEO[panel]
        rec = records.setdefault((panel, series), {})
        for x, y in pts:
            K = 1.0 + (x - g["xK1"]) / g["dxK"]
            v = g["v_anchor"] * 10 ** ((y - g["y_anchor"]) / g["ppd"])
            Kr = round(K)
            if abs(K - Kr) > 1e-3:
                continue
            d = rec.setdefault(Kr, {})
            if paint == "S":
                d["mean"] = v
            else:
                d["q10"] = min(d.get("q10", np.inf), v)
                d["q90"] = max(d.get("q90", -np.inf), v)
    out = [dict(panel=p, series=s, K=K, **vals)
           for (p, s), rec in sorted(records.items()) for K, vals in sorted(rec.items())]
    return pd.DataFrame(out)


ref = extract_original()
ref.to_csv("data/pdf_extracted_losses.csv", index=False)
cmp = res.merge(ref[ref.series != "ALISTA"], on=["panel", "series", "K"],
                suffixes=("", "_pdf"))
errs = []
for c in ["mean", "q10", "q90"]:
    errs.append((100 * (cmp[c] / cmp[f"{c}_pdf"] - 1)).abs())
err = pd.concat(errs, axis=1)
print("=== retained series vs original lasso_losses.pdf (|rel err| %) ===")
print(f"median {err.stack().median():.4f}   max {err.stack().max():.4f}")

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
    "xtick.major.width": 0, "xtick.minor.width": 0,
    "ytick.major.width": 0, "ytick.minor.width": 0,
})

PAGE_W, PAGE_H = 492.48, 267.4908946431
AX_W, AX_H, AX_Y0 = 197.0126941815, 181.2953679075, 64.6933469217
AX_X0_L, AX_X0_R = 51.1340382501, 288.2673058185

# shared y-limits: union of the original panels' ranges
# (left bottom 10^(-2.3135) ... right top 10^3)
YLIM = (10 ** (-2.0 + (64.6933469217 - 79.038) / 45.7610), 1e3)

fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
ax_l = fig.add_axes([AX_X0_L / PAGE_W, AX_Y0 / PAGE_H, AX_W / PAGE_W, AX_H / PAGE_H])
ax_r = fig.add_axes([AX_X0_R / PAGE_W, AX_Y0 / PAGE_H, AX_W / PAGE_W, AX_H / PAGE_H])

for ax, panel, title in [(ax_l, "in_dist", "In-distribution"),
                         (ax_r, "ood", "Out-of-distribution")]:
    for series in SERIES:
        d = res[(res.panel == panel) & (res.series == series)].sort_values("K")
        ax.fill_between(d.K, d.q10, d.q90, color=COLORS[series], alpha=0.2, lw=0)
        ax.plot(d.K, d["mean"], color=COLORS[series], marker=MARKERS[series],
                markersize=5, lw=1.5, zorder=3, label=LABELS[series])
    ax.set_yscale("log")
    ax.set_xlim(0.3, 15.7)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(2, 15, 2))
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    ax.minorticks_off()
    ax.set_xlabel("$K$")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)

ax_l.set_ylabel(r"Avg. $f(x^K) - f(x^\star)$")

handles, labels = ax_l.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 1.7 / PAGE_H), framealpha=0.8)

fig.savefig("lasso_losses_noalista.pdf")
print("wrote lasso_losses_noalista.pdf")
