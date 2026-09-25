"""Paper-style figures for the real-data (german.numer) LogReg experiment.

This is the ICLR addition: the DR-L2O paper has quad/lasso/pdlp figures but no
LogReg ones, so these are built to correspond to them -- same three series, same
colours and markers, same 1x2 in-distribution / out-of-distribution losses panel
and 2x3 fraction-solved grid.

Differences from quad/lasso/pdlp:

  * DR-L2O covers K = 1..15; L2O and OPT-PEP cover only K = {5, 10, 15}.
    The LogReg sweep trains ONE horizon per run, so each K is its own array
    task. DR-L2O was re-run across the full range once the per-task memory was
    sized from the measured leak (see run_learning_experiment.py); the L2O and
    OPT-PEP baselines have not been re-run, so their series are still three
    points. Curves are drawn over whatever K each series actually has, so a
    shorter baseline line is data coverage, not a gap in the plot.

  * No handcrafted reference curve. The quad/lasso/pdlp losses figures show
    only the three learned series, so the dashed grey GD(1/L) / Nesterov FGM
    line these panels used to carry has been dropped.

  * The solved thresholds are one decade lower. The paper uses
    eta in {1e-3, 1e-2, 1e-1}; LogReg's gap on german.numer spans ~1e-4..7e-2,
    so {1e-3, 1e-2, 1e-1} would saturate at 100% almost everywhere. See
    SOLVED_ETAS. The thresholding rule itself is the paper's: instance i counts
    as solved at level eta iff f(x_K_i) - f(x^*_i) <= eta * (1 + |f(x^*_i)|),
    the same relative rule as the lasso/pdlp figures and as
    build_logreg_table.stats. On german.numer f(x^*) in ~[0.35, 0.51], so the
    scaling is a mild (~1.4x) per-instance loosening rather than a reshuffle.

Schedule selection reuses learning/baselines/build_logreg_table.py rather than
reimplementing it: that module owns the per-framework SELECTION_METRIC rule
(validation loss for L2O/DR-L2O, training loss -- the worst-case PEP bound --
for OPT-PEP), and a second copy would drift from it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ICLR_OUT = HERE.parent                       # src/iclr_data_outputs
REPO = ICLR_OUT.parent.parent                # dro_pep
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'src'))

from _style import (ARCH_COLORS, ARCH_DISPLAY_NAMES, ARCH_MARKERS, ARCH_ORDER,  # noqa: E402
                    use_paper_style)
from learning.baselines import build_logreg_table as blt  # noqa: E402

K_VALS = list(range(1, 16))
# One decade below the paper's {1e-3, 1e-2, 1e-1}; see the module docstring.
SOLVED_ETAS = [1e-4, 1e-3, 1e-2]
WARMUP_ITERS = 5            # matches quad/lasso times.py
K_VALS_TIMES = list(range(1, 16))
# Every integer K=1..15 as a tick is unreadable at these panel widths. This is
# the tick set archive/quad/create_paper_plots.py uses for its K=1..15 sweeps,
# reused verbatim so the LogReg panels line up with the paper's.
X_TICKS = [2, 4, 6, 8, 10, 12, 14]

# Frac-solved grid layout (figure fractions of the 2.5 in tall figure). The
# legend hangs below `bottom` and savefig(bbox_inches='tight') crops to it.
FRAC_LAYOUT = dict(left=0.085, right=0.99, top=0.84, bottom=0.19,
                   hspace=0.24, wspace=0.16)
FRAC_LEGEND_Y = 0.045
FRAC_LEGEND_FONTSIZE = 10

# build_logreg_table's series names -> the arch keys _style uses.
SERIES_TO_ARCH = {'L2O': 'l2o', 'DR-L2O': 'ldro_pep', 'OPT-PEP': 'lpep'}
ARCH_TO_SERIES = {v: k for k, v in SERIES_TO_ARCH.items()}

ALG_CFG = {                       # cli name -> (config alg, display name)
    'gd': ('vanilla_gd', 'GD'),
    'fgm': ('nesterov_fgm', 'FGM'),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def default_runs_root():
    return ICLR_OUT


def default_data_dir():
    """Newest LogReg sample-creation bundle under iclr_data_outputs."""
    cands = sorted((ICLR_OUT / 'sample_creation_outputs' / 'LogReg').glob('*/*'))
    if not cands:
        raise SystemExit(
            'no LogReg sample-creation output under '
            f'{ICLR_OUT / "sample_creation_outputs" / "LogReg"}; pass --data-dir')
    return cands[-1]


def load_sets(data_dir):
    out = {}
    for name, fname in (('test', 'test_set.npz'), ('ood', 'ood_set.npz')):
        p = Path(data_dir) / fname
        if not p.is_file():
            raise SystemExit(f'missing {p}')
        d = np.load(p)
        out[name] = {k: d[k] for k in d.files}
    return out


def per_problem_losses(t_vec, beta_vec, ds):
    """f(x_K) - f* for every instance, via build_logreg_table's simulators."""
    if beta_vec is None:
        return np.asarray(blt.gd_losses(t_vec, ds))
    return np.asarray(blt.fgm_losses(t_vec, beta_vec, ds))


def solved_threshold(ds, eta):
    """Per-instance solved threshold eta * (1 + |f(x^*)|).

    The paper's relative rule (lasso/pdlp create_paper_plots.py, and
    build_logreg_table.stats), not an absolute gap: it divides out the
    instance-dependent magnitude of f(x^*). For quad, where f(x^*) = 0, it
    collapses to the absolute threshold the quad figures use.
    """
    return eta * (1.0 + np.abs(np.asarray(ds['f_opt_batch'])))


def select_all(runs_root, alg_cfg):
    """{arch: {K: (t, beta, meta)}} for one algorithm."""
    out = {}
    for series, arch in SERIES_TO_ARCH.items():
        for K in K_VALS:
            sel = blt.select_schedule(str(runs_root), series, alg_cfg, K)
            if sel is None:
                print(f'  warning: no {series} {alg_cfg} K={K} run found')
                continue
            _score, t, beta, meta = sel
            out.setdefault(arch, {})[K] = (t, beta, meta)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PANELS = [('test', 'In-distribution'), ('ood', 'Out-of-distribution')]
# Short row labels for the frac-solved grid: the rotated y-label sets the
# minimum panel height, so the abbreviation is what lets the grid shrink.
ROW_LABELS_SHORT = {'test': 'In-dist.', 'ood': 'Out-of-dist.'}


def _shared_legend(fig, axes_iter):
    """One de-duplicated legend below the panels, as the paper's figures do.

    Per-axes legends eat plot area at this figure size; the paper puts a single
    row underneath and lets bbox_inches='tight' crop to it.
    """
    handles, labels, seen = [], [], set()
    for ax in axes_iter:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                handles.append(h)
                labels.append(lab)
                seen.add(lab)
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.05), ncol=len(handles), frameon=True)


def make_losses_figure(scheds, sets, alg, out_dir):
    """1x2 mean final-iterate loss vs K, with [q10, q90] shading.

    Geometry and rcParams come from experiment_plots_icml/quad/losses.py --
    figsize (7, 3.5), that rc_context block, markersize 5, and a single shared
    legend under both panels. The point of the 7-inch width is that the paper
    includes these at \\textwidth (~7 in), so a 12-inch figure gets scaled DOWN
    by ~1.7x and its labels land that much smaller than every other figure in
    the paper. Authoring at final size keeps the fonts honest.
    """
    _cfg, disp = ALG_CFG[alg]
    log_floor = 1e-30           # q10 can underflow to 0 on a log axis
    rows = []
    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharex=True)
        for ax, (split, title) in zip(axes, PANELS):
            ds = sets[split]
            for arch in ARCH_ORDER:
                if arch not in scheds:
                    continue
                Ks, mean, q10, q90 = [], [], [], []
                for K in sorted(scheds[arch]):
                    t, beta, _ = scheds[arch][K]
                    lo = per_problem_losses(t, beta, ds)
                    Ks.append(K)
                    mean.append(lo.mean())
                    q10.append(np.quantile(lo, 0.1))
                    q90.append(np.quantile(lo, 0.9))
                    rows.append({'split': split, 'arch': ARCH_DISPLAY_NAMES[arch],
                                 'K': K, 'mean': lo.mean(),
                                 'q10': q10[-1], 'q90': q90[-1],
                                 'median': np.quantile(lo, 0.5)})
                ax.plot(Ks, mean, color=ARCH_COLORS[arch],
                        marker=ARCH_MARKERS[arch], markersize=5,
                        label=ARCH_DISPLAY_NAMES[arch])
                ax.fill_between(Ks, np.maximum(q10, log_floor), q90,
                                color=ARCH_COLORS[arch], alpha=0.2, linewidth=0)
            ax.set_yscale('log')
            ax.set_xlabel(r'$K$')
            ax.set_title(title)
            ax.set_xticks(X_TICKS)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(r'Avg. $f(x^K) - f(x^\star)$')
        # No suptitle: quad/lasso/pdlp losses.py label only the panels
        # ('In-distribution' / 'Out-of-distribution') and leave the experiment
        # name to the paper caption. Plain tight_layout, as they use.
        fig.tight_layout()
        _shared_legend(fig, axes.flat)
        pdf = out_dir / f'logreg_{alg}_losses.pdf'
        fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / f'logreg_{alg}_losses.csv', index=False)
    print(f'  wrote {pdf.name} + .csv')


def make_frac_solved_figure(scheds, sets, alg, out_dir):
    """2 x len(SOLVED_ETAS): fraction of instances with loss <= eta * (1+|f*|).

    Same geometry as the paper's frac-solved grids
    (experiment_plots_icml/quad/create_paper_plots.py): figsize (7, 3.1),
    axes.labelsize 10, shared legend below.
    """
    _cfg, disp = ALG_CFG[alg]
    rows = []
    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }):
        fig, axes = plt.subplots(2, len(SOLVED_ETAS), figsize=(7, 2.5),
                                 sharex=True, sharey=True)
        for r, (split, title) in enumerate(PANELS):
            ds = sets[split]
            for c, eta in enumerate(SOLVED_ETAS):
                ax = axes[r, c]
                thr = solved_threshold(ds, eta)
                for arch in ARCH_ORDER:
                    if arch not in scheds:
                        continue
                    Ks, frac = [], []
                    for K in sorted(scheds[arch]):
                        t, beta, _ = scheds[arch][K]
                        lo = per_problem_losses(t, beta, ds)
                        Ks.append(K)
                        frac.append(float(np.mean(lo <= thr)))
                        rows.append({'split': split, 'eta': eta,
                                     'arch': ARCH_DISPLAY_NAMES[arch], 'K': K,
                                     'frac_solved': frac[-1]})
                    ax.plot(Ks, frac, color=ARCH_COLORS[arch],
                            marker=ARCH_MARKERS[arch], markersize=5,
                            label=ARCH_DISPLAY_NAMES[arch])
                ax.set_ylim([0, 1.05])
                ax.set_yticks([0, 0.5, 1])
                ax.set_xticks(X_TICKS)
                ax.grid(True, alpha=0.3)
                if r == 0:
                    ax.set_title(rf'$\eta$ = {eta:g}')
                if r == 1:
                    ax.set_xlabel(r'$K$')
                if c == 0:
                    ax.set_ylabel(ROW_LABELS_SHORT[split])
        # Verbatim from the archived quad/lasso/pdlp frac-solved figures --
        # which experiment this is gets said in the paper caption, not here.
        # top=0.85 (not 0.80) also matches them, now that the title is short.
        fig.suptitle('Test set, fraction of problems solved', y=0.995)
        # Explicit layout instead of tight_layout: it leaves the panels as
        # tall as the 2.5 in figure allows once the suptitle, the eta titles,
        # the K-axis labels and the legend below have their room.
        fig.subplots_adjust(**FRAC_LAYOUT)
        handles, labels, seen = [], [], set()
        for ax in axes.flat:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    handles.append(h); labels.append(lab); seen.add(lab)
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, FRAC_LEGEND_Y), ncol=len(handles),
                   frameon=True, fontsize=FRAC_LEGEND_FONTSIZE)
        pdf = out_dir / f'logreg_{alg}_frac_problems_solved.pdf'
        fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        out_dir / f'logreg_{alg}_frac_problems_solved.csv', index=False)
    print(f'  wrote {pdf.name} + .csv')


def make_times_csv(scheds, alg, runs_root, out_dir):
    """mean +/- 2 sigma wall-clock per SGD iteration, as quad/lasso times.py."""
    rows = []
    for arch in ARCH_ORDER:
        for K in K_VALS_TIMES:
            if arch not in scheds or K not in scheds[arch]:
                continue
            csv_rel = scheds[arch][K][2]['csv']
            p = Path(runs_root) / csv_rel
            if not p.is_file():
                continue
            t = pd.read_csv(p, usecols=['iter_time'])['iter_time'].to_numpy()
            if t.size <= WARMUP_ITERS:
                continue
            t = t[WARMUP_ITERS:]
            rows.append({'Framework': ARCH_DISPLAY_NAMES[arch], 'K': K,
                         'Time': f'${t.mean():.3f} \\pm {2*t.std(ddof=1):.3f}$'})
    df = pd.DataFrame(rows)
    path = out_dir / f'logreg_{alg}_times.csv'
    df.to_csv(path, index=False)
    print(f'  wrote {path.name}')
    return df


def build(alg, runs_root=None, data_dir=None, out_dir=None):
    use_paper_style()
    runs_root = Path(runs_root or default_runs_root())
    data_dir = Path(data_dir or default_data_dir())
    out_dir = Path(out_dir or (ICLR_OUT / 'figures'))
    out_dir.mkdir(parents=True, exist_ok=True)
    alg_cfg, disp = ALG_CFG[alg]

    print(f'LogReg {disp} figures')
    print(f'  runs : {runs_root}')
    print(f'  data : {data_dir}')
    sets = load_sets(data_dir)
    scheds = select_all(runs_root, alg_cfg)
    if not scheds:
        raise SystemExit(f'no {disp} runs found under {runs_root}')
    for arch, byK in sorted(scheds.items()):
        src = {K: v[2]['csv'].split('/')[1] for K, v in byK.items()}
        print(f'  {ARCH_DISPLAY_NAMES[arch]:8s} K->run: {src}')

    make_losses_figure(scheds, sets, alg, out_dir)
    make_frac_solved_figure(scheds, sets, alg, out_dir)
    make_times_csv(scheds, alg, runs_root, out_dir)
    print(f'  -> {out_dir}')
