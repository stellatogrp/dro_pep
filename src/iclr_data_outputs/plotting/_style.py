"""Shared figure style for the ICLR plots.

Lifted verbatim from experiment_plots_icml/quad/create_paper_plots.py so the
regenerated figures are visually identical to the paper's.
"""
import matplotlib
import matplotlib.pyplot as plt

# Colour-blind-safe triple used throughout the paper.
ARCH_DISPLAY_NAMES = {'l2o': 'L2O', 'ldro_pep': 'DR-L2O', 'lpep': 'OPT-PEP'}
ARCH_COLORS = {'l2o': '#DC3220', 'ldro_pep': '#005AB5', 'lpep': '#00B32D'}
ARCH_MARKERS = {'l2o': 'o', 'ldro_pep': 's', 'lpep': '^'}
ARCH_ORDER = ['l2o', 'ldro_pep', 'lpep']

BASELINE_COLOR = '#666666'


def use_paper_style():
    """Apply the paper's rcParams, falling back if LaTeX is unavailable.

    HANDOFF.md pins matplotlib==3.10.8 to byte-match the paper: usetex
    tick-label baselines moved in 3.11, so a newer matplotlib shifts labels a
    few points even with identical rcParams. The figures are still correct,
    just not byte-identical.
    """
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 14,
        'figure.figsize': (12, 6),
    })
    try:                      # cheap probe: render one mathtext label
        fig = plt.figure()
        fig.gca().set_xlabel(r'$f(x^K)-f(x^\star)$')
        fig.canvas.draw()
        plt.close(fig)
    except Exception:
        plt.close('all')
        plt.rcParams.update({'text.usetex': False})
        print('note: LaTeX not available; falling back to mathtext '
              '(labels differ slightly from the paper)')
    if matplotlib.__version__ != '3.10.8':
        print(f'note: matplotlib {matplotlib.__version__}; the paper used '
              '3.10.8 (tick-label baselines moved in 3.11)')
