# lasso_intro_repro — exact reconstruction of `lasso_intro.pdf`

Self-contained reconstruction of the intro figure of the DR-L2O paper
(`papers/2026/dr-l2o/figures/lasso_intro.pdf`, uploaded to Overleaf by Vinit
on 2026-05-06). The original plotting script lives in the gitignored
`experiment_plots_icml/` directory on Vinit's machine and is not archived
anywhere; this folder rebuilds the figure from the raw run data on della.

The result is **numerically and geometrically exact**: all 25 text elements
of the PDF match to 0.01 pt, and all 416 curve/band vertices match to
0.005 pt (verified against the original by parsing both PDF content streams).

## Quick start

```bash
uv venv .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python reconstruct_lasso_intro.py   # -> lasso_intro_reconstructed.pdf
```

Rendering needs a LaTeX installation (`text.usetex`, Computer Modern).
The OOD test set (`data/test_sets/ood_set_normal3.npz`) ships pre-generated;
`regenerate_ood_set.py` rebuilds it deterministically if needed (~2 min of
CVXPY/Clarabel solves).

## What the figure shows / how it was made (reverse-engineered)

Two panels, test loss `f(x_K) - f*` of learned ISTA stepsize schedules on
LASSO (m=250, n=500, lambda=0.4, A columns unit-norm, seed 1000), K = 1..15,
mean line with 10th-90th quantile band, three frameworks:

| series  | schedule selection (per K)                                              |
|---------|-------------------------------------------------------------------------|
| L2O     | min-validation-loss SGD iterate across the eta sweep (N=1000 runs)       |
| DR-L2O  | eps = 10 **fixed** (not cross-validated), eta = 1e-3, min-validation row |
| OPT-PEP | eta = 1e-4 (validation-picked over {1e-5, 1e-4}), min-validation row     |

Statistics conventions (discovered by matching the PDF to 6+ digits):

- **In-distribution panel**: the two test instances on which the L2O
  schedules diverge (indices 111 and 189, loss >= initial loss) are excluded
  for **all** series; mean/q10/q90 over the remaining 248.
- **Out-of-distribution panel**: all 250 instances, no exclusion (that is
  why the L2O mean line and band blow off the top of the axes).
- Quantiles: `np.quantile` default (linear interpolation).

## Data provenance (all pulled from della-stellato, user vranjan)

- `data/runs/{l2o,ldro-pep,lpep}/` — Hydra configs + `progress.csv`
  (per-SGD-iterate training/validation loss and gamma schedule) of the
  final training sweeps with `training_sample_N=1000`:
  `/scratch/gpfs/BSTELLATO/vranjan/learn_dro_pep_out/learn_{l2o,dro,lpep}_outputs/Lasso/2026-05-0{3,4}`.
  (April sweeps used N=100 and do not correspond to the paper.)
- `data/test_sets/` — from
  `/scratch/gpfs/BSTELLATO/vranjan/out_of_sample_out/sample_creation_outputs/Lasso/2026-05-03/19-48-28`:
  `A_in_dist.npz`, in-distribution `test_set.npz` (250 instances,
  seed 20000, f* solved with CVXPY/CLARABEL on the cluster).
- `data/test_sets/ood_set_normal3.npz` — **regenerated locally** by
  `regenerate_ood_set.py`. The archived cluster OOD sets (uniform shift) are
  NOT what the figure uses; the figure's OOD set follows the paper
  (`x ~ N(0, 3^2)`, p_nonzero 0.1, noise std 0.01, seed 30000), produced by a
  newer code path (`x_ood_dist='normal'` in
  `src/learning_experiment_classes/lasso.py`, branch `neurips`) that Vinit
  ran locally without archiving the output. Generation is deterministic
  (JAX PRNG), and the regenerated set matches the original figure to 8
  significant digits.

## Files

- `reconstruct_lasso_intro.py` — selection + ISTA evaluation + statistics +
  plot. Also writes `data/selected_schedules.csv` (which run/row each
  plotted schedule comes from) and `data/reconstruction_vs_original.csv`
  (per-point relative errors vs the original PDF).
- `extract_pdf_curves.py` — parses the original PDF's content streams and
  recovers the plotted mean/q10/q90 values (`data/pdf_extracted.csv`);
  the ground truth used to verify the reconstruction.
- `regenerate_ood_set.py` — deterministic OOD test-set regeneration.

## Figure style (measured from the original PDF)

Page 492.48 x 198.371 pt, axes rects hard-coded in points; usetex Computer
Modern (ticks 10 pt, labels/titles 12 pt, legend 11 pt); colors
L2O `#DC3220`, DR-L2O `#005AB5`, OPT-PEP `#00B32D`; markers o/s/^ size 5,
lines 1.5 pt (zorder 3), bands alpha 0.2, grid alpha 0.3, tick marks drawn
with width 0; matplotlib pinned to 3.10.8 (usetex tick-label baselines moved
in 3.11).

## Caveats

- `data/reconstruction_vs_original.csv` shows max |rel. error| 0.04 %
  (median 0.000 %) on a handful of DR-L2O points; the plotted schedule there
  presumably came from a duplicate seed run in an unarchived state. The
  difference is ~0.005 pt in the figure.
- This folder is intentionally untracked; do not commit it to the dro_pep
  repo.
